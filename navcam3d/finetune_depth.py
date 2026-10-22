"""Fine-tune Depth Anything V2 Small on Curiosity Navcam stereo, and test whether it beats the pretrained model.

    python3 navcam3d/finetune_depth.py --train 520 1400 700 --test 1094 1098 1100

Training data: every stereo photo at the --train stops (download them with fetch_stop.py first). The target is
NASA's stereo depth; pixels without stereo, beyond 60 m, or on the rover are ignored.
Test: stops the model never saw, scored exactly like build_stop.py (fit the depth scaling on half the stereo
pixels, measure % error on the other half), for the pretrained and the fine-tuned model side by side.

Only the neck and head train (2.7 M of 24.8 M parameters); the image encoder stays frozen so a few hundred
Mars photos can't wreck what it learned from millions of Earth photos. The loss compares depth *shape*
(scale-and-shift invariant), because metres come from the stereo scaling anyway.

Writes data/models/da_v2_small_navcam.pt and docs/results/finetune.md (+ .json).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForDepthEstimation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import pds  # noqa: E402
from align import abs_rel, choose_align, fit_align  # noqa: E402
from build_stop import MAX_DEPTH, MODEL, complete, rover_body, to8  # noqa: E402

SIZE = 518
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
BINS = [(0, 4), (4, 8), (8, 15), (15, 30), (30, 60)]


def frames_for(sol):
    raw = ROOT / "data" / "raw" / f"sol{sol}"
    pairs = [(raw / x.name.replace("XYZLF", "RASLF"), x) for x in sorted(raw.glob("NLB_*XYZLF*.LBL"))]
    return [(r, x) for r, x in pairs if complete(r) and complete(x)]


def prepare(ras, xyz_lbl):
    img, _ = pds.read_product(ras)
    xyz, e = pds.read_product(xyz_lbl)
    cam, (R, t) = pds.CAHV(e), pds.rover_to_site(e)
    P_rov = (np.moveaxis(xyz, 0, -1).astype(np.float64) - t) @ R
    _, _, depth = cam.project(P_rov)
    valid = (np.abs(xyz).sum(0) > 0) & ~rover_body(P_rov) & (depth > 0.5) & (depth < MAX_DEPTH)
    depth = np.where(valid, depth, np.nan).astype(np.float32)
    g = np.asarray(Image.fromarray(to8(img)).resize((SIZE, SIZE), Image.BILINEAR), np.float32) / 255
    x = ((np.stack([g] * 3) - MEAN[:, None, None]) / STD[:, None, None]).astype(np.float32)
    inv = np.where(valid, 1 / np.nan_to_num(depth, nan=1), 0).astype(np.float32)
    w = np.asarray(Image.fromarray(valid.astype(np.float32)).resize((SIZE, SIZE), Image.BOX))
    inv_s = np.asarray(Image.fromarray(inv).resize((SIZE, SIZE), Image.BOX)) / np.maximum(w, 1e-6)
    return {"x": x, "inv": inv_s.astype(np.float32), "mask": w > 0.99, "depth": depth, "name": ras.stem}


def load_sets(train_sols, test_sols):
    """Prepared photos are cached, so a restart (e.g. after the shared GPU filled up) starts training at once."""
    cache = ROOT / "data" / "cache" / f"finetune_{'-'.join(map(str, train_sols))}_vs_{'-'.join(map(str, test_sols))}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return list(z["train"]), list(z["test"])
    train = [prepare(*f) for s in train_sols for f in frames_for(s)]
    train = [d for d in train if d["mask"].mean() > 0.05]
    test = [prepare(*f) for s in test_sols for f in frames_for(s)]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, train=np.array(train, dtype=object), test=np.array(test, dtype=object))
    return train, test


def wait_for_gpu(need_gb=2.6):
    """Other jobs share this laptop's 4 GB GPU; wait for a gap instead of crashing."""
    if not torch.cuda.is_available():
        return "cpu"
    while torch.cuda.mem_get_info()[0] / 1e9 < need_gb:
        print(f"  waiting for {need_gb} GB of free GPU memory ({torch.cuda.mem_get_info()[0] / 1e9:.1f} GB free)", flush=True)
        time.sleep(30)
    return "cuda"


def ssi_loss(pred, tgt, mask):
    """Scale-and-shift invariant L1: centre and scale both maps the same way, then compare."""
    loss = 0
    for p, g, m in zip(pred, tgt, mask):
        p, g = p[m], g[m]
        p = (p - p.median()) / (p - p.median()).abs().mean().clamp_min(1e-6)
        g = (g - g.median()) / (g - g.median()).abs().mean().clamp_min(1e-6)
        loss = loss + (p - g).abs().mean()
    return loss / len(pred)


@torch.no_grad()
def evaluate(model, data, dev):
    rng, rows = np.random.default_rng(0), []
    model.eval()
    for d in data:
        out = model(pixel_values=torch.tensor(d["x"])[None].to(dev)).predicted_depth.float()  # fp32: fp16 overflows
        rel = torch.nn.functional.interpolate(out[None], size=d["depth"].shape, mode="bicubic", align_corners=False)[0, 0].cpu().numpy()
        gt = d["depth"]
        valid = np.isfinite(gt)
        if valid.sum() < 5000:
            continue
        ys, xs = np.nonzero(valid)
        order = rng.permutation(len(ys))
        fit = np.zeros_like(valid)
        fit[ys[order[: len(ys) // 2]], xs[order[: len(ys) // 2]]] = True
        test = valid & ~fit
        method, align = choose_align(rel[fit], gt[fit], rng)
        row = {"photo": d["name"], "method": method,
               "plain": abs_rel(fit_align("plain", rel[fit], gt[fit])(rel[test]), gt[test]),
               "chosen": abs_rel(align(rel[test]), gt[test])}
        pred = align(rel)
        for lo, hi in BINS:
            k = test & (gt >= lo) & (gt < hi)
            row[f"{lo}-{hi} m"] = abs_rel(pred[k], gt[k]) if k.sum() > 500 else None
        rows.append(row)
    return rows


def summarise(rows):
    cols = ["plain", "chosen"] + [f"{lo}-{hi} m" for lo, hi in BINS]
    return {c: (float(np.median([r[c] for r in rows if r[c] is not None])) if any(r[c] is not None for r in rows) else None)
            for c in cols} | {"photos": len(rows)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, nargs="+", required=True)
    ap.add_argument("--test", type=int, nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--out", default=str(ROOT / "data" / "models" / "da_v2_small_navcam.pt"))
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    train, test = load_sets(a.train, a.test)
    print(f"{len(train)} training photos from sols {a.train}, {len(test)} test photos from sols {a.test} "
          f"({time.time() - t0:.0f} s to prepare)")
    if not train or not test:
        raise SystemExit("No photos found. Download the stops first: python3 navcam3d/fetch_stop.py <sols>")

    dev = wait_for_gpu() if dev == "cuda" else dev
    base = AutoModelForDepthEstimation.from_pretrained(MODEL).to(dev)
    before = evaluate(base, test, dev)
    del base

    model = AutoModelForDepthEstimation.from_pretrained(MODEL).to(dev)
    for p in model.backbone.parameters():
        p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    X = torch.tensor(np.stack([d["x"] for d in train]))
    Y = torch.tensor(np.stack([d["inv"] for d in train]))
    M = torch.tensor(np.stack([d["mask"] for d in train]))
    history, bad_steps = [], 0
    for ep in range(a.epochs):
        model.train()
        perm, total = torch.randperm(len(X)), 0.0
        for i in range(0, len(X), a.batch):
            b = perm[i:i + a.batch]
            x, y, m = X[b].to(dev), Y[b].to(dev), M[b].to(dev)
            if torch.rand(1) < 0.5:
                x, y, m = x.flip(-1), y.flip(-1), m.flip(-1)
            # full precision: in fp16 Depth Anything's output overflows to inf on this GPU and every step is lost
            pred = model(pixel_values=x).predicted_depth
            loss = ssi_loss(pred.float(), y, m)
            if not torch.isfinite(loss):
                bad_steps += 1
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total += loss.item() * len(b)
        history.append(total / len(X))
        print(f"epoch {ep + 1:2d}/{a.epochs}  training loss {history[-1]:.4f}  ({time.time() - t0:.0f} s)", flush=True)
    if bad_steps or not np.isfinite(history).all():
        raise SystemExit(f"Training went wrong ({bad_steps} steps with a non-finite loss). Not saving a broken model.")
    after = evaluate(model, test, dev)

    sb, sa = summarise(before), summarise(after)
    better = sum(x["chosen"] > y["chosen"] for x, y in zip(before, after))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    trained = {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("backbone.")}
    torch.save({"state_dict": trained, "train_sols": a.train, "test_sols": a.test, "epochs": a.epochs}, out)

    res = {"train_sols": a.train, "test_sols": a.test, "train_photos": len(train), "epochs": a.epochs,
           "history": history, "pretrained": sb, "finetuned": sa, "finetuned_better_on": better,
           "test_photos": len(after), "per_photo": {"pretrained": before, "finetuned": after}}
    docs = ROOT / "docs" / "results"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "finetune.json").write_text(json.dumps(res, indent=1))
    pct = lambda v: "–" if v is None else f"{v * 100:.1f}%"
    cols = ["plain", "chosen"] + [f"{lo}-{hi} m" for lo, hi in BINS]
    md = [f"# Fine-tuning Depth Anything V2 Small on Curiosity Navcam", "",
          f"- Trained on {len(train)} stereo photos from sols {', '.join(map(str, a.train))} "
          f"(neck + head only, {a.epochs} epochs, scale-and-shift-invariant loss).",
          f"- Tested on {len(after)} photos from sols {', '.join(map(str, a.test))}, which the model never saw.",
          "- Score: median % error against NASA stereo on held-out pixels, after scaling to metres on the other half.", "",
          "| Model | One plain fit | Fit chosen per photo | " + " | ".join(cols[2:]) + " |",
          "|---|---|---|" + "---|" * (len(cols) - 2)]
    for name, s in (("Pretrained (no training)", sb), ("Fine-tuned on Navcam", sa)):
        md.append(f"| {name} | " + " | ".join(pct(s[c]) for c in cols) + " |")
    md += ["", f"Fine-tuned was better on {better} of {len(after)} test photos (fit chosen per photo).", ""]
    (docs / "finetune.md").write_text("\n".join(md))
    print("\n".join(md))
    print("weights:", out, f"({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up live in logs
    main()
