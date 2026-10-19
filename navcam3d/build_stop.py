"""Turn Curiosity's Navcam photos at one or more stops into 3D ground models.

    python3 navcam3d/build_stop.py 1100              # one stop (downloads its photos first if needed)
    python3 navcam3d/build_stop.py 1100 1104 1108    # several stops; the AI model loads once

For every photo at the stop:
  1. NASA's XYZ file -> depth (distance in front of the camera) for every pixel stereo could match.
  2. Depth Anything V2 Small (pretrained, no training) -> depth for every pixel, unitless.
  3. Scale it to metres using the stereo pixels in the same photo. Three fits are tried and the one that does best
     on stereo pixels held back for checking is kept. Its error is measured on pixels the fit never saw.
  4. Fill only the pixels stereo missed, then turn every pixel into a 3D point using the camera model and the
     rover's recorded position.
All photos are then merged onto one 15 cm grid (stereo wins where it exists), spikes removed, small gaps filled,
and slope worked out. Output per stop:
  app/stops/sol<N>/scene.json, scene.b64.txt, img/   (loaded by the planner and the stop report page)
  data/models/sol<N>/ground_photo.glb, ground_slope.glb (open in Blender, Windows 3D Viewer, any glTF viewer)
"""
import argparse
import base64
import json
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image
from scipy import ndimage
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import fetch_stop  # noqa: E402
import pds  # noqa: E402

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"  # 25M parameters, Apache-2.0
CELL = 0.25          # slope map resolution, metres
GRID = 0.15          # ground surface resolution, metres
RADIUS = 24.0        # metres around the rover
MAX_DEPTH = 60.0     # stereo beyond this is uncertain (42 cm baseline); also the furthest Depth Anything may fill
SLOPE_CLASSES = [(0, 15, (47, 158, 98)), (15, 25, (226, 160, 40)), (25, 91, (214, 69, 65))]
rng = np.random.default_rng(0)


# ---------- turning Depth Anything's unitless output into metres ----------
def fit_align(method, r, z):
    """plain: 1/z = s*r + t.  weighted: same, scored as % error.  power: z = a * r^k (near + far in one photo)."""
    if method == "plain":
        s, t = np.linalg.lstsq(np.stack([r, np.ones_like(r)], 1), 1 / z, rcond=None)[0]
        return lambda R_: 1 / np.maximum(s * R_ + t, 1e-3)
    if method == "weighted":
        s, t = np.linalg.lstsq(np.stack([r * z, z], 1), np.ones_like(r), rcond=None)[0]
        return lambda R_: 1 / np.maximum(s * R_ + t, 1e-3)
    k, c = np.polyfit(np.log(np.maximum(r, 1e-3)), np.log(z), 1)
    return lambda R_: np.exp(c) * np.maximum(R_, 1e-3) ** k


def choose_align(r, z):
    cut = rng.random(len(r)) < 0.8
    best = min(("plain", "weighted", "power"),
               key=lambda m: np.mean(np.abs(fit_align(m, r[cut], z[cut])(r[~cut]) - z[~cut]) / z[~cut]))
    return best, fit_align(best, r, z)


# ---------- helpers ----------
def complete(lbl):
    img = lbl.with_suffix(".IMG")
    if not (lbl.exists() and img.exists()):
        return False
    e = pds.read_label(lbl)
    return img.stat().st_size == int(pds.get(e, "FILE_RECORDS")) * int(pds.get(e, "RECORD_BYTES"))


def to8(img):
    lo, hi = np.percentile(img[img > 0], [0.5, 99.5]) if (img > 0).any() else (0, 1)
    return (np.clip((img - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8)


def colorize(arr, cmap, vmin, vmax, mask=None):
    rgba = (colormaps[cmap]((arr - vmin) / max(vmax - vmin, 1e-6)) * 255).astype(np.uint8)
    if mask is not None:
        rgba[~mask, 3] = 0
    return Image.fromarray(rgba)


def rover_body(P_rov):
    """Points inside the rover's own footprint and above wheel level (arm, deck, wheels)."""
    return (P_rov[..., 0] > -1.6) & (P_rov[..., 0] < 1.9) & (np.abs(P_rov[..., 1]) < 1.5) & (P_rov[..., 2] < -0.25)


def write_glb(path, verts, rgb, faces, name):
    verts = verts.astype(np.float32)
    rgb = np.round(255 * (rgb / 255.0) ** 2.2)  # glTF vertex colours are linear, not sRGB
    bufs = [verts.tobytes(), rgb.astype(np.uint8).tobytes(), faces.astype(np.uint32).tobytes()]
    pad = [(-len(x)) % 4 for x in bufs]
    offs = np.cumsum([0] + [len(x) + p for x, p in zip(bufs, pad)])
    binary = b"".join(x + b"\0" * p for x, p in zip(bufs, pad))
    gltf = {
        "asset": {"version": "2.0", "generator": "gale-marswalk-planner"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "indices": 2, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0,
                                                "roughnessFactor": 1}, "doubleSided": True}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [{"buffer": 0, "byteOffset": int(offs[i]), "byteLength": len(bufs[i]),
                         "target": 34963 if i == 2 else 34962} for i in range(3)],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(verts), "type": "VEC3",
             "min": verts.min(0).tolist(), "max": verts.max(0).tolist()},
            {"bufferView": 1, "componentType": 5121, "normalized": True, "count": len(verts), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5125, "count": int(faces.size), "type": "SCALAR"}],
    }
    js = json.dumps(gltf).encode()
    js += b" " * ((-len(js)) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(binary)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(binary), 0x004E4942) + binary)


class Depth:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoImageProcessor.from_pretrained(MODEL)
        self.model = AutoModelForDepthEstimation.from_pretrained(MODEL).to(self.device).eval()

    def __call__(self, img8):
        rgb = Image.fromarray(np.stack([img8] * 3, axis=-1))
        with torch.no_grad():
            out = self.model(**self.proc(images=rgb, return_tensors="pt").to(self.device)).predicted_depth
            rel = torch.nn.functional.interpolate(out[None], size=img8.shape, mode="bicubic", align_corners=False)
        return rel[0, 0].cpu().numpy()


# ---------- one photo ----------
def process_frame(depth_model, ras_lbl, xyz_lbl):
    img, _ = pds.read_product(ras_lbl)
    xyz, e = pds.read_product(xyz_lbl)
    cam, (R, t) = pds.CAHV(e), pds.rover_to_site(e)
    P_rov = (np.moveaxis(xyz, 0, -1).astype(np.float64) - t) @ R
    _, _, stereo = cam.project(P_rov)
    valid = (np.abs(xyz).sum(0) > 0) & ~rover_body(P_rov) & (stereo < MAX_DEPTH)
    stereo = np.where(valid, stereo, np.nan)
    img8 = to8(img)
    rel = depth_model(img8)
    rays = cam.rays(*img.shape[::-1])
    n_valid = int(valid.sum())
    m = {"stereo_cover": n_valid / valid.size}
    filled, da_mask = stereo.copy(), np.zeros_like(valid)
    if n_valid > 5000:
        ys, xs = np.nonzero(valid)
        order = rng.permutation(n_valid)
        fit = np.zeros_like(valid)
        fit[ys[order[: n_valid // 2]], xs[order[: n_valid // 2]]] = True
        test = valid & ~fit
        method, align = choose_align(rel[fit], stereo[fit])
        da = align(rel)
        p, g = da[test], stereo[test]
        m["da_absrel"] = float(np.mean(np.abs(p - g) / g))
        m["da_within25"] = float(np.mean(np.maximum(p / g, g / p) < 1.25))
        m["da_absrel_plain"] = float(np.mean(np.abs(fit_align("plain", rel[fit], stereo[fit])(rel[test]) - g) / g))
        m["align"] = method
        cap = min(MAX_DEPTH, 1.25 * np.nanpercentile(stereo, 99))
        P_da = cam.C + rays * (np.nan_to_num(da) / (rays @ cam.A))[..., None]
        da_mask = ~valid & np.isfinite(da) & (da > 1.0) & (da < cap) & ~rover_body(P_da)
        filled = np.where(da_mask, da, filled)
    m["filled_cover"] = float(np.isfinite(filled).mean())
    P_site = (cam.C + rays * (np.nan_to_num(filled) / (rays @ cam.A))[..., None]) @ R.T + t
    east, north, up = P_site[..., 1] - t[1], P_site[..., 0] - t[0], -(P_site[..., 2] - t[2])
    have = np.isfinite(filled) & (np.hypot(east, north) < RADIUS + 2)
    sub = np.flatnonzero(have.ravel())[::3]
    gv = img.ravel()[sub].astype(float)  # one brightness scale for every photo, so merged ground has no seams
    gnorm = np.clip(128 + 42 * (gv - gv.mean()) / max(gv.std(), 1e-6), 0, 255)
    cloud = np.stack([east.ravel()[sub], north.ravel()[sub], up.ravel()[sub], da_mask.ravel()[sub], gnorm], 1)
    A_site = R @ cam.A
    m.update(id=ras_lbl.stem, azimuth=float(np.degrees(np.arctan2(A_site[1], A_site[0])) % 360),
             elevation=float(np.degrees(np.arcsin(-A_site[2]))))
    return m, cloud.astype(np.float32), (img8, stereo, rel, filled, valid), (R, t)


# ---------- one stop ----------
def build(sol, depth_model, fetch=True):
    raw = fetch_stop.RAW / f"sol{sol}"
    pairs = [(raw / x.name.replace("XYZLF", "RASLF"), x) for x in sorted(raw.glob("NLB_*XYZLF*.LBL"))]
    pairs = [(r, x) for r, x in pairs if complete(r) and complete(x)]
    if not pairs and fetch:
        fetch_stop.fetch(sol)
        return build(sol, depth_model, fetch=False)
    if not pairs:
        raise SystemExit(f"sol {sol}: no complete photo + XYZ pairs in {raw}")
    out = ROOT / "app" / "stops" / f"sol{sol}"
    if out.exists():
        shutil.rmtree(out)
    (out / "img").mkdir(parents=True)
    t0, frames, clouds, show = time.time(), [], [], None
    print(f"sol {sol}: {len(pairs)} stereo photos")
    for k, (ras, xyz) in enumerate(pairs):
        m, cloud, views, (R, t) = process_frame(depth_model, ras, xyz)
        frames.append({k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in m.items()})
        clouds.append(cloud)
        Image.fromarray(views[0]).resize((192, 192), Image.LANCZOS).save(out / "img" / f"f{k:02d}.jpg", quality=82)
        score = m["stereo_cover"] * (m["elevation"] < -15) * (m["filled_cover"] - m["stereo_cover"])
        if "da_absrel" in m and (show is None or score > show[0]):
            show = (score, k) + views
        print(f"  [{k + 1:2d}/{len(pairs)}] az {m['azimuth']:5.1f} el {m['elevation']:5.1f}  stereo "
              f"{m['stereo_cover']:.0%} -> filled {m['filled_cover']:.0%}  error {m.get('da_absrel', float('nan')):.1%}")

    # pipeline strip for the best low-looking photo
    if show is None:
        raise SystemExit(f"sol {sol}: no photo had enough stereo to scale Depth Anything")
    _, ks, img8, stereo, rel, filled, valid = show
    vmax, S = float(np.nanpercentile(stereo, 99)), (512, 512)
    Image.fromarray(img8).resize(S, Image.LANCZOS).save(out / "img" / "step1_photo.jpg", quality=88)
    colorize(np.nan_to_num(stereo), "turbo", 0, vmax, valid).resize(S).save(out / "img" / "step2_stereo.png")
    colorize(rel, "turbo_r", rel.min(), rel.max()).resize(S).save(out / "img" / "step3_da_raw.png")
    colorize(np.nan_to_num(filled), "turbo", 0, vmax, np.isfinite(filled)).resize(S).save(out / "img" / "step4_filled.png")

    pts = np.concatenate(clouds)
    st_all = pts[:, 3] == 0

    def grid_mean(cell, cols):
        n = int(2 * RADIUS / cell)
        ix = np.floor((pts[:, 0] + RADIUS) / cell).astype(int)
        iy = np.floor((RADIUS - pts[:, 1]) / cell).astype(int)
        ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
        cid, res = iy[ok] * n + ix[ok], []
        for src in (st_all[ok], ~st_all[ok]):
            cnt = np.bincount(cid[src], minlength=n * n).reshape(n, n)
            res.append((cnt, [np.bincount(cid[src], weights=pts[ok][src, c], minlength=n * n).reshape(n, n)
                              / np.maximum(cnt, 1) for c in cols]))
        (cs, vs), (cd, vd) = res
        merged = [np.where(cs > 0, a, np.where(cd > 0, b, np.nan)) for a, b in zip(vs, vd)]
        return n, cs, cd, merged  # stereo wins wherever any photo measured the cell

    # slope map (0.25 m), smoothed over ~0.75 m
    n, cs, cd, (height,) = grid_mean(CELL, [2])
    seen = (cs > 0) | (cd > 0)
    w = ndimage.gaussian_filter(seen.astype(float), 1.5)
    hs = ndimage.gaussian_filter(np.nan_to_num(height), 1.5) / np.maximum(w, 1e-6)
    gy, gx = np.gradient(hs, CELL)
    slope_raw = np.degrees(np.arctan(np.hypot(gx, gy)))
    slope = np.where(ndimage.binary_erosion(seen, iterations=2) & (w > 0.6), slope_raw, np.nan)
    light = np.clip(1 - 0.6 * (gx * 0.7 - gy * 0.7) / np.maximum(np.hypot(gx, gy), 1e-6)
                    * np.sin(np.radians(slope_raw)), 0.35, 1.25)
    hmin, hmax = np.nanpercentile(height, [1, 99])
    hrgb = colormaps["gist_earth"](0.15 + 0.7 * np.clip((hs - hmin) / (hmax - hmin), 0, 1))[..., :3] * light[..., None]
    Image.fromarray((np.dstack([np.clip(hrgb, 0, 1), seen]) * 255).astype(np.uint8)).resize(
        (n * 2, n * 2), Image.NEAREST).save(out / "img" / "map_height.png")
    cls = np.zeros((n, n, 4), np.uint8)
    for lo, hi, col in SLOPE_CLASSES:
        sel = (slope >= lo) & (slope < hi)
        cls[sel, :3], cls[sel, 3] = col, 235
    Image.fromarray(cls).resize((n * 2, n * 2), Image.NEAREST).save(out / "img" / "map_slope.png")
    sv = slope[np.isfinite(slope)]

    # one continuous ground surface (0.15 m)
    ng, cs2, cd2, (gh, gray_g) = grid_mean(GRID, [2, 4])
    src_g = np.where(cs2 > 0, 0, 1)
    data = np.isfinite(gh)
    _, (ny0, nx0) = ndimage.distance_transform_edt(~data, return_indices=True)
    spike = data & (np.abs(gh - ndimage.median_filter(gh[ny0, nx0], size=9)) > 0.6)  # bad estimates, rover parts
    gh[spike], data = np.nan, data & ~spike
    dist, (ny_, nx_) = ndimage.distance_transform_edt(~data, return_indices=True)
    fill = ~data & (dist * GRID <= 0.6)
    gh, gray_g, src_g = (np.where(fill, a[ny_, nx_], a) for a in (gh, gray_g, src_g))
    yy, xx = np.mgrid[0:ng, 0:ng]
    ce, cn = (xx + 0.5) * GRID - RADIUS, RADIUS - (yy + 0.5) * GRID
    ok_g = (data | fill) & (np.hypot(ce, cn) <= RADIUS)
    w_ = ndimage.gaussian_filter(ok_g.astype(float), 0.8)
    gh = ndimage.gaussian_filter(np.where(ok_g, gh, 0), 0.8) / np.maximum(w_, 1e-6)
    gidx = -np.ones((ng, ng), np.int64)
    gidx[ok_g] = np.arange(ok_g.sum())
    V = np.stack([ce[ok_g], gh[ok_g], -cn[ok_g]], 1)  # x = east, y = up, z = south (metres from the rover)
    a, b, c, d = gidx[:-1, :-1], gidx[1:, :-1], gidx[:-1, 1:], gidx[1:, 1:]
    tris = np.concatenate([np.stack([a, b, c], -1)[(a >= 0) & (b >= 0) & (c >= 0)],
                           np.stack([b, d, c], -1)[(b >= 0) & (d >= 0) & (c >= 0)]])
    vsrc = src_g[ok_g].astype(np.uint8)
    vgray = np.clip(np.nan_to_num(gray_g[ok_g], nan=128), 0, 255).astype(np.uint8)
    sl = slope[np.clip(np.floor((RADIUS - cn[ok_g]) / CELL).astype(int), 0, n - 1),
               np.clip(np.floor((ce[ok_g] + RADIUS) / CELL).astype(int), 0, n - 1)]
    vslope = np.where(np.isfinite(sl), np.clip(np.nan_to_num(sl), 0, 90), 255).astype(np.uint8)
    from_da = vsrc[tris].max(1) > 0
    tris = np.concatenate([tris[~from_da], tris[from_da]]).astype(np.uint32)
    n_st, n_da = int((~from_da).sum()), int(from_da.sum())

    blob = b"".join([np.round(V * 100).astype(np.int16).tobytes(), vgray.tobytes(), vsrc.tobytes(),
                     vslope.tobytes(), tris.tobytes()])
    (out / "scene.b64.txt").write_bytes(base64.b64encode(blob))
    models = ROOT / "data" / "models" / f"sol{sol}"
    models.mkdir(parents=True, exist_ok=True)
    write_glb(models / "ground_photo.glb", V, np.repeat(vgray[:, None], 3, 1), tris, f"sol_{sol}_ground")
    slope_rgb = np.full((len(V), 3), 125, np.uint8)
    for lo, hi, col in SLOPE_CLASSES:
        slope_rgb[(vslope >= lo) & (vslope < hi)] = col
    write_glb(models / "ground_slope.glb", V, slope_rgb, tris, f"sol_{sol}_slope")

    scored = [f for f in frames if "da_absrel" in f]
    stop = json.loads((raw / "stop.json").read_text()) if (raw / "stop.json").exists() else {}
    meta = {
        "sol": sol, "time": pds.get(pds.read_label(pairs[0][1]), "START_TIME"),
        "site_drive": pairs[0][1].stem[18:25], "stop": stop,
        "rover_heading": float(np.degrees(np.arctan2(R[1, 0], R[0, 0])) % 360),  # forward, degrees from north
        "frames": frames, "show_frame": ks,
        "counts": {"vertices": int(len(V)), "faces": int(len(tris)), "faces_stereo": n_st},
        "layout": ["pos:int16x3 cm", "gray:uint8", "src:uint8", "slope:uint8", "index:uint32x3"],
        "frame_face_ranges": [[0, n_st, n_da]],
        "map": {"cell": CELL, "radius": RADIUS, "height_min": float(hmin), "height_max": float(hmax),
                "area_m2": float(seen.sum() * CELL * CELL),
                "slope_share": {"under15": float(np.mean(sv < 15)), "15to25": float(np.mean((sv >= 15) & (sv < 25))),
                                "over25": float(np.mean(sv >= 25))},
                "da_only_share": float(np.mean(cs[seen] == 0))},
        "summary": {
            "median_da_absrel": float(np.median([f["da_absrel"] for f in scored])),
            "median_da_absrel_plain": float(np.median([f["da_absrel_plain"] for f in scored])),
            "worst_da_absrel": float(np.max([f["da_absrel"] for f in scored])),
            "worst_da_absrel_plain": float(np.max([f["da_absrel_plain"] for f in scored])),
            "mean_stereo_cover": float(np.mean([f["stereo_cover"] for f in frames])),
            "mean_filled_cover": float(np.mean([f["filled_cover"] for f in frames])),
            "frames_with_stereo": len(scored),
        },
    }
    (out / "scene.json").write_text(json.dumps(meta, indent=1))
    shutil.copy(HERE / "stop_report.html", out / "index.html")
    print(f"sol {sol}: {len(V):,} ground points, {len(tris):,} triangles, median error "
          f"{meta['summary']['median_da_absrel']:.1%}, {meta['map']['area_m2']:.0f} m² mapped "
          f"({time.time() - t0:.0f} s) -> {out.relative_to(ROOT)}")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sols", type=int, nargs="+")
    ap.add_argument("--no-fetch", action="store_true", help="only use photos already in data/raw/")
    a = ap.parse_args()
    depth_model = Depth()
    for sol in a.sols:
        build(sol, depth_model, fetch=not a.no_fetch)


if __name__ == "__main__":
    main()
