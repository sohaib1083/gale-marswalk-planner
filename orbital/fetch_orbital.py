"""Download a small window of NASA/USGS orbital data around one Curiosity stop, without the huge full files.

Sources (USGS Astrogeology, MSL Gale merged mosaics v3, equirectangular, Mars sphere R = 3,396,190 m):
  DEM   MSL_Gale_DEM_Mosaic_1m_v3.tif          1 m/pixel heights, 3.9 GB, one compressed row per strip
  ORTHO MSL_Gale_Orthophoto_Mosaic_25cm_v3.tif  25 cm/pixel greyscale, 24 GB, 128 x 128 tiles

Both files sit on Amazon S3 and accept byte-range requests, so we read the file directory, work out which
rows/tiles cover our window, download only those bytes, and decode them with Pillow (libtiff).

Run:  python3 fetch_orbital.py [--lat -4.6728315 --lon 137.369321 --size 2000]
Writes data/cache/dem_1m.npy, data/cache/ortho_50cm.png and data/cache/window.json
"""
import argparse
import concurrent.futures as cf
import io
import json
import sys
import math
import struct
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

S3 = "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/Mars/MSL/"
DEM_URL, ORTHO_URL = S3 + "MSL_Gale_DEM_Mosaic_1m_v3.tif", S3 + "MSL_Gale_Orthophoto_Mosaic_25cm_v3.tif"
R_MARS = 3396190.0
CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE.mkdir(exist_ok=True)


def get_range(url, start, end, tries=6):
    """Bytes [start, end] inclusive, retrying slow or dropped connections."""
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            data = urllib.request.urlopen(req, timeout=120).read()
            if len(data) == end - start + 1:
                return data
        except Exception:
            pass
        time.sleep(1 + k)
    raise RuntimeError(f"could not download bytes {start}-{end}")


def file_size(url):
    req = urllib.request.Request(url, method="HEAD")
    return int(urllib.request.urlopen(req, timeout=60).headers["Content-Length"])


# ---------- reading a TIFF directory (classic or BigTIFF) from byte ranges ----------
TYPES = {1: ("B", 1), 2: ("c", 1), 3: ("H", 2), 4: ("I", 4), 12: ("d", 8), 16: ("Q", 8)}


def read_ifd(url):
    head = get_range(url, 0, 15)
    big = struct.unpack("<H", head[2:4])[0] == 43
    off = struct.unpack("<Q", head[8:16])[0] if big else struct.unpack("<I", head[4:8])[0]
    n_fmt, n_sz, e_sz = ("<Q", 8, 20) if big else ("<H", 2, 12)
    n = struct.unpack(n_fmt, get_range(url, off, off + n_sz - 1))[0]
    raw = get_range(url, off + n_sz, off + n_sz + n * e_sz - 1)
    tags = {}
    for i in range(n):
        e = raw[i * e_sz:(i + 1) * e_sz]
        if big:
            tag, typ, cnt = struct.unpack("<HHQ", e[:12]); val = e[12:20]; inline = 8
        else:
            tag, typ, cnt = struct.unpack("<HHI", e[:8]); val = e[8:12]; inline = 4
        fmt, sz = TYPES.get(typ, ("B", 1))
        if cnt * sz <= inline:
            tags[tag] = (fmt, cnt, None, val[:cnt * sz])
        else:
            ptr = struct.unpack("<Q" if big else "<I", val)[0]
            tags[tag] = (fmt, cnt, ptr, None)
    return tags


def tag_values(url, tags, tag, first=0, count=None):
    """Values of one tag; for long arrays fetch only entries [first, first+count)."""
    fmt, cnt, ptr, inline = tags[tag]
    sz = TYPES[[k for k, v in TYPES.items() if v[0] == fmt][0]][1]
    if inline is not None:
        return struct.unpack("<" + fmt * cnt, inline)
    count = cnt if count is None else count
    raw = get_range(url, ptr + first * sz, ptr + (first + count) * sz - 1)
    return struct.unpack("<" + fmt * count, raw)


def make_tiff(width, height, chunks, tags_extra):
    """A minimal little-endian TIFF around already-compressed strips/tiles, for Pillow to decode."""
    data = b"".join(chunks)
    header = 8
    offsets, pos = [], header
    for c in chunks:
        offsets.append(pos); pos += len(c)
    entries = {256: (4, [width]), 257: (4, [height]), **tags_extra}
    is_tiled = 322 in entries
    entries[324 if is_tiled else 273] = (4, offsets)
    entries[325 if is_tiled else 279] = (4, [len(c) for c in chunks])
    extra, ifd_entries = b"", []
    ifd_off = pos
    n = len(entries)
    after = ifd_off + 2 + 12 * n + 4
    for tag in sorted(entries):
        typ, vals = entries[tag]
        fmt = {3: "H", 4: "I"}[typ]
        packed = struct.pack("<" + fmt * len(vals), *vals)
        if len(packed) <= 4:
            ifd_entries.append(struct.pack("<HHI", tag, typ, len(vals)) + packed.ljust(4, b"\0"))
        else:
            ifd_entries.append(struct.pack("<HHII", tag, typ, len(vals), after + len(extra)))
            extra += packed
    ifd = struct.pack("<H", n) + b"".join(ifd_entries) + struct.pack("<I", 0)
    return b"II*\0" + struct.pack("<I", ifd_off) + data + ifd + extra


def parallel(jobs, fn, workers=8, label=""):
    out, t0 = [None] * len(jobs), time.time()
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, j): i for i, j in enumerate(jobs)}
        for k, f in enumerate(cf.as_completed(futs), 1):
            out[futs[f]] = f.result()
            if k % max(1, len(jobs) // 10) == 0:
                print(f"  {label}: {k}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=-4.6728315)   # Curiosity, sol 1100 (site 49, drive 2902)
    ap.add_argument("--lon", type=float, default=137.369321)
    ap.add_argument("--size", type=float, default=2000.0)      # window edge, metres
    a = ap.parse_args()
    X, Y = R_MARS * math.radians(a.lon), R_MARS * math.radians(a.lat)  # equirectangular, centre lon 0
    half = a.size / 2

    # ---- DEM: 1 m, one LZW-compressed row per strip ----
    tags = read_ifd(DEM_URL)
    width = tag_values(DEM_URL, tags, 256)[0]
    sx, sy, _ = tag_values(DEM_URL, tags, 33550)
    tie = tag_values(DEM_URL, tags, 33922)
    x0, y0 = tie[3], tie[4]
    r0, r1 = int((y0 - (Y + half)) / sy), int((y0 - (Y - half)) / sy)
    c0, c1 = int((X - half - x0) / sx), int((X + half - x0) / sx)
    offs = tag_values(DEM_URL, tags, 273, r0, r1 - r0)
    cnts = tag_values(DEM_URL, tags, 279, r0, r1 - r0)
    print(f"DEM rows {r0}-{r1}, cols {c0}-{c1}: downloading {sum(cnts) / 1e6:.0f} MB")
    strips = parallel(list(zip(offs, cnts)), lambda oc: get_range(DEM_URL, oc[0], oc[0] + oc[1] - 1), label="DEM rows")
    tif = make_tiff(width, len(strips), strips, {258: (3, [32]), 259: (3, [5]), 262: (3, [1]), 277: (3, [1]),
                                                   278: (4, [1]), 284: (3, [1]), 339: (3, [3])})
    dem = np.array(Image.open(io.BytesIO(tif)), dtype=np.float32)[:, c0:c1]
    dem[dem < -30000] = np.nan
    np.save(CACHE / "dem_1m.npy", dem)
    print(f"DEM window {dem.shape}, heights {np.nanmin(dem):.1f} to {np.nanmax(dem):.1f} m")

    # ---- ORTHO: 25 cm, PackBits 128 x 128 tiles, palette greyscale ----
    tags = read_ifd(ORTHO_URL)
    W = tag_values(ORTHO_URL, tags, 256)[0]
    TW = tag_values(ORTHO_URL, tags, 322)[0]
    osx, osy, _ = tag_values(ORTHO_URL, tags, 33550)
    otie = tag_values(ORTHO_URL, tags, 33922)
    cmap = np.array(tag_values(ORTHO_URL, tags, 320)).reshape(3, -1)[0] >> 8   # palette: red channel = grey
    ntx = math.ceil(W / TW)
    pr0, pr1 = int((otie[4] - (Y + half)) / osy), int((otie[4] - (Y - half)) / osy)
    pc0, pc1 = int((X - half - otie[3]) / osx), int((X + half - otie[3]) / osx)
    tr0, tr1, tc0, tc1 = pr0 // TW, pr1 // TW + 1, pc0 // TW, pc1 // TW + 1
    rows = []
    for tr in range(tr0, tr1):
        first = tr * ntx + tc0
        rows.append((tag_values(ORTHO_URL, tags, 324, first, tc1 - tc0), tag_values(ORTHO_URL, tags, 325, first, tc1 - tc0)))
    print(f"ORTHO tiles {tr1 - tr0} x {tc1 - tc0}: downloading {sum(sum(c) for _, c in rows) / 1e6:.0f} MB")

    def fetch_row(oc):
        o, c = oc
        lo, hi = min(o), max(x + y for x, y in zip(o, c)) - 1
        blob = get_range(ORTHO_URL, lo, hi)          # one request per tile row (tiles are stored in order)
        return [blob[x - lo:x - lo + y] for x, y in zip(o, c)]

    tile_rows = parallel(rows, fetch_row, label="photo tile rows")
    chunks = [t for row in tile_rows for t in row]
    tif = make_tiff((tc1 - tc0) * TW, (tr1 - tr0) * TW, chunks,
                    {258: (3, [8]), 259: (3, [32773]), 262: (3, [1]), 277: (3, [1]), 284: (3, [1]),
                     322: (3, [TW]), 323: (3, [TW])})
    idx = np.array(Image.open(io.BytesIO(tif)))
    grey = cmap[idx].astype(np.uint8)
    y_off, x_off = pr0 - tr0 * TW, pc0 - tc0 * TW
    grey = grey[y_off:y_off + (pr1 - pr0), x_off:x_off + (pc1 - pc0)]
    small = Image.fromarray(grey).resize((grey.shape[1] // 2, grey.shape[0] // 2), Image.LANCZOS)  # 50 cm/pixel
    small.save(CACHE / "ortho_50cm.png")
    print(f"photo window {grey.shape} at 25 cm -> saved {small.size} at 50 cm")

    json.dump({"lat": a.lat, "lon": a.lon, "size_m": a.size, "center_xy": [X, Y],
               "dem": {"x0": x0 + c0 * sx, "y0": y0 - r0 * sy, "px": sx},
               "ortho": {"x0": otie[3] + pc0 * osx, "y0": otie[4] - pr0 * osy, "px": osx * 2},
               "radius": R_MARS, "projection": "equirectangular, centre longitude 0, Mars 2000 sphere"},
              open(CACHE / "window.json", "w"), indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up live in logs
    main()
