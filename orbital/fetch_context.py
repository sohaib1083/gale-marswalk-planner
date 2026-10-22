"""The zoomed-out levels: the whole of Mars, and the whole of the USGS Gale map, for the globe -> crater -> walk flow.

    python3 orbital/fetch_context.py

Writes app/world/:
  mars_color.jpg      4096 x 2048 global colour map (Viking MDIM 2.1 colour mosaic, NASA Mars Trek tiles)
  gale_color.jpg      Gale Crater colour at ~230 m/pixel (same mosaic, zoom level 7)
  gale_dem.b64.txt    heights every 50 m across the USGS Gale DEM (1 m mosaic sampled every 50th row/column)
  gale_shade.jpg      shaded relief of that grid, tinted with the Viking colours
  context.json        extents, the full Curiosity path, and landing sites for the globe
"""
import base64
import concurrent.futures as cf
import io
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fetch_orbital import DEM_URL, R_MARS, get_range, make_tiff, read_ifd, tag_values  # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "app" / "world"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "data" / "cache"
TREK = "https://trek.nasa.gov/tiles/Mars/EQ/Mars_Viking_MDIM21_ClrMosaic_global_232m/1.0.0//default/default028mm/{z}/{row}/{col}.jpg"
TRAVERSE = "https://mars.nasa.gov/mmgis-maps/MSL/Layers/json/MSL_traverse.json"
STEP = 50  # metres between samples of the 1 m DEM

# landing sites for context on the globe (planetocentric, east longitude -180..180)
SITES = [
    ("Curiosity · Gale Crater", -4.5895, 137.4417, True), ("Perseverance · Jezero Crater", 18.4447, 77.4508, False),
    ("Opportunity · Meridiani Planum", -1.9462, -5.5266, False), ("Spirit · Gusev Crater", -14.5684, 175.4726, False),
    ("InSight · Elysium Planitia", 4.5024, 135.6234, False), ("Zhurong · Utopia Planitia", 25.066, 109.925, False),
    ("Viking 1 · Chryse Planitia", 22.27, -47.95, False), ("Viking 2 · Utopia Planitia", 47.64, 134.29, False),
    ("Pathfinder · Ares Vallis", 19.13, -33.22, False),
]


def tile(z, row, col):
    url = TREK.format(z=z, row=row, col=col)
    for _ in range(5):
        try:
            return Image.open(io.BytesIO(urllib.request.urlopen(url, timeout=60).read())).convert("RGB")
        except Exception:
            continue
    return Image.new("RGB", (256, 256), (120, 80, 60))


def mosaic(z, rows, cols):
    jobs = [(r, c) for r in rows for c in cols]
    with cf.ThreadPoolExecutor(8) as ex:
        tiles = list(ex.map(lambda rc: tile(z, *rc), jobs))
    img = Image.new("RGB", (256 * len(cols), 256 * len(rows)))
    for (r, c), t in zip(jobs, tiles):
        img.paste(t, ((c - cols[0]) * 256, (r - rows[0]) * 256))
    return img


def main():
    # 1. whole planet: level 3 = 16 x 8 tiles of 256 px, -180..180, 90..-90
    mosaic(3, range(8), range(16)).save(OUT / "mars_color.jpg", quality=86)
    print("globe texture: 4096 x 2048")

    # 2. Gale at level 7 (256 x 128 tiles, 1.40625 degrees each), 135.0..140.6 E, -2.8..-8.4
    deg = 360 / 256
    cols = range(int((135.0 + 180) / deg), int((140.6 + 180) / deg) + 1)
    rows = range(int((90 + 2.8) / deg), int((90 + 8.4) / deg) + 1)
    gale = mosaic(7, rows, cols)
    gale_ext = {"west": -180 + cols[0] * deg, "east": -180 + (cols[-1] + 1) * deg,
                "north": 90 - rows[0] * deg, "south": 90 - (rows[-1] + 1) * deg}
    gale.save(OUT / "gale_color.jpg", quality=86)
    print(f"Gale colour: {gale.size}, extent {gale_ext}")

    # 3. the whole USGS Gale DEM, every 50th row and column (one compressed row per strip)
    tags = read_ifd(DEM_URL)
    width, height = tag_values(DEM_URL, tags, 256)[0], tag_values(DEM_URL, tags, 257)[0]
    tie = tag_values(DEM_URL, tags, 33922)
    x0, y0 = tie[3], tie[4]
    rows_wanted = list(range(STEP // 2, height, STEP))
    offs = tag_values(DEM_URL, tags, 273)
    cnts = tag_values(DEM_URL, tags, 279)
    print(f"DEM: {len(rows_wanted)} of {height} rows, {sum(cnts[r] for r in rows_wanted) / 1e6:.0f} MB")
    with cf.ThreadPoolExecutor(8) as ex:
        strips = list(ex.map(lambda r: get_range(DEM_URL, offs[r], offs[r] + cnts[r] - 1), rows_wanted))
    tif = make_tiff(width, len(strips), strips, {258: (3, [32]), 259: (3, [5]), 262: (3, [1]), 277: (3, [1]),
                                                   278: (4, [1]), 284: (3, [1]), 339: (3, [3])})
    full = np.array(Image.open(io.BytesIO(tif)), dtype=np.float32)
    dem = full[:, STEP // 2::STEP]
    dem[dem < -30000] = np.nan
    lo = float(np.nanmin(dem))
    dm = np.where(np.isfinite(dem), np.round((dem - lo) * 10), 65535).astype("<u2")  # decimetres above lowest point
    (OUT / "gale_dem.b64.txt").write_bytes(base64.b64encode(dm.tobytes()))
    print(f"Gale DEM grid {dem.shape} at {STEP} m, {lo:.0f} to {np.nanmax(dem):.0f} m")

    # 4. shaded relief tinted with Viking colour
    h = np.nan_to_num(dem, nan=lo)
    gy, gx = np.gradient(h, STEP)
    az, alt = math.radians(315), math.radians(35)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    shade = np.clip(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect), 0, 1)
    lat0 = math.degrees((y0 - (STEP // 2)) / R_MARS)
    lon0 = math.degrees((x0 + STEP // 2) / R_MARS)
    lats = lat0 - np.arange(dem.shape[0]) * math.degrees(STEP / R_MARS)
    lons = lon0 + np.arange(dem.shape[1]) * math.degrees(STEP / R_MARS)
    g = np.asarray(gale, dtype=np.float32) / 255
    gi = np.clip(((gale_ext["north"] - lats) / (gale_ext["north"] - gale_ext["south"]) * g.shape[0]).astype(int), 0, g.shape[0] - 1)
    gj = np.clip(((lons - gale_ext["west"]) / (gale_ext["east"] - gale_ext["west"]) * g.shape[1]).astype(int), 0, g.shape[1] - 1)
    tint = g[gi][:, gj]
    tint = tint / np.maximum(tint.mean(axis=2, keepdims=True), 1e-3) * 0.62   # keep the hue, drop baked-in shading
    rgb = np.clip(tint * (0.35 + 0.9 * shade[..., None]), 0, 1)
    rgb[~np.isfinite(dem)] = 0.08
    Image.fromarray((rgb * 255).astype(np.uint8)).resize((dem.shape[1] * 2, dem.shape[0] * 2), Image.BICUBIC).save(
        OUT / "gale_shade.jpg", quality=86)

    # 5. Curiosity's whole path (lon, lat, elevation), thinned to ~1 point every 15 m
    trav = json.loads(urllib.request.urlopen(TRAVERSE, timeout=300).read())
    pts = []
    for f in trav["features"]:
        geo = f["geometry"]
        for line in ([geo["coordinates"]] if geo["type"] == "LineString" else geo["coordinates"]):
            for lon, lat, *rest in line:
                if not pts or math.hypot(lon - pts[-1][0], lat - pts[-1][1]) * math.radians(1) * R_MARS > 15:
                    pts.append([round(lon, 6), round(lat, 6), round(rest[0] if rest else 0, 1)])
    print(f"Curiosity path: {len(pts)} points")

    json.dump({
        "globe": {"texture": "world/mars_color.jpg", "source": "Viking MDIM 2.1 colour mosaic via NASA Mars Trek"},
        "gale_color": {"texture": "world/gale_color.jpg", **gale_ext},
        "gale_dem": {"file": "world/gale_dem.b64.txt", "shade": "world/gale_shade.jpg", "rows": int(dem.shape[0]),
                     "cols": int(dem.shape[1]), "step_m": STEP, "height_min": lo, "unit": "dm",
                     "north": float(lats[0]), "west": float(lons[0]), "dlat": math.degrees(STEP / R_MARS),
                     "dlon": math.degrees(STEP / R_MARS),
                     "source": "USGS Astrogeology MSL Gale merged DEM 1 m v3, sampled every 50 m"},
        "path": pts, "sites": [{"name": n, "lat": la, "lon": lo_, "main": m} for n, la, lo_, m in SITES],
    }, open(OUT / "context.json", "w"), separators=(",", ":"))
    print("wrote", ", ".join(f"{p.name} {p.stat().st_size / 1e6:.1f} MB" for p in sorted(OUT.iterdir())))


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up live in logs
    main()
