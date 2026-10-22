"""Turn the downloaded window + NASA feeds into the files the planner app loads.

Run after fetch_orbital.py (and after building any stops with navcam3d/build_stop.py):
    python3 orbital/build_app_data.py
Writes app/: map.jpg, terrain.b64.txt, data.json (lists every stop in app/stops/ as a 3D close-up)
"""
import base64
import json
import sys
import math
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CACHE, APP = ROOT / "data" / "cache", ROOT / "app"
APP.mkdir(exist_ok=True)
FEEDS = {
    "waypoints": "https://mars.nasa.gov/mmgis-maps/MSL/Layers/json/MSL_waypoints.json",
    "traverse": "https://mars.nasa.gov/mmgis-maps/MSL/Layers/json/MSL_traverse.json",
    "weather": "https://mars.nasa.gov/rss/api/?feed=weather&category=msl&feedtype=json",
}
GRID = 2  # metres per cell for routing and 3D


def feed(name):
    p = CACHE / f"{name}.json"
    if not p.exists():
        p.write_bytes(urllib.request.urlopen(FEEDS[name], timeout=120).read())
    return json.loads(p.read_text())


win = json.loads((CACHE / "window.json").read_text())
R = win["radius"]
dem1 = np.load(CACHE / "dem_1m.npy")
n_m = dem1.shape[0]


def to_px(lon, lat):
    """lon/lat -> metres from the window's top-left corner (x right = east, y down = south)."""
    x, y = R * math.radians(lon), R * math.radians(lat)
    return x - win["dem"]["x0"], win["dem"]["y0"] - y


def results_summary(closeups):
    """Headline numbers for the page, all measured by this repo's scripts."""
    out = {"stops": len(closeups), "photos": sum(c["photos"] for c in closeups),
           "area_m2": round(sum(c["area_m2"] for c in closeups)),
           "rocks": sum(len(c["hazards"]) for c in closeups),
           "tall_rocks": sum(1 for c in closeups for h in c["hazards"] if h[2] >= 0.5),
           "median_error": float(np.median([c["error"] for c in closeups])) if closeups else None}
    ft = ROOT / "docs" / "results" / "finetune.json"
    if ft.exists():
        f = json.loads(ft.read_text())
        out["finetune"] = {"train_sols": f["train_sols"], "test_sols": f["test_sols"], "train_photos": f["train_photos"],
                           "test_photos": f["test_photos"], "pretrained": f["pretrained"]["chosen"],
                           "finetuned": f["finetuned"]["chosen"], "pretrained_plain": f["pretrained"]["plain"],
                           "finetuned_plain": f["finetuned"]["plain"], "better_on": f["finetuned_better_on"]}
    return out


# ---- terrain grid (2 m), stored as centimetres above the lowest point ----
h, w = (dem1.shape[0] // GRID) * GRID, (dem1.shape[1] // GRID) * GRID
dem = np.nanmean(dem1[:h, :w].reshape(h // GRID, GRID, w // GRID, GRID), axis=(1, 3))
hmin = float(np.nanmin(dem))
cm = np.where(np.isfinite(dem), np.round((dem - hmin) * 100), 65535).astype("<u2")
(APP / "terrain.b64.txt").write_bytes(base64.b64encode(cm.tobytes()))
print(f"terrain {cm.shape} cells of {GRID} m, heights {hmin:.1f} to {np.nanmax(dem):.1f} m, "
      f"{np.isnan(dem).mean():.1%} missing")

# ---- map photo: 1 m per pixel JPEG ----
ortho = Image.open(CACHE / "ortho_50cm.png")
Image.fromarray(np.array(ortho)).resize((w, h), Image.LANCZOS).convert("L").save(APP / "map.jpg", quality=84)

# ---- Curiosity's path and stops inside the window ----
size = win["size_m"]
inside = lambda x, y: 0 <= x < w and 0 <= y < h
stops = []
for f in feed("waypoints")["features"]:
    p = f["properties"]
    x, y = to_px(p["lon"], p["lat"])
    if inside(x, y):
        stops.append({"sol": p["sol"], "site": p["site"], "drive": p["drive"], "x": round(x, 1), "y": round(y, 1),
                      "elev": round(p["elev_geoid"], 2), "yaw": p["yaw"], "km": p["dist_km"]})
path = []
for f in feed("traverse")["features"]:
    geo = f["geometry"]
    lines = [geo["coordinates"]] if geo["type"] == "LineString" else geo["coordinates"]
    for line in lines:
        seg = []
        for lon, lat, *_ in line:
            x, y = to_px(lon, lat)
            if inside(x, y):
                seg.append([round(x, 1), round(y, 1)])
            elif seg:
                path.append(seg); seg = []
        if len(seg) > 1:
            path.append(seg)
print(f"{len(stops)} Curiosity stops and {sum(len(s) for s in path)} path points inside the window "
      f"(sols {min(s['sol'] for s in stops)}-{max(s['sol'] for s in stops)})")

# ---- weather: Curiosity REMS, every sol reported ----
keep = ("sol", "terrestrial_date", "season", "ls", "min_temp", "max_temp", "min_gts_temp", "max_gts_temp",
        "pressure", "atmo_opacity", "sunrise", "sunset", "local_uv_irradiance_index")
weather = {}
for s in feed("weather")["soles"]:
    if s.get("max_temp") not in (None, "--"):
        weather[s["sol"]] = {k: s.get(k) for k in keep}
print(f"weather for {len(weather)} sols")

# ---- every stop already turned into 3D by navcam3d/build_stop.py ----
closeups = []
for sj in sorted((APP / "stops").glob("sol*/scene.json")):
    m = json.loads(sj.read_text())
    st = m.get("stop") or {}
    if "lon" not in st:
        continue
    x, y = to_px(st["lon"], st["lat"])
    if inside(x, y):
        closeups.append({"sol": m["sol"], "folder": f"stops/{sj.parent.name}", "x": round(x, 1), "y": round(y, 1),
                         "elev": st["elev_geoid"], "photos": len(m["frames"]),
                         "error": round(m["summary"]["median_da_absrel"], 4), "area_m2": round(m["map"]["area_m2"]),
                         "model": m.get("depth_model", "Depth Anything V2 Small (pretrained)"),
                         # rocks/ledges in map metres: [x, y, height above ground, 1 if estimated by Depth Anything]
                         "hazards": [[round(x + h["east"], 2), round(y + h["south"], 2), h["height"], int(h["estimated"])]
                                     for h in m.get("hazards", [])]})
print(f"{len(closeups)} stops with 3D close-ups ({sum(len(c['hazards']) for c in closeups)} rocks/ledges):",
      ", ".join(f"sol {c['sol']}" for c in closeups))
data = {
    "window": {"lat": win["lat"], "lon": win["lon"], "size_m": size, "grid_m": GRID, "cols": w // GRID,
               "rows": h // GRID, "height_min": hmin, "map_px_m": 1,
               "source": "USGS Astrogeology MSL Gale merged DEM 1 m v3 and orthophoto 25 cm v3"},
    "stops": stops, "path": path, "weather": weather,
    "closeups": closeups,
    "radiation": {"msv_per_day": 0.64, "source": "Curiosity RAD, Hassler et al., Science 2014"},
    "results": results_summary(closeups),
}
(APP / "data.json").write_text(json.dumps(data, separators=(",", ":")))
print("app files:", ", ".join(f"{p.name} {p.stat().st_size / 1e6:.1f} MB" for p in sorted(APP.iterdir()) if p.is_file()))
