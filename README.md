# Gale Marswalk Planner

Plan an astronaut's walk on real Mars terrain, and see the ground at Curiosity's stops in 3D.

- **Orbital map + 3D terrain** of a 2 km × 2 km patch of Gale Crater (USGS/HiRISE, 1 m heights, 25 cm photo).
- **Route planner:** tap a start and an end; it finds the quickest walk that stays off steep ground, then gives
  distance, time on foot, climb, steepest step, a height profile, a sunset check and the radiation dose.
- **Conditions:** Curiosity's own weather (REMS) for any sol, and radiation (RAD).
- **Ground close-ups:** Curiosity's Navcam photos at each stop turned into 3D ground. Stereo is used where NASA has
  it, and Depth Anything V2 fills the gaps. Open one stop, or all stops placed where they really are.

Built by the AstroNox team for NASA Space Apps 2026 (Karachi), challenge *Interplanetary Survival Guide:
Martian Map*. This is a **rehearsal build** (Oct 2026). The hackathon rules say project work starts Sat 14 Nov,
09:00, so rebuild during the event rather than submitting this code as is.

## Run it

```
cd app
python3 -m http.server 8000        # open http://localhost:8000
```

The app ships with data for the sol 1100 area, so it works straight away with no downloads.

## Add more stops (this is how you scale to 1000+ photos)

```
pip install -r requirements.txt
python3 navcam3d/build_stop.py 1085 1087 1093 1094 1098 1100   # download + build each stop
python3 orbital/build_app_data.py                              # refresh the app's stop list
```

For each sol, `navcam3d/fetch_stop.py` does the following:
1. Looks up where Curiosity parked in NASA's rover location feed.
2. Lists that sol in NASA's PDS Navcam archive and downloads every left-Navcam stereo photo taken at that stop,
   with its XYZ file. Downloads resume after drops.

`build_stop.py` then turns them into one 3D ground model per stop.

| Photos | Stops | Download | Build time (laptop, 4 GB GPU) |
|---|---|---|---|
| ~20 | 1 | ~280 MB | ~1 min |
| ~200 | 10 | ~3 GB | ~10 min |
| ~1000 | 50 | ~15 GB | ~50 min |

The download is the slow part, at 1–8 MB/s from NASA's server. Run it on a cloud machine (Colab, Kaggle) for
big batches; the GPU work is small. Which stops matter? The ones along the walk you want to show.
Curiosity's stops inside the map window are in `app/data.json` → `stops`.

## Rebuild the orbital map for another place in Gale

```
python3 orbital/fetch_orbital.py --lat -4.6728315 --lon 137.369321 --size 2000   # ~235 MB, ~5 min
python3 orbital/build_app_data.py
```

It reads only the needed bytes out of USGS's 3.9 GB height map and 24 GB photo mosaic (HTTP range requests).

## How the 3D ground is made (one photo)

```
Navcam photo ──► Depth Anything V2 (unitless depth) ──┐
NASA XYZ file ──► stereo depth (metres, with holes) ──┴─► scale DA to metres on stereo pixels ──► fill holes
                                                                                                   │
camera model (CAHV) + rover position from the label ──► every pixel becomes a 3D point ◄────────────┘
                                                                     │
                       all photos at the stop ──► one ground surface on a 15 cm grid ──► slope, mesh, .glb
```

- **Scaling Depth Anything:** three fits are tried per photo (plain, weighted, power law). It keeps the one that
  scores best on stereo pixels held back for checking. Error is then measured on pixels the fit never saw.
- **Result at sol 1100:** median error 5.2% over 19 photos. One plain fit had a worst photo of 75%; the chosen
  fit brings that to 13%.
- **Checks against NASA's own numbers:**
  - XYZ re-projects to within 0.1 px.
  - Rebuilt points match NASA's XYZ to within 0.5 mm.
  - The rover heading we compute (219.4°) equals NASA's location feed.
- **Fine-tuning Depth Anything is optional.** `notebooks/02_finetune_or_not.ipynb` tests whether it helps
  (train on two sites, test on a third). It runs on a free Colab T4.

## Layout

| Path | What |
|---|---|
| `app/` | The planner (single page, three.js). `stops/sol<N>/` holds each stop's 3D model and report page |
| `navcam3d/` | Rover photos → 3D: `fetch_stop.py`, `build_stop.py`, `pds.py` (NASA file reader, camera model), `stop_report.html` |
| `orbital/` | Orbital map: `fetch_orbital.py` (range-reads USGS mosaics), `build_app_data.py` (map, heights, path, weather, stop list) |
| `notebooks/` | `01_photo_to_3d.ipynb` (every step on one photo, outputs saved), `02_finetune_or_not.ipynb` |
| `docs/learn-3d.md` | Beginner course: stereo, XYZ files, camera models, frames, Depth Anything, meshes |
| `data/` | Downloads, caches and `.glb` models (not in git) |

## Data and credits

| Data | Source |
|---|---|
| Navcam photos + XYZ | NASA/JPL-Caltech, MSL Navcam RDRs, PDS Imaging Node (`MSLNAV_1XXX`) |
| Heights 1 m, photo 25 cm | USGS Astrogeology, MSL Gale merged DEM v3 and orthophoto v3 (HiRISE: NASA/JPL/University of Arizona; CTX: NASA/JPL/MSSS; HRSC: ESA/DLR/FU Berlin) |
| Curiosity path and stops | NASA rover location feed (`mars.nasa.gov/mmgis-maps/MSL/Layers/json/`) |
| Weather | Curiosity REMS (Centro de Astrobiología), via `mars.nasa.gov/rss/api/?feed=weather&category=msl` |
| Radiation | Curiosity RAD: 0.64 mSv/day on the surface (Hassler et al., *Science*, 2014) |
| Depth model | Depth Anything V2 Small (Apache-2.0) |

## Limits

- Walking speed (Tobler's rule, 3 km/h on flat) and slope limits are example values, not NASA EVA rules.
- 1 m orbital heights miss rocks under ~2 m. Close-ups show them, but only at stops that have been built.
- Depth Anything fills are estimates, especially far from the rover. Where any photo has stereo, stereo wins.
- Next: the AI4Mars terrain classifier (sand / soil / bedrock / big rock) painted onto the close-ups and route cost.
