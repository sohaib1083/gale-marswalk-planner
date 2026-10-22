# Status and roadmap

Last updated: 22 Oct 2026. Numbers in this file come from runs of the code in this repo.

## Where we are

| Area | Result | How we know |
|---|---|---|
| Geometry from NASA files | 3D points land within **0.1 px** of their pixel; rebuilt points match NASA's XYZ to **0.5 mm**; our computed rover heading (219.4°) equals NASA's location feed | `notebooks/01_photo_to_3d.ipynb`, `docs/learn-3d.md` |
| Depth Anything V2 Small, no training | median **4.6 %** error vs NASA stereo across 12 stops (per stop 3.6–6.7 %), on pixels the scaling never saw | `app/stops/sol*/scene.json` → `summary` |
| Scaling matters | one plain fit: worst photo **75 %** error; best of three fits chosen per photo: **13 %** | sol 1100 report page |
| Fine-tuning on Navcam | trained on 55 photos from 3 distant sites, tested on 57 unseen photos: **5.1 % → 5.2 %** overall (no gain), but **11.2 % → 8.6 % at 15–30 m**; better on 31 of 57 photos. Pretrained stays the default until more training data (300+ photos) shows a real gain | [results/finetune.md](results/finetune.md), `navcam3d/finetune_depth.py` |
| Stops in 3D | **12 stops (sols 1085–1112), 220 Navcam photos, 21,040 m² of ground at 15 cm** | `navcam3d/build_stop.py`, `app/data.json` → `results` |
| Hazards | **446 rocks and ledges ≥ 25 cm** found, **118 taller than 50 cm**; routes go round the tall ones | `navcam3d/align.py` → `find_hazards`, `app/planner.js` |
| Route planner | A* over 2 m cells, slope limit, rock costs, Tobler walking speed | `app/planner.js`, 6 tests |
| Zoom levels | Mars globe (Viking colour) → Gale Crater 3D (USGS heights every 50 m, Curiosity's whole route) → 2 km planning area (1 m heights, 25 cm photo) → stops (15 cm) | live app |

## What is still weak (be honest with judges)

1. **Coverage:** 12 stops (220 photos) along 212 m of Curiosity's driving, out of about 5,000 sols and 580,000 Navcam photos. The pipeline
   scales (one command per list of sols), but every stop costs about 280 MB of downloads.
2. **Terrain type:** no sand / soil / bedrock layer yet. The AI4Mars classifier (the other subteam) should be painted
   onto the close-ups and added to the route cost (sand = slow, risky).
3. **Hazards are geometric only:** rocks and ledges at least 25 cm tall, from 15 cm ground grids. Where only Depth
   Anything saw the ground (no stereo), they are estimates and marked so.
4. **Orbital heights miss small rocks:** 1 m posts can't see rocks under ~2 m except where a close-up exists.
5. **Walking speed and slope limits** are example values, not NASA EVA rules.
6. **Weather and radiation** are what Curiosity measured on a sol, used as a stand-in forecast.

## Production readiness

| Item | State |
|---|---|
| Tests | `pytest -q` (NASA reader, camera model, scaling, hazards, TIFF rebuild) + `node --test app/tests/` (routing) |
| CI | `.github/workflows/ci.yml` runs both on every push, plus an app-data check |
| Hosting | GitHub Pages from `main`, updates on push: https://sohaib1083.github.io/gale-marswalk-planner/app/ |
| Reproducible builds | pinned `requirements.txt`; every data step is a script; downloads resume and retry |
| Failure handling | one bad stop or a dropped NASA listing no longer stops a batch; the app shows plain errors if data fails to load |
| Licence and credits | MIT for code; NASA/USGS public-domain data credited in README and on the page |
| Performance | one shared WebGL context for all four 3D views; frames are drawn only when the view changes; each 3D stop loads only when needed (phones and data-saver browsers ask first, about 3.5 MB per stop) |
| Mobile | proper viewport and standards mode; checked at 390 px wide (no sideways scroll; globe framed for tall screens) |
| Graphics failures | if the browser drops the 3D context (low graphics memory), the view says so and recovers by itself |
| Still to do | one-time token permission so CI can be pushed (`gh auth refresh -h github.com -s workflow`); an offline bundle for the venue Wi-Fi; AI4Mars terrain layer |

## Road to submission (14–15 Nov)

1. Pick the walk to showcase and build every stop along it (`navcam3d/build_stop.py <sols>`).
2. Add the AI4Mars terrain layer from the classifier subteam.
3. Use the fine-tuned depth model only if `results/finetune.md` shows it beats the pretrained one.
4. Record the 30-second video with `tools/record_demo.py`; write the 7 slides from this file.
5. Rules: Space Apps says project work starts Sat 14 Nov, 09:00. Treat this repo as preparation and rebuild the
   submitted version during the event.
