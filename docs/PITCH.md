# 7-slide outline (Space Apps accepts 7 slides or a 30-second video)

Challenge: *Interplanetary Survival Guide: Martian Map*. Every number below comes from this repo's scripts.

## 1. The problem
The first people on Mars will walk. Orbital maps show where to go but miss what is underfoot: a 1 m height map
can't see a 40 cm rock or a ledge. An astronaut needs a route that is safe at their scale, plus the
conditions on the day.

## 2. What we built
**Gale Marswalk Planner:** one app, four zoom levels, all real NASA data.
- **Mars globe** (Viking colour).
- **Gale Crater in 3D**, with Curiosity's whole route since 2012.
- **A 2 km planning area**: 1 m heights and a 25 cm HiRISE photo.
- **Ground close-ups** rebuilt from Curiosity's own photos at 15 cm.

Live: https://sohaib1083.github.io/gale-marswalk-planner/app/

## 3. Plan a walk
Tap a start and an end. The planner returns the quickest route that:
- never crosses ground steeper than the limit you set, and
- goes round rocks taller than 50 cm.

You get distance, time on foot (Tobler's rule, 3 km/h on flat ground), climb, steepest step, a height profile,
a sunset check, Curiosity's measured weather for the sol and the radiation dose (RAD, 0.64 mSv/sol).

## 4. Rover photos into 3D ground
For each Navcam photo, NASA's stereo XYZ file gives true depth where both cameras matched. Depth Anything V2
fills the rest, scaled to metres on the stereo pixels in the same photo. Results:
- **12 stops**, **220 photos**, **21,040 m²** of ground.
- **4.6 %** median depth error on pixels the scaling never saw.

Geometry checks against NASA's own numbers: points reproject to within 0.1 px; our rover heading matches
NASA's feed to 0.1°.

## 5. Hazards the orbit can't see
**446 rocks and ledges** found in the close-ups, **118 of them taller than 50 cm**. Routes count the small ones
they step past and go round the tall ones. Slope classes are under 15° easy, 15–25° careful, over 25° no go.

## 6. Did training help?
We fine-tuned Depth Anything on 55 Navcam stereo photos from three distant parts of Gale, then tested it on
57 photos from stops it never saw.
- **Overall:** no gain (5.1 % → 5.2 %).
- **15–30 m:** error fell from 11.2 % to 8.6 %, exactly where the pretrained model struggled.

We report this as it is. The pretrained model stays the default until more training data shows a clear win.
The bigger gain came from scaling: choosing how to scale depth per photo cut the worst photo's error from
75 % to 13 %.

## 7. What's next
- The AI4Mars terrain classifier (sand / soil / bedrock / big rock) as a route cost.
- Close-ups along any route on demand (the pipeline already takes a list of sols).
- Real EVA limits from NASA's human-factors guidance.
- An offline bundle for field use.
