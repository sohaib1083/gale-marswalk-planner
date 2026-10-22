# Fine-tuning Depth Anything V2 Small on Curiosity Navcam

- Trained on 55 stereo photos from sols 520, 1400, 603 (neck + head only, 10 epochs, scale-and-shift-invariant loss).
- Tested on 57 photos from sols 1094, 1098, 1100, which the model never saw.
- Score: median % error against NASA stereo on held-out pixels, after scaling to metres on the other half.

| Model | One plain fit | Fit chosen per photo | 0-4 m | 4-8 m | 8-15 m | 15-30 m | 30-60 m |
|---|---|---|---|---|---|---|---|
| Pretrained (no training) | 5.4% | 5.1% | 2.0% | 3.2% | 6.9% | 11.2% | 28.7% |
| Fine-tuned on Navcam | 5.7% | 5.2% | 1.8% | 3.3% | 7.4% | 8.6% | 29.5% |

Fine-tuned was better on 31 of 57 test photos (fit chosen per photo).
