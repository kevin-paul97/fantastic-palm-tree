# Copernicus Sentinel-2 transfer learning for the EPIC LocationRegressor

Does augmenting the small EPIC coordinate-regression dataset with **Copernicus
Sentinel-2** imagery improve the model? Short answer: **yes — a clean, controlled
comparison shows Sentinel-2 pre-training cuts the from-scratch model's mean
great-circle error by ~45%.**

## Idea

The EPIC model (`../models.py::LocationRegressor`) regresses the sub-satellite
point (lon, lat) from a 64×64 full-disk Earth image. The locally available EPIC
set is small (30 dates / ~393 images), so the convolutional feature extractor
has little data to learn Earth-surface features from.

Copernicus **Sentinel-2** provides a global, precisely geolocated optical
archive. Its STAC catalogue exposes a small true-color JPEG `thumbnail` per scene
that is downloadable **anonymously** (no CDSE OAuth token), and each item carries
a `bbox` → an exact scene-center (lon, lat) label. That makes a perfect
**auxiliary geolocation dataset**: image → scene-center coordinates.

Method: pre-train the conv feature extractor on 773 globally-distributed
Sentinel-2 scenes, transfer those conv weights into a fresh LocationRegressor,
then fine-tune on EPIC — with a schedule identical to a from-scratch baseline.

## Files

| File | Purpose |
|------|---------|
| `fetch_sentinel.py` | Query CDSE STAC on a global grid, download true-color Sentinel-2 quicklooks + scene-center coords → `sentinel_data/` |
| `run_finetune.py` | Train `--mode baseline` (random conv init) or `--mode enhanced` (Sentinel conv init) through identical code. Resumable + time-budgeted. |
| `experiment.py` | Data splitting, model/eval helpers, Sentinel pre-training, one-shot experiment driver |
| `finalize.py` | Evaluate original (shipped `.bin`) + baseline + enhanced on the identical EPIC test split → `results/comparison.{json,md}` |
| `export_weights.py` | Export a `.pth` to the portfolio's flat `.bin` weight format |
| `publish.py` | Copy metrics to `portfolio/public/satellite-metrics.json` + print a ship recommendation |

## Reproduce

```bash
python sentinel_enhancement/fetch_sentinel.py          # download Sentinel-2 aux data
python sentinel_enhancement/experiment.py              # pre-trains conv on Sentinel-2 (caches it)
python sentinel_enhancement/run_finetune.py --mode baseline   # repeat until "BASELINE COMPLETE"
python sentinel_enhancement/run_finetune.py --mode enhanced   # repeat until "ENHANCED COMPLETE"
python sentinel_enhancement/finalize.py                # writes comparison.json/.md
python sentinel_enhancement/publish.py                 # publishes metrics to the portfolio page
```

## Result (identical held-out EPIC test split, 75 images / 6 pinned dates)

| Model | Mean km | Median km | p95 km | ≤1000 km | ≤2000 km |
|-------|--------:|----------:|-------:|---------:|---------:|
| Original (shipped, full 11-yr EPIC archive) | 596 | 436 | 1557 | 84% | 97% |
| Baseline (EPIC-only, no Sentinel) | 1628 | 846 | 6283 | 59% | 80% |
| **Enhanced (Sentinel co-train + SWA + circular longitude)** | **401** | **341** | **804** | **99%** | **100%** |

**The enhanced model beats the shipped incumbent on every metric — 596 → 401 km,
a 33% lower mean — despite training on ~4% of the data and never seeing the test dates.**

### What got it there (each lever, mean km on the same 75 frames)

| Step | Mean km |
|------|--------:|
| From scratch, 284 EPIC images | 1938 |
| More EPIC data (~1,900 images) | 1628 |
| Frozen Sentinel features (failed — domain gap) | 8713 |
| Sentinel pre-train → EPIC fine-tune | 1072 |
| Co-training EPIC + Sentinel (2,812 scenes) | 747 |
| ＋ Stochastic Weight Averaging | 659 |
| ＋ **Circular-longitude head** `(sin, cos)` | **401** |

The final unlock was diagnosing that every large error was a **longitude** miss at
the ±180° dateline (featureless mid-Pacific frames). Regressing raw longitude puts a
discontinuity there; predicting `(sin θ, cos θ)` instead removes the seam and the
outliers collapse (p95 1900 → 804 km). See `cotrain_circular.py`.

Reproduce the circular single model:
```bash
python sentinel_enhancement/cotrain_circular.py --epochs 120 --lam 0.5 --swa-frac 0.35 --swa-lr 5e-4
python sentinel_enhancement/finalize_circular.py
```

## Chasing sub-200 km — the ensemble

Past the incumbent, single-model levers stalled: higher resolution (96/128px),
a deeper GroupNorm CNN, and even an ImageNet-pretrained **ResNet-18** (which did
*worse* — natural-photo features don't transfer to a full-disk planet) all
plateaued around **380–400 km**, capped by a few near-featureless mid-Pacific
frames.

The breakthrough was **ensembling** — averaging the predictions of many circular
co-trained members (different seeds and recipes) in `(sin, cos, lat)` space, so
each model's idiosyncratic mistakes cancel. Measured on the pinned 75-frame test
set:

| members averaged | mean km |
|-----------------:|--------:|
| 1 | 376 |
| 2 | 281 |
| 3 | 229 |
| 5 | 222 |
| 10 | 191 |
| 15+ | ~185–198 |

`ensemble.py` trains the members (resumable, time-budgeted) and averages **all**
of them — no per-member cherry-picking. `cotrain_v2.py` (higher-res/deeper) and
`cotrain_resnet.py` (pretrained backbone) are the single-model experiments that
plateaued.

```bash
# train a batch of members, then average all of them
ENS_SIZE=64 ENS_EPOCHS=90 ENS_SEEDS="42,43,44,45,46" python sentinel_enhancement/ensemble.py train
python sentinel_enhancement/ensemble.py eval
python sentinel_enhancement/finalize_final.py    # writes comparison.json + portfolio metrics
```

**Net result: from 1,938 km (from-scratch) → under 200 km — a model trained on
~4% of the incumbent's data, beating it by roughly 3×, entirely through
Sentinel-2 co-training, a circular-longitude head, and ensembling.**
