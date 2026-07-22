# EPIC coordinate regression — Sentinel-2 transfer-learning experiment

Evaluated on an identical held-out EPIC test split (75 images, 6 dates).

| Model | Mean km | Median km | p95 km | ≤1000 km | ≤2000 km |
|-------|--------:|----------:|-------:|---------:|---------:|
| Original (shipped) | 596 | 436 | 1557 | 84% | 97% |
| Baseline (EPIC-only) | 1628 | 846 | 6283 | 59% | 80% |
| Enhanced (Sentinel→EPIC) | 659 | 416 | 1920 | 80% | 95% |

**Clean comparison (identical data/schedule):** Sentinel pre-training improved mean error by **+969 km (+59.5%)** (baseline 1628 → enhanced 659).