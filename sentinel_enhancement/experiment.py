"""
Controlled experiment: does Copernicus Sentinel-2 transfer learning improve the
EPIC LocationRegressor?

Three models, evaluated on an IDENTICAL held-out EPIC test split:

  1. original  - the shipped portfolio checkpoint (models/regressor_final.pth)
  2. baseline  - LocationRegressor trained from scratch on EPIC-train only
  3. enhanced  - LocationRegressor whose conv feature-extractor is pre-trained on
                 Sentinel-2 (image -> scene-center lon/lat), then fine-tuned on
                 the SAME EPIC-train split with the SAME schedule as `baseline`.

`baseline` vs `enhanced` is the clean, apples-to-apples test of the Sentinel
contribution: identical data, epochs, lr, seed - the ONLY difference is whether
the conv weights were initialised from Sentinel pre-training.

Outputs:
  results/comparison.json
  results/comparison.md
  models/regressor_baseline.pth
  models/regressor_enhanced.pth
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from models import LocationRegressor          # noqa: E402
from datasets import CoordinateNormalizer      # noqa: E402

HERE = Path(__file__).resolve().parent
SENT_DIR = HERE / "sentinel_data"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
MODELS = ROOT / "models"

IMAGES_DIR = ROOT / "images"
META_DIR = ROOT / "combined"

IMG_SIZE = 64
SEED = 42

# Global normalizer: model learns to emit raw degrees directly (matches the
# portfolio TS port, which treats the raw output as [lon, lat]).
NORM = CoordinateNormalizer()  # full-globe [-180,180] / [-90,90]


def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.use_deterministic_algorithms(False)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),  # RGB [0,1]
])


# ────────────────────────── datasets ──────────────────────────

class PairDataset(Dataset):
    """Generic (image_path, [lon,lat]) dataset."""
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        path, (lon, lat) = self.pairs[i]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        return TF(img), torch.tensor([lon, lat], dtype=torch.float32)


def load_epic_pairs_by_date():
    by_date = {}
    for date_dir in sorted(IMAGES_DIR.iterdir()):
        if not date_dir.is_dir():
            continue
        meta = META_DIR / f"{date_dir.name}.json"
        if not meta.exists():
            continue
        data = json.loads(meta.read_text())
        coord = {}
        for item in data:
            name = item.get("image")
            c = item.get("centroid_coordinates", {})
            if name and c.get("lat") is not None and c.get("lon") is not None:
                coord[name] = (float(c["lon"]), float(c["lat"]))
        pairs = []
        for png in sorted(date_dir.glob("*.png")):
            if png.stem in coord:
                pairs.append((str(png), coord[png.stem]))
        if pairs:
            by_date[date_dir.name] = pairs
    return by_date


# Test set is PINNED so it stays identical as more EPIC training data is added,
# keeping every model comparable to the shipped model's score on this same set.
FIXED_TEST_DATES = [
    "2026-03-31", "2026-04-05", "2026-04-09",
    "2026-04-14", "2026-04-17", "2026-04-21",
]


def epic_splits():
    """Date-level split with a PINNED test set. All non-test dates go to train,
    with a deterministic 10% held out for validation/model-selection."""
    by_date = load_epic_pairs_by_date()
    all_dates = sorted(by_date.keys())
    test_dates = [d for d in FIXED_TEST_DATES if d in by_date]
    pool = [d for d in all_dates if d not in set(test_dates)]

    rng = random.Random(SEED)
    rng.shuffle(pool)
    n_val = max(1, round(len(pool) * 0.10))
    val_dates = pool[:n_val]
    train_dates = pool[n_val:]

    def flat(ds):
        return [p for d in ds for p in by_date[d]]

    return (flat(train_dates), flat(val_dates), flat(test_dates),
            {"train_dates": sorted(train_dates), "val_dates": sorted(val_dates),
             "test_dates": sorted(test_dates)})


def load_sentinel_pairs():
    labels_f = SENT_DIR / "labels.json"
    if not labels_f.exists():
        return []
    labels = json.loads(labels_f.read_text())
    pairs = []
    for fname, (lon, lat) in labels.items():
        path = SENT_DIR / "images" / fname
        if path.exists():
            pairs.append((str(path), (float(lon), float(lat))))
    return pairs


# ────────────────────────── train / eval ──────────────────────────

def make_model(device):
    m = LocationRegressor(input_channels=3, conv_channels=[64, 128, 256],
                          kernel_size=3, pool_size=4, activation="tanh",
                          hidden_dim=128, output_dim=2, dropout_rate=0.2)
    return m.to(device)


def train(model, train_loader, val_loader, device, epochs, lr, tag,
          step_size=20, gamma=0.5):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
    crit = nn.MSELoss()
    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for ep in range(epochs):
        model.train()
        tot = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(NORM.normalize(out), NORM.normalize(y))
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()

        model.eval()
        vtot = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                vtot += crit(NORM.normalize(out), NORM.normalize(y)).item()
        vavg = vtot / max(1, len(val_loader))
        if vavg < best_val:
            best_val = vavg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep == 0 or (ep + 1) % 10 == 0 or ep == epochs - 1:
            print(f"  [{tag}] epoch {ep+1}/{epochs}  train={tot/max(1,len(train_loader)):.5f}  "
                  f"val={vavg:.5f}  best={best_val:.5f}", flush=True)

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    preds, trues = [], []
    for x, y in test_loader:
        out = model(x.to(device)).cpu()
        preds.append(out)
        trues.append(y)
    preds = torch.cat(preds); trues = torch.cat(trues)
    hav = NORM.compute_haversine_distance(preds, trues)
    deg = NORM.compute_coordinate_error_degrees(preds, trues)
    return {
        "n": int(len(hav)),
        "mean_haversine_km": float(hav.mean()),
        "median_haversine_km": float(hav.median()),
        "p25_haversine_km": float(hav.quantile(0.25)),
        "p75_haversine_km": float(hav.quantile(0.75)),
        "p95_haversine_km": float(hav.quantile(0.95)),
        "mean_error_deg": float(deg.mean()),
        "median_error_deg": float(deg.median()),
        "pct_within_500km": float((hav <= 500).float().mean() * 100),
        "pct_within_1000km": float((hav <= 1000).float().mean() * 100),
        "pct_within_2000km": float((hav <= 2000).float().mean() * 100),
    }


def load_checkpoint_into(model, path, device):
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd)
    return model


# Order must match WEIGHT_LAYOUT in the portfolio's epic-cnn.ts / export_weights.py
_BIN_LAYOUT = [
    ("conv_layers.0.0.weight", (64, 3, 3, 3)), ("conv_layers.0.0.bias", (64,)),
    ("conv_layers.1.0.weight", (128, 64, 3, 3)), ("conv_layers.1.0.bias", (128,)),
    ("conv_layers.2.0.weight", (256, 128, 3, 3)), ("conv_layers.2.0.bias", (256,)),
    ("fc_layers.0.weight", (128, 256)), ("fc_layers.0.bias", (128,)),
    ("fc_layers.3.weight", (2, 128)), ("fc_layers.3.bias", (2,)),
]


def load_bin_into(model, bin_path, device):
    """Load the exact shipped portfolio weights (.bin: [uint32 count][float32...])."""
    import struct
    raw = Path(bin_path).read_bytes()
    count = struct.unpack("<I", raw[:4])[0]
    flat = np.frombuffer(raw, dtype="<f4", offset=4).copy()
    assert flat.size == count, f"{flat.size} != {count}"
    sd = model.state_dict()
    off = 0
    for key, shape in _BIN_LAYOUT:
        n = int(np.prod(shape))
        sd[key] = torch.from_numpy(flat[off:off + n].reshape(shape)).float()
        off += n
    model.load_state_dict(sd)
    return model


def transfer_conv_weights(dst, src_state):
    """Copy only conv_layers.* from a pre-trained state dict into dst."""
    dst_sd = dst.state_dict()
    copied = 0
    for k, v in src_state.items():
        if k.startswith("conv_layers.") and k in dst_sd and dst_sd[k].shape == v.shape:
            dst_sd[k] = v.clone()
            copied += 1
    dst.load_state_dict(dst_sd)
    return copied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epic-epochs", type=int, default=60)
    ap.add_argument("--sentinel-epochs", type=int, default=35)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--finetune-lr", type=float, default=6e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--force", action="store_true", help="retrain even if caches exist")
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}", flush=True)
    set_seed()

    tr, va, te, split_info = epic_splits()
    print(f"EPIC split -> train {len(tr)}  val {len(va)}  test {len(te)}", flush=True)
    print(f"  test dates: {split_info['test_dates']}", flush=True)

    def loader(pairs, shuffle):
        return DataLoader(PairDataset(pairs), batch_size=args.batch_size,
                          shuffle=shuffle, num_workers=0)

    epic_tr = loader(tr, True)
    epic_va = loader(va, False)
    epic_te = loader(te, False)

    results = {"split": split_info, "models": {}}

    # 1) ORIGINAL portfolio model — evaluate the EXACT shipped .bin if present,
    #    else fall back to the regressor_final.pth checkpoint.
    ship_bin = Path(os.environ.get(
        "PORTFOLIO_BIN",
        str(Path.home() / "Development/portfolio/public/models/regressor_weights.bin")))
    orig_path = MODELS / "regressor_final.pth"
    if ship_bin.exists():
        set_seed()
        orig = make_model(device)
        load_bin_into(orig, ship_bin, device)
        results["models"]["original"] = evaluate(orig, epic_te, device)
        results["original_source"] = str(ship_bin)
        print(f"ORIGINAL (shipped .bin): {results['models']['original']['mean_haversine_km']:.0f} km mean", flush=True)
    elif orig_path.exists():
        set_seed()
        orig = make_model(device)
        load_checkpoint_into(orig, orig_path, device)
        results["models"]["original"] = evaluate(orig, epic_te, device)
        results["original_source"] = str(orig_path)
        print(f"ORIGINAL: {results['models']['original']['mean_haversine_km']:.0f} km mean", flush=True)

    # 2) BASELINE (EPIC only, from scratch) — reuse cached checkpoint if present.
    baseline_ckpt = MODELS / "regressor_baseline.pth"
    set_seed()
    baseline = make_model(device)
    if baseline_ckpt.exists() and not args.force:
        print("\n== Loading cached BASELINE ==", flush=True)
        load_checkpoint_into(baseline, baseline_ckpt, device)
    else:
        print("\n== Training BASELINE (EPIC-only, from scratch) ==", flush=True)
        train(baseline, epic_tr, epic_va, device, args.epic_epochs, args.lr, "baseline")
        torch.save({"model_state_dict": baseline.state_dict()}, baseline_ckpt)
    results["models"]["baseline"] = evaluate(baseline, epic_te, device)
    print(f"BASELINE: {results['models']['baseline']['mean_haversine_km']:.0f} km mean", flush=True)

    # 3) ENHANCED (Sentinel-pretrained conv -> EPIC fine-tune)
    sent_pairs = load_sentinel_pairs()
    print(f"\nSentinel aux pairs: {len(sent_pairs)}", flush=True)
    if len(sent_pairs) >= 100:
        pre_ckpt = MODELS / "_sentinel_pretrained.pth"
        set_seed()
        pre = make_model(device)
        if pre_ckpt.exists() and not args.force:
            print("== Loading cached Sentinel-2 pre-trained conv ==", flush=True)
            load_checkpoint_into(pre, pre_ckpt, device)
        else:
            print("== Pre-training conv on Sentinel-2 ==", flush=True)
            rng = random.Random(SEED); rng.shuffle(sent_pairs)
            n_val = max(16, int(len(sent_pairs) * 0.1))
            s_va, s_tr = sent_pairs[:n_val], sent_pairs[n_val:]
            train(pre, loader(s_tr, True), loader(s_va, False), device,
                  args.sentinel_epochs, args.lr, "sentinel-pretrain")
            torch.save({"model_state_dict": pre.state_dict()}, pre_ckpt)
        pre_state = {k: v.detach().cpu().clone() for k, v in pre.state_dict().items()}

        print("== Fine-tuning ENHANCED on EPIC ==", flush=True)
        set_seed()
        enhanced = make_model(device)
        n_copied = transfer_conv_weights(enhanced, pre_state)
        print(f"  transferred {n_copied} conv tensors from Sentinel pre-training", flush=True)
        train(enhanced, epic_tr, epic_va, device, args.epic_epochs, args.finetune_lr, "enhanced")
        results["models"]["enhanced"] = evaluate(enhanced, epic_te, device)
        torch.save({"model_state_dict": enhanced.state_dict()}, MODELS / "regressor_enhanced.pth")
        print(f"ENHANCED: {results['models']['enhanced']['mean_haversine_km']:.0f} km mean", flush=True)
    else:
        print("Not enough Sentinel data yet — skipping enhanced model.", flush=True)

    # ── report ──
    results["config"] = vars(args)
    (RESULTS / "comparison.json").write_text(json.dumps(results, indent=2))
    write_markdown(results)
    print(f"\nWrote {RESULTS/'comparison.json'} and comparison.md", flush=True)


def write_markdown(results):
    m = results["models"]
    order = [k for k in ("original", "baseline", "enhanced") if k in m]
    lines = ["# EPIC coordinate regression — Sentinel-2 transfer-learning experiment", ""]
    lines.append(f"Evaluated on an identical held-out EPIC test split "
                 f"({m[order[0]]['n']} images, {len(results['split']['test_dates'])} dates).")
    lines.append("")
    lines.append("| Model | Mean km | Median km | p95 km | ≤1000 km | ≤2000 km |")
    lines.append("|-------|--------:|----------:|-------:|---------:|---------:|")
    label = {"original": "Original (shipped)", "baseline": "Baseline (EPIC-only)",
             "enhanced": "Enhanced (Sentinel→EPIC)"}
    for k in order:
        r = m[k]
        lines.append(f"| {label[k]} | {r['mean_haversine_km']:.0f} | "
                     f"{r['median_haversine_km']:.0f} | {r['p95_haversine_km']:.0f} | "
                     f"{r['pct_within_1000km']:.0f}% | {r['pct_within_2000km']:.0f}% |")
    lines.append("")
    if "baseline" in m and "enhanced" in m:
        b, e = m["baseline"]["mean_haversine_km"], m["enhanced"]["mean_haversine_km"]
        delta = b - e
        pct = 100 * delta / b if b else 0
        verdict = "improved" if delta > 0 else "did NOT improve"
        lines.append(f"**Clean comparison (identical data/schedule):** Sentinel pre-training "
                     f"{verdict} mean error by **{delta:+.0f} km ({pct:+.1f}%)** "
                     f"(baseline {b:.0f} → enhanced {e:.0f}).")
    (RESULTS / "comparison.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
