"""
Export a LocationRegressor checkpoint to the flat binary format consumed by the
portfolio TS port (epic-cnn.ts):

    [uint32 little-endian count][float32 little-endian x count]

The float order MUST match WEIGHT_LAYOUT in src/lib/epic-cnn.ts:
    c1w c1b c2w c2b c3w c3b f1w f1b f2w f2b
"""

import sys
import struct
from pathlib import Path
import torch

LAYOUT = [
    "conv_layers.0.0.weight", "conv_layers.0.0.bias",
    "conv_layers.1.0.weight", "conv_layers.1.0.bias",
    "conv_layers.2.0.weight", "conv_layers.2.0.bias",
    "fc_layers.0.weight", "fc_layers.0.bias",
    "fc_layers.3.weight", "fc_layers.3.bias",
]
EXPECTED = 403970  # = sum of WEIGHT_LAYOUT in epic-cnn.ts (the "402,178" code comment there is stale)


def export(in_path, out_path):
    sd = torch.load(in_path, map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]

    floats = []
    for key in LAYOUT:
        if key not in sd:
            raise KeyError(f"missing {key} in checkpoint {in_path}")
        floats.append(sd[key].detach().float().flatten().contiguous())
    flat = torch.cat(floats)
    count = flat.numel()
    if count != EXPECTED:
        raise ValueError(f"param count {count} != expected {EXPECTED}")

    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", count))
        f.write(flat.numpy().astype("<f4").tobytes())
    print(f"wrote {out_path} ({count} params, {Path(out_path).stat().st_size} bytes)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python export_weights.py <in.pth> <out.bin>")
        sys.exit(1)
    export(sys.argv[1], sys.argv[2])
