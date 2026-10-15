"""Encode one held-port source pool and its two target scenes with matched encoders."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, "E:/Hermes/paper_tracking/sensitivity/oof_reliability_check")
from general_domain_ssl import backbone, normalize, rgb  # noqa: E402

PIXELS = Path("D:/SAR_completion_cache_20261010/source_inputs128.npy")


def required_rows(port, root):
    all_rows = []
    for path in root.glob(port.replace(" ", "_") + "__*/split.npz"):
        with np.load(path) as saved:
            all_rows.extend([saved[key].ravel() for key in
                             ("source_fit", "query", "support_same", "support_cross")])
    assert all_rows and len(all_rows) % 4 == 0
    return np.unique(np.concatenate(all_rows))


@torch.inference_mode()
def encode_r50(model, pixels, rows, output):
    zero = np.lib.format.open_memmap(output / "zero.npy", mode="w+", dtype="float32", shape=(len(rows), 2048))
    c4 = np.lib.format.open_memmap(output / "c4.npy", mode="w+", dtype="float32", shape=(len(rows), 2048))
    for start in range(0, len(rows), 32):
        batch = rows[start:start + 32]
        pair = torch.from_numpy(np.array(pixels[batch], dtype=np.float32)).cuda() / 255.
        x = normalize(rgb(F.interpolate(pair, (112, 112), mode="bilinear",
                                        align_corners=False, antialias=True)))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            views = [model(torch.rot90(x, turn, (-2, -1))).float() for turn in range(4)]
        zero[start:start + len(batch)] = views[0].cpu().numpy()
        c4[start:start + len(batch)] = torch.stack(views).mean(0).cpu().numpy()
        if start % 1024 == 0:
            print(output.name, "R50", start, "/", len(rows), flush=True)
    zero.flush()
    c4.flush()


@torch.inference_mode()
def encode_r18(model, pixels, rows, output, name):
    result = np.lib.format.open_memmap(output / (name + ".npy"), mode="w+", dtype="float32",
                                       shape=(len(rows), 512))
    for start in range(0, len(rows), 64):
        batch = rows[start:start + 64]
        pair = torch.from_numpy(np.array(pixels[batch], dtype=np.float32)).cuda() / 255.
        x = normalize(rgb(F.interpolate(pair, (112, 112), mode="bilinear",
                                        align_corners=False, antialias=True)))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            value = model(x).float()
        result[start:start + len(batch)] = value.cpu().numpy()
        if start % 2048 == 0:
            print(output.name, name, start, "/", len(rows), flush=True)
    result.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("port")
    parser.add_argument("--root", type=Path, default=HERE)
    args = parser.parse_args()
    output = args.root / args.port.replace(" ", "_")
    output.mkdir(exist_ok=True)
    rows = required_rows(args.port, args.root)
    np.save(output / "rows.npy", rows)
    pixels = np.load(PIXELS, mmap_mode="r")
    torch.set_num_threads(2)
    r50, _ = backbone("r50")
    encode_r50(r50.cuda().eval(), pixels, rows, output)
    del r50
    torch.cuda.empty_cache()
    r18, _ = backbone("r18")
    encode_r18(r18.cuda().eval(), pixels, rows, output, "G18")
    checkpoint = torch.load(output / "source_only_r18_best.pt", map_location="cpu", weights_only=False)
    r18.load_state_dict(checkpoint["backbone"], strict=True)
    encode_r18(r18.cuda().eval(), pixels, rows, output, "D18")


if __name__ == "__main__":
    main()
