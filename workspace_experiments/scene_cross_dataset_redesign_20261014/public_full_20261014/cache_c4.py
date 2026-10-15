"""Encode four right-angle views with the frozen public-fold ImageNet R50."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, "E:/Hermes/paper_tracking/sensitivity/oof_reliability_check")
from general_domain_ssl import backbone, normalize, rgb  # noqa: E402

PIXELS = {
    "oss1": ROOT / "public_expansion" / "oss1_inputs128.npy",
    "oss2": Path("D:/SAR_completion_cache_20261010/oss2_inputs128.npy"),
}


@torch.inference_mode()
def encode(dataset):
    torch.set_num_threads(2)
    model, width = backbone("r50")
    model = model.cuda().eval()
    pixels = np.load(PIXELS[dataset], mmap_mode="r")
    result = np.empty((len(pixels), width), dtype=np.float32)
    zero = np.empty_like(result)
    for start in range(0, len(pixels), 32):
        pair = torch.from_numpy(np.array(pixels[start:start + 32], dtype=np.float32)).cuda() / 255.
        x = normalize(rgb(F.interpolate(pair, (112, 112), mode="bilinear",
                                        align_corners=False, antialias=True)))
        views = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for turn in range(4):
                views.append(model(torch.rot90(x, turn, (-2, -1))).float())
        zero[start:start + len(pair)] = views[0].cpu().numpy()
        result[start:start + len(pair)] = torch.stack(views).mean(0).cpu().numpy()
        if start % 512 == 0:
            print(dataset, start, "/", len(pixels), flush=True)
    np.save(HERE / f"{dataset}_c4.npy", result)
    np.save(HERE / f"{dataset}_zero.npy", zero)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=PIXELS)
    encode(parser.parse_args().dataset)
