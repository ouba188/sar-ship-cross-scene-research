"""Cache all OSS1 chips in manifest order for acquisition-held episodes."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


HERE = Path(__file__).resolve().parent
MANIFEST = Path("E:/Hermes/paper_tracking/sensitivity/opensarship1_transfer_20261007/target_manifest.csv")


def main():
    manifest = pd.read_csv(MANIFEST, usecols=["sample_id", "chip_path_vv", "chip_path_vh"])
    np.save(HERE / "oss1_ids.npy", manifest.sample_id.to_numpy(str))
    pixels = np.lib.format.open_memmap(HERE / "oss1_inputs128.npy", mode="w+",
                                       dtype=np.uint8, shape=(len(manifest), 2, 128, 128))
    for i, row in enumerate(manifest.itertuples(index=False)):
        for channel, path in enumerate((row.chip_path_vv, row.chip_path_vh)):
            with Image.open(path) as image:
                pixels[i, channel] = np.asarray(image.convert("L").resize(
                    (128, 128), Image.Resampling.BILINEAR))
        if (i + 1) % 250 == 0:
            print(f"cached {i + 1}/{len(manifest)}", flush=True)
    pixels.flush()


if __name__ == "__main__":
    main()
