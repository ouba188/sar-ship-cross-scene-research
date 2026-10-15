"""Compare existing source caches; no encoder execution or fitted models."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
WORK = ROOT.parent
SOURCE = WORK / "visual_reliable_baseline/artifacts/eight_class_adaptive_20260916"
SENSITIVITY = Path("E:/Hermes/paper_tracking/sensitivity")


def distances(a, b, rows):
    relative, cosine, absolute = [], [], []
    for start in range(0, len(rows), 512):
        ix = rows[start:start + 512]
        x, y = np.asarray(a[ix], float), np.asarray(b[ix], float)
        nx, ny = np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1)
        relative.extend(np.linalg.norm(x-y, axis=1) / np.maximum(ny, 1e-12))
        cosine.extend(np.sum(x*y, axis=1) / np.maximum(nx*ny, 1e-12))
        absolute.extend(np.max(np.abs(x-y), axis=1))
    return np.array(relative), np.array(cosine), np.array(absolute)


def main():
    frame = pd.read_csv(SOURCE / "manifest.csv", keep_default_na=False,
                        usecols=["instance_id", "port", "class_id", "old_cache_index",
                                 "image_relpath_vv", "image_relpath_vh"])
    rows = np.union1d(*[np.load(WORK / "ports_public_data_study_20261012" /
                               f"fold_{fold}/fits/source_rows.npy") for fold in (2, 3)])
    files = {
        "fresh_raw": ROOT / "features/original/raw.npy",
        "legacy_raw": SOURCE / "pooled_features.npy",
        "zero_raw": SENSITIVITY / "c4_base_trial_20261009/pooled_zero.npy",
        "fresh_c4": ROOT / "features/original/c4.npy",
        "legacy_c4": SENSITIVITY / "c4_base_trial_20261009/pooled_c4.npy",
        "fresh_extra": ROOT / "features/original/extra.npy",
        "legacy_extra": SENSITIVITY / "neural_complement_pilot_20261007/frozen_replay/source_extra.npy",
    }
    arrays = {key: np.load(path, mmap_mode="r") for key, path in files.items()}
    pairs = [("fresh_raw", "legacy_raw"), ("fresh_raw", "zero_raw"),
             ("zero_raw", "legacy_raw"), ("fresh_c4", "legacy_c4"),
             ("fresh_extra", "legacy_extra")]
    table = frame.iloc[rows].copy()
    table.insert(0, "row", rows)
    table["origin"] = np.where(table.old_cache_index.to_numpy(int) >= 0,
                                "pooled_from_shared", "new_feature_batch")
    summaries = []
    for left, right in pairs:
        tag = left + "_vs_" + right
        relative, cosine, absolute = distances(arrays[left], arrays[right], rows)
        table[tag + "_relative"] = relative
        table[tag + "_cosine"] = cosine
        for name, mask in [("all", np.ones(len(rows), bool)),
                           ("pooled_from_shared", table.origin.to_numpy() == "pooled_from_shared"),
                           ("new_feature_batch", table.origin.to_numpy() == "new_feature_batch")]:
            r, c, a = relative[mask], cosine[mask], absolute[mask]
            summaries.append(dict(pair=tag, scope=name, rows=int(mask.sum()),
                                  relative_mean=float(r.mean()), relative_max=float(r.max()),
                                  relative_p99=float(np.quantile(r, .99)),
                                  relative_over002=int(np.sum(r > .02)),
                                  cosine_min=float(c.min()), max_absolute=float(a.max())))
    table.to_csv(OUT / "source_provenance_rows.csv", index=False)
    table.nlargest(20, "fresh_raw_vs_legacy_raw_relative").to_csv(
        OUT / "source_provenance_worst20.csv", index=False)
    result = dict(rows=len(rows), summaries=summaries,
                  interpretation="Cache comparisons only; numerical execution cause not isolated.")
    (OUT / "SOURCE_PROVENANCE.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
