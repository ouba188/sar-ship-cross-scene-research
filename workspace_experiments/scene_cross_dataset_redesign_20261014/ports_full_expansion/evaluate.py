"""Apply the matched public full-readout implementation to held 24-port scenes."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "public_full_20261014"))
from run import evaluate  # noqa: E402

MANIFEST = Path("E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv")


def load(port, fold, root):
    source = root / port.replace(" ", "_")
    path = root / fold / "split.npz"
    feature_rows = np.load(source / "rows.npy")
    local = lambda rows: np.searchsorted(feature_rows, rows)
    with np.load(path) as arrays:
        train = local(arrays["ssl_train"])
        val = local(arrays["source_validation"])
        query = local(arrays["query"])
        support = {(origin, draw): local(arrays["support_" + origin][draw])
                   for origin in ("same", "cross") for draw in range(5)}
    metadata = pd.read_csv(MANIFEST, usecols=["class_id", "port", "mmsi", "product"],
                           dtype={"mmsi": str})
    acquisition = metadata["product"].str.split("_").map(lambda parts: "_".join((parts[0], parts[6], parts[7])))
    source_train = pd.DataFrame({"class_id": metadata.class_id.iloc[feature_rows[train]].to_numpy(int),
                                 "mmsi": metadata.mmsi.iloc[feature_rows[train]].to_numpy(str),
                                 "acquisition": acquisition.iloc[feature_rows[train]].to_numpy(str)})
    source_val = pd.DataFrame({"class_id": metadata.class_id.iloc[feature_rows[val]].to_numpy(int),
                               "mmsi": metadata.mmsi.iloc[feature_rows[val]].to_numpy(str),
                               "acquisition": acquisition.iloc[feature_rows[val]].to_numpy(str)})
    query_frame = pd.DataFrame({"mmsi": metadata.mmsi.iloc[feature_rows[query]].to_numpy(str)})
    assert set(source_train.class_id) == set(source_val.class_id) == set(range(8))
    assert metadata.port.iloc[feature_rows[np.r_[train, val]]].ne(port).all()
    assert metadata.port.iloc[feature_rows[query]].eq(port).all()
    assert not set(source_train.mmsi) & set(metadata.mmsi.iloc[feature_rows[query]])
    assert not set(source_val.mmsi) & set(metadata.mmsi.iloc[feature_rows[query]])
    return dict(dataset="ports", fold=fold, train=train, val=val, query=query, support=support,
                labels=metadata.class_id.iloc[feature_rows].to_numpy(int),
                raw={name: np.load(source / (name + ".npy"), mmap_mode="r")
                     for name in ("G18", "D18")}
                    | {"G50": np.load(source / "zero.npy", mmap_mode="r")},
                c4=np.load(source / "c4.npy", mmap_mode="r"),
                zero=np.load(source / "zero.npy", mmap_mode="r"),
                source_train=source_train, source_val=source_val, query_frame=query_frame,
                rows=feature_rows, port=port, relation_task_cap=16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("port")
    parser.add_argument("--root", type=Path, default=HERE)
    args = parser.parse_args()
    for folder in sorted(args.root.glob(args.port.replace(" ", "_") + "__*/split.npz")):
        print("FOLD", folder.parent.name, flush=True)
        data = load(args.port, folder.parent.name, args.root)
        with threadpool_limits(limits=2):
            rows, detail = evaluate(data, None)
        for row in rows:
            row["port"] = args.port
        pd.DataFrame(rows).to_csv(folder.parent / "full_metrics.csv", index=False)
        (folder.parent / "full_fit.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")
        print(pd.DataFrame(rows).groupby(["origin", "arm"])[["chip_oa", "chip_ba"]].mean().round(2).to_string(), flush=True)


if __name__ == "__main__":
    main()
