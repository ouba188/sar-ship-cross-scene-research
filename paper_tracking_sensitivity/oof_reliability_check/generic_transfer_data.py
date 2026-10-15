"""Load existing PACS/OfficeHome paired embeddings for one outer LODO fold.

These are two encoders of the same RGB image, not two sensors. View B was
supervised on all three outer source domains: head-only inner LODO is conditional
on that encoder and is NOT encoder-level OOF. No models are trained here.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DOMAINS = {
    "pacs": ("art_painting", "cartoon", "photo", "sketch"),
    "officehome": ("Art", "Clipart", "Product", "Real World"),
}


def load_generic_transfer(dataset, target_domain, *, data_dir=None, feature_dir=None):
    """Return full-row arrays plus source/target indices, without class selection.

    ``data_dir`` and ``feature_dir`` override the dataset-specific directories.
    Feature arrays are read-only memory maps. Labels retain the parquet class
    IDs; class_names maps those IDs to the embedded Hugging Face label names.
    row_ids are positions in the original extraction order, not image identities.
    Callers must keep target query labels out of fitting and gate construction.
    """
    if dataset not in DOMAINS:
        raise ValueError("dataset must be 'pacs' or 'officehome'")
    if target_domain not in DOMAINS[dataset]:
        raise ValueError(f"unknown {dataset} target domain: {target_domain!r}")

    pacs = dataset == "pacs"
    data_dir = Path(data_dir) if data_dir is not None else (
        Path("E:/Hermes/xdomain_data") / ("pacs" if pacs else "office_home"))
    feature_dir = Path(feature_dir) if feature_dir is not None else (
        Path("E:/Hermes/paper_tracking") / ("xdom_pacs" if pacs else "xdom_generic"))
    files = ([data_dir / "train-00000-of-00001.parquet"] if pacs
             else sorted(data_dir.glob("*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet files in {data_dir}")

    frames, class_names = [], None
    for path in files:
        metadata = pq.read_schema(path).metadata
        names = tuple(json.loads(metadata[b"huggingface"])["info"]["features"]["label"]["names"])
        if class_names is not None and names != class_names:
            raise ValueError("parquet shards have different class mappings")
        class_names = names
        frames.append(pd.read_parquet(path, columns=["label", "domain"]))
    frame = pd.concat(frames, ignore_index=True)
    labels = frame.label.to_numpy(dtype=np.int64)
    domains = frame.domain.to_numpy(dtype=str)
    n_classes = 7 if pacs else 65
    if len(class_names) != n_classes or not np.array_equal(np.unique(labels), np.arange(n_classes)):
        raise ValueError(f"{dataset} must retain all {n_classes} original class IDs")
    if tuple(np.unique(domains)) != DOMAINS[dataset]:
        raise ValueError(f"{dataset} must contain its four original domains")

    a_path = feature_dir / ("featsA.npy" if pacs else "featsA_officehome.npy")
    b_path = feature_dir / (f"featsB_{target_domain}.npy" if pacs
                            else f"featsB_officehome_{target_domain}.npy")
    base = np.load(a_path, mmap_mode="r", allow_pickle=False)
    extra = np.load(b_path, mmap_mode="r", allow_pickle=False)
    for path, array, width in ((a_path, base, 2048), (b_path, extra, 512)):
        if array.shape != (len(frame), width) or array.dtype != np.dtype("float32"):
            raise ValueError(f"{path}: expected float32 shape {(len(frame), width)}, "
                             f"got {array.dtype} {array.shape}")

    source_rows = np.flatnonzero(domains != target_domain)
    target_rows = np.flatnonzero(domains == target_domain)
    source_domains = [d for d in DOMAINS[dataset] if d != target_domain]
    extraction_script = feature_dir / ("run_pacs.py" if pacs else "run_xdom_generic.py")
    provenance = {
        "dataset": dataset,
        "dataset_repository": "flwrlabs/pacs" if pacs else "flwrlabs/office-home",
        "parquet_files": [str(p) for p in files],
        "parquet_rows": [len(f) for f in frames],
        "base_feature_file": str(a_path),
        "extra_feature_file": str(b_path),
        "extraction_script": str(extraction_script),
        "row_alignment": "Original parquet row order; OfficeHome concatenates sorted shard filenames. "
                         "Both feature arrays use that order. No sample IDs are stored in the NPY files.",
        "pairing": "Two encoders of the same RGB image; no independent second sensor.",
        "view_a": "Frozen ResNet50, ResNet50_Weights.IMAGENET1K_V1, fc=Identity, 2048 dimensions.",
        "view_b": "ResNet18, ResNet18_Weights.IMAGENET1K_V1, supervised source fine-tuning, "
                  "12 epochs, Adam lr=0.001, batch=128, horizontal flip; fc=Identity, 512 dimensions.",
        "eval_resize": "Resize(224)" if pacs else "Resize((224, 224))",
        "normalization": "ImageNet RGB mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)",
        "target_domain": target_domain,
        "view_b_supervised_domains": source_domains,
        "encoder_access_evidence": "Existing extraction script excludes the named target domain. "
                                   "Feature caches alone do not independently prove training access.",
        "encoder_training_seed": None,
        "inner_encoder_oof": False,
        "inner_oof_scope": "Refitting heads on this B cache is conditional on a source-supervised "
                           "encoder that already saw the inner held-out source labels.",
        "historical_development_dataset": True,
        "prior_usage": "2026-10-05 ridge/pair-key and full-chain evaluations, 20% handoff; "
                       + ("PACS retained 7 classes." if pacs else
                          "OfficeHome first used 65 classes; full-chain used global top-8 classes."),
    }
    return dict(dataset=dataset, target_domain=target_domain,
                base_features=base, extra_features=extra, labels=labels,
                domains=domains, class_names=class_names, n_classes=n_classes,
                row_ids=np.arange(len(frame)), source_rows=source_rows,
                target_rows=target_rows, provenance=provenance)
