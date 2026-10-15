import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from generic_transfer_data import DOMAINS, load_generic_transfer


def write_labels(path, labels, domains, names):
    table = pa.table({"label": labels, "domain": domains})
    metadata = {b"huggingface": json.dumps({"info": {"features": {
        "label": {"names": names, "_type": "ClassLabel"},
    }}}).encode()}
    pq.write_table(table.replace_schema_metadata(metadata), path)


@pytest.fixture(params=["pacs", "officehome"])
def cached_dataset(tmp_path, request):
    dataset = request.param
    n_classes = 7 if dataset == "pacs" else 65
    names = [f"class_{c}" for c in range(n_classes)]
    # Reverse class/domain order so sorting rows or factorizing labels breaks pairing.
    labels = np.tile(np.arange(n_classes)[::-1], 4)
    domains = np.repeat(DOMAINS[dataset][::-1], n_classes)
    n = len(labels)
    if dataset == "pacs":
        write_labels(tmp_path / "train-00000-of-00001.parquet", labels, domains, names)
    else:
        chunks = np.array_split(np.arange(n), 3)
        for i in (2, 0, 1):
            rows = chunks[i]
            write_labels(tmp_path / f"train-{i:05d}-of-00003.parquet",
                         labels[rows], domains[rows], names)
    base = np.zeros((n, 2048), dtype=np.float32)
    base[:, 0] = np.arange(n)
    np.save(tmp_path / ("featsA.npy" if dataset == "pacs" else "featsA_officehome.npy"), base)
    for fold, domain in enumerate(DOMAINS[dataset]):
        extra = np.zeros((n, 512), dtype=np.float32)
        extra[:, 0] = np.arange(n) + 1000 * (fold + 1)
        filename = (f"featsB_{domain}.npy" if dataset == "pacs"
                    else f"featsB_officehome_{domain}.npy")
        np.save(tmp_path / filename, extra)
    return dataset, tmp_path, labels, domains, names


def test_preserves_pairing_labels_and_selects_outer_specific_b(cached_dataset, monkeypatch):
    dataset, root, labels, domains, names = cached_dataset
    read_parquet = pd.read_parquet

    def labels_only(path, **kwargs):
        assert kwargs["columns"] == ["label", "domain"]
        return read_parquet(path, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", labels_only)
    for fold, target in enumerate(DOMAINS[dataset]):
        data = load_generic_transfer(dataset, target, data_dir=root, feature_dir=root)
        np.testing.assert_array_equal(data["labels"], labels)
        np.testing.assert_array_equal(data["domains"], domains)
        np.testing.assert_array_equal(data["base_features"][:, 0], np.arange(len(labels)))
        np.testing.assert_array_equal(data["extra_features"][:, 0],
                                      np.arange(len(labels)) + 1000 * (fold + 1))
        assert data["class_names"] == tuple(names)
        assert data["n_classes"] == len(names)
        np.testing.assert_array_equal(data["source_rows"], np.flatnonzero(domains != target))
        np.testing.assert_array_equal(data["target_rows"], np.flatnonzero(domains == target))
        np.testing.assert_array_equal(data["row_ids"], np.arange(len(labels)))
        assert not data["base_features"].flags.writeable
        assert not data["extra_features"].flags.writeable
        provenance = data["provenance"]
        assert not provenance["inner_encoder_oof"]
        assert provenance["historical_development_dataset"]
        assert target not in provenance["view_b_supervised_domains"]
        assert len(provenance["view_b_supervised_domains"]) == 3
        json.dumps(provenance, allow_nan=False)


@pytest.mark.parametrize("view", ["A", "B"])
def test_rejects_row_mismatch(cached_dataset, view):
    dataset, root, labels, domains, names = cached_dataset
    target = DOMAINS[dataset][0]
    filename = (("featsA.npy" if dataset == "pacs" else "featsA_officehome.npy") if view == "A"
                else (f"featsB_{target}.npy" if dataset == "pacs"
                      else f"featsB_officehome_{target}.npy"))
    width = 2048 if view == "A" else 512
    np.save(root / filename, np.zeros((len(labels) - 1, width), dtype=np.float32))
    with pytest.raises(ValueError, match="expected float32 shape"):
        load_generic_transfer(dataset, target, data_dir=root, feature_dir=root)


def test_rejects_inconsistent_officehome_label_mapping(tmp_path):
    names = [f"class_{c}" for c in range(65)]
    write_labels(tmp_path / "train-0.parquet", [0], ["Art"], names)
    write_labels(tmp_path / "train-1.parquet", [0], ["Clipart"], names[::-1])
    with pytest.raises(ValueError, match="different class mappings"):
        load_generic_transfer("officehome", "Art", data_dir=tmp_path, feature_dir=tmp_path)


def test_rejects_unknown_dataset_or_target():
    with pytest.raises(ValueError, match="dataset must"):
        load_generic_transfer("fmow", "Africa")
    with pytest.raises(ValueError, match="target domain"):
        load_generic_transfer("officehome", "Real_World")
