"""Port-held splits and identity isolation; metadata never enters query features."""
from collections.abc import Sequence

import numpy as np


RECORD_FIELDS = ("id", "label", "vessel_id", "port", "group_id")


def _validate_records(records):
    seen = set()
    for row in records:
        for name in RECORD_FIELDS:
            if name not in row or row[name] is None or row[name] == "":
                raise ValueError(f"Record needs nonempty {name}")
        if row["id"] in seen:
            raise ValueError(f"Duplicate observation id: {row['id']}")
        seen.add(row["id"])


def _identities(records, indices):
    return ({records[i]["vessel_id"] for i in indices},
            {records[i]["group_id"] for i in indices})


def split_lopo(records: Sequence[dict], target_ports, validation_ports):
    """Return original row indices for train/validation/test and excluded rows.

    Reserve target ports first, then validation ports. Lower-priority rows sharing
    either a vessel or an acquisition group with retained higher-priority rows
    are excluded. The final splits are disjoint on both identities. Selection,
    source teachers and statistics must use ``train`` only; source validation
    must not be used to build the prototype bank.

    ``missing_source_classes`` is the sorted union of validation/test labels
    absent from train. A closed-set runner must reject or explicitly report this
    situation; this function does not silently remove those test examples.
    """
    _validate_records(records)
    targets, validation = set(target_ports), set(validation_ports)
    if not targets or not validation or targets & validation:
        raise ValueError("Use nonempty, disjoint target_ports and validation_ports")
    known_ports = {r["port"] for r in records}
    if not (targets | validation) <= known_ports:
        raise ValueError("Requested held ports are absent from records")

    test = [i for i, row in enumerate(records) if row["port"] in targets]
    test_vessels, test_groups = _identities(records, test)
    valid = [i for i, row in enumerate(records)
             if row["port"] in validation
             and row["vessel_id"] not in test_vessels
             and row["group_id"] not in test_groups]
    valid_vessels, valid_groups = _identities(records, valid)
    train = [i for i, row in enumerate(records)
             if row["port"] not in targets | validation
             and row["vessel_id"] not in test_vessels | valid_vessels
             and row["group_id"] not in test_groups | valid_groups]
    retained = set(train + valid + test)
    train_classes = {records[i]["label"] for i in train}
    missing = {
        name: sorted({records[i]["label"] for i in ids} - train_classes)
        for name, ids in (("validation", valid), ("test", test))
    }
    return {
        "train": np.asarray(train, dtype=np.int64),
        "validation": np.asarray(valid, dtype=np.int64),
        "test": np.asarray(test, dtype=np.int64),
        "excluded": np.asarray([i for i in range(len(records)) if i not in retained], dtype=np.int64),
        "missing_source_classes": sorted(set(missing["validation"] + missing["test"])),
        "missing_by_split": missing,
    }


def legal_reference_mask(query_rows: Sequence[dict], reference_rows: Sequence[dict]):
    """Boolean [query, reference] mask; query labels and port names are unused.

    Query records only need vessel_id and group_id here. Inference on novel,
    unidentified ships can construct its own all-legal mask; do not invent an
    identity shared by all unknown observations.
    """
    return np.asarray([
        [query["vessel_id"] != ref["vessel_id"]
         and query["group_id"] != ref["group_id"] for ref in reference_rows]
        for query in query_rows
    ], dtype=bool).reshape(len(query_rows), len(reference_rows))


def validate_bank_source(bank_records: Sequence[dict], source_records: Sequence[dict]):
    """Reject out-of-source or changed provenance and repeated source vessels."""
    _validate_records(source_records)
    _validate_records(bank_records)
    source = {r["id"]: r for r in source_records}
    vessels = set()
    for row in bank_records:
        original = source.get(row["id"])
        if original is None or any(row[k] != original[k] for k in RECORD_FIELDS):
            raise ValueError(f"Prototype is outside the declared source rows: {row['id']}")
        if row["vessel_id"] in vessels:
            raise ValueError(f"Repeated prototype vessel: {row['vessel_id']}")
        vessels.add(row["vessel_id"])

