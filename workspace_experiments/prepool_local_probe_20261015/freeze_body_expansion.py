"""Freeze acquisition expansion and nested matched support budgets before scoring."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
OUT = HERE / "body_context_expansion"
CONTEXT = PROJECT / "context_condition_study_20261015"
REGION = PROJECT / "region_cue_replication_20261013"
sys.path.insert(0, str(CONTEXT))
from audit_ports_context import chip_overlap

SEED = 20261016


def counts(frame):
    return frame.class_id.value_counts().reindex(range(8), fill_value=0).to_numpy(int)


def tokens(capacity, classes, limit=64):
    used = np.zeros(8, int)
    result = []
    while len(result) < limit:
        available = [c for c in classes if used[c] < capacity[c]]
        if not available:
            break
        c = min(available, key=lambda c: (used[c], -capacity[c], c))
        result.append(c)
        used[c] += 1
    return result


def remove_overlap(pool, query, coords):
    qpos = coords.loc[query.sample_id]
    return pool.loc[[not any(chip_overlap(coords.loc[sid], q) for _, q in qpos.iterrows())
                     for sid in pool.sample_id]]


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT / "frozen.json").exists():
        print("Frozen inventory already exists; no resampling.")
        return
    old = json.loads((CONTEXT / "target_episodes.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(PROJECT / "visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv",
        usecols=["instance_id", "class_id", "mmsi", "port", "product"], dtype={"mmsi": str})
    manifest["sample_id"] = manifest.instance_id
    manifest["row_index"] = np.arange(len(manifest))
    manifest["acquisition"] = manifest["product"].str.split("_").map(lambda p: "_".join((p[0], p[6], p[7])))
    far_ok = set(pd.read_csv(CONTEXT / "geometry_by_sample.csv").query("far_ok").sample_id)
    positions = pd.read_csv(REGION / "source_region_metadata.csv", usecols=[
        "sample_id", "resolved_crop_left_vv", "resolved_crop_top_vv", "resolved_crop_left_vh", "resolved_crop_top_vh"])
    shape = pd.read_csv(REGION / "ports/frame.csv", usecols=[
        "sample_id", "scene_id", "native_h_vv", "native_w_vv", "native_h_vh", "native_w_vh"])
    coords = positions.merge(shape, on="sample_id", validate="one_to_one").set_index("sample_id")
    initial_ids = {sid for e in old for sid in e["query"] + [
        x for draw in e["draws"] for origin in ("same", "cross") for x in draw[origin]]}
    initial_ships = set(manifest.loc[manifest.sample_id.isin(initial_ids), "mmsi"])
    initial_acqs = {e["fold"].split("__", 1)[1] for e in old}
    all_old_role_acqs = set(manifest.loc[manifest.sample_id.isin(initial_ids), "acquisition"])
    episodes, inventory, used_frames, pools = [], [], [], []
    for port_index, port in enumerate(dict.fromkeys(e["port"] for e in old)):
        episode = next(e for e in old if e["port"] == port)
        folder = Path(episode["folder"])
        source = pd.concat([pd.read_csv(folder / name, dtype={"mmsi": str})
                            for name in ("source_train.csv", "source_validation.csv")])
        target = manifest.loc[manifest.port.eq(port) & manifest.sample_id.isin(far_ok)
            & ~manifest.mmsi.isin(source.mmsi) & ~manifest.acquisition.isin(source.acquisition)].copy()
        target = target.loc[target.mmsi.fillna("").str.fullmatch("[1-9][0-9]{8}")]
        groups = {acq: group.sort_values("sample_id").drop_duplicates("mmsi")
                  for acq, group in target.groupby("acquisition", sort=True)}
        options = []
        for acq_index, (acq, group) in enumerate(groups.items()):
            record = dict(port=port, acquisition=acq, ships=len(group), classes=group.class_id.nunique(),
                          initial_query_acquisition=acq in initial_acqs,
                          initial_any_role_acquisition=acq in all_old_role_acqs, eligible=False)
            inventory.append(record)
            if acq in initial_acqs or len(group) < 40 or group.class_id.nunique() < 3:
                continue
            rng = np.random.default_rng(SEED + port_index*10000 + acq_index)
            fraction = min(1/3, 64 / len(group))
            query_ids = []
            for _, cls in group.groupby("class_id", sort=True):
                query_ids.extend(rng.choice(cls.sample_id, max(1, int(np.ceil(len(cls)*fraction))), replace=False))
            query = group.loc[group.sample_id.isin(query_ids)]
            same = remove_overlap(group.loc[~group.mmsi.isin(query.mmsi)], query, coords)
            if len(query) < 16 or len(same) < 16:
                continue
            choices = []
            for cross_acq, other in groups.items():
                if cross_acq == acq:
                    continue
                cross = other.loc[~other.mmsi.isin(query.mmsi)]
                capacity = np.minimum(counts(same), counts(cross))
                classes = [c for c in sorted(query.class_id.unique()) if capacity[c] > 0]
                if len(classes) < 3 or capacity[classes].sum() < 16:
                    continue
                choices.append((len(classes), int(capacity[classes].sum()), len(cross), cross_acq,
                                cross, tokens(capacity, classes)))
            if not choices:
                continue
            _, _, _, cross_acq, cross, order = max(choices, key=lambda x:x[:4])
            budgets = [b for b in (16, 32, 64) if b <= len(order)]
            record.update(eligible=True, query=len(query), same=len(same), cross=len(cross),
                          max_budget=max(budgets), cross_acquisition=cross_acq)
            options.append((record, query, same, cross, order, acq_index))
        rng = np.random.default_rng(SEED + port_index)
        selected = sorted(rng.choice(len(options), min(4, len(options)), replace=False).tolist())
        print(port, "eligible acquisitions", len(options), "selected", len(selected), flush=True)
        for i in selected:
            record, query, same, cross, order, acq_index = options[i]
            record["selected"] = True
            budgets = [b for b in (16, 32, 64) if b <= len(order)]
            draws = []
            for draw in range(5):
                pair = {}
                for k, (origin, pool) in enumerate((("same", same), ("cross", cross))):
                    rng = np.random.default_rng(SEED + port_index*10000 + acq_index*100 + draw*2 + k)
                    available = {c: list(rng.permutation(part.sample_id.to_numpy(str)))
                                 for c, part in pool.groupby("class_id")}
                    pair[origin] = [available[c].pop() for c in order]
                draws.append(pair)
            assert not set(source.mmsi) & set(pd.concat([query, same, cross]).mmsi)
            for draw in draws:
                for b in budgets:
                    left = manifest.set_index("sample_id").loc[draw["same"][:b]]
                    right = manifest.set_index("sample_id").loc[draw["cross"][:b]]
                    np.testing.assert_array_equal(counts(left), counts(right))
                    assert not set(query.mmsi) & (set(left.mmsi) | set(right.mmsi))
            episodes.append(dict(port=port, fold=port.replace(" ", "_")+"__"+record["acquisition"],
                query_acquisition=record["acquisition"], cross_acquisition=record["cross_acquisition"],
                source_folder=str(folder), old_reference_fold=episode["fold"],
                query=query.sample_id.to_list(), budgets=budgets, draws=draws,
                query_ships_seen_in_initial=int(query.mmsi.isin(initial_ships).sum()),
                query_acquisition_seen_as_initial_support=record["initial_any_role_acquisition"]))
            used_ids = set(query.sample_id) | {sid for draw in draws for origin in ("same", "cross") for sid in draw[origin]}
            used_frames.append(target.loc[target.sample_id.isin(used_ids)])
        pools.append(dict(port=port, reference=episode["fold"], source_folder=str(folder),
                          eligible=len(options), selected=len(selected)))
    assert episodes
    used = pd.concat(used_frames).drop_duplicates("sample_id").sort_values("row_index")
    used.to_csv(OUT / "required_samples.csv", index=False)
    pd.DataFrame(inventory).to_csv(OUT / "candidate_inventory.csv", index=False)
    (OUT / "frozen.json").write_text(json.dumps(dict(episodes=episodes, pools=pools,
        seed=SEED, stage="acquisition_expansion_same_five_ports", no_target_outcome_selection=True),
        indent=2, ensure_ascii=False), encoding="utf-8")
    print("FROZEN", len(episodes), "scenes", sum(len(e["query"]) for e in episodes),
          "queries", len(used), "unique chips", flush=True)


if __name__ == "__main__":
    main()
