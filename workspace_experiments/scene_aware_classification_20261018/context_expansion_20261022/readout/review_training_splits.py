"""Exercise the real split function with output redirected inside readout; no fits."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    spec = importlib.util.spec_from_file_location('reviewed_training', HERE.parent/'train_expanded.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.HERE = HERE/'training_review'
    mod.HERE.mkdir(exist_ok=True)
    t = pd.read_csv(HERE/'targets.csv', dtype={'mmsi': str})
    edges = pd.read_csv(HERE/'neighbors.csv')
    with np.load(HERE/'features.npz') as f:
        mask, fg, bg = f['mask'], f['fg_valid'], f['bg_valid']
    folds, bins = mod.split_data(t)
    identity, y, groups = t.mmsi.to_numpy(), t.class_id.to_numpy(), t.pass_group.to_numpy()
    rows, oof_rows, seen_tests = [], [], []
    loc = dict(zip(t.instance_id, t.index))
    for fold, (train, val, test) in enumerate(folds):
        seen_tests.extend(test.tolist())
        for a, b in ((train, val), (train, test), (val, test)):
            assert not set(identity[a]).intersection(identity[b])
            assert not set(groups[a]).intersection(groups[b])
        for role, ix in (('train', train), ('validation', val), ('test', test)):
            assert len(ix) and set(y[ix]) == {0, 1, 2}
            rows.append(dict(fold=fold, role=role, n=len(ix), mmsi=len(set(identity[ix])), passes=len(set(groups[ix])),
                             context=int(mask[ix].any(1).sum()), classes=json.dumps(pd.Series(y[ix]).value_counts().to_dict())))
        seen_oof = []
        for b in sorted(set(bins[train])):
            held = train[bins[train] == b]
            fit = train[(bins[train] != b) & ~np.isin(identity[train], identity[held])]
            assert not set(identity[fit]).intersection(identity[held])
            assert not set(groups[fit]).intersection(groups[held])
            assert set(y[fit]) == {0, 1, 2} and len(fit) >= 33
            seen_oof.extend(held.tolist())
            oof_rows.append(dict(fold=fold, inner_bin=int(b), held=len(held), fit=len(fit), fit_passes=len(set(groups[fit]))))
        assert sorted(seen_oof) == sorted(train.tolist())
        unique = edges[edges.instance_id.isin(set(t.iloc[train].instance_id))].drop_duplicates(['product', 'detection_id'])
        ii, jj = unique.instance_id.map(loc).to_numpy(), unique.slot.to_numpy(int)
        assert int(fg[ii, jj].sum()) >= 32 and int(bg[ii, jj].sum()) >= 32
    assert sorted(seen_tests) == list(range(len(t)))
    pd.DataFrame(rows).to_csv(mod.HERE/'coverage.csv', index=False)
    pd.DataFrame(oof_rows).to_csv(mod.HERE/'oof_coverage.csv', index=False)
    result = dict(targets=len(t), folds=len(folds), inner_oof_splits=len(oof_rows),
                  actual_split_function_executed=True, all_outer_roles_nonempty_three_classes=True,
                  acquisition_and_mmsi_overlap=0, each_target_tested_once=True, each_source_target_oof_once=True,
                  inner_oof_fit_min=min(r['fit'] for r in oof_rows), role_projection_at_least32_unique_valid_nodes=True,
                  training_or_model_fits_run=False,
                  scope='Split and OOF indices checked against current saved data; training source read only; no head optimization or score replay')
    (mod.HERE/'checks.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
