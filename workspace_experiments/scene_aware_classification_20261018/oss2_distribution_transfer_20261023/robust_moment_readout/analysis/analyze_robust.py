"""Analyze frozen robust moment heads; no fitting or feature extraction."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
ROBUST = HERE.parent
ROOT = ROBUST.parent
sys.path.insert(0, str(ROOT / 'analysis'))
import analyze_oss2 as previous

common = previous.common
plt = previous.plt
ARMS = ['local', 'additive']
KEY = ['role', 'seed', 'fold', 'instance_id']
REQUIRED = ['predictions.csv', 'selections.csv', 'selection_curves.csv',
            'selection_frozen.json', 'summary.json']


def scores(frame, prefix='s'):
    return frame[[f'{prefix}{c}' for c in range(8)]].to_numpy(float)


def comparison_frame(new, old, reference):
    a = new.set_index(KEY).sort_index().copy()
    b = old.set_index(KEY).sort_index()
    assert a.index.is_unique and b.index.is_unique
    assert a.index.equals(b.index), 'Prediction partitions differ'
    for col in ['class_id', 'mmsi', 'pass_group', 'product', 'valid_vv', 'valid_vh']:
        np.testing.assert_array_equal(a[col], b[col])
    np.testing.assert_array_equal(scores(a, 'b'), scores(b, 'b'))
    original_b = scores(a, 'b')
    ref = original_b if reference == 'B' else scores(b)
    for c in range(8):
        a[f'strong_b{c}'] = original_b[:, c]
        a[f'b{c}'] = ref[:, c]
    a['reference'] = reference
    a['reference_epoch'] = -1 if reference == 'B' else b.epoch.to_numpy()
    a['base_prediction'] = ref.argmax(1)
    a['prediction'] = scores(a).argmax(1)
    a['net'] = a.prediction.eq(a.class_id).astype(int) - a.base_prediction.eq(a.class_id).astype(int)
    a['outcome'] = np.select([a.net.eq(1), a.net.eq(-1), a.prediction.eq(a.class_id)],
                             ['rescued', 'harmed', 'CC'], default='WW')
    a['zero_epoch'] = a.epoch.eq(0)
    return a.reset_index()


def learning_tables():
    heads, traces = [], []
    for version, folder in [('original', ROOT), ('robust', ROBUST)]:
        selection = pd.read_csv(folder / 'selections.csv')
        curve = pd.read_csv(folder / 'selection_curves.csv')
        selection = selection[selection.arm.isin(ARMS)]
        curve = curve[curve.arm.isin(ARMS)]
        assert len(selection) == 36
        for row in selection.itertuples():
            g = curve[curve.arm.eq(row.arm) & curve.seed.eq(row.seed) & curve.fold.eq(row.fold)].sort_values('epoch')
            assert g.epoch.is_unique and set(g.epoch) == set(range(0, 201, 5))
            start, end = g.iloc[0], g.iloc[-1]
            chosen = g[g.epoch.eq(row.epoch)].iloc[0]
            finite_loss = g.loc[g.train_loss.notna(), ['epoch', 'train_loss']]
            assert np.isfinite(finite_loss.train_loss).all()
            item = dict(version=version, arm=row.arm, seed=row.seed, fold=row.fold,
                        epoch=row.epoch, zero_epoch=row.epoch == 0,
                        train_loss_step5=float(finite_loss.iloc[0].train_loss),
                        train_loss_step200=float(end.train_loss),
                        train_loss_change_5_to_200=float(end.train_loss - finite_loss.iloc[0].train_loss))
            for field in ['J', 'OA', 'BA', 'NLL', 'Brier']:
                item[f'initial_{field}'] = float(start[field])
                item[f'selected_{field}'] = float(chosen[field])
                item[f'selected_delta_{field}'] = float(chosen[field] - start[field])
                item[f'final_{field}'] = float(end[field])
            heads.append(item)
        curve = curve.copy()
        curve['version'] = version
        traces.append(curve)
    h = pd.DataFrame(heads)
    h.to_csv(HERE / 'learning_heads.csv', index=False)
    pd.concat(traces, ignore_index=True).to_csv(HERE / 'learning_curves.csv', index=False)
    index = ['arm', 'seed', 'fold']
    new = h[h.version.eq('robust')].drop(columns='version')
    old = h[h.version.eq('original')].drop(columns='version')
    paired = new.merge(old, on=index, suffixes=('_robust', '_original'), validate='one_to_one')
    paired['returned_to_B'] = paired.zero_epoch_robust & ~paired.zero_epoch_original
    paired['activated_from_B'] = ~paired.zero_epoch_robust & paired.zero_epoch_original
    paired['both_zero'] = paired.zero_epoch_robust & paired.zero_epoch_original
    paired.to_csv(HERE / 'learning_paired.csv', index=False)
    h.groupby(['version', 'arm', 'seed']).agg(
        heads=('fold', 'size'), zero_epochs=('zero_epoch', 'sum'),
        median_epoch=('epoch', 'median'), mean_loss5=('train_loss_step5', 'mean'),
        mean_loss200=('train_loss_step200', 'mean'),
        mean_selected_J_gain=('selected_delta_J', 'mean')
    ).reset_index().to_csv(HERE / 'learning_summary.csv', index=False)


def render(g, table):
    out = HERE / 'panels'
    out.mkdir(exist_ok=True)
    table = table.set_index('sample_id')
    cases, missing = [], []
    for arm in ARMS:
        sub = g[g.arm.eq(arm) & g.seed.eq(17) & g.role.eq('test') & g.reference.eq('original_' + arm)]
        for outcome in ['rescued', 'harmed', 'CC', 'WW']:
            hit = sub[sub.outcome.eq(outcome)].sort_values(['instance_id', 'fold'])
            if hit.empty:
                missing.append(dict(arm=arm, outcome=outcome))
                continue
            q = hit.iloc[0]
            native = table.loc[q.instance_id]
            y = int(q.class_id)
            b = q[[f'strong_b{c}' for c in range(8)]].to_numpy(float)
            old = q[[f'b{c}' for c in range(8)]].to_numpy(float)
            new = q[[f's{c}' for c in range(8)]].to_numpy(float)
            oldr, newr = previous.highest_other(old, y), previous.highest_other(new, y)
            fig, axes = plt.subplots(2, 3, figsize=(15, 10), layout='constrained')
            dimensions = []
            for j, pol in enumerate(['vv', 'vh']):
                path = native[f'chip_path_{pol}']
                with Image.open(path) as im:
                    raw = np.array(im)
                padded, footprint = previous.native_image(path)
                h, w = raw.shape
                top, left = (224-h)//2, (224-w)//2
                np.testing.assert_array_equal(np.rint(padded[0, top:top+h, left:left+w]*255).astype(np.uint8), raw)
                dimensions.append(f'{pol}:{w}x{h}')
                axes[j, 0].imshow(raw, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                axes[j, 0].set_title(f'{pol.upper()} original {w}x{h}; uint8')
                axes[j, 1].imshow(padded[0], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
                axes[j, 1].set_title(f'224 native pad; no resize/crop; footprint={footprint.mean():.4f}')
                for ax in axes[j, :2]:
                    ax.set_xlabel('native pixel x')
                    ax.set_ylabel('native pixel y')
            x = np.arange(8)
            for offset, score, label, color in [(-.25, b, 'Frozen B', '#aaaaaa'),
                    (0, old, 'Original ' + arm, '#496979'), (.25, new, 'Robust ' + arm, '#158964')]:
                axes[0, 2].bar(x+offset, score, width=.24, label=label, color=color)
            axes[0, 2].legend(fontsize=8)
            axes[0, 2].set_title('Actual scores; original head is case reference')
            axes[1, 2].bar(x, new-old, color=['#158964' if c == y else '#bd5264' for c in x])
            axes[1, 2].axhline(0, color='black', lw=.6)
            axes[1, 2].set_title('Robust - original scores; not physical attribution')
            note = 'ZERO EPOCH: fallback to B, not learned gain' if q.zero_epoch else 'source-selected nonzero epoch'
            fig.suptitle(f'OSS2 robust {arm} vs original {arm} / {outcome} / seed17 / fold{int(q.fold)}\n'
                f'{q.instance_id} / {native.class_name} / {native.pass_group}\n'
                f'epoch {int(q.reference_epoch)} -> {int(q.epoch)}; {note}\n'
                f'winner {old.argmax()} -> {new.argmax()}; rival {oldr} -> {newr}; '
                f'margin {old[y]-old[oldr]:+.5f} -> {new[y]-new[newr]:+.5f}', fontsize=10)
            panel = out / f'{arm}_{outcome}.png'
            fig.savefig(panel, dpi=100)
            plt.close(fig)
            cases.append(dict(arm=arm, reference='original_' + arm, seed=17, role='test',
                fold=int(q.fold), instance_id=q.instance_id, class_id=y, class_name=native.class_name,
                mmsi=native.mmsi, pass_group=native.pass_group, outcome=outcome,
                original_epoch=int(q.reference_epoch), robust_epoch=int(q.epoch),
                zero_epoch=bool(q.zero_epoch), B_winner=int(b.argmax()), old_winner=int(old.argmax()),
                final_winner=int(new.argmax()), old_rival=oldr, final_rival=newr,
                old_margin=old[y]-old[oldr], final_margin=new[y]-new[newr],
                true_delta=new[y]-old[y], oldrival_delta=new[oldr]-old[oldr],
                finalrival_delta=new[newr]-old[newr], dimensions=';'.join(dimensions),
                image_vv=native.chip_path_vv, image_vh=native.chip_path_vh, panel=str(panel),
                selection='seed17 test first instance_id then fold; robust versus original same arm'))
    pd.DataFrame(cases).to_csv(HERE / 'review_cases.csv', index=False)
    return cases, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-inputs', action='store_true')
    args = parser.parse_args()
    missing = [name for name in REQUIRED if not (ROBUST / name).is_file()]
    if args.check_inputs:
        print(json.dumps(dict(ready=not missing, missing=missing), indent=2))
        return
    if missing:
        raise SystemExit('Awaiting completed training outputs: ' + ', '.join(missing))
    summary = json.loads((ROBUST / 'summary.json').read_text(encoding='utf-8'))
    frozen = json.loads((ROBUST / 'selection_frozen.json').read_text(encoding='utf-8'))
    assert summary['heads'] == frozen['heads'] == 36
    assert not frozen['test_labels_used_for_selection']
    new = pd.read_csv(ROBUST / 'predictions.csv', dtype={'mmsi': str})
    old = pd.read_csv(ROOT / 'predictions.csv', dtype={'mmsi': str})
    table = pd.read_csv(ROOT / 'cohort.csv', dtype={'mmsi': str})
    assert set(new.arm) == set(ARMS) and set(new.seed) == {17, 29, 43}
    assert set(new.fold) == set(range(6)) and set(new.role) == {'validation', 'test'}
    assert np.isfinite(new[[f'{p}{c}' for p in ['b', 's'] for c in range(8)]]).all().all()
    for _, sub in new[new.role.eq('test')].groupby(['arm', 'seed']):
        assert len(sub) == len(table) and sub.instance_id.is_unique
    np.testing.assert_allclose(scores(new[new.epoch.eq(0)]), scores(new[new.epoch.eq(0)], 'b'), rtol=0, atol=0)
    learning_tables()
    comparisons = []
    for arm in ARMS:
        for reference in ['B', 'original_' + arm, 'mean_only', 'extra_score']:
            oldarm = arm if reference == 'B' or reference.startswith('original_') else reference
            comparisons.append(comparison_frame(new[new.arm.eq(arm)], old[old.arm.eq(oldarm)], reference))
    g = pd.concat(comparisons, ignore_index=True)
    pooled, strata, classfold, cis, fallback = [], [], [], [], []
    for (role, arm, seed, reference), sub in g.groupby(['role', 'arm', 'seed', 'reference']):
        meta = dict(role=role, arm=arm, seed=seed, reference=reference)
        pooled.append(dict(**meta, **common.metrics(sub)))
        for unit in ['class_id', 'fold', 'pass_group', 'product', 'valid_vh']:
            for value, h in sub.groupby(unit):
                strata.append(dict(**meta, unit=unit, name=value, **common.metrics(h)))
        for (fold, cls), h in sub.groupby(['fold', 'class_id']):
            classfold.append(dict(**meta, fold=fold, class_id=cls, **common.metrics(h)))
        for zero, h in sub.groupby('zero_epoch'):
            fallback.append(dict(**meta, zero_epoch=bool(zero), **common.metrics(h)))
        if role == 'test':
            cis.extend(dict(**meta, **r) for r in common.intervals(sub, sub.net.to_numpy(), 'robust_' + arm + '-' + reference))
    for name, rows in [('pooled_metrics', pooled), ('stratified_metrics', strata),
                       ('class_fold_metrics', classfold), ('paired_intervals', cis), ('epoch_strata', fallback)]:
        pd.DataFrame(rows).to_csv(HERE / f'{name}.csv', index=False)
    sf = pd.DataFrame(strata)
    for unit in ['class_id', 'fold', 'pass_group', 'product', 'valid_vh']:
        sf[sf.unit.eq(unit)].to_csv(HERE / f'{unit}_metrics.csv', index=False)
    same = g[g.reference.eq('original_' + g.arm)]
    same[['role', 'arm', 'seed', 'fold', 'instance_id', 'class_id', 'epoch', 'reference_epoch',
          'zero_epoch', 'base_prediction', 'prediction', 'net', 'outcome']].to_csv(HERE / 'same_arm_changes.csv', index=False)
    cases, absent = render(g, table)
    (HERE / 'analysis_summary.json').write_text(json.dumps(dict(training_runs=0,
        compared_heads=36, prediction_rows=len(new), paired_ci_rows=len(cis),
        cases=len(cases), unique_case_objects=len({r['instance_id'] for r in cases}),
        missing_outcomes=absent, images_viewed=False, pixels_resized=0, pixels_cropped=0,
        case_reference='original same-name arm, not B',
        ci='Fixed test predictions, pass/MMSI 2000 paired cluster draws; no fit/selection uncertainty',
        learning='source selection curves first; step5/200 logged losses are pre-update, validation scores post-update'), indent=2), encoding='utf-8')
    print(pd.DataFrame(pooled)[['role', 'arm', 'seed', 'reference', 'net', 'delta_OA', 'delta_BA', 'delta_NLL', 'delta_Brier']].to_string(index=False))


if __name__ == '__main__':
    main()
