"""Bounded independent review of frozen fold-2 geometry outputs."""
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, accuracy_score

OUT = Path(__file__).resolve().parent
MAIN = OUT.parent
VARIANTS = ('original', 'range_geometry', 'axis_sham')
FOLD = 2


def stat(y, prediction, probability):
    p = np.asarray(probability, dtype=float)
    loss = np.einsum('ij,ij->i', p, p) - 2*p[np.arange(len(y)), y] + 1
    return dict(oa=100*accuracy_score(y, prediction), ba=100*balanced_accuracy_score(y, prediction),
                brier_macro=float(np.mean([loss[y == c].mean() for c in np.unique(y)])))


def main():
    # Exercise only the actual pure transform; importing the runner would import model code.
    tree = ast.parse((MAIN / 'slc_geometry.py').read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'corrected')
    ns = dict(np=np, Image=Image, VARIANTS=VARIANTS[1:])
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(MAIN / 'slc_geometry.py'), 'exec'), ns)
    fixture = np.zeros((21, 21), np.uint8)
    fixture[7:14, 4:17] = 255
    x, xi = ns['corrected'](Image.fromarray(fixture), .25, 'range_geometry')
    y, yi = ns['corrected'](Image.fromarray(fixture), .25, 'axis_sham')
    xx, yy = np.asarray(x), np.asarray(y)
    assert np.count_nonzero(xx.max(1)) == 7 and np.count_nonzero(xx.max(0)) < 13
    assert np.count_nonzero(yy.max(0)) == 13 and np.count_nonzero(yy.max(1)) < 7
    assert xi['content_pixels'] == yi['content_pixels'] == 105
    assert xi['fill'] == yi['fill'] == 0
    controls = [pd.read_csv(MAIN / 'geometry_features' / v / 'controls.csv') for v in VARIANTS[1:]]
    keys = ['sample_id', 'native_side', 'compressed_side', 'content_pixels', 'fill', 'vh_fill', 'ratio']
    pd.testing.assert_frame_equal(controls[0][keys], controls[1][keys])
    assert len(controls[0]) == 999
    outputs = {}
    for variant in VARIANTS:
        folder = MAIN / 'predictions/original' / variant / f'fold_{FOLD}/oss2'
        with np.load(folder / 'predictions.npz') as z:
            saved = {k: z[k] for k in z.files}
        outputs[variant] = (saved, np.load(folder / 'probabilities.npy', mmap_mode='r'))
    ref = outputs['original'][0]
    metadata = pd.read_csv(OUT / 'public_metadata.csv', dtype=str, keep_default_na=False)
    frame = metadata.loc[metadata.dataset.eq('oss2')].set_index('sample_id').loc[ref['ids']]
    labels = frame.class_id.to_numpy(int)
    slc = frame.product_type.eq('SLC').to_numpy()
    arm_index = {name: list(ref['arms']).index(name) for name in ['R', 'T']}
    expert = list(ref['heads']).index('expert')
    prior = list(ref['heads']).index('prior')
    for saved, p in outputs.values():
        for k in ['ids', 'support', 'eval_masks', 'arms', 'heads']:
            np.testing.assert_array_equal(saved[k], ref[k])
        np.testing.assert_array_equal(p[:, :, prior, ~slc], outputs['original'][1][:, :, prior, ~slc])
    metric_table = pd.read_csv(MAIN / 'geometry_metrics.csv')
    effect_table = pd.read_csv(MAIN / 'geometry_effects.csv')
    pair_table = pd.read_csv(MAIN / 'geometry_pairs.csv')
    checks, denominators, coupling = [], [], []

    def compare(table, key, values):
        mask = np.ones(len(table), bool)
        for k, value in key.items():
            mask &= table[k].eq(value).to_numpy()
        rows = table.loc[mask]
        assert len(rows) == 1, (key, len(rows))
        actual = rows.iloc[0]
        for field, expected in values.items():
            error = abs(float(actual[field])-float(expected))
            checks.append(dict(**key, metric=field, absolute_error=error))
            assert error < 1e-9, (key, field, error)

    pairs = pd.read_csv(OUT / 'oss2_grdh_slc_pairs.csv', keep_default_na=False)
    pairs = pairs.loc[pairs.same_label].copy()
    assert len(pairs) == 233
    gi = frame.index.get_indexer(pairs.sample_id_grdh)
    si = frame.index.get_indexer(pairs.sample_id_slc)
    assert (gi >= 0).all() and (si >= 0).all()
    for draw, query in enumerate(ref['eval_masks']):
        for scope, mask in [('original', query), ('query_SLC', query & slc)]:
            for arm, ai in arm_index.items():
                values = {}
                for variant, (saved, p) in outputs.items():
                    value = stat(labels[mask], saved['predictions'][draw, ai, expert, mask], p[draw, ai, expert, mask])
                    values[variant] = value
                    compare(metric_table, dict(fold=FOLD, draw=draw, scope=scope, arm=arm, head='expert', variant=variant), value)
                for a, b in [('range_geometry', 'original'), ('axis_sham', 'original'), ('range_geometry', 'axis_sham')]:
                    compare(effect_table, dict(fold=FOLD, draw=draw, scope=scope, arm=arm, head='expert', contrast=a+'-'+b),
                            dict(delta_ba=values[a]['ba']-values[b]['ba'], delta_oa=values[a]['oa']-values[b]['oa'],
                                 risk_gain_macro=values[b]['brier_macro']-values[a]['brier_macro']))
        eligible = query[gi] & query[si]
        clean = pairs.noise_xml_grdh.astype(str).eq('0').to_numpy() & pairs.noise_xml_slc.astype(str).eq('0').to_numpy()
        for scope, mask in [('paired_all', eligible), ('paired_both_clean', eligible & clean)]:
            g, s = gi[mask], si[mask]
            np.testing.assert_array_equal(labels[g], labels[s])
            denominators.append(dict(fold=FOLD, draw=draw, scope=scope, pairs=len(g),
                                     classes=len(np.unique(labels[g])), class_counts=json.dumps(np.bincount(labels[g], minlength=8).tolist())))
            for variant, (saved, p) in outputs.items():
                for arm, ai in arm_index.items():
                    gm = stat(labels[g], saved['predictions'][draw, ai, expert, g], p[draw, ai, expert, g])
                    sm = stat(labels[s], saved['predictions'][draw, ai, expert, s], p[draw, ai, expert, s])
                    compare(pair_table, dict(fold=FOLD, draw=draw, scope=scope, variant=variant, arm=arm, head='expert'),
                            dict(grdh_ba=gm['ba'], slc_ba=sm['ba'], grdh_oa=gm['oa'], slc_oa=sm['oa'],
                                 grdh_brier=gm['brier_macro'], slc_brier=sm['brier_macro'], gap=sm['ba']-gm['ba'], pairs=len(g)))
        for variant in VARIANTS[1:]:
            for arm, ai in arm_index.items():
                before = outputs['original'][1][draw, ai, expert, query & ~slc]
                after = outputs[variant][1][draw, ai, expert, query & ~slc]
                coupling.append(dict(draw=draw, variant=variant, arm=arm,
                                     grdh_query_probability_max_change=float(np.max(np.abs(after-before))),
                                     slc_support_rows=int(slc[ref['support'][draw]].sum())))
    pd.DataFrame(checks).to_csv(OUT / 'geometry_independent_numeric_checks.csv', index=False)
    pd.DataFrame(denominators).to_csv(OUT / 'geometry_pair_denominators_fold2.csv', index=False)
    pd.DataFrame(coupling).to_csv(OUT / 'geometry_support_coupling_fold2.csv', index=False)
    receipt = dict(status='passed', fold=FOLD, arms=['R', 'T'], head='expert', draws=5,
                   scopes=['original', 'query_SLC', 'paired_all', 'paired_both_clean'], scalar_checks=len(checks),
                   maximum_numeric_error=max(x['absolute_error'] for x in checks), transform_fixture_passed=True,
                   matched_control_rows=999, grdh_prior_bitexact=True, source_fits=0, gpu_runs=0,
                   notes='Independent sklearn OA/BA and algebraic Brier from frozen probabilities; no imports of model runner')
    (OUT / 'GEOMETRY_IMPLEMENTATION_VERIFIED.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))
    print(pd.DataFrame(denominators).to_string(index=False))


if __name__ == '__main__':
    main()
