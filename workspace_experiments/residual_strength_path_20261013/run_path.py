"""Frozen logit-strength paths with matched source-only selection."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.special import softmax

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
OLD = ROOT / 'classwise_utility_adaptation_20261012'
STUDY = ROOT / 'same_vessel_condition_study_20261010'
EXTERNAL = ROOT / 'frozen_candidate_coverage_20261012/budget/strong_external/oss2/oss2'
TEMP = Path('C:/Users/25791/AppData/Local/Temp')
ALPHAS = (0., .25, .5, .75, 1.)
MODES = ('joint', 'global')
VALUES = ('oa', 'ba', 'macro_brier', 'delta_oa', 'delta_ba', 'delta_macro_brier',
          'rescue_ba', 'harm_ba', 'risk_gain_unchanged', 'risk_gain_changed',
          'endpoint_rescue_kept_ba', 'endpoint_rescue_lost_ba', 'novel_rescue_ba',
          'endpoint_harm_kept_ba', 'endpoint_harm_avoided_ba', 'novel_harm_ba')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def archive(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def path_name(mode, alpha):
    return mode + '_a' + str(alpha).replace('.', 'p')


def fusion(base, expert):
    return .5 * (softmax(base, axis=1) + softmax(expert, axis=1))


def path_logits(raw, end, alpha):
    return raw + alpha * (end - raw)


def measures(y, p, baseline):
    classes, inverse, count = np.unique(y, return_inverse=True, return_counts=True)
    weight = 1. / (len(classes) * count[inverse])
    pred, old_pred = p.argmax(1), baseline.argmax(1)
    correct, old_correct = pred == y, old_pred == y
    truth = np.eye(p.shape[1])[y]
    loss = np.square(p - truth).sum(1)
    old_loss = np.square(baseline - truth).sum(1)
    rescue = (~old_correct) & correct
    harm = old_correct & (~correct)
    changed = pred != old_pred
    result = dict(oa=100 * correct.mean(), ba=100 * (weight * correct).sum(),
                  macro_brier=(weight * loss).sum(),
                  delta_oa=100 * (correct.astype(float) - old_correct).mean(),
                  delta_ba=100 * (weight * (correct.astype(float) - old_correct)).sum(),
                  delta_macro_brier=(weight * (loss - old_loss)).sum(),
                  rescue_ba=100 * (weight * rescue).sum(),
                  harm_ba=100 * (weight * harm).sum(),
                  changed_n=int(changed.sum()),
                  risk_gain_unchanged=(weight * (old_loss - loss) * ~changed).sum(),
                  risk_gain_changed=(weight * (old_loss - loss) * changed).sum())
    np.testing.assert_allclose(result['rescue_ba'] - result['harm_ba'], result['delta_ba'], atol=1e-12)
    np.testing.assert_allclose(result['risk_gain_unchanged'] + result['risk_gain_changed'],
                               -result['delta_macro_brier'], atol=1e-12)
    perclass = []
    for cls in classes:
        mask = y == cls
        perclass.append(dict(class_id=int(cls), n=int(mask.sum()),
                             recall=100 * correct[mask].mean(),
                             rescue_ba=100 * rescue[mask].mean() / len(classes),
                             harm_ba=100 * harm[mask].mean() / len(classes)))
    return result, perclass


def freeze():
    started = time.perf_counter()
    taus = read_json(OLD / 'FROZEN_CHOICE.json')['taus']
    units, checks = [], []

    def save(key, raw, base, prior, residual, endpoints, gates, refs, ids):
        values = dict(raw=fusion(base, raw), **ids)
        np.testing.assert_allclose(raw, prior + residual, atol=1e-10, rtol=0)
        for mode in MODES:
            for alpha in ALPHAS:
                score = path_logits(raw, endpoints[mode], alpha)
                direct = prior + (1 + alpha * (gates[mode] - 1)) * residual
                error = float(abs(score - direct).max())
                np.testing.assert_allclose(score, direct, atol=1e-10, rtol=0)
                values[path_name(mode, alpha)] = fusion(base, score)
                if alpha in (0., 1.):
                    ref = values['raw'] if alpha == 0 else fusion(base, endpoints[mode])
                    np.testing.assert_allclose(values[path_name(mode, alpha)], ref, atol=1e-12, rtol=0)
                    np.testing.assert_array_equal(values[path_name(mode, alpha)].argmax(1), ref.argmax(1))
                checks.append(dict(**key, mode=mode, alpha=alpha, logit_path_error=error))
        for name, score in refs.items():
            values[name] = fusion(base, score)
        dest = OUT / 'predictions' / key['family'] / key['context'] / (key['unit'] + '.npz')
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, **values)
        units.append(dict(**key, path=str(dest), references=list(refs)))

    for family in ('selection', 'source'):
        for path in sorted((OLD / family).glob('*/*.npz')):
            if path.stem.endswith('_oof'):
                continue
            z = archive(path)
            arms = {m: m + '_t' + str(taus[m]).replace('.', 'p') for m in MODES}
            key = dict(family=family, context=path.parent.name, unit=path.stem, budget=len(z['support_ids']))
            save(key, z['raw'], z['base'], z['prior'], z['residual'],
                 {m: z[a] for m, a in arms.items()}, {m: z['gate_' + a] for m, a in arms.items()},
                 {a: z[a] for a in ('count', 'fixed_scaling')},
                 {a: z[a] for a in ('support_ids', 'query_ids')})
    for budget in (16, 128):
        for draw in range(5):
            unit = f'{draw}_{budget}'
            z = archive(EXTERNAL / (unit + '.npz'))
            new = archive(TEMP / 'SAR_classwise_utility_20261012/oss2/oss2' / (unit + '.npz'))
            gates = archive(OLD / 'target_oof/oss2/oss2' / (unit + '.npz'))
            count = archive(TEMP / 'SAR_prior_protection_20261012/oss2' / (unit + '.npz'))
            for key in ('query_ids', 'support_ids'):
                np.testing.assert_array_equal(z[key], new[key])
                np.testing.assert_array_equal(z[key], count[key])
            save(dict(family='external', context='oss2', unit=unit, budget=budget),
                 z['expert_raw'], z['base'], new['prior'], z['expert_raw'] - new['prior'],
                 {m: new[m] for m in MODES}, {m: gates['gate_' + m] for m in MODES},
                 dict(count=count['count'], simple=z['expert_simple'], initial=z['expert_initial']),
                 {a: z[a] for a in ('support_ids', 'query_ids')})
    assert len(units) == 118
    write_json('PREDICTIONS_FROZEN.json', dict(units=units, alphas=ALPHAS, taus=taus,
               labels_used=False, new_fits=0, seconds=time.perf_counter() - started))
    pd.DataFrame(checks).to_csv(OUT / 'path_checks.csv', index=False)
    print('FROZEN', len(units), 'units', flush=True)


def evaluate(family):
    frozen = read_json(OUT / 'PREDICTIONS_FROZEN.json')
    if family != 'selection':
        assert (OUT / 'FROZEN_ALPHA.json').exists()
    manifest = (read_json(EXTERNAL / 'PREDICTIONS_FROZEN.json')['target_manifest']
                if family == 'external' else STUDY / 'observations.csv')
    labels = pd.read_csv(manifest, usecols=['class_id']).class_id.to_numpy(int)
    previous = pd.read_csv(OLD / ('target_metrics.csv' if family == 'external' else family + '_metrics.csv'))
    records, classes, replays = [], [], []
    for unit in frozen['units']:
        if unit['family'] != family:
            continue
        z = archive(unit['path'])
        key = {k: unit[k] for k in ('family', 'context', 'unit', 'budget')}
        y = labels[z['query_ids']]
        arms = [('reference', -1., a) for a in ['raw'] + unit['references']]
        arms += [(m, a, path_name(m, a)) for m in MODES for a in ALPHAS]
        for mode, alpha, arm in arms:
            score, cl = measures(y, z[arm], z['raw'])
            if mode != 'reference':
                _, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
                weight = 100. / (len(counts) * counts[inverse])
                old_correct = z['raw'].argmax(1) == y
                end_correct = z[path_name(mode, 1.)].argmax(1) == y
                correct = z[arm].argmax(1) == y
                end_rescue, end_harm = (~old_correct) & end_correct, old_correct & (~end_correct)
                score.update(endpoint_rescue_kept_ba=float(weight @ (end_rescue & correct)),
                    endpoint_rescue_lost_ba=float(weight @ (end_rescue & ~correct)),
                    novel_rescue_ba=float(weight @ (~old_correct & ~end_correct & correct)),
                    endpoint_harm_kept_ba=float(weight @ (end_harm & ~correct)),
                    endpoint_harm_avoided_ba=float(weight @ (end_harm & correct)),
                    novel_harm_ba=float(weight @ (old_correct & end_correct & ~correct)))
                np.testing.assert_allclose(score['endpoint_rescue_kept_ba'] + score['novel_rescue_ba'], score['rescue_ba'], atol=1e-12)
                np.testing.assert_allclose(score['endpoint_harm_kept_ba'] + score['novel_harm_ba'], score['harm_ba'], atol=1e-12)
            row_key = dict(**key, mode=mode, alpha=alpha, arm=arm)
            records.append(dict(**row_key, **score))
            classes.extend(dict(**row_key, **v) for v in cl)
            if mode == 'reference' or alpha == 1.:
                if family == 'external':
                    old_arm = arm if mode == 'reference' else mode
                    matched = previous[(previous.dataset == 'oss2') & (previous.layer == 'fusion') &
                        (previous.draw == int(unit['unit'].split('_')[0])) &
                        (previous.budget == unit['budget']) & (previous.arm == old_arm)]
                else:
                    old_arm = arm if mode == 'reference' else mode + '_t' + str(frozen['taus'][mode]).replace('.', 'p')
                    matched = previous[(previous.context == unit['context']) &
                        (previous.unit == unit['unit']) & (previous.arm == old_arm)]
                assert len(matched) == 1
                errors = [abs(score[k] - float(matched.iloc[0][k])) for k in ('oa', 'ba', 'macro_brier')]
                np.testing.assert_allclose([score[k] for k in ('oa', 'ba', 'macro_brier')],
                    matched.iloc[0][['oa', 'ba', 'macro_brier']].to_numpy(float), atol=1e-10, rtol=0)
                replays.append(dict(**row_key, error=max(errors)))
    f = pd.DataFrame(records)
    f.to_csv(OUT / (family + '_metrics.csv'), index=False)
    pd.DataFrame(classes).to_csv(OUT / (family + '_classes.csv'), index=False)
    pd.DataFrame(replays).to_csv(OUT / (family + '_replays.csv'), index=False)
    f.groupby(['budget', 'mode', 'alpha', 'arm'])[list(VALUES)].mean().reset_index().to_csv(
        OUT / (family + '_summary.csv'), index=False)
    if family == 'selection':
        choice = {}
        for mode in MODES:
            by_budget = f[f['mode'] == mode].groupby(['budget', 'alpha']).ba.mean()
            candidate = by_budget.groupby('alpha').mean()
            choice[mode] = float(candidate[candidate >= candidate.max() - 1e-12].index.min())
        write_json('FROZEN_ALPHA.json', dict(alphas=choice,
            criterion='source FIT full fusion BA, equal budget then equal unit; numerical ties smallest alpha',
            held_used=False, external_used=False))
        print('SOURCE-SELECTED', choice, flush=True)
    choice = read_json(OUT / 'FROZEN_ALPHA.json')['alphas']
    selected = f[(f['mode'] == 'reference') |
                 f.apply(lambda x: x['mode'] in choice and x.alpha == choice[x['mode']], axis=1)].copy()
    selected['policy'] = np.where(selected['mode'] == 'reference', selected.arm, selected['mode'] + '_selected')
    selected.to_csv(OUT / (family + '_selected.csv'), index=False)
    print(family, selected.groupby(['budget', 'policy'])[list(VALUES[:7])].mean().to_string(), flush=True)


if __name__ == '__main__':
    freeze()
    for family in ('selection', 'source', 'external'):
        evaluate(family)
    write_json('EVALUATION_COMPLETE.json', dict(source_fit_units=81, source_held_units=27,
               external_units=10, external_budgets=[16, 128], new_fits=0, main_model_changed=False))
