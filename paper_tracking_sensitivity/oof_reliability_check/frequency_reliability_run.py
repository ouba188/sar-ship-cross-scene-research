"""Predict before reading evaluation labels; report all prespecified arms."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from prepare_learned_frequency import (OUT as EXP, ROOT, SOURCE, PARENT,
    DATASETS, load_npz, load_pickle)
from cross_dataset_c4 import route
from frequency_reliability_core import (kernels, coefficients, strict_loo,
    ce_strength, local_strength, point_ce, balanced_ce, balanced_accuracy)

OUT = ROOT / 'frequency_reliability_20261009'
MODES = ('baseline', 'learned', 'uniform_half', 'loo_ce', 'local', 'source_selected')


def write(path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def router(info, data):
    if info['dataset'] == 'self24ports':
        from receiver_consistent_handoff import confidence_parts
        from evidence_aware_handoff import predict_gate, route_scores
        gate = load_pickle(PARENT/info['task']/'source_models.pkl')['gates']['joint']
        def routing(score, expert, labels, ev):
            em, eh = confidence_parts(expert)
            bm, bh = confidence_parts(score)
            result = route_scores(score, expert, predict_gate(gate, np.c_[em, bm, eh, bh]),
                labels, ev, data['route_ids'], budget=.05)
            return result['predictions'], result['selected']
    else:
        gate = load_pickle(Path(info['model_path']))['gate']
        def routing(score, expert, labels, ev):
            return route(score, expert, gate, labels, ev, data['route_ids'])
    return routing


def predict(folder):
    info = json.loads((folder/'PREPARED.json').read_text())
    out = OUT/'targets'/info['dataset']/info['task']
    out.mkdir(parents=True, exist_ok=True)
    if (out/'COMPLETE.json').exists():
        return
    selection = json.loads((OUT/'source'/info['dataset']/info['task']/'SELECTION.json').read_text())
    rule = selection['rule']
    assert 'local_ba' in selection['episodes'][0]
    started = time.perf_counter()
    data = load_npz(folder/'target.npz')
    emb = load_npz(folder/'selective_embedding.npz')
    np.testing.assert_array_equal(emb['rows'], data['rows'])
    embedding = emb['embedding'].astype(float)
    routing = router(info, data)
    with np.load(folder/'policies.npz') as p:
        arm = list(p['modes']).index('selective')
        expected = {k: p[k][:, arm] for k in ('base_predictions', 'predictions', 'selected')}
    assert data['support'].shape == (5, 64)
    shape = (5, len(MODES), len(data['rows']))
    base = np.empty(shape, dtype=np.int8)
    full = np.empty(shape, dtype=np.int8)
    frozen_full = np.empty(shape, dtype=np.int8)
    selected = np.empty(shape, dtype=bool)
    local_gammas = np.empty((5, len(data['rows'])), dtype=np.float32)
    diagnostics, loo_archive = [], {}
    for draw, support in enumerate(data['support']):
        sy, ev = data['support_labels'][draw], data['eval_masks'][draw]
        assert len(np.unique(support)) == 64 and not ev[support].any()
        k0, k1 = kernels(data, embedding, support, info['eta'])
        l0, l1 = strict_loo(k0[support], k1[support], data['prior'][support], sy)
        gamma = ce_strength(l0, l1, sy)
        utility = point_ce(l0, sy)-point_ce(l1, sy)
        local_gamma = local_strength(k1-k0, utility)
        s0 = data['prior'] + k0 @ coefficients(k0[support], sy, data['prior'][support])
        s1 = data['prior'] + k1 @ coefficients(k1[support], sy, data['prior'][support])
        delta = s1-s0
        scale = {'baseline': 0., 'learned': 1., 'uniform_half': .5,
                 'loo_ce': gamma, 'local': local_gamma[:, None]}
        scale['source_selected'] = scale[rule]
        for i, mode in enumerate(MODES):
            score = s0 + scale[mode]*delta
            if mode == 'learned' or (mode == 'source_selected' and rule == 'learned'):
                score = s1
            bp = score.argmax(axis=1)
            pred, sel = routing(score, data['expert'][draw], sy, ev)
            assert not sel[~ev].any() and sel.sum() <= int(.05*ev.sum())
            if mode in ('baseline', 'learned'):
                references = (data['expected_base'][draw], data['expected_full'][draw], data['expected_selected'][draw]) if mode == 'baseline' else (
                    expected['base_predictions'][draw], expected['predictions'][draw], expected['selected'][draw])
                for actual, reference in zip((bp, pred, sel), references):
                    np.testing.assert_array_equal(actual, reference, err_msg=info['task']+' '+mode)
            base[draw, i], full[draw, i], selected[draw, i] = bp, pred, sel
            frozen_full[draw, i] = np.where(expected['selected'][draw], data['expert'][draw].argmax(1), bp)
        local_gammas[draw] = local_gamma
        diagnostics.append(dict(draw=draw, rule=rule, gamma=gamma,
            local_gamma_mean=float(local_gamma[ev].mean()),
            local_gamma_std=float(local_gamma[ev].std()),
            local_gamma_p10=float(np.quantile(local_gamma[ev], .1)),
            local_gamma_p90=float(np.quantile(local_gamma[ev], .9)),
            loo_utility_positive=int(np.sum(utility > 0)), loo_utility_negative=int(np.sum(utility < 0)),
            loo_utility_mean=float(utility.mean()),
            loo_ce_old=balanced_ce(l0, sy), loo_ce_new=balanced_ce(l1, sy),
            loo_ce_shrunk=balanced_ce(l0+gamma*(l1-l0), sy),
            loo_ba_old=balanced_accuracy(l0, sy), loo_ba_new=balanced_accuracy(l1, sy),
            loo_ba_shrunk=balanced_accuracy(l0+gamma*(l1-l0), sy)))
        for key, values in [('old', l0), ('new', l1), ('utility', utility)]:
            loo_archive[f'{key}_{draw}'] = values
        print(f'TARGET {info["task"]} draw{draw} gamma={gamma:.4f} local={local_gamma[ev].mean():.4f} replay exact', flush=True)
    np.savez_compressed(out/'policies.npz', modes=np.array(MODES), base_predictions=base,
        predictions=full, frozen_predictions=frozen_full, selected=selected,
        support=data['support'], eval_masks=data['eval_masks'], rows=data['rows'],
        gamma=np.array([d['gamma'] for d in diagnostics]), local_gamma=local_gammas)
    np.savez_compressed(out/'loo.npz', **loo_archive)
    pd.DataFrame(diagnostics).to_csv(out/'diagnostics.csv', index=False)
    write(out/'COMPLETE.json', dict(dataset=info['dataset'], task=info['task'], rule=rule,
        draws=5, support_n=64, budget=.05, baseline_exact=True, learned_exact=True,
        target_eval_labels_read=False, seconds=time.perf_counter()-started))


def statistics(y, pred):
    good = y == pred
    recall = [good[y == c].mean() for c in np.unique(y)]
    f1 = []
    for c in range(8):
        tp = np.sum((y == c) & (pred == c))
        denom = np.sum(y == c)+np.sum(pred == c)
        f1.append(2*tp/denom if denom else 0.)
    return 100*good.mean(), 100*np.mean(recall), 100*np.mean(f1)


def report(folders):
    # The only target evaluation-label reads are below this all-predictions gate.
    markers = sorted((OUT/'targets').glob('*/*/COMPLETE.json'))
    if len(markers) != 25:
        raise ValueError('Report requires all25 target predictions already frozen')
    source_y = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id']).class_id.to_numpy()
    metrics, classes, associations, diag, selection_rows = [], [], [], [], []
    for folder in folders:
        info = json.loads((folder/'PREPARED.json').read_text())
        meta = dict(dataset=info['dataset'], task=info['task'])
        out = OUT/'targets'/info['dataset']/info['task']
        p = load_npz(out/'policies.npz')
        y = source_y[p['rows']] if info['dataset'] == 'self24ports' else pd.read_csv(
            ROOT/DATASETS[info['dataset']][0]/'target_manifest.csv', usecols=['class_id']).class_id.to_numpy()
        sy = load_npz(folder/'target.npz')['support_labels']
        sel = json.loads((OUT/'source'/info['dataset']/info['task']/'SELECTION.json').read_text())
        selection_rows.append(dict(**meta, rule=sel['rule'], source_delta_ba=sel['source_mean_delta_ba'],
            **{m+'_ba': np.mean([r[m+'_ba'] for r in sel['episodes']]) for m in ('learned', 'loo_ce', 'local')}))
        dd = pd.read_csv(out/'diagnostics.csv')
        for k, v in meta.items():
            dd[k] = v
        diag.append(dd)
        with np.load(folder/'policies.npz') as original:
            old_index, learned_index = list(original['modes']).index('baseline'), list(original['modes']).index('selective')
            original_scores = original['scores'][:, [old_index, learned_index]].astype(float)
        for draw, ev in enumerate(p['eval_masks']):
            for stage, key in [('base', 'base_predictions'), ('full', 'predictions'), ('frozen_route', 'frozen_predictions')]:
                ref, old = p[key][draw, 1], p[key][draw, 0]
                ref_oa, ref_ba, _ = statistics(y[ev], ref[ev])
                old_oa, old_ba, _ = statistics(y[ev], old[ev])
                rescued, harmed = ev & (ref == y) & (old != y), ev & (ref != y) & (old == y)
                for i, mode in enumerate(MODES):
                    pred = p[key][draw, i]
                    oa, ba, f1 = statistics(y[ev], pred[ev])
                    metrics.append(dict(**meta, draw=draw, stage=stage, mode=mode,
                        oa=oa, ba=ba, macro_f1_8=f1, delta_oa=oa-ref_oa, delta_ba=ba-ref_ba,
                        delta_old_oa=oa-old_oa, delta_old_ba=ba-old_ba,
                        rescues=int(np.sum(ev & (pred == y) & (ref != y))),
                        harms=int(np.sum(ev & (pred != y) & (ref == y))),
                        changes=int(np.sum(ev & (pred != ref))),
                        learned_rescues=int(rescued.sum()), retained_rescues=int(np.sum(rescued & (pred == y))),
                        learned_harms=int(harmed.sum()), repaired_harms=int(np.sum(harmed & (pred == y))),
                        admissions=int(p['selected'][draw, i].sum()) if stage == 'full' else 0))
                    for c in np.unique(y[ev]):
                        mask = ev & (y == c)
                        classes.append(dict(**meta, draw=draw, stage=stage, mode=mode, class_id=int(c),
                            support_n=int(np.sum(sy[draw] == c)), n=int(mask.sum()),
                            recall=100*float(np.mean(pred[mask] == c)),
                            delta_recall=100*float(np.mean(pred[mask] == c)-np.mean(ref[mask] == c))))
            # Descriptive evidence only: no target-derived thresholds or parameter updates.
            s0, s1 = original_scores[draw]
            gain = point_ce(s0[ev], y[ev])-point_ce(s1[ev], y[ev])
            gamma = p['local_gamma'][draw, ev]
            cov = np.corrcoef(gamma, gain)[0, 1] if gamma.std() and gain.std() else np.nan
            for mode, scale in [('learned', 1.), ('uniform_half', .5), ('loo_ce', p['gamma'][draw]), ('local', gamma[:, None])]:
                score = s0[ev]+scale*(s1-s0)[ev]
                associations.append(dict(**meta, draw=draw, mode=mode,
                    eval_balanced_ce=balanced_ce(score, y[ev]),
                    local_gamma_utility_pearson=float(cov),
                    gamma_on_positive_utility=float(gamma[gain > 0].mean()) if np.any(gain > 0) else np.nan,
                    gamma_on_negative_utility=float(gamma[gain < 0].mean()) if np.any(gain < 0) else np.nan))
    m = pd.DataFrame(metrics)
    m.to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'class_metrics.csv', index=False)
    pd.DataFrame(associations).to_csv(OUT/'utility_diagnostics.csv', index=False)
    pd.concat(diag, ignore_index=True).to_csv(OUT/'support_diagnostics.csv', index=False)
    pd.DataFrame(selection_rows).to_csv(OUT/'source_selection.csv', index=False)
    values = ['oa', 'ba', 'macro_f1_8', 'delta_oa', 'delta_ba', 'delta_old_oa', 'delta_old_ba']
    task = m.groupby(['dataset', 'task', 'stage', 'mode'])[values].mean().reset_index()
    task.to_csv(OUT/'task_means.csv', index=False)
    summary = task.groupby(['dataset', 'stage', 'mode'])[values].mean().reset_index()
    summary.to_csv(OUT/'summary.csv', index=False)
    m.groupby(['dataset', 'draw', 'stage', 'mode'])[values].mean().to_csv(OUT/'draw_metrics.csv')
    counts = ['rescues', 'harms', 'changes', 'learned_rescues', 'retained_rescues', 'learned_harms', 'repaired_harms', 'admissions']
    m.groupby(['dataset', 'stage', 'mode'])[counts].sum().to_csv(OUT/'events.csv')
    write(OUT/'STATUS.json', dict(complete=True, tasks=25, draws=125,
        all_baseline_and_learned_predictions_and_routes_exact=True,
        source_episodes=100, source_rule_choices=pd.DataFrame(selection_rows).rule.value_counts().to_dict(),
        target_eval_labels_only_in_report=True, new_maps=0, no_gpu=True,
        local_added_before_target_readout=True, independent_confirmation=False))
    print(summary[summary.stage.eq('full')].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['predict', 'report', 'all'], default='all')
    args = parser.parse_args()
    folders = [p.parent for p in sorted((EXP/'tasks').rglob('EVALUATED.json'))]
    assert len(folders) == 25
    if args.stage != 'report':
        for folder in folders:
            predict(folder)
    if args.stage != 'predict':
        report(folders)
