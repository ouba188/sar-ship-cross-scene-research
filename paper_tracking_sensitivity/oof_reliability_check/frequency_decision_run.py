"""Gate-level held-source-port selection and frozen target endpoint decisions."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from prepare_learned_frequency import OUT as EXP, ROOT, SOURCE, DATASETS, load_npz
from frequency_reliability_run import router, statistics
from frequency_reliability_core import kernels, coefficients, local_strength, point_ce
from frequency_decision_core import decision_features, fit_benefit, predict_benefit, endpoint_scores

OUT = ROOT/'frequency_decision_20261009'
CANDIDATES = ('decision_only', 'decision_support')
MODES = ('baseline', 'learned', 'previous_local', *CANDIDATES, 'source_selected')


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def source_fit(folder):
    info = json.loads((folder/'PREPARED.json').read_text())
    out = OUT/'source'/info['dataset']/info['task']
    if (out/'SELECTION.json').exists():
        return
    prepared = json.loads((out/'PREPARED.json').read_text())
    d = load_npz(out/'examples.npz')
    f, y, ep = d['features'], d['labels'], d['episode']
    old, new = d['old'], d['new']
    a, b = old.argmax(1), new.argmax(1)
    utility = (b == y).astype(float)-(a == y)
    conflict = a != b
    widths = dict(decision_only=prepared['decision_width'], decision_support=f.shape[1])
    oof, records = np.zeros((len(y), 2)), []
    for held in np.unique(ep):
        train, valid = (ep != held) & conflict, ep == held
        for j, mode in enumerate(CANDIDATES):
            model = fit_benefit(f[train, :widths[mode]], utility[train], d['weights'][train])
            value = predict_benefit(model, f[valid, :widths[mode]])
            oof[valid, j] = value
            score = endpoint_scores(old[valid], new[valid], value)
            _, ba, _ = statistics(y[valid], score.argmax(1))
            _, baseline_ba, _ = statistics(y[valid], b[valid])
            rec = dict(episode=int(held), mode=mode, n=int(valid.sum()), train_conflicts=int(train.sum()),
                ba=ba, learned_ba=baseline_ba, delta_ba=ba-baseline_ba,
                source_rejected=int(np.sum(score.argmax(1) != b[valid])))
            records.append(rec)
    validation = pd.DataFrame(records)
    gains = validation.groupby('mode').delta_ba.mean().to_dict()
    best = max(('learned', *CANDIDATES), key=lambda m: 0. if m == 'learned' else gains[m])
    models = {m: fit_benefit(f[conflict, :widths[m]], utility[conflict], d['weights'][conflict]) for m in CANDIDATES}
    with (out/'models.pkl').open('wb') as handle:
        pickle.dump(models, handle)
    np.savez_compressed(out/'oof.npz', benefit=oof, utility=utility, disagreement=conflict,
        episode=ep, labels=y, old_predictions=a, new_predictions=b)
    validation.to_csv(out/'validation.csv', index=False)
    write(out/'SELECTION.json', dict(dataset=info['dataset'], task=info['task'], rule=best,
        source_delta_ba=gains, decision_width=prepared['decision_width'], features=prepared['feature_names'],
        source_queries=len(y), source_conflicts=int(conflict.sum()), source_rescues=int(np.sum(utility > 0)),
        source_harms=int(np.sum(utility < 0)), source_ports=len(np.unique(ep)),
        validation='Each gate fold excludes its held source episode port; representation not independently held',
        no_target_labels=True))
    print(f'DECISION FIT {info["task"]}: {best}, source BA gains {gains}', flush=True)


def target_predict(folder):
    info = json.loads((folder/'PREPARED.json').read_text())
    out = OUT/'targets'/info['dataset']/info['task']
    out.mkdir(parents=True, exist_ok=True)
    if (out/'COMPLETE.json').exists():
        return
    src = OUT/'source'/info['dataset']/info['task']
    selection = json.loads((src/'SELECTION.json').read_text())
    with (src/'models.pkl').open('rb') as handle:
        models = pickle.load(handle)
    data, emb = load_npz(folder/'target.npz'), load_npz(folder/'selective_embedding.npz')
    np.testing.assert_array_equal(emb['rows'], data['rows'])
    embedding = emb['embedding'].astype(float)
    previous = ROOT/'frequency_reliability_20261009/targets'/info['dataset']/info['task']
    prior_policy, prior_loo = load_npz(previous/'policies.npz'), load_npz(previous/'loo.npz')
    original = load_npz(folder/'policies.npz')
    arm = list(original['modes']).index('selective')
    routing = router(info, data)
    shape = (5, len(MODES), len(data['rows']))
    base, full, frozen = [np.empty(shape, np.int8) for _ in range(3)]
    selected = np.zeros(shape, bool)
    benefits = np.empty((5, 2, len(data['rows'])), np.float32)
    for draw, support in enumerate(data['support']):
        sy, ev = data['support_labels'][draw], data['eval_masks'][draw]
        np.testing.assert_array_equal(support, prior_policy['support'][draw])
        if len(np.unique(support)) != 64 or ev[support].any():
            raise ValueError('Support budget/split mismatch')
        k0, k1 = kernels(data, embedding, support, info['eta'])
        s0 = data['prior']+k0@coefficients(k0[support], sy, data['prior'][support])
        s1 = data['prior']+k1@coefficients(k1[support], sy, data['prior'][support])
        l0, l1 = prior_loo[f'old_{draw}'], prior_loo[f'new_{draw}']
        f, names, width = decision_features(s0, s1, (k1-k0)/.25, sy, l0, l1)
        if names != selection['features'] or width != selection['decision_width']:
            raise ValueError('Source/target feature contract mismatch')
        local_gamma = local_strength(k1-k0, point_ce(l0, sy)-point_ce(l1, sy))
        score = dict(baseline=s0, learned=s1,
            previous_local=s0+local_gamma[:, None]*(s1-s0))
        for j, mode in enumerate(CANDIDATES):
            features = f[:, :width] if mode == 'decision_only' else f
            benefit = predict_benefit(models[mode], features)
            benefits[draw, j] = benefit
            score[mode] = endpoint_scores(s0, s1, benefit)
        score['source_selected'] = score[selection['rule']]
        for j, mode in enumerate(MODES):
            bp = score[mode].argmax(1)
            pred, sel = routing(score[mode], data['expert'][draw], sy, ev)
            if sel[~ev].any() or sel.sum() > int(.05*ev.sum()):
                raise ValueError('Expert handoff cap exceeded')
            if mode in ('baseline', 'learned', 'previous_local'):
                prev = list(prior_policy['modes']).index('local' if mode == 'previous_local' else mode)
                np.testing.assert_array_equal(bp, prior_policy['base_predictions'][draw, prev])
                np.testing.assert_array_equal(pred, prior_policy['predictions'][draw, prev])
                np.testing.assert_array_equal(sel, prior_policy['selected'][draw, prev])
            base[draw, j], full[draw, j], selected[draw, j] = bp, pred, sel
            frozen[draw, j] = np.where(original['selected'][draw, arm], data['expert'][draw].argmax(1), bp)
        print(f'DECISION TARGET {info["task"]} draw{draw}', flush=True)
    np.savez_compressed(out/'policies.npz', modes=np.array(MODES), base_predictions=base,
        predictions=full, frozen_predictions=frozen, selected=selected, benefit=benefits,
        rows=data['rows'], support=data['support'], eval_masks=data['eval_masks'])
    write(out/'COMPLETE.json', dict(dataset=info['dataset'], task=info['task'], rule=selection['rule'],
        draws=5, labels=64, handoff_fraction=.05, target_eval_labels_read=False, endpoint_policy=True))


def report(folders, output=OUT, candidates=CANDIDATES):
    markers = sorted((output/'targets').glob('*/*/COMPLETE.json'))
    if len(markers) != 25:
        raise ValueError('Freeze all target predictions before label readout')
    ys = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id']).class_id.to_numpy()
    records, classes, aucs, source_records = [], [], [], []
    for folder in folders:
        info = json.loads((folder/'PREPARED.json').read_text())
        meta = dict(dataset=info['dataset'], task=info['task'])
        out = output/'targets'/info['dataset']/info['task']
        p = load_npz(out/'policies.npz')
        y = ys[p['rows']] if info['dataset'] == 'self24ports' else pd.read_csv(
            ROOT/DATASETS[info['dataset']][0]/'target_manifest.csv', usecols=['class_id']).class_id.to_numpy()
        source = json.loads((output/'source'/info['dataset']/info['task']/'SELECTION.json').read_text())
        source_records.append(dict(**meta, rule=source['rule'], queries=source['source_queries'],
            conflicts=source['source_conflicts'], rescues=source['source_rescues'], harms=source['source_harms'],
            **source['source_delta_ba']))
        for draw, ev in enumerate(p['eval_masks']):
            for stage, key in [('base', 'base_predictions'), ('full', 'predictions'), ('frozen_route', 'frozen_predictions')]:
                ref, old = p[key][draw, 1], p[key][draw, 0]
                roa, rba, _ = statistics(y[ev], ref[ev])
                rescued, harmed = ev & (ref == y) & (old != y), ev & (ref != y) & (old == y)
                for j, mode in enumerate(p['modes']):
                    pred = p[key][draw, j]
                    oa, ba, f1 = statistics(y[ev], pred[ev])
                    records.append(dict(**meta, draw=draw, stage=stage, mode=mode,
                        oa=oa, ba=ba, macro_f1_8=f1, delta_oa=oa-roa, delta_ba=ba-rba,
                        changes=int(np.sum(ev & (pred != ref))),
                        rescued=int(np.sum(ev & (pred == y) & (ref != y))),
                        harmed=int(np.sum(ev & (pred != y) & (ref == y))),
                        original_rescues=int(rescued.sum()), lost_original_rescues=int(np.sum(rescued & (pred != y))),
                        original_harms=int(harmed.sum()), repaired_original_harms=int(np.sum(harmed & (pred == y)))))
                    for c in np.unique(y[ev]):
                        mask = ev & (y == c)
                        classes.append(dict(**meta, draw=draw, stage=stage, mode=mode, class_id=int(c), n=int(mask.sum()),
                            recall=100*float(np.mean(pred[mask] == c)),
                            delta_recall=100*float(np.mean(pred[mask] == c)-np.mean(ref[mask] == c))))
                event = rescued | harmed
                for j, mode in enumerate(candidates):
                    value = p['benefit'][draw, j]
                    aucs.append(dict(**meta, draw=draw, stage=stage, mode=mode,
                        n=int(event.sum()), auc=float(roc_auc_score(rescued[event], value[event]))
                            if rescued.any() and harmed.any() else np.nan,
                        rescue_benefit=float(value[rescued].mean()) if rescued.any() else np.nan,
                        harm_benefit=float(value[harmed].mean()) if harmed.any() else np.nan))
    m = pd.DataFrame(records)
    m.to_csv(output/'metrics.csv', index=False)
    pd.DataFrame(classes).to_csv(output/'class_metrics.csv', index=False)
    pd.DataFrame(source_records).to_csv(output/'source_selection.csv', index=False)
    if aucs:
        pd.DataFrame(aucs).to_csv(output/'utility_metrics.csv', index=False)
    fields = ['oa', 'ba', 'macro_f1_8', 'delta_oa', 'delta_ba']
    task = m.groupby(['dataset', 'task', 'stage', 'mode'])[fields].mean().reset_index()
    task.to_csv(output/'task_means.csv', index=False)
    summary = task.groupby(['dataset', 'stage', 'mode'])[fields].mean().reset_index()
    summary.to_csv(output/'summary.csv', index=False)
    m.groupby(['dataset', 'draw', 'stage', 'mode'])[fields].mean().to_csv(output/'draw_metrics.csv')
    counts = ['changes', 'rescued', 'harmed', 'original_rescues', 'lost_original_rescues', 'original_harms', 'repaired_original_harms']
    m.groupby(['dataset', 'stage', 'mode'])[counts].sum().to_csv(output/'events.csv')
    if aucs:
        pd.DataFrame(aucs).groupby(['dataset', 'stage', 'mode', 'task'])[['auc', 'rescue_benefit', 'harm_benefit']].mean().groupby(
            ['dataset', 'stage', 'mode']).mean().to_csv(output/'utility_summary.csv')
    write(output/'STATUS.json', dict(complete=True, tasks=25, draws=125,
        endpoint_policy=True, original_paths_replayed=True, target_query_labels_not_in_fit=True,
        independent_confirmation=False, selection_counts=pd.DataFrame(source_records).rule.value_counts().to_dict()))
    print(summary[summary.stage.eq('full')].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['source', 'target', 'report', 'all'], default='all')
    args = parser.parse_args()
    folders = [p.parent for p in sorted((EXP/'tasks').rglob('EVALUATED.json'))]
    if args.stage in ('source', 'all'):
        for folder in folders:
            source_fit(folder)
    if args.stage in ('target', 'all'):
        for folder in folders:
            target_predict(folder)
    if args.stage in ('report', 'all'):
        report(folders)
