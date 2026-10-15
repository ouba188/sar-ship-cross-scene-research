"""Trace frequency gains through support geometry, score terms, and routing."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_learned_frequency import OUT as EXP, ROOT, SOURCE, PARENT, DATASETS, load_npz, load_pickle
from conditional_direction_residual import fit_predict
from cross_dataset_c4 import route

OUT = EXP/'mechanism_20261009'


def dual_coefficients(gram, labels, prior, strength=8.):
    _, inverse, count = np.unique(labels, return_inverse=True, return_counts=True)
    root = np.sqrt(len(labels)/(len(count)*count[inverse]))
    desired = 2*np.eye(prior.shape[1])[labels]-1
    return root[:, None]*np.linalg.solve(root[:, None]*gram*root[None, :]
        +strength*np.eye(len(labels)), root[:, None]*(desired-prior))


def parts(data, embedding, support, labels, eta):
    phi, prior, mean, conditions = [data[k] for k in ('phi', 'prior', 'mean', 'conditions')]
    rbf = np.exp(-np.sum((conditions[:, None]-conditions[support][None])**2, axis=2)/.5)
    km = (mean@mean[support].T)*rbf
    k0 = phi@phi[support].T+eta*km
    kf = (embedding@embedding[support].T)*rbf
    a0 = dual_coefficients(k0[support], labels, prior[support])
    a1 = dual_coefficients(k0[support]+.25*kf[support], labels, prior[support])
    base = fit_predict(phi, prior, mean, conditions, support, labels, eta, conditional=True)
    refit = k0@(a1-a0)
    direct = .25*kf@a1
    full = base+refit+direct
    direct_reference = prior+phi@(phi[support].T@a1)+eta*km@a1+.25*kf@a1
    np.testing.assert_allclose(full, direct_reference, atol=2e-11, rtol=2e-11)
    constant_query = np.broadcast_to(embedding[support].mean(0)@embedding[support].T, kf.shape)*rbf
    variants = dict(baseline=base, learned_frequency=full,
        old_path_refit_only=base+refit,
        direct_path_only=base+direct,
        query_frequency_removed=base+refit+.25*constant_query@a1)
    for name, extra in [('duplicate_mean', km), ('condition_only', rbf)]:
        alpha = dual_coefficients(k0[support]+.25*extra[support], labels, prior[support])
        variants[name] = prior+k0@alpha+.25*extra@alpha
    # Support-only diagonal change isolates a numerical regularization effect.
    ad = dual_coefficients(k0[support]+.25*np.eye(len(support)), labels, prior[support])
    variants['diagonal_only'] = prior+k0@ad
    variants['diagonal_only'][support] += .25*ad
    return variants, dict(refit=refit, direct=direct, k0=k0, kf=kf, km=km, condition=rbf)


def metrics(y, pred):
    good = y == pred
    return 100*good.mean(), 100*np.mean([good[y == c].mean() for c in np.unique(y)])


def balanced_average(values, y, mask):
    return float(np.mean([values[mask & (y == c)].mean() for c in np.unique(y[mask])])) if mask.any() else np.nan


def analyze(folder):
    info = json.loads((folder/'PREPARED.json').read_text())
    out = OUT/info['dataset']/info['task']
    out.mkdir(parents=True, exist_ok=True)
    if (out/'COMPLETE.json').exists() and 'changed_from_learned' in pd.read_csv(out/'metrics.csv', nrows=0).columns:
        return
    data, p = load_npz(folder/'target.npz'), load_npz(folder/'policies.npz')
    e = load_npz(folder/'selective_embedding.npz')['embedding'].astype(float)
    if info['dataset'] == 'self24ports':
        y = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id']).class_id.to_numpy()[data['rows']]
        from receiver_consistent_handoff import confidence_parts
        from evidence_aware_handoff import predict_gate, route_scores
        gate = load_pickle(PARENT/info['task']/'source_models.pkl')['gates']['joint']
        def routing(score, expert, labels, ev):
            em, eh = confidence_parts(expert)
            bm, bh = confidence_parts(score)
            ret = route_scores(score, expert, predict_gate(gate, np.c_[em, bm, eh, bh]),
                               labels, ev, data['route_ids'], budget=.05)
            return ret['predictions'], ret['selected']
    else:
        y = pd.read_csv(ROOT/DATASETS[info['dataset']][0]/'target_manifest.csv', usecols=['class_id']).class_id.to_numpy()
        gate = load_pickle(Path(info['model_path']))['gate']
        def routing(score, expert, labels, ev):
            return route(score, expert, gate, labels, ev, data['route_ids'])
    records, margins, neighbors, classes = [], [], [], []
    meta = dict(dataset=info['dataset'], task=info['task'])
    for draw, support in enumerate(data['support']):
        ev, sy = data['eval_masks'][draw], data['support_labels'][draw]
        variants, values = parts(data, e, support, sy, info['eta'])
        ref = variants['baseline'].argmax(1)
        learned = variants['learned_frequency'].argmax(1)
        np.testing.assert_array_equal(ref, p['base_predictions'][draw, 0])
        arm = list(p['modes']).index('selective')
        np.testing.assert_array_equal(learned, p['base_predictions'][draw, arm])
        for mode, score in variants.items():
            full, selected = routing(score, data['expert'][draw], sy, ev)
            if selected[~ev].any() or selected.sum() > int(.05*ev.sum()):
                raise ValueError('Budget mismatch')
            if mode == 'learned_frequency':
                np.testing.assert_array_equal(full, p['predictions'][draw, arm])
                np.testing.assert_array_equal(selected, p['selected'][draw, arm])
            for stage, pred in [('base', score.argmax(1)), ('full', full)]:
                oa, ba = metrics(y[ev], pred[ev])
                learned_ref = learned if stage == 'base' else p['predictions'][draw, arm]
                records.append(dict(**meta, draw=draw, mode=mode, stage=stage, oa=oa, ba=ba,
                    changed_from_learned=int(np.sum(ev & (pred != learned_ref))),
                    lost_correct_vs_learned=int(np.sum(ev & (learned_ref == y) & (pred != y))),
                    gained_correct_vs_learned=int(np.sum(ev & (learned_ref != y) & (pred == y)))))
                if mode in ('baseline', 'learned_frequency', 'condition_only', 'query_frequency_removed'):
                    for cl in np.unique(y[ev]):
                        mask = ev & (y == cl)
                        classes.append(dict(**meta, draw=draw, mode=mode, stage=stage, class_id=int(cl),
                                            recall=100*np.mean(pred[mask] == cl), count=int(mask.sum())))
        groups = dict(rescued=ev & (ref != y) & (learned == y),
                      harmed=ev & (ref == y) & (learned != y), all=ev)
        rival = np.where(ref != y, ref, learned)
        rr = np.arange(len(y))
        for group, mask in groups.items():
            if group == 'all':
                continue
            for component in ('refit', 'direct'):
                change = values[component][rr, y]-values[component][rr, rival]
                margins.append(dict(**meta, draw=draw, group=group, component=component, count=int(mask.sum()),
                    mean_true_margin_delta=float(change[mask].mean()) if mask.any() else np.nan,
                    positive_fraction=float(np.mean(change[mask] > 0)) if mask.any() else np.nan,
                    sum_true_margin_delta=float(change[mask].sum())))
        supported = np.isin(y, sy)
        for name in ('k0', 'km', 'kf', 'condition'):
            kernel = values[name]
            nearest = kernel.argmax(1)
            hit = (sy[nearest] == y).astype(float)
            top = np.argpartition(-kernel, 4, axis=1)[:, :5]
            purity = np.mean(sy[top] == y[:, None], axis=1)
            for group, mask0 in groups.items():
                mask = mask0 & supported
                neighbors.append(dict(**meta, draw=draw, group=group, kernel=name,
                    query_n=int(mask0.sum()), support_present_n=int(mask.sum()),
                    top1_same_class_macro=100*balanced_average(hit, y, mask),
                    top5_purity_macro=100*balanced_average(purity, y, mask),
                    top1_same_class_micro=100*float(hit[mask].mean()) if mask.any() else np.nan,
                    top5_purity_micro=100*float(purity[mask].mean()) if mask.any() else np.nan))
        print(f'TRACE {info["task"]} draw{draw} exact', flush=True)
    for name, rows in [('metrics', records), ('margin_parts', margins), ('neighbors', neighbors), ('classes', classes)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv', index=False)
    (out/'COMPLETE.json').write_text(json.dumps(dict(**meta, draws=5, exact_replay=True,
        no_refit_network=True, no_target_query_fitting=True, diagnostic_only=True), indent=2))


def report():
    folders = [p.parent for p in sorted(OUT.glob('*/*/COMPLETE.json'))]
    for name, group, fields in [
        ('metrics', ['dataset', 'stage', 'mode'], ['oa', 'ba']),
        ('neighbors', ['dataset', 'group', 'kernel'], ['top1_same_class_macro', 'top5_purity_macro']),
        ('margin_parts', ['dataset', 'group', 'component'], ['mean_true_margin_delta', 'positive_fraction']),
        ('classes', ['dataset', 'stage', 'mode', 'class_id'], ['recall'])]:
        data = pd.concat([pd.read_csv(p/f'{name}.csv') for p in folders], ignore_index=True)
        data.to_csv(OUT/f'{name}_all.csv', index=False)
        summary = data.groupby(group+['task'])[fields].mean().groupby(group).mean().reset_index()
        summary.to_csv(OUT/f'{name}_summary.csv', index=False)
        if name == 'metrics':
            print(summary[summary.stage.eq('full')].to_string(index=False), flush=True)
            data.groupby(['dataset', 'stage', 'mode'])[['changed_from_learned',
                'lost_correct_vs_learned', 'gained_correct_vs_learned']].sum().to_csv(OUT/'intervention_events.csv')
    (OUT/'READOUT.json').write_text(json.dumps(dict(tasks=len(folders), expected=25,
        complete=len(folders) == 25, independent_confirmation=False,
        query_frequency_removed='Support-fitted alpha unchanged; replace query frequency embedding by support mean',
        condition_only='Refit with extra beta*condition RBF, no added frequency features',
        diagonal_only='Extra support diagonal only, zero query-support extra similarities',
        score_identity='new-old = K_old*(alpha_new-alpha_old) + beta*K_frequency*alpha_new',
        limitation='Exact model computation attribution and interventions, not physical causal identification'), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for marker in sorted((EXP/'tasks').rglob('EVALUATED.json')):
        if args.tasks and marker.parent.name not in args.tasks:
            continue
        analyze(marker.parent)
    report()
