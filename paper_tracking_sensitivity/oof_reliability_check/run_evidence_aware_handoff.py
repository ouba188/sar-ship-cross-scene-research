"""Source-selected stable score calibration and evidence-aware handoff gates."""

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '2'

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evidence_aware_handoff import (ALPHAS, select_recalibration, gate_features,
                                  fit_gate, predict_gate, route_scores)
from receiver_consistent_handoff import gate_scores
from run_generic_transfer import metrics
from sar_joint_completion import SOURCE, write_json


ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
PARENT = ROOT/'module_completion_20261008/ports_projection'
TRACE = ROOT/'latent_usage_attribution_20261008'
OUT = ROOT/'evidence_aware_handoff_20261008'
ARMS = ('reference', 'stable_recalibration', 'source_scale_control', 'matched_score_gate', 'evidence_gate')


def archive(path):
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k] for k in saved.files}


def protocol():
    record = dict(date='2026-10-08', task='Use previously implicit stable evidence, not add feature bank',
        arms=list(ARMS), labels_per_target_draw=64, target_draws=5, budget=.05,
        stable_recalibration='Base64 + alpha * candidate-span source-base-head contribution',
        source_scale_control='Base64 + alpha_control * entire source-base-head prior',
        alpha_grid=list(ALPHAS), alpha_selection='Equal source-held-port mean Base64 BA; ties prefer alpha0 then grid order',
        source_episodes='Original inner identity/product-purged train rows; fixed64 seed0 supports; other rows validate',
        candidate_discovery='Refit source raw statistics, pair directions, nuisance penalty and SVD excluding inner held port',
        source_heads='Cached inner geometry; base ridge1 refit; original expert source head; ridge8 target64 adaptation',
        encoder='Frozen current per-outer-port M3 tail; saw source episode labels, NEVER outer target labels',
        source_gate='Logistic C1, informative eligible query outcomes; equal port then true-class weights',
        score_gate_features='expert/base margin,expert/base entropy,total predicted-pair advantage',
        evidence_gate_features='Same five plus stable-component predicted-pair advantage',
        remainder='Total minus stable advantage, not an extra linearly redundant gate feature',
        gate_admission='Same disagreement,observed expertclass,positive score,floor(.05*Nquery),row-ID tie order',
        controls='Recalibration arms use old gate; two refit gates use unchanged Base64 and expert',
        combined='Not in first-stage arms; consider only after separate gains are read out',
        frozen_target_expert=True, query_model_selection=False, independent_confirmation=False,
        original_model_unchanged=True, reconstruction='Target latent trace inherits documented recovered-PCA approximation')
    path = OUT/'PROTOCOL.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != record:
            raise ValueError('Changed protocol must use new output')
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(path, record)


def train_source(port):
    folder = OUT/'ports'/port
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'SOURCE_FITTED.json').exists():
        with (folder/'source_models.pkl').open('rb') as f:
            return pickle.load(f)
    descriptor = json.loads((OUT/'source_episodes'/port/'COMPLETE.json').read_text(encoding='utf-8'))
    episodes = [archive(Path(item['path'])) for item in descriptor['episodes']]
    calibration = select_recalibration(episodes)
    groups = {key: [] for key in ('base', 'expert', 'labels', 'eligible', 'weights', 'score_features', 'evidence_features')}
    for episode in episodes:
        b, e = episode['base_scores'], episode['expert_scores']
        y, query = episode['labels'], episode['query']
        if query[episode['support']].any():
            raise ValueError('Support label reused as source gate query outcome')
        observed = np.bincount(y[episode['support']], minlength=8) > 0
        eligible = (b.argmax(1) != e.argmax(1)) & observed[e.argmax(1)]
        features = gate_features(b, e, episode['stable_base'], episode['stable_expert'])
        _, inverse, count = np.unique(y[query], return_inverse=True, return_counts=True)
        weights = 1./(len(episodes)*len(count)*count[inverse])
        values = dict(base=b[query], expert=e[query], labels=y[query], eligible=eligible[query], weights=weights,
            score_features=features['base_features'][query], evidence_features=features['evidence_features'][query])
        for key, value in values.items():
            groups[key].append(value)
    bank = {key: np.concatenate(value) for key, value in groups.items()}
    if not np.isclose(bank['weights'].sum(), 1.):
        raise ValueError('Source episode weighting is not normalized')
    models = {}
    for name in ('score', 'evidence'):
        models[name] = fit_gate(bank[name+'_features'], bank['base'], bank['expert'], bank['labels'],
                                bank['eligible'], bank['weights'])
    result = dict(calibration=calibration, models=models)
    with (folder/'source_models.pkl').open('wb') as f:
        pickle.dump(result, f)
    write_json(folder/'SOURCE_FITTED.json', dict(port=port, episodes=len(episodes),
        source_query_rows=len(bank['labels']), calibration=calibration,
        source_models='source_models.pkl', outer_target_labels_used=0,
        note='Head/direction held-port; M3 encoder source-trained, not inner-OOF encoder'))
    print(f'SOURCE FIT {port}: stable alpha={calibration["stable_alpha"]}, '
          f'scale alpha={calibration["scale_alpha"]}, queries={len(bank["labels"])}', flush=True)
    return result


def freeze_target(port, fitted, labels):
    folder = OUT/'ports'/port
    if (folder/'TARGET_FROZEN.json').exists():
        return
    old = archive(PARENT/port/'matched_joint/joint_policies.npz')
    trace = archive(TRACE/'ports'/port/'trace.npz')
    np.testing.assert_array_equal(old['query_rows'], trace['query_rows'])
    np.testing.assert_array_equal(old['source_rows'], trace['source_rows'])
    np.testing.assert_array_equal(old['support_rows'], trace['support_rows'])
    np.testing.assert_array_equal(old['eval_mask'], trace['eval_mask'])
    with (PARENT/port/'source_model.pkl').open('rb') as f:
        gate = pickle.load(f)['gate']
    with np.load(ROOT/'pred_cache'/f'{port}.npz') as saved:
        source_prior = saved['base_logits'][old['query_rows']].astype(float)
    names = list(trace['component_names'])
    csource = trace['contributions'][:, names.index('base_source')]
    cb = csource+trace['contributions'][:, names.index('base_adaptation')]
    ce = sum(trace['contributions'][:, names.index(k)] for k in
             ('expert_source', 'expert_retained', 'expert_complement'))
    records, scores, bases = [], [], []
    alpha, control = fitted['calibration']['stable_alpha'], fitted['calibration']['scale_alpha']
    for draw, ev in enumerate(old['eval_mask']):
        b, e = old['base_scores'][draw], old['expert_scores'][draw]
        views = dict(reference=b, stable_recalibration=b+alpha*csource[draw],
                     source_scale_control=b+control*source_prior,
                     matched_score_gate=b, evidence_gate=b)
        features = gate_features(b, e, cb[draw], ce[draw])
        gates = {name: gate_scores(views[name], e, gate) for name in ARMS[:3]}
        gates['matched_score_gate'] = predict_gate(fitted['models']['score'], features['base_features'])
        gates['evidence_gate'] = predict_gate(fitted['models']['evidence'], features['evidence_features'])
        support_labels = labels[old['support_rows'][draw]]
        predictions = []
        for name in ARMS:
            r = route_scores(views[name], e, gates[name], support_labels, ev, old['query_rows'])
            if name == 'reference':
                np.testing.assert_array_equal(r['predictions'], old['predictions'][draw])
                np.testing.assert_array_equal(r['selected'], old['selected'][draw])
            if r['cap'] != old['cap'][draw] or r['selected'][~ev].any():
                raise ValueError('Budget or support exclusion changed')
            predictions.append(r)
        records.append(predictions)
        scores.append(np.stack([gates[k] for k in ARMS]))
        bases.append(np.stack([views[k] for k in ARMS]))
    np.savez_compressed(folder/'policies.npz', query_rows=old['query_rows'], source_rows=old['source_rows'],
        support_rows=old['support_rows'], eval_mask=old['eval_mask'], arm_names=np.asarray(ARMS),
        expert_scores=old['expert_scores'], base_scores=np.stack(bases), gate_scores=np.stack(scores),
        **{k: np.stack([[r[k] for r in draw] for draw in records]) for k in records[0][0]})
    write_json(folder/'TARGET_FROZEN.json', dict(port=port, arms=list(ARMS), draws=5,
        reference_replay_exact=True, stable_alpha=alpha, source_scale_alpha=control))
    print('TARGET FROZEN '+port, flush=True)


def evaluate(inventory, labels):
    rows, coefficients = [], []
    for item in inventory:
        port = item['port']
        p = archive(OUT/'ports'/port/'policies.npz')
        y = labels[p['query_rows']]
        fitted = json.loads((OUT/'ports'/port/'SOURCE_FITTED.json').read_text(encoding='utf-8'))
        coefficients.append(dict(port=port, stable_alpha=fitted['calibration']['stable_alpha'],
            source_scale_alpha=fitted['calibration']['scale_alpha']))
        for draw, ev in enumerate(p['eval_mask']):
            ref = p['predictions'][draw, 0]
            for i, arm in enumerate(ARMS):
                pred = p['predictions'][draw, i]
                base = p['base_predictions'][draw, i]
                m, bm = metrics(y[ev], pred[ev], 8), metrics(y[ev], base[ev], 8)
                rows.append(dict(port=port, draw=draw, arm=arm, acc=m['acc'], ba=m['ba'],
                    base_acc=bm['acc'], base_ba=bm['ba'], selected=int(p['selected'][draw,i].sum()),
                    query_rows=int(ev.sum()), rescued=int(np.sum(ev & (pred == y) & (base != y))),
                    harmed=int(np.sum(ev & (pred != y) & (base == y))),
                    prediction_change=float(np.mean(pred[ev] != ref[ev]))))
    df = pd.DataFrame(rows)
    df.to_csv(OUT/'metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(coefficients).to_csv(OUT/'source_selected_parameters.csv', index=False)
    pm = df.groupby(['port', 'arm']).mean(numeric_only=True).drop(columns='draw').reset_index()
    pm.to_csv(OUT/'port_means.csv', index=False, encoding='utf-8-sig')
    means = pm.groupby('arm').mean(numeric_only=True).reset_index()
    means.to_csv(OUT/'equal_port_means.csv', index=False, encoding='utf-8-sig')
    contrasts = []
    for a, b in (('stable_recalibration', 'reference'), ('stable_recalibration', 'source_scale_control'),
                 ('evidence_gate', 'reference'), ('evidence_gate', 'matched_score_gate'),
                 ('matched_score_gate', 'reference')):
        av, bv = pm[pm.arm == a].set_index('port'), pm[pm.arm == b].set_index('port')
        delta = av.ba-bv.ba
        contrasts.append(dict(candidate=a, control=b, delta_ba_pp=float(100*delta.mean()),
            delta_oa_pp=float(100*(av.acc-bv.acc).mean()), positive_ports=int((delta>1e-12).sum()),
            negative_ports=int((delta < -1e-12).sum())))
    write_json(OUT/'COMPLETE.json', dict(ports=23, draws=5, arms=list(ARMS), rows=len(df),
        means=means.to_dict('records'), contrasts=contrasts, main_model_unchanged=True,
        target_query_selection=False, independent_confirmation=False))
    print(means.to_string(index=False), flush=True)
    print(json.dumps(contrasts, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ports', nargs='*')
    parser.add_argument('--protocol-only', action='store_true')
    args = parser.parse_args()
    protocol()
    if args.protocol_only:
        return
    if (OUT/'COMPLETE.json').exists():
        raise ValueError('Preserve completed experiment')
    inventory = json.loads((ROOT/'oof_base_adapt_results_20261007/POLICIES_SAVED.json').read_text(encoding='utf-8'))
    labels = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id']).class_id.to_numpy()
    with threadpool_limits(limits=2):
        for item in inventory:
            port = item['port']
            if args.ports and port not in args.ports:
                continue
            if not (OUT/'source_episodes'/port/'COMPLETE.json').exists():
                print('SOURCE EPISODES NOT READY '+port, flush=True)
                continue
            fitted = train_source(port)
            freeze_target(port, fitted, labels)
    if all((OUT/'ports'/item['port']/'TARGET_FROZEN.json').exists() for item in inventory):
        write_json(OUT/'ALL_TARGETS_FROZEN.json', dict(ports=23, arms=list(ARMS)))
        evaluate(inventory, labels)
    else:
        print('PARTIAL FREEZE; no target query metrics read', flush=True)


if __name__ == '__main__':
    main()
