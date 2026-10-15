"""Class-generic same64 / 5% LODO, conditional on existing source-trained encoders."""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

from generic_transfer_data import DOMAINS, load_generic_transfer
from receiver_consistent_handoff import confidence_parts
from residual_expert import source_geometry
from run_opensarship2_transfer import ARMS, freeze_draw, target_geometry, write_json


def fit_heads(base, extra, labels, rows, n_classes):
    pca = PCA(n_components=128, random_state=0).fit(base[rows])
    x = pca.transform(base[rows])
    center, scale = x.mean(0), x.std(0)+1e-6
    xs = (x-center)/scale
    b = RidgeClassifier(alpha=1., class_weight='balanced').fit(xs, labels[rows])
    pca_ssl = PCA(n_components=64, random_state=0).fit(extra[rows])
    joint = np.column_stack((xs, pca_ssl.transform(extra[rows])))
    _, ec, es = source_geometry(joint, np.arange(len(rows)))
    _, bc, bs = source_geometry(xs, np.arange(len(rows)))
    e = RidgeClassifier(alpha=1., class_weight='balanced').fit((joint-ec)/es, labels[rows])
    if not np.array_equal(b.classes_, np.arange(n_classes)) or not np.array_equal(e.classes_, b.classes_):
        raise ValueError('source training fold lacks required classes')
    return dict(pca=pca, pca_ssl=pca_ssl, center=center, scale=scale, base=b, expert=e,
        expert_center=ec, expert_scale=es, residual_center=bc, residual_scale=bs)


def fit_gate(base_logits, expert_logits, labels, domains):
    bm, bh = confidence_parts(base_logits)
    em, eh = confidence_parts(expert_logits)
    x = np.column_stack((em, bm, eh, bh))
    weights = np.empty(len(labels))
    unique_domains = np.unique(domains)
    for domain in unique_domains:
        mask = domains == domain
        classes, inverse, counts = np.unique(labels[mask], return_inverse=True, return_counts=True)
        weights[mask] = 1./(len(unique_domains)*len(classes)*counts[inverse])
    center = np.average(x, axis=0, weights=weights)
    scale = np.maximum(np.sqrt(np.average((x-center)**2, axis=0, weights=weights)), 1e-6)
    features = (x-center)/scale
    b, e = base_logits.argmax(1), expert_logits.argmax(1)
    delta = (e == labels).astype(float)-(b == labels).astype(float)
    informative = (b != e) & (delta != 0)
    if np.unique(delta[informative]).size != 2:
        raise ValueError('source informative choices lack both signs')
    lr = LogisticRegression(C=1., max_iter=2000, solver='lbfgs', tol=1e-9)
    lr.fit(features[informative], (delta[informative] > 0).astype(int),
           sample_weight=len(labels)*weights[informative])
    if lr.n_iter_.max() >= lr.max_iter:
        raise ValueError('source gate did not converge')
    return dict(center=center, scale=scale, lr=np.r_[lr.coef_[0], lr.intercept_[0]],
                source_disagreements=int((b != e).sum()), source_informative=int(informative.sum()))


def metrics(labels, predictions, n_classes):
    matrix = confusion_matrix(labels, predictions, labels=np.arange(n_classes))
    count = matrix.sum(1)
    present = count > 0
    recall = np.divide(matrix.diagonal(), count, out=np.full(n_classes, np.nan), where=present)
    return dict(acc=float(matrix.trace()/matrix.sum()), ba=float(recall[present].mean()),
                class_recall=[float(x) if np.isfinite(x) else None for x in recall])


def run_fold(out, data):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'summary.json').exists():
        raise ValueError('completed fold must not be overwritten')
    base, extra, labels = data['base_features'], data['extra_features'], data['labels']
    source, target = data['source_rows'], data['target_rows']
    domains, k = data['domains'], data['n_classes']
    write_json(out/'protocol.json', dict(provenance=data['provenance'],
        n_classes=k, source_rows=len(source), target_rows=len(target),
        labels_per_draw=64, seeds=list(range(5)), budget=.05, base_pca=128, extra_pca=64,
        source_ridge=1., residual_ridge=8., gate_lr_c=1.,
        source_oof='three inner leave-one-source-domain-out HEAD folds; source-supervised encoder fixed',
        no_target_query_model_selection=True, no_target_frequency_class_filter=True,
        metric_scope='query only, support excluded', independent_confirmation=False))
    pd.DataFrame(dict(row_id=target, class_id=labels[target], domain=domains[target])).to_csv(
        out/'target_manifest.csv', index=False)
    oof_b = np.full((len(source), k), np.nan, dtype=np.float32)
    oof_e = np.full_like(oof_b, np.nan)
    for held_domain in np.unique(domains[source]):
        held = source[domains[source] == held_domain]
        train = source[domains[source] != held_domain]
        model = fit_heads(base, extra, labels, train, k)
        b, e, _, _ = target_geometry(model, base[held], extra[held])
        positions = np.searchsorted(source, held)
        oof_b[positions], oof_e[positions] = b, e
        np.savez_compressed(out/f'head_oof_{held_domain}.npz', train_rows=train,
                            held_rows=held, base=b, expert=e)
        print(f'HEAD OOF {data["dataset"]}/{data["target_domain"]} held={held_domain}', flush=True)
    if not np.isfinite(oof_b).all() or not np.isfinite(oof_e).all():
        raise ValueError('incomplete source head-OOF')
    gate = fit_gate(oof_b, oof_e, labels[source], domains[source])
    model = fit_heads(base, extra, labels, source, k)
    model['gate'] = gate
    with (out/'source_model.pkl').open('wb') as handle:
        pickle.dump(model, handle)
    prior_b, prior_e, phi_b, phi_e = target_geometry(model, base[target], extra[target])
    draws = []
    for seed in range(5):
        support = np.random.default_rng(seed).choice(len(target), 64, replace=False)
        draws.append(freeze_draw(prior_b, prior_e, phi_b, phi_e, gate, support,
                                 labels[target[support]], target))
    np.savez_compressed(out/'policies.npz', **{key: np.stack([d[key] for d in draws]) for key in draws[0]},
        arm_names=np.asarray(ARMS), seeds=np.arange(5), target_rows=target,
        prior_base_scores=prior_b, prior_expert_scores=prior_e)
    write_json(out/'POLICIES_SAVED.json', dict(draws=5, class_count=k,
        labels_per_draw=64, all_policies_saved_before_query_readout=True))
    records = []
    for seed, policy in enumerate(draws):
        ev, selected = policy['eval_masks'], policy['selected']
        if selected[~ev].any() or selected.sum() > policy['caps']:
            raise ValueError('support/budget violation')
        if not np.array_equal(policy['predictions'][2] != policy['predictions'][1], selected):
            raise ValueError('handoff prediction changes differ from admissions')
        for ai, arm in enumerate(ARMS):
            records.append(dict(seed=seed, arm=arm,
                **metrics(labels[target][ev], policy['predictions'][ai, ev], k),
                query_rows=int(ev.sum()), admitted_rows=int(selected.sum()) if ai == 2 else 0,
                cap=int(policy['caps']), observed_support_classes=int(policy['observed_classes'].sum())))
    write_json(out/'metrics.json', records)
    pd.DataFrame([{key: value for key, value in r.items() if key != 'class_recall'}
                  for r in records]).to_csv(out/'metrics.csv', index=False)
    means = [dict(arm=arm, **{key: float(np.mean([r[key] for r in records if r['arm'] == arm]))
                             for key in ('acc', 'ba', 'admitted_rows')}) for arm in ARMS]
    differences = {key: [100*(records[3*s+2][key]-records[3*s+1][key]) for s in range(5)]
                   for key in ('acc', 'ba')}
    summary = dict(dataset=data['dataset'], target_domain=data['target_domain'],
        n_classes=k, target_rows=len(target), means=means, handoff_draw_deltas_pp=differences,
        handoff_mean_deltas_pp={key: float(np.mean(v)) for key, v in differences.items()},
        ba_positive_draws=int(np.count_nonzero(np.asarray(differences['ba']) > 0)),
        observed_support_classes=[int(d['observed_classes'].sum()) for d in draws],
        inner_encoder_oof=False, independent_confirmation=False)
    write_json(out/'summary.json', summary)
    print(f'RESULT {data["dataset"]}/{data["target_domain"]}: {summary["handoff_mean_deltas_pp"]}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=tuple(DOMAINS), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    with threadpool_limits(limits=4):
        for domain in DOMAINS[args.dataset]:
            data = load_generic_transfer(args.dataset, domain)
            summaries.append(run_fold(args.out/domain.replace(' ', '_'), data))
    macro = [dict(arm=arm, **{key: float(np.mean([s['means'][i][key] for s in summaries]))
                             for key in ('acc', 'ba')}) for i, arm in enumerate(ARMS)]
    write_json(args.out/'summary.json', dict(dataset=args.dataset, domains=summaries,
        domain_macro_means=macro, aggregation='equal-domain average of five support-draw means',
        mean_handoff_delta_oa_pp=100*(macro[2]['acc']-macro[1]['acc']),
        mean_handoff_delta_ba_pp=100*(macro[2]['ba']-macro[1]['ba']),
        positive_domains=sum(s['handoff_mean_deltas_pp']['ba'] > 0 for s in summaries),
        inner_encoder_oof=False, independent_confirmation=False))
    print(f'DONE {args.dataset}', flush=True)


if __name__ == '__main__':
    main()
