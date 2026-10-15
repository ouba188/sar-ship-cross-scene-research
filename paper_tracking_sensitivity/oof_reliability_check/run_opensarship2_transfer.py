"""Frozen same-64 / 5%-handoff transfer from existing source to public OSS2."""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits

from build_oof_predictions import exclude_identities, fit_predict_heads, load_manifest
from budgeted_residual_handoff import ranked_mask, score_policy
from receiver_consistent_handoff import gate_scores
from residual_expert import fit_correction, source_geometry
from run_oof_reliability import fit_reliability

WS = Path('E:/\u4e34\u65f6\u4f1a\u8bdd/visual_reliable_baseline')
SOURCE = WS/'artifacts/eight_class_adaptive_20260916'
TARGET = Path('D:/Documents/project_root/project_data/public_opensarship2/opensarship2')
ARMS = ('original_base', 'base_visual64', 'visual_handoff5')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def prepare_source(out):
    manifest = load_manifest(SOURCE/'manifest.csv')
    target = pd.read_csv(TARGET/'metadata/opensarship2_manifest_native.csv', dtype=str, keep_default_na=False)
    keep = np.ones(len(manifest['labels']), dtype=bool)
    overlaps = {}
    for source_key, target_key in [('mmsi', 'mmsi_or_ship_id'), ('product', 'scene_id')]:
        identifiers = target[target_key].to_numpy(dtype=str)
        identifiers = np.unique(identifiers[identifiers != ''])
        hit = np.isin(manifest[source_key], identifiers)
        keep &= ~hit
        overlaps[source_key] = dict(source_rows=int(hit.sum()), shared_ids=int(len(
            np.intersect1d(manifest[source_key][hit], identifiers))))
    rows = np.flatnonzero(keep)
    np.savez_compressed(out/'source_split.npz', source_rows=rows, excluded_source_rows=np.flatnonzero(~keep))
    write_json(out/'source_split.json', dict(source_rows=len(rows), excluded_source_rows=int((~keep).sum()),
        overlaps=overlaps, policy='Purge source identities against ALL 9120 target rows before fitting; existing frozen encoders are not retrained.'))
    base = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
    with np.load(WS/'b2_ssl_feats_38091.npz') as archive:
        extra = archive['F'].astype(np.float32)
    return manifest, rows, base, extra


def fit_source(out, manifest, rows, base, extra):
    labels, ports = manifest['labels'], manifest['ports']
    cache = out/'source_oof'
    cache.mkdir(exist_ok=True)
    oof_base = np.full((len(rows), 8), np.nan, dtype=np.float32)
    oof_expert = oof_base.copy()
    for number, port in enumerate(np.unique(ports[rows])):
        held = rows[ports[rows] == port]
        all_held = np.flatnonzero(ports == port)
        train = exclude_identities(rows[ports[rows] != port], all_held, manifest)
        path = cache/f'{port}.npz'
        started = time.monotonic()
        if path.exists():
            with np.load(path) as d:
                if not np.array_equal(d['train_rows'], train) or not np.array_equal(d['held_rows'], held):
                    raise ValueError('saved source fold belongs to a different cohort')
                b, e = d['base'], d['expert']
        else:
            b, e = fit_predict_heads(base, extra, labels, train, held)
            np.savez_compressed(path, train_rows=train, held_rows=held, base=b, expert=e)
        oof_base[np.searchsorted(rows, held)] = b
        oof_expert[np.searchsorted(rows, held)] = e
        print(f'OOF {number+1}/{len(np.unique(ports[rows]))} {port} '
              f'train={len(train)} held={len(held)} {time.monotonic()-started:.1f}s', flush=True)
    if not np.isfinite(oof_base).all() or not np.isfinite(oof_expert).all():
        raise ValueError('incomplete source OOF coverage')
    gate = fit_reliability(oof_base, oof_expert, labels[rows], ports[rows])
    np.savez_compressed(out/'source_oof.npz', rows=rows, base=oof_base, expert=oof_expert,
                        labels=labels[rows], ports=ports[rows])
    pca = PCA(n_components=128, random_state=0).fit(base[rows])
    x = pca.transform(base[rows])
    center, scale = x.mean(0), x.std(0)+1e-6
    xs = (x-center)/scale
    model_b = RidgeClassifier(alpha=1., class_weight='balanced').fit(xs, labels[rows])
    pca_ssl = PCA(n_components=64, random_state=0).fit(extra[rows])
    z = pca_ssl.transform(extra[rows])
    joint = np.column_stack((xs, z))
    phi_e, expert_center, expert_scale = source_geometry(joint, np.arange(len(rows)))
    phi_b, residual_center, residual_scale = source_geometry(xs, np.arange(len(rows)))
    model_e = RidgeClassifier(alpha=1., class_weight='balanced').fit(
        (joint-expert_center)/expert_scale, labels[rows])
    if not np.array_equal(model_b.classes_, np.arange(8)) or not np.array_equal(model_e.classes_, np.arange(8)):
        raise ValueError('source class-column alignment differs')
    model = dict(pca=pca, pca_ssl=pca_ssl, center=center, scale=scale, base=model_b,
                 expert=model_e, expert_center=expert_center, expert_scale=expert_scale,
                 residual_center=residual_center, residual_scale=residual_scale, gate=gate)
    with (out/'source_model.pkl').open('wb') as f:
        pickle.dump(model, f)
    print('SOURCE MODEL SAVED', flush=True)
    return model


def target_geometry(model, base, extra):
    xs = ((model['pca'].transform(base)-model['center'])/model['scale']).astype(np.float32)
    joint = np.column_stack((xs, model['pca_ssl'].transform(extra).astype(np.float32)))
    xe = (joint-model['expert_center'])/model['expert_scale']
    xb = (xs-model['residual_center'])/model['residual_scale']
    phi_b = np.column_stack((xb/np.sqrt(128), np.ones(len(base))))
    phi_e = np.column_stack((xe/np.sqrt(192), np.ones(len(base))))
    return (model['base'].decision_function(xs).astype(np.float32).astype(float),
            model['expert'].decision_function(xe).astype(np.float32).astype(float), phi_b, phi_e)


def freeze_draw(prior_b, prior_e, phi_b, phi_e, gate, support, support_labels, row_ids):
    """Only the 64 support labels enter this function; query truth is absent."""
    if len(support) != 64 or len(np.unique(support)) != 64 or len(support_labels) != 64:
        raise ValueError('requires exactly 64 unique support annotations')
    evaluation = ~np.isin(np.arange(len(row_ids)), support)
    cap = int(np.floor(.05*evaluation.sum()))
    coef_b = fit_correction(phi_b[support], support_labels, prior_b[support])
    coef_e = fit_correction(phi_e[support], support_labels, prior_e[support])
    adapted_b, adapted_e = prior_b+phi_b@coef_b, prior_e+phi_e@coef_e
    b, e = adapted_b.argmax(1), adapted_e.argmax(1)
    score = gate_scores(adapted_b, adapted_e, gate)
    observed = np.bincount(support_labels, minlength=prior_b.shape[1]) > 0
    eligible = evaluation & (b != e) & observed[e] & (score > 0)
    selected = ranked_mask(score, eligible, row_ids, cap)
    predictions = np.stack((prior_b.argmax(1), b, np.where(selected, e, b)))
    return dict(support_rows=support, eval_masks=evaluation, caps=np.asarray(cap),
        predictions=predictions, selected=selected, gate_scores=score,
        base_correction=coef_b, expert_correction=coef_e,
        base_scores=adapted_b, expert_scores=adapted_e,
        observed_classes=observed)


def freeze(out, model):
    target = pd.read_csv(out/'target_manifest.csv', keep_default_na=False)
    with np.load(out/'features.npz') as f:
        if not np.array_equal(f['sample_ids'], target.sample_id.to_numpy(dtype=str)):
            raise ValueError('target features/manifest order differs')
        prior_b, prior_e, phi_b, phi_e = target_geometry(model, f['paired_r50'], f['vh_ssl'])
    draws = []
    for seed in range(5):
        support = np.random.default_rng(seed).choice(len(target), 64, replace=False)
        support_labels = target.iloc[support].class_id.to_numpy(dtype=int)
        draws.append(freeze_draw(prior_b, prior_e, phi_b, phi_e, model['gate'], support,
                                 support_labels, target.target_manifest_row.to_numpy()))
    np.savez_compressed(out/'policies.npz', **{k: np.stack([d[k] for d in draws]) for k in draws[0]},
        arm_names=np.asarray(ARMS), seeds=np.arange(5), sample_ids=target.sample_id.to_numpy(dtype=str),
        prior_base_scores=prior_b, prior_expert_scores=prior_e)
    write_json(out/'POLICIES_SAVED.json', dict(draws=5, arms=list(ARMS), target_rows=len(target),
        labels_per_draw=64, split='row-level random support; same-vessel query rows can remain',
        independent_confirmation=False, no_target_query_model_selection=True))
    print('ALL FIVE POLICIES SAVED BEFORE QUERY READOUT', flush=True)


def evaluate(out, *, target_name='public OpenSARShip2 mapped eight-class cohort',
             encoder_scope='Existing frozen paired R50 and F0921-domain SSL R18; source identity purge applies to heads/gate.'):
    target = pd.read_csv(out/'target_manifest.csv', keep_default_na=False)
    y = target.class_id.to_numpy(dtype=int)
    with np.load(out/'policies.npz') as archive:
        p = {k: archive[k] for k in archive.files}
    rows = []
    for i in range(5):
        ev, selected = p['eval_masks'][i], p['selected'][i]
        if selected[~ev].any() or selected.sum() > p['caps'][i]:
            raise ValueError('support/budget violation')
        if not np.array_equal(p['predictions'][i, 2] != p['predictions'][i, 1], selected):
            raise ValueError('handoff changes differ from admissions')
        for ai, arm in enumerate(ARMS):
            pred = p['predictions'][i, ai]
            rows.append(dict(seed=i, arm=arm, **score_policy(pred, p['predictions'][i, 0], y, ev),
                evaluated_rows=int(ev.sum()), cap=int(p['caps'][i]),
                admitted_rows=int(selected.sum()) if ai == 2 else 0,
                admission_fraction=float(selected.sum()/ev.sum()) if ai == 2 else 0.,
                support_class_counts=np.bincount(y[p['support_rows'][i]], minlength=8).tolist()))
    write_json(out/'metrics.json', rows)
    pd.DataFrame([{k: v for k, v in r.items() if k not in ('class_recall', 'support_class_counts')}
                  for r in rows]).to_csv(out/'metrics.csv', index=False)
    means = []
    for arm in ARMS:
        arm_rows = [r for r in rows if r['arm'] == arm]
        recalls = []
        for c in range(8):
            present = [r['class_recall'][c] for r in arm_rows if r['class_recall'][c] is not None]
            recalls.append(float(np.mean(present)) if present else None)
        means.append(dict(arm=arm, class_recall=recalls,
            **{m: float(np.mean([r[m] for r in arm_rows])) for m in ('ba', 'acc', 'admission_fraction')}))
    contrasts = []
    for a, b in [(2, 1), (1, 0), (2, 0)]:
        differences = [100*(rows[3*i+a]['ba']-rows[3*i+b]['ba']) for i in range(5)]
        contrasts.append(dict(candidate=ARMS[a], control=ARMS[b], mean_delta_ba_pp=float(np.mean(differences)),
                              draw_deltas_pp=differences, positive_draws=int(np.sum(np.asarray(differences)>0))))
    summary = dict(target=target_name, target_rows=len(target),
        evaluated_rows_per_draw=len(target)-64, target_labels_per_draw=64, draws=5, means=means,
        comparisons=contrasts, statistical_unit='one dataset, five overlapping support draws; not five independent datasets',
        independent_confirmation=False, interpretation='Cross-dataset transfer under frozen parameters, not new blind data.',
        encoder_scope=encoder_scope,
        cost='5% expert admission, not 5% feature inference cost or 5% changes relative to original base')
    write_json(out/'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=('source', 'finish', 'all'), default='all')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'summary.json').exists():
        raise ValueError('completed experiment must not be overwritten')
    write_json(args.out/'RUN_CONFIG.json', dict(date='2026-10-07', arms=list(ARMS),
        seeds=list(range(5)), labels_per_draw=64, support_sampling='uniform without replacement, no class balancing',
        base_pca=128, ssl_pca=64, source_ridge=1., residual_ridge=8., gate_lr_c=1.,
        budget=.05, gate_threshold=0., source_identity_purge=True,
        target_split='row-level, matching current main experiment',
        no_quality_filter=True, no_target_tuning=True, original_results_untouched=True))
    with threadpool_limits(limits=4):
        if args.stage in ('source', 'all'):
            manifest, rows, base, extra = prepare_source(args.out)
            model = fit_source(args.out, manifest, rows, base, extra)
        else:
            with (args.out/'source_model.pkl').open('rb') as f:
                model = pickle.load(f)
        if args.stage in ('finish', 'all'):
            freeze(args.out, model)
            evaluate(args.out)


if __name__ == '__main__':
    main()
