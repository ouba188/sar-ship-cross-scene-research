"""Source-defined pair directions and fixed-readout task sensitivity, no fits."""

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '2'

import itertools
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from run_protected_drift_activation import OUT as PARENT, ROOT, SOURCE, make_views
from sar_joint_completion import write_json

OUT = ROOT / 'task_stable_sensitivity_20261009'
ARMS = ('matched_baseline', 'stable_only', 'joint')


def pair_geometry(z, y, domains, a, b, minimum=5):
    """Equal-port contrasts on the same ports for both classes."""
    cells = []
    for port in np.unique(domains):
        ia = np.flatnonzero((domains == port) & (y == a))
        ib = np.flatnonzero((domains == port) & (y == b))
        if min(len(ia), len(ib)) >= minimum:
            cells.append((z[ia].mean(0), z[ib].mean(0)))
    if len(cells) < 3:
        return None
    means = np.asarray(cells)
    centers = means.mean(0)
    delta = centers[0] - centers[1]
    gap = np.linalg.norm(delta)
    if gap <= 1e-12:
        return None
    direction = delta / gap
    drift = (means - centers).reshape(-1, z.shape[1])
    drift_sd = np.sqrt(np.mean(np.square(drift @ direction)))
    return dict(direction=direction, midpoint=centers.mean(0), delta=delta,
                drift=drift, source_ports=len(cells), source_gap=float(gap),
                port_sd_along_task=float(drift_sd),
                port_sd_over_gap=float(drift_sd / gap),
                total_port_rms_over_gap=float(np.sqrt(np.square(drift).sum(1).mean()) / gap))


def binary_ba(y, margin, a, b):
    pred = np.where(margin >= 0, a, b)
    return float(.5 * (np.mean(pred[y == a] == a) + np.mean(pred[y == b] == b)))


def held_geometry(z, y, a, b, geometry):
    u, center = geometry['direction'], geometry['midpoint']
    delta = z[y == a].mean(0) - z[y == b].mean(0)
    signed = (z - center) @ u
    norm = np.linalg.norm(delta)
    return dict(auc=float(roc_auc_score(y == a, signed)),
                fixed_midpoint_ba=binary_ba(y, signed, a, b),
                target_gap_ratio=float(delta @ u / geometry['source_gap']),
                target_cosine=float(delta @ u / norm) if norm > 0 else np.nan,
                reversed=bool(delta @ u < 0),
                midpoint_shift_over_gap=float(((z[y == a].mean(0) + z[y == b].mean(0)) / 2
                                               - center) @ u / geometry['source_gap']))


def stable_jacobians(g):
    raw = g['qs'].T * g['feature_scale']
    pca = g['model']['pca'].components_
    xs = (raw @ pca.T) / g['model']['scale']
    complement = (raw - (raw @ pca.T) @ pca) / g['stats']['discarded_scale'][:2048]
    return xs, complement


def readout_gradient(g, heads, policies, support_views, arm, draw, head):
    r = g['stable_rank']
    xs, comp = stable_jacobians(g)
    coef = heads[arm + '_' + head]['coef']
    grad = xs @ coef[:128] + coef[192:192+r] / g['stable_scale'][:, None]
    beta = policies[arm + '_' + head + '_coefficients'][draw]
    activate = arm != 'matched_baseline'
    if head == 'base':
        grad += (xs / g['base_scale'] / np.sqrt(128)) @ beta[:128]
        if activate:
            grad += beta[128:128+r] / g['stable_scale'][:, None] / np.sqrt(r)
    else:
        v = support_views['common'][arm == 'joint']
        dk = .5 * ((xs / g['model']['expert_scale'][:128] / np.sqrt(192))
                    @ v['retained'][:, :128].T
                    + (comp / np.sqrt(2560)) @ v['complement'][:, :2048].T)
        if activate:
            dk += .25 * support_views['stable'].T / g['stable_scale'][:, None] / np.sqrt(r)
        grad += dk @ beta
    return grad


def direct_scores(bank, views, support_views, heads, p, arm, draw, head):
    h = heads[arm + '_' + head]
    beta = p[arm + '_' + head + '_coefficients'][draw]
    score = bank @ h['coef'] + h['intercept']
    suppress, activate = arm == 'joint', arm != 'matched_baseline'
    if head == 'base':
        return score + views['phis'][suppress, activate] @ beta
    q, s = views['common'][suppress], support_views['common'][suppress]
    kernel = .5 * (q['retained'] @ s['retained'].T + q['complement'] @ s['complement'].T) + 1.
    if activate:
        kernel += .25 * views['stable'] @ support_views['stable'].T
    return score + kernel @ beta


def verify_gradients(g, bank, raw, extra, heads, p, support_views, gradients):
    selected = np.flatnonzero(p['eval_mask'][0])[:3]
    before_bank = np.asarray(bank[selected], dtype=float)
    before_raw = np.asarray(raw[p['query_rows'][selected]], dtype=float)
    before = make_views(before_bank, before_raw, extra[selected], g)
    r = g['stable_rank']
    shift = np.linspace(-.5, .5, r)
    after_bank = before_bank.copy()
    xs, _ = stable_jacobians(g)
    after_bank[:, :128] += shift @ xs
    after_bank[:, 192:192+r] += shift / g['stable_scale']
    after_raw = before_raw + (shift @ g['qs'].T) * g['feature_scale']
    after = make_views(after_bank, after_raw, extra[selected], g)
    replay, derivative = 0., 0.
    for arm in ARMS:
        index = list(p['arm_names']).index(arm)
        for head in ('base', 'expert'):
            original = direct_scores(before_bank, before, support_views, heads, p, arm, 0, head)
            changed = direct_scores(after_bank, after, support_views, heads, p, arm, 0, head)
            err = np.max(np.abs(original - p[head + '_scores'][0, index, selected]))
            replay = max(replay, float(err))
            err = np.max(np.abs(changed - original - shift @ gradients[arm, head]))
            derivative = max(derivative, float(err))
    if max(replay, derivative) > 3e-6:
        raise ValueError(f'Readout reconstruction failed: replay={replay}, derivative={derivative}')
    return replay, derivative


def run_port(folder, frame, raw):
    with (folder / 'geometry.pkl').open('rb') as f:
        g = pickle.load(f)
    with (folder / 'source_models.pkl').open('rb') as f:
        heads = pickle.load(f)['heads']
    with np.load(folder / 'policies.npz') as f:
        p = {k: f[k] for k in f.files}
    srows, trows = np.load(folder / 'source_rows.npy'), np.load(folder / 'target_rows.npy')
    np.testing.assert_array_equal(trows, p['query_rows'])
    np.testing.assert_array_equal(srows, p['source_rows'])
    if (frame.port.to_numpy()[srows] == folder.name).any():
        raise ValueError('Target rows in source-defined pair geometry')
    sb = np.load(folder / 'source_bank.npy', mmap_mode='r')
    tb = np.load(folder / 'target_bank.npy', mmap_mode='r')
    r = g['stable_rank']
    sz = np.asarray(sb[:, 192:192+r]) * g['stable_scale'] + g['stable_center']
    tz = np.asarray(tb[:, 192:192+r]) * g['stable_scale'] + g['stable_center']
    spaces = {'stable': (sz, tz), 'ordinary_rank_matched': (sb[:, :r], tb[:, :r])}
    y, domains = frame.class_id.to_numpy(), frame.port.to_numpy()
    geometry = {}
    source_records, query_records, decision_records = [], [], []
    for name, (s, _) in spaces.items():
        for a, b in itertools.combinations(range(8), 2):
            v = pair_geometry(s, y[srows], domains[srows], a, b)
            if v is not None:
                geometry[name, a, b] = v
                source_records.append(dict(port=folder.name, space=name, a=a, b=b, rank=r,
                    **{k: v[k] for k in ('source_ports', 'source_gap', 'port_sd_along_task',
                                         'port_sd_over_gap', 'total_port_rms_over_gap')}))
    extra = np.load(ROOT / 'module_completion_20261008/ports_projection' / folder.name
                    / 'matched_joint/joint_extra.npy', mmap_mode='r')
    replay = derivative = 0.
    for draw, ev in enumerate(p['eval_mask']):
        support = np.searchsorted(trows, p['support_rows'][draw])
        np.testing.assert_array_equal(trows[support], p['support_rows'][draw])
        if ev[support].any():
            raise ValueError('Support rows enter query metrics')
        support_views = make_views(tb[support], raw[trows[support]], extra[support], g)
        gradients = {(arm, head): readout_gradient(g, heads, p, support_views, arm, draw, head)
                     for arm in ARMS for head in ('base', 'expert')}
        if draw == 0:
            replay, derivative = verify_gradients(g, tb, raw, extra, heads, p, support_views, gradients)
        for (space, a, b), geo in geometry.items():
            use = ev & np.isin(y[trows], [a, b])
            labels = y[trows][use]
            if min(np.count_nonzero(labels == a), np.count_nonzero(labels == b)) < 5:
                continue
            z = spaces[space][1][use]
            key = dict(port=folder.name, draw=draw, a=a, b=b)
            query_records.append(dict(**key, space=space, n_a=int(sum(labels == a)),
                                      n_b=int(sum(labels == b)), **held_geometry(z, labels, a, b, geo)))
            if space != 'stable':
                continue
            coordinate = (z - geo['midpoint']) @ geo['direction']
            for (arm, head), grad in gradients.items():
                j = list(p['arm_names']).index(arm)
                scores = p[head + '_scores'][draw, j, use]
                margin = scores[:, a] - scores[:, b]
                pair_grad = grad[:, a] - grad[:, b]
                removed = coordinate * (geo['direction'] @ pair_grad)
                response = geo['drift'] @ pair_grad
                before = binary_ba(labels, margin, a, b)
                after = binary_ba(labels, margin - removed, a, b)
                decision_records.append(dict(**key, arm=arm, head=head,
                    pair_ba=before, direction_removed_pair_ba=after,
                    removal_delta_pp=100 * (after-before),
                    pair_prediction_flip=float(np.mean((margin >= 0) != (margin-removed >= 0))),
                    source_class_margin_response=float(geo['delta'] @ pair_grad),
                    same_class_port_margin_rms=float(np.sqrt(np.mean(np.square(response))))))
    return source_records, query_records, decision_records, dict(port=folder.name, rank=r,
        max_score_replay_error=replay, max_raw_perturbation_derivative_error=derivative)


def summarize():
    source = pd.read_csv(OUT / 'source_pair_geometry.csv')
    query = pd.read_csv(OUT / 'target_pair_metrics.csv')
    decisions = pd.read_csv(OUT / 'decision_sensitivity.csv')
    qpm = query.groupby(['port', 'space', 'a', 'b']).mean(numeric_only=True).drop(columns='draw').reset_index()
    qpm.to_csv(OUT / 'target_port_pair_means.csv', index=False)
    matched = qpm.merge(source[['port', 'space', 'a', 'b', 'port_sd_over_gap',
                                 'total_port_rms_over_gap']], on=['port', 'space', 'a', 'b'])
    summary = matched.groupby(['space', 'a', 'b']).mean(numeric_only=True).reset_index()
    summary['target_ports'] = qpm.groupby(['space', 'a', 'b']).size().to_numpy()
    summary.to_csv(OUT / 'pair_summary.csv', index=False)
    dpm = decisions.groupby(['port', 'arm', 'head', 'a', 'b']).mean(numeric_only=True).drop(columns='draw').reset_index()
    ds = dpm.groupby(['arm', 'head', 'a', 'b']).mean(numeric_only=True).reset_index()
    ds.to_csv(OUT / 'decision_pair_summary.csv', index=False)
    # Each pair averages its eligible ports; the overall diagnostic averages pairs.
    summary.drop(columns=['a', 'b']).groupby('space').mean(numeric_only=True).to_csv(OUT / 'equal_pair_summary.csv')
    ds.drop(columns=['a', 'b']).groupby(['arm', 'head']).mean(numeric_only=True).to_csv(OUT / 'decision_overall.csv')
    plot(summary, ds)
    return summary, ds


def plot(summary, decisions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    stable = summary[summary.space == 'stable'].copy()
    control = summary[summary.space == 'ordinary_rank_matched'].set_index(['a', 'b'])
    stable = stable.sort_values(['a', 'b'])
    names = [f'{a}-{b}' for a, b in zip(stable.a, stable.b)]
    x = np.arange(len(names))
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    axes[0].bar(x, stable.port_sd_over_gap, color='#197d80')
    axes[0].set_ylabel('Source port SD / class gap')
    axes[0].set_title('Task-specific candidate stability; source-defined directions, no new training')
    axes[1].plot(x, 100*stable.auc, 'o-', label='Candidate stable space', color='#197d80')
    axes[1].plot(x, [100*control.loc[a, b].auc for a, b in zip(stable.a, stable.b)],
                 's--', label='Same-rank ordinary PCA coordinates', color='#a63b63')
    axes[1].axhline(50, color='#777777', linewidth=.8)
    axes[1].set_ylabel('Held-port pair AUC (%)')
    axes[1].legend()
    for head, offset, color in [('base', -.18, '#197d80'), ('expert', .18, '#a63b63')]:
        d = decisions[(decisions.arm == 'stable_only') & (decisions['head'] == head)].set_index(['a', 'b'])
        axes[2].bar(x+offset, [d.loc[a, b].removal_delta_pp for a, b in zip(stable.a, stable.b)],
                    width=.36, label=head, color=color)
    axes[2].axhline(0, color='#777777', linewidth=.8)
    axes[2].set_ylabel('Pair BA change after\ndirection removal (pp)')
    axes[2].legend()
    axes[2].set_xticks(x, names, rotation=45)
    axes[2].set_xlabel('Class pair (not an eight-class end-to-end accuracy test)')
    fig.savefig(OUT / 'task_sensitivity.png', dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(date='2026-10-09', new_training=False, main_model_unchanged=True,
        source_direction='Mean class contrast, equal weight over common source ports with >=5 samples each class; >=3 common ports',
        spaces=['Raw orthonormal Qs coordinates', 'First r existing standardized PCA coordinates, practical same-rank comparator'],
        query='Five saved query masks excluding original64 supports; >=5 per class; all available pairs',
        frozen_models=list(ARMS), sensitivity='Pair margin changes; R50 stable-direction query-only projection to source pair midpoint; support coefficients fixed',
        weights='Draw mean within port/pair; eligible ports equal per pair; pairs equal in overall diagnostic',
        boundary='Existing development data, source-fitted descriptive geometry; no causal or absolute-invariance claim; not full eight-class or handoff gains',
        no_target_selection=True, diagnostic_target_labels_only=True)
    write_json(OUT / 'PROTOCOL.json', protocol)
    frame = pd.read_csv(SOURCE / 'manifest.csv', usecols=['class_id', 'port'], keep_default_na=False)
    raw = np.load(SOURCE / 'pooled_features.npy', mmap_mode='r')
    folders = [p for p in sorted((PARENT / 'ports').iterdir()) if (p / 'policies.npz').exists()]
    sr, qr, dr, checks = [], [], [], []
    with threadpool_limits(limits=2):
        for folder in folders:
            s, q, d, check = run_port(folder, frame, raw)
            sr.extend(s)
            qr.extend(q)
            dr.extend(d)
            checks.append(check)
            print(json.dumps(check), flush=True)
    for name, records in [('source_pair_geometry', sr), ('target_pair_metrics', qr),
                          ('decision_sensitivity', dr), ('verification', checks)]:
        pd.DataFrame(records).to_csv(OUT / (name + '.csv'), index=False)
    summary, _ = summarize()
    write_json(OUT / 'COMPLETE.json', dict(ports=len(folders), source_records=len(sr),
        query_records=len(qr), decision_records=len(dr), new_model_fits=0,
        max_score_replay_error=max(c['max_score_replay_error'] for c in checks),
        max_raw_perturbation_derivative_error=max(c['max_raw_perturbation_derivative_error'] for c in checks)))
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
