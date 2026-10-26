"""Matched M2 and fixed-matcher M3 readout refits; no metric training."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRANSFER = ROOT / 'public_m123_cross_dataset_20261026'
REPAIR = TRANSFER / 'normalization_repair_20261026'
EVIDENCE = ROOT / 'paper_evidence_completion_20261026'
sys.path.insert(0, str(ROOT / 'oss1_m123_replication_20261026'))
import train_frontend as front
import train_m3 as m3
SEEDS = (17, 29, 43)


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def load(path):
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}


def scale2(raw, ids):
    x = raw[ids]
    return dict(mu=x.mean((0, 1), keepdims=True),
                scale=np.maximum(x.std((0, 1), keepdims=True), 1e-8),
                active=(np.ptp(x.reshape(-1, x.shape[-1]), axis=0) > 0)[None, None])


def gate_raw(cached, base, dv, dh, available):
    values = [base-base.mean(1, keepdims=True), dv, dh]
    for s in (base, base+dv, base+dh):
        other = np.stack([np.delete(s, c, 1).max(1) for c in range(4)], 1)
        values.append(s-other)
    values.extend(m3.softmax(s, axis=1) for s in (base, base+dv, base+dh))
    values.extend(np.broadcast_to(available[:, p, None], base.shape) for p in range(2))
    result = cached.copy()
    result[..., :11] = np.stack(values, -1)
    return result


def train():
    torch.set_num_threads(2)
    start = time.perf_counter()
    receipts, replays = [], []
    write(HERE/'PROTOCOL.json', dict(sources=['oss1', 'oss2'], seeds=SEEDS,
        base='regional_mlp fit_only versus original corrected M1',
        M2='original regional_raw 69->12->1, fit_train/fit_validation,200 steps, strict constant-column mask',
        M3='fixed source-trained PairMetric, original32 bank and local cache; refit JointHead, VV, centeredVH, alpha,185-dim frequency gate',
        M3_training='utility_train/selection; heads200steps; gate100steps; AdamW .003/.01; alpha0,.1,...,1',
        selection='exact (OA+BA)/2; retain earliest strict optimum including zero',
        target_labels='masked during all fits; evaluation after all predictions saved',
        scope='readout refit conditional on historical frozen matcher, not entire M3 training',
        base_training_predictions_oof=False, cpu_threads=2, new_architecture=False))
    for source, target in [('oss1', 'oss2'), ('oss2', 'oss1')]:
        direction = f'{source}-to-{target}'
        root = ROOT/f'{source}_m123_replication_20261026'
        table, fit, val = front.load_table(root/'stack_table.csv')
        # M3 sees source utility/selection labels only; held-out evaluation stays masked.
        labels = pd.read_csv(root/'stack_table.csv', usecols=['class_id', 'role'])
        y = labels.class_id.to_numpy(int, copy=True)
        y[labels.role.eq('evaluation')] = -1
        utility = np.flatnonzero(labels.role.eq('utility_train'))
        selection = np.flatnonzero(labels.role.eq('selection'))
        cross = pd.read_csv(TRANSFER/direction/'target_table.csv',
            usecols=['obs_id', 'mmsi', 'pass_group', 'role'], dtype={'mmsi': str})
        sums = front.aligned(root/'frontend/summaries.npz', table)
        csums = front.aligned(TRANSFER/direction/'summaries/summaries.npz', cross)
        strong = load(EVIDENCE/f'classical_fit_only/{source}/source_predictions.npz')
        cstrong = load(EVIDENCE/f'classical_fit_only/{source}/cross_predictions.npz')
        np.testing.assert_array_equal(strong['obs_id'], table.obs_id)
        np.testing.assert_array_equal(cstrong['obs_id'], cross.obs_id)
        for seed in SEEDS:
            src = load(REPAIR/f'source_{source}/seed{seed}/predictions.npz')
            dst = load(REPAIR/f'{direction}/seed{seed}/predictions.npz')
            original = load(root/f'frontend/scores_seed{seed}.npz')
            np.testing.assert_array_equal(original['obs_id'], table.obs_id)
            np.testing.assert_array_equal(dst['obs_id'], cross.obs_id)
            local = np.concatenate([src['local'], dst['local']])
            available = np.concatenate([src['available'], dst['available']]).astype(np.float32)
            cached_gate = np.concatenate([src['gate_raw'], dst['gate_raw']])
            for branch in ('old', 'strong'):
                out = HERE/direction/f'seed{seed}'/branch
                (out/'models').mkdir(parents=True, exist_ok=True)
                sb = src['m1'] if branch == 'old' else strong[f'regional_mlp_seed{seed}']
                cb = dst['m1'] if branch == 'old' else cstrong[f'regional_mlp_seed{seed}']
                sr = front.regional_raw(sb, sums['summary1'], sums['quality'], sums['geometry'])
                cr = front.regional_raw(cb, csums['summary1'], csums['quality'], csums['geometry'])
                sc = scale2(sr, fit)
                sx, cx = m3.transform(sr, sc), m3.transform(cr, sc)
                torch.manual_seed(seed)
                head = front.upstream.RegionHead()
                s2 = front.train_head(out, 'M2', seed, head, sx, sb, table, fit, val, sc['mu'], sc['scale'])
                c2 = cb+front.upstream.predict(head, cx)
                np.savez_compressed(out/'M2_scaler.npz', **sc)
                if branch == 'old':
                    for side, actual, expected in [('source', s2, src['s12']), ('cross', c2, dst['s12'])]:
                        err = float(abs(actual-expected).max())
                        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=0)
                        np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
                        replays.append(dict(source=source, seed=seed, side=side, max_error=err))
                base = np.concatenate([s2, c2]).astype(np.float32)
                raw = m3.local_raw(local, base)
                scaler = m3.scale_fit(raw, utility)
                x = m3.transform(raw, scaler)
                torch.manual_seed(seed)
                joint = m3.JointHead()
                dj, sj = m3.fit_head('joint_refit', joint, x, base, y, utility, selection, out)
                null = m3.transform(np.zeros((1, 1, 15)), scaler).reshape(15)
                deltas, steps = [], []
                for pol, name in enumerate(('vv', 'vh_centered')):
                    torch.manual_seed(seed)
                    polar = m3.PolarHead(pol, centered=bool(pol), null=null)
                    torch.nn.init.zeros_(polar.net[-1].weight)
                    torch.nn.init.zeros_(polar.net[-1].bias)
                    d, step = m3.fit_head(name, polar, x, base, y, utility, selection, out)
                    deltas.append(d)
                    steps.append(step)
                dv, dh = deltas
                alpha = m3.choose_alpha(base, dv, dh, available, y, selection)
                weights = available*np.array([alpha, 1-alpha], np.float32)
                gr = gate_raw(cached_gate, base, dv, dh, available)
                freq, fw, gs = m3.fit_gate('frequency', seed, gr, base, dv, dh, available,
                                           y, utility, selection, alpha, out)
                scores = dict(bare=np.concatenate([sb, cb]), M2=base, joint=base+dj,
                    vv=base+dv*available[:, :1], vh=base+dh*available[:, 1:],
                    B_fixed=base+weights[:, :1]*dv+weights[:, 1:]*dh, frequency=freq)
                extra = dict(delta_vv=dv, delta_vh=dh, weight_frequency=fw, gate_raw=gr)
                for side, sl, tab in [('source', slice(0, len(table)), table), ('cross', slice(len(table), None), cross)]:
                    np.savez_compressed(out/f'{side}_predictions.npz', obs_id=tab.obs_id.to_numpy(),
                        **{k: v[sl] for k, v in scores.items()}, **{k: v[sl] for k, v in extra.items()})
                torch.save(scaler, out/'M3_readout_scaler.pt')
                sel = json.loads((out/f'selection_M2_{seed}.json').read_text())
                receipt = dict(source=source, seed=seed, branch=branch, M2_step=sel['epoch'],
                    joint_step=sj, vv_step=steps[0], vh_step=steps[1], alpha=alpha, frequency_step=gs,
                    M2_source_constant_columns=int((~sc['active']).sum()),
                    base_fit_OA=float((sb[fit].argmax(1)==y[fit]).mean()),
                    base_fit_validation_OA=float((sb[val].argmax(1)==y[val]).mean()))
                receipts.append(receipt)
                write(out/'selection.json', receipt)
                print('SAVED', source, seed, branch, flush=True)
    pd.DataFrame(receipts).to_csv(HERE/'selections.csv', index=False)
    write(HERE/'replay.json', replays)
    write(HERE/'predictions_saved.json', dict(seconds=time.perf_counter()-start,
        all12_configurations_saved=True, target_metrics_read=False, metric_frozen=True))


def evaluate():
    assert (HERE/'predictions_saved.json').exists()
    metrics, pairs, classes = [], [], []
    for source, target in [('oss1', 'oss2'), ('oss2', 'oss1')]:
        direction = f'{source}-to-{target}'
        for seed in SEEDS:
            for branch in ('old', 'strong'):
                out = HERE/direction/f'seed{seed}'/branch
                for side in ('source', 'cross'):
                    tab = pd.read_csv(ROOT/f'{source}_m123_replication_20261026/stack_table.csv' if side=='source'
                                      else TRANSFER/direction/'target_table.csv', dtype={'mmsi': str})
                    z = load(out/f'{side}_predictions.npz')
                    np.testing.assert_array_equal(z['obs_id'], tab.obs_id)
                    for role, ix in tab.groupby('role').groups.items():
                        ix = np.asarray(ix)
                        yy = tab.iloc[ix].class_id.to_numpy(int)
                        meta = dict(source=source, side=side, role=role, seed=seed, branch=branch)
                        for arm in ('bare', 'M2', 'joint', 'vv', 'vh', 'B_fixed', 'frequency'):
                            metrics.append(dict(**meta, arm=arm, **m3.metrics(yy, z[arm][ix])))
                        for before, after in [('bare', 'M2'), ('M2', 'joint'), ('M2', 'B_fixed'),
                                               ('M2', 'frequency'), ('bare', 'frequency'), ('B_fixed', 'frequency')]:
                            a, b = z[before][ix].argmax(1)==yy, z[after][ix].argmax(1)==yy
                            row = dict(**meta, before=before, after=after, n=len(ix),
                                rescue=int((b&~a).sum()), harm=int((a&~b).sum()),
                                OA_delta_pp=float((b.astype(float)-a).mean()*100),
                                BA_delta_pp=float(np.mean([(b[yy==c].astype(float)-a[yy==c]).mean() for c in np.unique(yy)])*100))
                            pairs.append(row)
                            for c in np.unique(yy):
                                take = yy==c
                                classes.append(dict(**meta, before=before, after=after, class_id=int(c), n=int(take.sum()),
                                    rescue=int((b&~a&take).sum()), harm=int((a&~b&take).sum()),
                                    before_recall=float(a[take].mean()), after_recall=float(b[take].mean())))
    frame = pd.DataFrame(metrics)
    frame.to_csv(HERE/'metrics.csv', index=False)
    summary = frame.groupby(['source', 'side', 'role', 'branch', 'arm']).agg(
        OA=('OA', 'mean'), BA=('BA', 'mean'), OA_std=('OA', 'std'), BA_std=('BA', 'std'),
        NLL=('NLL', 'mean'), Brier=('Brier', 'mean'), n=('n', 'first')).reset_index()
    summary.to_csv(HERE/'summary.csv', index=False)
    pd.DataFrame(pairs).to_csv(HERE/'paired.csv', index=False)
    pd.DataFrame(classes).to_csv(HERE/'per_class.csv', index=False)
    print(summary.loc[summary.role.eq('evaluation')].to_string(index=False), flush=True)


if __name__ == '__main__':
    train()
    evaluate()
