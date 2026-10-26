"""Matched summary factorial and actual mass-weighted regional pooling ablations."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import importlib.util
import time
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

prior = module('readout_bypass', ROOT/'raw_region_bypass_20261026/run.py')
aligned = module('aligned_training', ROOT/'m3_aligned_training_20261026/run.py')
base, m3, tr = prior.base, aligned.m3, aligned.tr
ARMS = ('full', 'mean_meta', 'attention_only', 'mean_only', 'raw_only', 'merged_raw', 'body_raw')
SEEDS = (17, 29, 43)
DEST = HERE/'readout'


def inputs(raw, local, auxiliary, arm):
    x = base.representations(raw, local)['regional']
    aux = auxiliary.copy()
    summary = aux[:, :72].reshape(-1, 2, 3, 4, 3)
    if arm in ('mean_meta', 'mean_only'):
        summary[..., 1:] = 0
    if arm in ('attention_only', 'mean_only'):
        aux[:, 72:] = 0
    if arm in ('raw_only', 'merged_raw', 'body_raw'):
        aux[:] = 0
    if arm in ('merged_raw', 'body_raw'):
        ids = np.arange(20 if arm == 'merged_raw' else 8)
        mass = local['mass'][:, :, ids]
        pooled = (local['layer1'][:, :, ids]*mass[..., None]).sum(2)/np.maximum(mass.sum(2)[..., None], 1e-8)
        # Three identical slots remove region identity without changing head size.
        x[:, :-2] = np.repeat(pooled[:, :, None], 3, axis=2).reshape(len(x), -1)
    return x, aux


def refit(b, cache, table, seed, out):
    utility = np.flatnonzero(table.role.eq('utility_train'))
    selection = np.flatnonzero(table.role.eq('selection'))
    y = table.class_id.to_numpy(int, copy=True)
    y[table.role.eq('evaluation')] = -1
    local, av = cache['local'], cache['available']
    sc = m3.scale_fit(m3.local_raw(local, b), utility)
    x = m3.transform(m3.local_raw(local, b), sc)
    null = m3.transform(np.zeros((1, 1, 15)), sc).reshape(15)
    deltas, steps = [], []
    for p, name in enumerate(('vv', 'vh_centered')):
        torch.manual_seed(seed)
        head = m3.PolarHead(p, centered=bool(p), null=null)
        torch.nn.init.zeros_(head.net[-1].weight)
        torch.nn.init.zeros_(head.net[-1].bias)
        d, step = m3.fit_head(name, head, x, b, y, utility, selection, out)
        deltas.append(d); steps.append(step)
    dv, dh = deltas
    alpha = m3.choose_alpha(b, dv, dh, av, y, selection)
    weights = av*np.array([alpha, 1-alpha], np.float32)
    gr = aligned.old.previous.gate_raw(cache['gate_raw'], b, dv, dh, av)
    final, fw, gs = m3.fit_gate('frequency', seed, gr, b, dv, dh, av, y, utility, selection, alpha, out)
    torch.save(sc, out/'M3_readout_scaler.pt')
    return dict(bare=b, B_fixed=b+weights[:, :1]*dv+weights[:, 1:]*dh,
                frequency=final, delta_vv=dv, delta_vh=dh, weight_frequency=fw), dict(vv_step=steps[0], vh_step=steps[1], alpha=alpha, frequency_step=gs)


def train():
    torch.set_num_threads(2)
    DEST.mkdir(parents=True, exist_ok=True)
    checks, receipts = [], []
    start = time.perf_counter()
    for source, target_name in (('oss1', 'oss2'), ('oss2', 'oss1')):
        root, cross, table, target, raw, eraw, local, elocal = base.dataset(source)
        fit = np.flatnonzero(table.role.eq('fit_train'))
        val = np.flatnonzero(table.role.eq('fit_validation'))
        for field in ('mmsi', 'pass_group'):
            assert not set(table.iloc[fit][field]) & set(table.iloc[val][field])
            assert not set(table.loc[table.role.ne('evaluation'), field]) & set(target[field])
        aux, eaux = prior.prior.summary_inputs(root, cross, table, target)
        direction = f'{source}-to-{target_name}'
        frozen = {}
        for seed in SEEDS:
            pair = [tr.load_npz(aligned.HERE/direction/f'seed{seed}/{side}_predictions.npz') for side in ('source', 'cross')]
            for z, tab in zip(pair, (table, target)):
                np.testing.assert_array_equal(z['obs_id'], tab.obs_id)
            frozen[seed] = {key: np.concatenate([z[key] for z in pair]) for key in ('local', 'available', 'gate_raw', 'refs', 'frequency', 'bare')}
        for arm in ARMS:
            out = DEST/source/arm
            out.mkdir(parents=True, exist_ok=True)
            x, a = inputs(raw, local, aux, arm)
            ex, ea = inputs(eraw, elocal, eaux, arm)
            sc, sa = base.standardize(x, fit), base.standardize(a, fit)
            np.savez_compressed(out/'raw_scaler.npz', **sc)
            np.savez_compressed(out/'summary_scaler.npz', **sa)
            x, ex = base.transform(x, sc), base.transform(ex, sc)
            a, ea = base.transform(a, sa), base.transform(ea, sa)
            for seed in SEEDS:
                folder = out/f'seed{seed}'
                folder.mkdir(exist_ok=True)
                # Reuse exactly the retained base optimizer/initialization, including zero auxiliary columns.
                sb, cb, receipt = prior.train(x, ex, a, ea, None, table, fit, val, seed, 'raw_summary', folder)
                b = np.concatenate([sb, cb]).astype(np.float32)
                if arm == 'full':
                    np.testing.assert_array_equal(b, frozen[seed]['bare'])
                if arm == 'raw_only':
                    for side, actual in (('source', sb), ('cross', cb)):
                        expected = tr.load_npz(prior.HERE/source/f'{side}_predictions.npz')[f'raw_only_seed{seed}']
                        # Extra zero columns change GEMM rounding, not the information or selected rule.
                        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
                        np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
                        old_ck = tr.load_checkpoint(prior.HERE/source/f'raw_only_seed{seed}.pt')
                        assert receipt['step'] == old_ck['step']
                        checks.append(dict(source=source, seed=seed, side=side,
                                           raw_widened_replay_error=float(abs(actual-expected).max()),
                                           same_selected_step_and_predictions=True))
                scores, tail = refit(b, frozen[seed], table, seed, folder)
                if arm == 'full':
                    np.testing.assert_array_equal(scores['frequency'], frozen[seed]['frequency'])
                    checks.append(dict(source=source, seed=seed, full_base_and_final_exact=True))
                for side, sl, tab in (('source', slice(0, len(table)), table), ('cross', slice(len(table), None), target)):
                    np.savez_compressed(folder/f'{side}_predictions.npz', obs_id=tab.obs_id.to_numpy(),
                                        **{k: v[sl] for k,v in scores.items()})
                receipts.append(dict(**receipt, candidate=arm, source=source, **tail))
                print('FROZEN_READOUT', source, arm, seed, tail, flush=True)
    pd.DataFrame(receipts).to_csv(DEST/'training.csv', index=False)
    tr.write_json(DEST/'checks.json', checks)
    tr.write_json(DEST/'predictions_saved.json', dict(configurations=len(receipts), seconds=time.perf_counter()-start,
                  target_labels_read=False, matcher_frozen=True, base_and_dependents_refit=True))


def evaluate():
    assert (DEST/'predictions_saved.json').exists()
    rows, pairs, classes = [], [], []
    for source, target in (('oss1', 'oss2'), ('oss2', 'oss1')):
        for side in ('source', 'cross'):
            table = (pd.read_csv(ROOT/f'{source}_m123_replication_20261026/stack_table.csv', dtype={'mmsi':str})
                     if side=='source' else tr.read_table(tr.HERE/f'{source}-to-{target}/target_table.csv', labels=True))
            y = table.class_id.to_numpy(int)
            for seed in SEEDS:
                z = {arm:tr.load_npz(DEST/source/arm/f'seed{seed}/{side}_predictions.npz') for arm in ARMS}
                for arm in ARMS:
                    np.testing.assert_array_equal(z[arm]['obs_id'], table.obs_id)
                    for endpoint in ('bare', 'B_fixed', 'frequency'):
                        for role, ix in table.groupby('role').indices.items():
                            meta = dict(source=source, side=side, role=role, seed=seed, arm=arm, endpoint=endpoint)
                            score = z[arm][endpoint][ix]
                            rows.append(dict(**meta, **base.classification_metrics(y[ix], score)))
                            for c in range(4):
                                selected = y[ix]==c
                                classes.append(dict(**meta, class_id=c, n=int(selected.sum()),
                                    recall=float((score.argmax(1)[selected]==c).mean()) if selected.any() else np.nan))
                    for endpoint in ('bare', 'frequency'):
                        for role, ix in table.groupby('role').indices.items():
                            old = z['full'][endpoint][ix].argmax(1)==y[ix]
                            new = z[arm][endpoint][ix].argmax(1)==y[ix]
                            pairs.append(dict(source=source, side=side, role=role, seed=seed, arm=arm, endpoint=endpoint,
                                              rescue=int((new&~old).sum()), harm=int((old&~new).sum())))
    frame = pd.DataFrame(rows)
    frame.to_csv(DEST/'metrics.csv', index=False)
    frame.groupby(['source','side','role','arm','endpoint'])[['OA','BA','NLL','Brier']].agg(['mean','std']).to_csv(DEST/'summary.csv')
    pd.DataFrame(pairs).to_csv(DEST/'paired.csv', index=False)
    pd.DataFrame(classes).to_csv(DEST/'per_class.csv', index=False)
    tr.write_json(DEST/'evaluation_complete.json', dict(configurations=42, native_review_pending=True))
    print(frame.query("side=='cross'").groupby(['source','arm','endpoint'])[['OA','BA']].mean().to_string(), flush=True)


if __name__ == '__main__':
    train()
    evaluate()
