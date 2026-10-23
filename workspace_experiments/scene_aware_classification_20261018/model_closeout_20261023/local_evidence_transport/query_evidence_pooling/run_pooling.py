"""Frozen query-token aggregation controls, retaining actual reference identity."""
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
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'support_transfer_learning'))
from run import load, REGIONS, TAU, UP, metrics, exact_j


def extract(x, mass, bank, idx):
    """Return per-query-token/per-reference log kernels before query pooling."""
    kernels, areas, refs, valid = [], [], [], []
    with torch.inference_mode():
        for p in range(2):
            for lo, hi in REGIONS:
                s = torch.tensor(x[bank, p, lo:hi], device='cuda')
                b = mass[bank, p, lo:hi].copy()
                present = b.sum(-1) > 0
                b /= np.maximum(b.sum(-1, keepdims=True), 1e-20)
                logb = torch.tensor(np.where(b > 0, np.log(np.maximum(b, 1e-30)), -1e4), device='cuda')
                values = []
                for first in range(0, len(idx), 64):
                    q = torch.tensor(x[idx[first:first+64], p, lo:hi], device='cuda')
                    cosine = torch.einsum('bid,crjd->bicrj', q, s) / TAU
                    values.append(torch.logsumexp(cosine + logb[None, None], -1).cpu().numpy())
                k = np.concatenate(values)
                kernels.append(np.pad(k, ((0,0), (0,8-hi+lo), (0,0), (0,0)), constant_values=-1e4))
                a = mass[idx, p, lo:hi].copy()
                a /= np.maximum(a.sum(-1, keepdims=True), 1e-20)
                areas.append(np.pad(a, ((0,0), (0,8-hi+lo))))
                pi = present / np.maximum(present.sum(-1, keepdims=True), 1)
                refs.append(pi)
                valid.append((a.sum(-1) > 0)[:,None] & present.any(-1)[None])
    return np.stack(kernels, 1), np.stack(areas, 1), np.stack(refs), np.stack(valid, 1)


def aggregate(k, a, pi, valid):
    logr = np.where(pi > 0, np.log(np.maximum(pi, 1e-30)), -1e4)
    loga = np.where(a > 0, np.log(np.maximum(a, 1e-30)), -1e4)
    token = logsumexp(k + logr[None,:,None], axis=-1)
    joint = logsumexp(token + loga[:,:,:,None], axis=2)
    query_log = np.sum(a[:,:,:,None] * token, axis=2)
    by_reference = np.sum(a[:,:,:,None,None] * k, axis=2)
    coherent = logsumexp(by_reference + logr[None], axis=-1)
    common = logsumexp(token, axis=-1, keepdims=True) - np.log(8)
    mixed_token = np.logaddexp(token, common) - np.log(2)
    mixture = np.sum(a[:,:,:,None] * mixed_token, axis=2)
    region = dict(joint=joint, query_log=query_log, reference_coherent=coherent, shared_mixture=mixture)
    scores = {}
    for name, values in region.items():
        values = np.where(valid, values, 0)
        region[name] = values
        score = values.sum(1) / np.maximum(valid.sum(1), 1)
        scores[name] = (score - score.mean(-1, keepdims=True)).astype(np.float32)
    responsibility = np.exp(token + loga[:,:,:,None] - joint[:,:,None,:])
    responsibility *= valid[:,:,None]
    return scores, region, token, responsibility


def main():
    tick = time.perf_counter()
    torch.set_num_threads(2)
    table, x, mass, bank = load()
    idx = np.flatnonzero(table.role.isin(['utility_train', 'selection', 'evaluation']))
    k, a, pi, valid = extract(x, mass, bank, idx)
    values, region, token, responsibility = aggregate(k, a, pi, valid)
    previous = np.load(ROOT / 'support_transfer_learning/local_scores.npz')
    np.testing.assert_array_equal(idx, previous['idx'])
    err = float(np.max(np.abs(values['joint'] - previous['uniform'])))
    assert err < 5e-6, err
    np.savez_compressed(HERE / 'details.npz', k=k, a=a, pi=pi, valid=valid, token=token,
                        responsibility=responsibility, **{name+'_region':v for name,v in region.items()})
    np.savez_compressed(HERE / 'local_scores.npz', idx=idx, bank=bank, **values)
    y = table.class_id.to_numpy()
    role = table.iloc[idx].role.to_numpy()
    tr, va, te = [np.flatnonzero(role == r) for r in ('utility_train', 'selection', 'evaluation')]
    scale = float(np.sqrt(np.mean(values['joint'][tr] ** 2)))
    direct, full, selections, pairs, classes = [], [], [], [], []
    for name, score in values.items():
        for label, ids in [('utility_train',tr), ('selection',va), ('evaluation',te)]:
            direct.append(dict(arm=name, role=label, **metrics(y[idx[ids]], score[ids])))
        for c in range(8):
            mask = y[idx[te]] == c
            classes.append(dict(arm=name,class_id=c,n=int(mask.sum()),correct=int((score[te][mask].argmax(-1)==c).sum())))
    for seed in (17,29,43):
        base = np.load(UP / f'strategies_seed{seed}.npz')['logits'][:,2].astype(np.float32)
        preds = dict(full=base)
        gains = {}
        for name, score in values.items():
            best, gain = exact_j(y[idx[va]], base[idx[va]].argmax(-1)), 0.
            for w in (0.,.1,.25,.5,1.,2.):
                s = base[idx[va]] + w * score[va] / scale
                objective = exact_j(y[idx[va]], s.argmax(-1))
                selections.append(dict(seed=seed,arm=name,gain=w,**metrics(y[idx[va]],s)))
                if objective > best:
                    best, gain = objective, w
            gains[name] = gain
            s = base.copy()
            s[idx] += gain * score / scale
            preds[name] = s
        for name, score in values.items():
            s = base.copy()
            s[idx] += gains['joint'] * score / scale
            preds[name+'_common'] = s
        old = preds['joint'][idx[te]].argmax(-1) == y[idx[te]]
        for name, s in preds.items():
            full.append(dict(seed=seed,arm=name,gain=0 if name=='full' else gains['joint' if name.endswith('_common') else name],
                             **metrics(y[idx[te]],s[idx[te]])))
            correct = s[idx[te]].argmax(-1) == y[idx[te]]
            pairs.append(dict(seed=seed,arm=name,rescued=int((correct&~old).sum()),harmed=int((~correct&old).sum()),net=int(correct.sum()-old.sum())))
        np.savez_compressed(HERE / f'predictions_{seed}.npz', **preds)
    pd.DataFrame(direct).to_csv(HERE/'direct_metrics.csv',index=False)
    pd.DataFrame(full).to_csv(HERE/'full_metrics.csv',index=False)
    pd.DataFrame(selections).to_csv(HERE/'selection_curve.csv',index=False)
    pd.DataFrame(pairs).to_csv(HERE/'paired_results.csv',index=False)
    pd.DataFrame(classes).to_csv(HERE/'class_results.csv',index=False)
    concentration = []
    for j in te:
        for g in range(6):
            c = int(y[idx[j]])
            if not valid[j,g,c]:
                continue
            r = responsibility[j,g,:,c]
            area = a[j,g]
            concentration.append(dict(obs_id=int(table.iloc[idx[j]].obs_id),group=g,class_id=c,
                                      max_share=float(r.max()),effective_tokens=float(1/(r*r).sum()),
                                      area_max=float(area.max()),area_effective=float(1/(area*area).sum())))
    pd.DataFrame(concentration).to_csv(HERE/'concentration.csv',index=False)
    receipt=dict(seconds=time.perf_counter()-tick,joint_replay_error=err,scale=scale,
                 no_training=True,no_new_selection_protocol=True,source_only=True)
    (HERE/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(pd.DataFrame(direct).query("role=='evaluation'").to_string(index=False))
    print(pd.DataFrame(full).groupby('arm')[['OA','BA','NLL','Brier']].mean().to_string())
    print(receipt,flush=True)


if __name__ == '__main__':
    main()
