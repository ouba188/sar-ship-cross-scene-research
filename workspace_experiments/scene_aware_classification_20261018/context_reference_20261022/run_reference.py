"""Source-only background-reference readout with matched information controls."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent/'context_expansion_20261022'
sys.path.insert(0, str(HERE.parent/'context_relation_pilot_20261022'))
from run_ports24 import Baseline, weights, risk, metrics

ARMS = ['own_matched', 'external', 'same_scene_replacement', 'constant_external']


def background_inputs(t, f, mom, edges):
    active = mom['neighbor_valid'] & f['mask'][:, :, None]
    count = active.sum(1)
    external = []
    for key in ['mean', 'second']:
        external.append((mom['neighbor_'+key]*active[..., None]).sum(1)/np.maximum(count[..., None], 1))
    eligible = (count > 0).all(1) & mom['target_valid'].all(1)
    donors = np.arange(len(t))
    changed = np.zeros(len(t), bool)
    loc = dict(zip(t.instance_id, range(len(t))))
    windows = {}
    keys = {}
    for name, group in edges.groupby('instance_id'):
        windows[loc[name]] = [json.loads(w) for w in group.read_window_vv]
        keys[loc[name]] = set(group.detection_id)
    rows = []
    for product, group in t.groupby('product'):
        ids = sorted([i for i in group.index if eligible[i]], key=lambda i: t.iloc[i].instance_id)
        for k, i in enumerate(ids):
            pts = np.asarray(json.loads(t.iloc[i].target_points))
            lo, hi = pts.min(0), pts.max(0)
            for shift in range(1, len(ids)):
                j = ids[(k+shift) % len(ids)]
                overlaps = any(x < hi[0] and x+w > lo[0] and y < hi[1] and y+h > lo[1]
                               for x, y, w, h in windows[j])
                if not overlaps:
                    donors[i] = j
                    changed[i] = True
                    break
            j = donors[i]
            rows.append(dict(instance_id=t.iloc[i].instance_id, donor=t.iloc[j].instance_id,
                             product=product, changed=bool(changed[i]),
                             original_objects=len(keys[i]), donor_objects=len(keys[j]),
                             shared_objects=len(keys[i]&keys[j]), identical_objects=keys[i] == keys[j]))
    pd.DataFrame(rows).to_csv(HERE/'replacement_map.csv', index=False)
    return dict(own_matched=(mom['target_mean'].astype(float), mom['target_second'].astype(float)),
                external=tuple(external), same_scene_replacement=tuple(a[donors] for a in external)), eligible, changed


def reference_parameters(mom, refs, fit, eligible):
    mu0 = mom['target_mean'][fit].mean(0, dtype=np.float64)
    second0 = mom['target_second'][fit].mean(0, dtype=np.float64)
    var0 = np.maximum(second0-mu0**2, 0)
    ix = fit[eligible[fit]]
    assert len(ix)
    const = tuple(x[ix].mean(0) for x in refs['external'])
    return mu0, var0, const


def feature_change(h, mu, second, mu0, var0, eligible):
    var = np.maximum(second-mu**2, 0)
    active = var0 > 1e-8
    scale = np.sqrt(var0/np.maximum(var, np.maximum(.01*var0, 1e-12))).clip(.25, 4.)
    scale = np.where(active, scale, 1.)
    shape = h.reshape(-1, 2, 2, 64)
    change = mu0[:, None, :] + scale[:, None, :]*(shape-mu[:, None, :]) - shape if mu.ndim == 2 else (
        mu0[None, :, None, :] + scale[:, :, None, :]*(shape-mu[:, :, None, :]) - shape)
    change *= active[None, :, None, :]
    change[~eligible] = 0
    return change.reshape(-1, 256)


def score_changes(base, change):
    jac = (base.head.coef_/base.post.scale_[None, :]) @ base.pca.components_
    jac = jac/base.scale.scale_[None, :]*3
    out = np.zeros((len(change), 16, 3), np.float64)
    for p in range(2):
        for g in range(8):
            ix = np.r_[p*128+np.arange(g*8, (g+1)*8), p*128+64+np.arange(g*8, (g+1)*8)]
            out[:, p*8+g] = change[:, ix] @ jac[:, ix].T
    return out


def all_changes(base, h, mom, refs, fit, eligible):
    mu0, var0, const = reference_parameters(mom, refs, fit, eligible)
    result, raw = {}, {}
    for arm in ARMS:
        mu, second = const if arm == 'constant_external' else refs[arm]
        raw[arm] = feature_change(h, mu, second, mu0, var0, eligible)
        result[arm] = score_changes(base, raw[arm])
    return result, raw, (mu0, var0, const)


def main():
    torch.set_num_threads(2)
    start = time.perf_counter()
    t = pd.read_csv(EXP/'readout/targets.csv', dtype={'mmsi': str})
    f = np.load(EXP/'readout/features.npz')
    mom = np.load(HERE/'moments/moments.npz')
    edges = pd.read_csv(EXP/'readout/neighbors.csv')
    assert np.array_equal(f['instance_ids'], mom['instance_ids'])
    refs, eligible, changed = background_inputs(t, f, mom, edges)
    y, identity, h = t.class_id.to_numpy(), t.mmsi.to_numpy(), f['target'].astype(float)
    allocation = pd.read_csv(EXP/'pass_assignment.csv').set_index('pass_group').group_bin
    bins = t.pass_group.map(allocation).to_numpy()
    records, predictions, curves, alpha_rows, checks = [], [], [], [], []
    for fold in range(6):
        old = np.load(EXP/f'fold{fold}_inputs.npz')
        train, val, test = [old[k] for k in ['train', 'validation', 'test']]
        alpha = float(old['alpha'])
        base = Baseline(h[train], y[train], identity[train], alpha)
        bs = base.predict(h)
        assert np.allclose(bs, old['base'], atol=2e-4)
        ds, raw, reference = all_changes(base, h, mom, refs, train, eligible)
        oof = np.zeros_like(bs)
        od = {a: np.zeros_like(ds[a]) for a in ARMS}
        for b in sorted(set(bins[train])):
            held = train[bins[train] == b]
            fit = train[(bins[train] != b) & ~np.isin(identity[train], identity[held])]
            assert not set(identity[fit]) & set(identity[held])
            local = Baseline(h[fit], y[fit], identity[fit], alpha)
            oof[held] = local.predict(h[held])
            local_ds, _, _ = all_changes(local, h, mom, refs, fit, eligible)
            for arm in ARMS:
                od[arm][held] = local_ds[arm][held]
        assert np.allclose(oof[train], old['oof'][train], atol=2e-4)
        np.savez_compressed(HERE/f'fold{fold}_cache.npz', base=bs, oof=oof, train=train, validation=val,
                            test=test, eligible=eligible, replacement_changed=changed,
                            **{a: ds[a] for a in ARMS}, **{'oof_'+a: od[a] for a in ARMS})
        w = torch.tensor(weights(y[train], identity[train]), dtype=torch.float64)
        yt = torch.tensor(y[train], dtype=torch.long)
        for arm in ARMS:
            td = torch.tensor(od[arm][train], dtype=torch.float64)
            tb = torch.tensor(oof[train], dtype=torch.float64)
            a = torch.nn.Parameter(torch.zeros(16, dtype=torch.float64))
            opt = torch.optim.Adam([a], lr=.01)
            best = np.zeros(16)
            best_epoch = 0
            best_risk = risk(y[val], bs[val])
            for epoch in range(1, 201):
                opt.zero_grad()
                logits = tb+torch.einsum('ngc,g->nc', td, a)
                loss = (torch.nn.functional.cross_entropy(logits, yt, reduction='none')*w).mean()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    a.clamp_(0, 1)
                if epoch % 5 == 0:
                    aa = a.detach().numpy()
                    vr = risk(y[val], bs[val]+np.einsum('ngc,g->nc', ds[arm][val], aa))
                    curves.append(dict(fold=fold, arm=arm, epoch=epoch, train_loss=float(loss.detach()), val_nll=vr))
                    if vr < best_risk-1e-6:
                        best_risk, best_epoch, best = vr, epoch, aa.copy()
            delta = np.einsum('ngc,g->nc', ds[arm], best)
            score = bs+delta
            assert not delta[~eligible].any()
            # Independent implementation: run transformed feature vectors through the fitted A.
            aa = np.repeat(best.reshape(2, 8), 8, axis=1)
            aa = np.repeat(aa[:, None, :], 2, axis=1).reshape(256)
            replay = base.predict(h[test]+raw[arm][test]*aa)
            err = float(np.abs(replay-score[test]).max())
            assert err < 1e-7
            checks.append(dict(fold=fold, arm=arm, feature_score_max_error=err))
            for g, value in enumerate(best):
                alpha_rows.append(dict(fold=fold, arm=arm, group=g, polarization=['VV', 'VH'][g//8], alpha=value))
            for scope, ix in [('all', test), ('context', test[eligible[test]]),
                              ('replaceable', test[changed[test]])]:
                bp, pp = bs[ix].argmax(1), score[ix].argmax(1)
                records.append(dict(fold=fold, arm=arm, scope=scope, n=len(ix), epoch=best_epoch, val_nll=best_risk,
                                    parameters=16, **metrics(y[ix], score[ix]),
                                    rescued=int(((bp != y[ix]) & (pp == y[ix])).sum()),
                                    harmed=int(((bp == y[ix]) & (pp != y[ix])).sum())))
            for i in test:
                predictions.append(dict(fold=fold, arm=arm, instance_id=t.iloc[i].instance_id, class_id=int(y[i]),
                    prediction=int(score[i].argmax()), base_prediction=int(bs[i].argmax()), eligible=bool(eligible[i]),
                    replacement_changed=bool(changed[i]),
                    **{f's{c}':float(score[i,c]) for c in range(3)}, **{f'b{c}':float(bs[i,c]) for c in range(3)}))
            print('fold', fold, arm, 'epoch', best_epoch, 'OA/BA', records[-3]['OA'], records[-3]['BA'], flush=True)
        pd.DataFrame(records).to_csv(HERE/'metrics.csv', index=False)
        pd.DataFrame(predictions).to_csv(HERE/'predictions.csv', index=False)
        pd.DataFrame(alpha_rows).to_csv(HERE/'learned_strength.csv', index=False)
        pd.DataFrame(curves).to_csv(HERE/'curves.csv', index=False)
        pd.DataFrame(checks).to_csv(HERE/'verification.csv', index=False)
    p = pd.DataFrame(predictions)
    pooled = []
    for arm, part in p.groupby('arm'):
        for scope, q in [('all', part), ('context', part[part.eligible]), ('replaceable', part[part.replacement_changed])]:
            pooled.append(dict(arm=arm, scope=scope, n=len(q), **metrics(q.class_id.to_numpy(), q[['s0','s1','s2']].to_numpy()),
                               rescued=int(((q.base_prediction != q.class_id) & (q.prediction == q.class_id)).sum()),
                               harmed=int(((q.base_prediction == q.class_id) & (q.prediction != q.class_id)).sum())))
    pd.DataFrame(pooled).to_csv(HERE/'pooled.csv', index=False)
    summary = dict(seconds=time.perf_counter()-start, targets=len(t), eligible=int(eligible.sum()),
                   replaceable=int(changed.sum()), heads=24, deterministic=True, device='cpu',
                   max_replay_error=max(r['feature_score_max_error'] for r in checks))
    (HERE/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(pd.DataFrame(pooled).query('scope == "all"').to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
