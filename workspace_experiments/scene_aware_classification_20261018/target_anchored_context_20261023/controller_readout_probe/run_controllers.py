"""Fixed controller-readout controls; parent scores and group evidence stay frozen."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
from run_anchored import EXP, Anchored, SEEDS, weights, risk, metrics

READOUTS = {'whole64': 'target', 'foreground64': 'target_fg', 'background64': 'target_bg'}
ARMS = list(READOUTS) + ['constant_own_full']
READOUT_NORM = float(np.sqrt(32))


def prepare_projection(fold, t, f):
    path = HERE/'projections'/f'fold{fold}.npz'
    if path.exists():
        cached = dict(np.load(path))
        np.testing.assert_array_equal(cached['instance_ids'], t.instance_id)
        return cached
    parent = np.load(PARENT/f'fold{fold}_inputs.npz')
    own = np.load(PARENT/f'fold{fold}_own_input.npz')['env']
    train = parent['train']
    # Projection sees only source whole features, never labels or region fit rows.
    scale = StandardScaler().fit(f['target'][train])
    pca = PCA(n_components=64, svd_solver='full').fit(scale.transform(f['target'][train]))
    layout = own[:, 64:]
    assert np.array_equal(layout, np.broadcast_to(layout[:1], layout.shape))
    out = dict(instance_ids=t.instance_id.to_numpy(dtype=str), train=train,
        scale_mean=scale.mean_, scale_scale=scale.scale_, pca_mean=pca.mean_,
        pca_components=pca.components_, explained_variance_ratio=pca.explained_variance_ratio_,
        layout=layout, parent_env64_norm=np.linalg.norm(own[:, :64], axis=1),
        constant=own[train].mean(0, dtype=np.float64).astype(np.float32))
    for arm, key in READOUTS.items():
        values = pca.transform(scale.transform(f[key]))
        norm = np.linalg.norm(values, axis=1, keepdims=True)
        valid = np.ones(len(t), bool) if key == 'target' else f[key+'_valid']
        assert (norm[valid] > 1e-8).all()
        unit = (values/np.maximum(norm, 1e-8)).astype(np.float32)
        unit[~valid] = 0
        assert np.isfinite(unit).all() and not unit[~valid].any()
        np.testing.assert_allclose(np.linalg.norm(unit[valid], axis=1), 1, atol=2e-6)
        out[arm+'_unit'], out[arm+'_valid'] = unit, valid
    np.savez_compressed(path, **out)
    return out


def train_arm(arm, readout_norm, t, projections):
    output = HERE/arm
    output.mkdir(exist_ok=True)
    if (output/'summary.json').exists():
        saved = json.loads((output/'summary.json').read_text(encoding='utf-8'))
        assert saved['readout_norm'] == readout_norm
        print('completed arm retained:', arm, flush=True)
        return
    started = time.perf_counter()
    y, identities = t.class_id.to_numpy(), t.mmsi.to_numpy()
    rows, records, curves, checks, gates, norms = [], [], [], [], [], []
    for fold in range(6):
        d = dict(np.load(PARENT/f'fold{fold}_inputs.npz'))
        projection = projections[fold]
        train, val, test = [d[k] for k in ('train', 'validation', 'test')]
        np.testing.assert_array_equal(train, projection['train'])
        for a, b in ((train,val), (train,test), (val,test)):
            assert not set(identities[a]) & set(identities[b])
            assert not set(t.iloc[a].pass_group) & set(t.iloc[b].pass_group)
        if arm == 'constant_own_full':
            env = np.broadcast_to(projection['constant'], (len(t),79)).copy()
            assert np.array_equal(env, np.broadcast_to(env[:1], env.shape))
            valid_features = np.ones(len(t), bool)
        else:
            env = np.c_[projection[arm+'_unit']*readout_norm, projection['layout']].astype(np.float32)
            valid_features = projection[arm+'_valid']
            np.testing.assert_array_equal(env[:,64:], projection['layout'])
        norms.append(dict(fold=fold, arm=arm, valid_features=int(valid_features.sum()),
            env64_norm_min=float(np.linalg.norm(env[:,:64], axis=1).min()),
            env64_norm_max=float(np.linalg.norm(env[:,:64], axis=1).max()),
            parent_env64_norm_min=float(projection['parent_env64_norm'].min()),
            parent_env64_norm_max=float(projection['parent_env64_norm'].max())))
        np.savez_compressed(output/f'fold{fold}_env.npz', env=env, valid_features=valid_features,
                            train=train, validation=val, test=test)
        ee = torch.tensor(env, dtype=torch.float32, device='cuda')
        q = torch.tensor(d['q'], dtype=torch.float32, device='cuda')
        oq = torch.tensor(d['oq'], dtype=torch.float32, device='cuda')
        oo = torch.tensor(d['oof'], dtype=torch.float32, device='cuda')
        yy = torch.tensor(y, dtype=torch.long, device='cuda')
        w = torch.tensor(weights(y[train], identities[train]), dtype=torch.float32, device='cuda')
        active = torch.ones(len(t), dtype=torch.float32, device='cuda')
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = Anchored().cuda()
            assert sum(p.numel() for p in model.parameters()) == 3616
            optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
            best = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            best_risk, best_epoch = risk(y[val], d['base'][val]), 0
            with torch.no_grad():
                assert not model(ee, q, active).any()
            for epoch in range(1,201):
                optimizer.zero_grad()
                s = oo[train]+model(ee[train], oq[train], active[train])
                loss = (torch.nn.functional.cross_entropy(s, yy[train], reduction='none')*w).mean()
                loss.backward()
                optimizer.step()
                if epoch % 5 == 0:
                    with torch.no_grad():
                        s = d['base'][val]+model(ee[val], q[val], active[val]).cpu().numpy()
                    vr = risk(y[val], s)
                    curves.append(dict(fold=fold, arm=arm, seed=seed, epoch=epoch,
                                       loss=float(loss.detach()), val_nll=vr))
                    if vr < best_risk-1e-6:
                        best_risk, best_epoch = vr, epoch
                        best = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            model.load_state_dict(best)
            with torch.no_grad():
                delta = model(ee, q, active).cpu().numpy()
                coefficients = (.5*torch.tanh(model.net(ee))).cpu().numpy()
                null = model(ee[test], torch.zeros_like(q[test]), active[test]).cpu().numpy()
            assert np.isfinite(delta).all() and not null.any()
            if best_epoch == 0:
                assert not delta.any()
            if arm == 'constant_own_full':
                assert np.array_equal(coefficients, np.broadcast_to(coefficients[:1], coefficients.shape))
            score = d['base']+delta
            path = HERE/'models'/f'fold{fold}_{arm}_{seed}.pt'
            torch.save(best, path)
            replay = Anchored().cuda().eval()
            replay.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
            with torch.no_grad():
                back = d['base'][test]+replay(ee[test], q[test], active[test]).cpu().numpy()
            error = float(np.abs(back-score[test]).max())
            assert error < 1e-5
            np.testing.assert_array_equal(back.argmax(1), score[test].argmax(1))
            checks.append(dict(fold=fold, arm=arm, seed=seed, replay_error=error, top1_changes=0,
                               activated_test=len(test), zero_evidence_exact=True))
            bp, pp = d['base'][test].argmax(1), score[test].argmax(1)
            records.append(dict(fold=fold, arm=arm, seed=seed, epoch=best_epoch, val_nll=best_risk,
                parameters=3616, **metrics(y[test], score[test]),
                rescued=int(((bp != y[test]) & (pp == y[test])).sum()),
                harmed=int(((bp == y[test]) & (pp != y[test])).sum())))
            for g in range(32):
                gates.append(dict(fold=fold, arm=arm, seed=seed, group=g,
                    mean=float(coefficients[test,g].mean()), sd=float(coefficients[test,g].std()),
                    minimum=float(coefficients[test,g].min()), maximum=float(coefficients[test,g].max())))
            for i in test:
                rows.append(dict(fold=fold, arm=arm, seed=seed, instance_id=t.iloc[i].instance_id,
                    class_id=int(y[i]), prediction=int(score[i].argmax()), base_prediction=int(d['base'][i].argmax()),
                    active=True, has_external=bool(d['has'][i]), valid_features=bool(valid_features[i]),
                    **{f's{c}':float(score[i,c]) for c in range(3)},
                    **{f'b{c}':float(d['base'][i,c]) for c in range(3)}))
            print(arm, fold, seed, 'epoch', best_epoch, 'net', records[-1]['rescued']-records[-1]['harmed'], flush=True)
        for name, values in [('metrics',records), ('predictions',rows), ('curves',curves),
                             ('verification',checks), ('gates',gates), ('env_norms',norms)]:
            pd.DataFrame(values).to_csv(output/(name+'.csv'), index=False)
    p = pd.DataFrame(rows)
    assert len(p) == len(t)*3 and not p.duplicated(['seed','instance_id']).any()
    pooled = []
    for seed, q0 in p.groupby('seed'):
        for scope, q in [('all',q0),('external',q0[q0.has_external]),('no_external',q0[~q0.has_external])]:
            pooled.append(dict(arm=arm, seed=seed, scope=scope, n=len(q),
                **metrics(q.class_id.to_numpy(), q[['s0','s1','s2']].to_numpy()),
                net=int(q.prediction.eq(q.class_id).sum()-q.base_prediction.eq(q.class_id).sum())))
    pd.DataFrame(pooled).to_csv(output/'pooled.csv', index=False)
    summary = dict(arm=arm, heads=18, targets=len(t), readout_norm=readout_norm,
        seconds=time.perf_counter()-started, max_replay_error=max(c['replay_error'] for c in checks),
        parent_q_base_oof_unchanged=True, source_only=True, all_targets_active=True,
        constant_from_training_start=arm == 'constant_own_full')
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(pd.DataFrame(pooled).loc[lambda x:x.scope.eq('all')].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arms', nargs='+', choices=ARMS, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    for folder in ('projections','models'):
        (HERE/folder).mkdir(exist_ok=True)
    t = pd.read_csv(EXP/'readout/targets.csv', dtype={'mmsi':str})
    f = np.load(EXP/'readout/features.npz')
    np.testing.assert_array_equal(t.instance_id, f['instance_ids'])
    projections = [prepare_projection(fold, t, f) for fold in range(6)]
    for arm in args.arms:
        train_arm(arm, None if arm == 'constant_own_full' else READOUT_NORM, t, projections)


if __name__ == '__main__':
    main()
