"""Matched uncentered group gating, preserving parent baseline score caches."""
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
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
from run_anchored import EXP, STRONG, Anchored, View, KINDS, combine, fit_head, weights, risk, metrics, SEEDS

ARMS = {'uncentered_external': 'anchored', 'uncentered_own_full': 'own_regions_full'}


def refit_origin(raw, names, fit, y, ids, alpha):
    # Preserve float32 fitting, including the same randomized PCA seed and head.
    views = {name: View(raw[name], fit, name) for name in names}
    z = combine({name: views[name].transform(raw[name]) for name in names}, names)
    head = fit_head(z, y, ids, fit, alpha)
    z64 = combine({name: views[name].transform(raw[name].astype(np.float64)) for name in names}, names)
    direct = head.predict(z64)*3
    centered = np.zeros((len(y), 32, 3), np.float64)
    uncentered = np.zeros_like(centered)
    origin = np.zeros((32, 3), np.float64)
    reference, offset, details = {}, 0, {}
    for name in names:
        v, h = views[name], raw[name]
        dim = v.pca.n_components_
        norm = np.sqrt(dim) if len(names) > 1 else 1.
        w = ((head.coef_[:, offset:offset+dim].astype(np.float64)/v.post.scale_[None, :])
             @ v.pca.components_.astype(np.float64))/v.scale.scale_[None, :]*3/norm
        center = h[fit].mean(0, dtype=np.float64)
        reference[name] = v.transform(center[None])
        if name == 'stem':
            groups = [np.r_[p*128+np.arange(g*8, (g+1)*8),
                            p*128+64+np.arange(g*8, (g+1)*8)] for p in range(2) for g in range(8)]
            start = 0
        else:
            groups, start = np.array_split(np.arange(h.shape[1]), 16), 16
        for j, cols in enumerate(groups):
            group = start+j
            origin[group] = w[:, cols] @ center[cols]
            centered[:, group] = (h[:, cols]-center[cols]) @ w[:, cols].T
            uncentered[:, group] = h[:, cols].astype(np.float64) @ w[:, cols].T
        details[name+'_W'] = w
        details[name+'_h0'] = center
        offset += dim
    bias = head.predict(combine(reference, names))[0]*3
    raw_intercept = bias-origin.sum(0)
    errors = dict(centered_group_error=float(np.max(np.abs(centered.sum(1)+bias-direct))),
                  raw_group_error=float(np.max(np.abs(uncentered.sum(1)+raw_intercept-direct))),
                  qraw_origin_error=float(np.max(np.abs(uncentered-centered-origin))))
    assert max(errors.values()) < 1e-7, errors
    return direct, centered, origin, raw_intercept, details, errors


def prepare(fold, raw, t, bins):
    d = dict(np.load(PARENT/f'fold{fold}_inputs.npz'))
    strong = np.load(STRONG/f'fold{fold}_base.npz')
    old = np.load(EXP/f'fold{fold}_inputs.npz')
    train, val, test = [d[k] for k in ('train', 'validation', 'test')]
    names = KINDS[str(strong['selected_kind'])]
    alpha = float(strong['selected_alpha'])
    y, ids = t.class_id.to_numpy(), t.mmsi.to_numpy()
    for role in ('train', 'validation', 'test'):
        np.testing.assert_array_equal(d[role], strong[role])
        np.testing.assert_array_equal(d[role], old[role])
    for a, b in ((train, val), (train, test), (val, test)):
        assert not set(ids[a]) & set(ids[b])
        assert not set(bins[a]) & set(bins[b])
    fits = [('outer', train, np.arange(len(t)))]
    for b in sorted(set(bins[train])):
        held = train[bins[train] == b]
        fit = train[(bins[train] != b) & ~np.isin(ids[train], ids[held])]
        assert not set(ids[fit]) & set(ids[held])
        fits.append((f'inner{b}', fit, held))
    origins_oof = np.zeros_like(d['oq'])
    coverage = np.zeros(len(t), int)
    checks, origin_rows = [], []
    for role, fit, held in fits:
        score, qq, origin, intercept, details, errors = refit_origin(raw, names, fit, y, ids, alpha)
        original_q = d['q'][held] if role == 'outer' else d['oq'][held]
        original_score = d['base'][held] if role == 'outer' else d['oof'][held]
        qerr = float(np.max(np.abs(qq[held]-original_q)))
        serr = float(np.max(np.abs(score[held]-original_score)))
        assert qerr < 1e-7 and serr < 3e-4, (role, qerr, serr)
        np.testing.assert_array_equal(score[held].argmax(1), original_score.argmax(1))
        if role == 'outer':
            outer_origin = origin
        else:
            origins_oof[held] = origin
            coverage[held] += 1
        np.savez_compressed(HERE/'origins'/f'fold{fold}_{role}.npz', fit=fit, held=held,
            origin=origin, raw_intercept=intercept, selected_names=np.asarray(names), selected_alpha=alpha, **details)
        checks.append(dict(fold=fold, fit_role=role, fit_n=len(fit), held_n=len(held),
            parent_q_error=qerr, parent_score_error=serr, parent_top1_changes=0, **errors))
        for g in range(32):
            origin_rows.append(dict(fold=fold, fit_role=role, group=g,
                **{f'origin{c}':float(origin[g, c]) for c in range(3)}))
        print('origin', fold, role, 'fit', len(fit), 'replay', qerr, flush=True)
    np.testing.assert_array_equal(coverage[train], np.ones(len(train)))
    assert not coverage[np.setdiff1d(np.arange(len(t)), train)].any()
    qraw, oqraw = d['q']+outer_origin, d['oq']+origins_oof
    own = np.load(PARENT/f'fold{fold}_own_input.npz')['env']
    assert np.isfinite(qraw).all() and np.isfinite(oqraw).all() and np.isfinite(own).all()
    np.savez_compressed(HERE/f'fold{fold}_inputs.npz', base=d['base'], oof=d['oof'],
        qraw=qraw, oqraw=oqraw, origin=outer_origin, oof_origin=origins_oof,
        train=train, validation=val, test=test, has=d['has'])
    return d, own, qraw, oqraw, checks, origin_rows


def main():
    torch.set_num_threads(2)
    started = time.perf_counter()
    for folder in ('models', 'origins'):
        (HERE/folder).mkdir(exist_ok=True)
    t = pd.read_csv(EXP/'readout/targets.csv', dtype={'mmsi': str})
    f = np.load(EXP/'readout/features.npz')
    deep = np.load(STRONG/'deep_features.npz')
    np.testing.assert_array_equal(t.instance_id, f['instance_ids'])
    np.testing.assert_array_equal(t.instance_id, deep['instance_ids'])
    raw = {'stem': f['target'], 'layer3': deep['layer3'], 'layer4': deep['layer4']}
    assert all(v.dtype == np.float32 for v in raw.values())
    alloc = pd.read_csv(EXP/'pass_assignment.csv').set_index('pass_group').group_bin
    bins = t.pass_group.map(alloc).to_numpy()
    y, ids = t.class_id.to_numpy(), t.mmsi.to_numpy()
    records, rows, curves, checks, origin_rows, replay_rows = [], [], [], [], [], []
    for fold in range(6):
        d, own, qraw, oqraw, verify, origins = prepare(fold, raw, t, bins)
        checks.extend(verify)
        origin_rows.extend(origins)
        train, val, test = [d[k] for k in ('train', 'validation', 'test')]
        oo = torch.tensor(d['oof'], dtype=torch.float32, device='cuda')
        q = torch.tensor(qraw, dtype=torch.float32, device='cuda')
        oq = torch.tensor(oqraw, dtype=torch.float32, device='cuda')
        yy = torch.tensor(y, dtype=torch.long, device='cuda')
        w = torch.tensor(weights(y[train], ids[train]), dtype=torch.float32, device='cuda')
        for seed in SEEDS:
            for arm in ARMS:
                external = arm == 'uncentered_external'
                env = torch.tensor(d['env'] if external else own, dtype=torch.float32, device='cuda')
                active = d['has'] if external else np.ones(len(t), bool)
                valid = torch.tensor(active, dtype=torch.float32, device='cuda')
                torch.manual_seed(seed)
                model = Anchored().cuda()
                opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
                best = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                br, be = risk(y[val], d['base'][val]), 0
                with torch.no_grad():
                    assert not model(env, q, valid).any()
                for epoch in range(1, 201):
                    opt.zero_grad()
                    score = oo[train]+model(env[train], oq[train], valid[train])
                    loss = (torch.nn.functional.cross_entropy(score, yy[train], reduction='none')*w).mean()
                    loss.backward()
                    opt.step()
                    if epoch % 5 == 0:
                        with torch.no_grad():
                            score = d['base'][val]+model(env[val], q[val], valid[val]).cpu().numpy()
                        rr = risk(y[val], score)
                        curves.append(dict(fold=fold, arm=arm, seed=seed, epoch=epoch,
                                           loss=float(loss.detach()), val_nll=rr))
                        if rr < br-1e-6:
                            br, be = rr, epoch
                            best = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                model.load_state_dict(best)
                with torch.no_grad():
                    delta = model(env, q, valid).cpu().numpy()
                assert np.isfinite(delta).all() and not delta[~active].any()
                if be == 0:
                    assert not delta.any()
                score = d['base']+delta
                path = HERE/'models'/f'fold{fold}_{arm}_{seed}.pt'
                torch.save(best, path)
                replay = Anchored().cuda().eval()
                replay.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
                with torch.no_grad():
                    back = d['base'][test]+replay(env[test], q[test], valid[test]).cpu().numpy()
                error = float(np.max(np.abs(back-score[test])))
                assert error < 1e-5
                np.testing.assert_array_equal(back.argmax(1), score[test].argmax(1))
                replay_rows.append(dict(fold=fold, arm=arm, seed=seed, error=error,
                    top1_changes=0, inactive_exact=True, zero_step_exact=True))
                bp, pp = d['base'][test].argmax(1), score[test].argmax(1)
                records.append(dict(fold=fold, arm=arm, seed=seed, epoch=be, val_nll=br,
                    parameters=sum(p.numel() for p in model.parameters()),
                    **metrics(y[test], score[test]),
                    rescued=int(((bp != y[test]) & (pp == y[test])).sum()),
                    harmed=int(((bp == y[test]) & (pp != y[test])).sum())))
                for i in test:
                    rows.append(dict(fold=fold, arm=arm, seed=seed, instance_id=t.iloc[i].instance_id,
                        class_id=int(y[i]), prediction=int(score[i].argmax()), base_prediction=int(d['base'][i].argmax()),
                        active=bool(active[i]), has_external=bool(d['has'][i]),
                        **{f's{c}':float(score[i,c]) for c in range(3)},
                        **{f'b{c}':float(d['base'][i,c]) for c in range(3)}))
                print('head', fold, arm, seed, 'epoch', be, 'net', records[-1]['rescued']-records[-1]['harmed'], flush=True)
        for name, table in [('metrics', records), ('predictions', rows), ('curves', curves),
                            ('origin_verification', checks), ('group_origins', origin_rows), ('replay', replay_rows)]:
            pd.DataFrame(table).to_csv(HERE/(name+'.csv'), index=False)
    p = pd.DataFrame(rows)
    assert len(records) == 36 and not p.duplicated(['arm', 'seed', 'instance_id']).any()
    assert p.groupby(['arm', 'seed']).size().eq(len(t)).all()
    pooled = []
    for (arm, seed), part in p.groupby(['arm', 'seed']):
        for scope, q0 in [('all', part), ('external', part[part.has_external]), ('no_external', part[~part.has_external])]:
            pooled.append(dict(arm=arm, seed=seed, scope=scope, n=len(q0),
                **metrics(q0.class_id.to_numpy(), q0[['s0','s1','s2']].to_numpy()),
                net=int(q0.prediction.eq(q0.class_id).sum()-q0.base_prediction.eq(q0.class_id).sum())))
    pd.DataFrame(pooled).to_csv(HERE/'pooled.csv', index=False)
    summary = dict(seconds=time.perf_counter()-started, heads=36, source_reader_refits=len(checks), targets=len(t),
        seeds=SEEDS, steps=200, max_group_error=max(v['raw_group_error'] for v in checks),
        max_parent_q_error=max(v['parent_q_error'] for v in checks), max_head_replay_error=max(v['error'] for v in replay_rows),
        parent_baseline_scores_unchanged=True, source_only=True, zero_step_fallback=True,
        group_definition='Original channels: stem groups0..15 tie corresponding mean/max; deep groups16..31; no PCA-axis grouping.',
        comparison='Uncentered gating implicitly modulates the origin contribution; not claimed equivalent to every SE network.')
    (HERE/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(pd.DataFrame(pooled).loc[lambda q:q.scope.eq('all')].to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
