"""Four matched M3 operators on the retained raw_summary base.

Training/prediction and label-based evaluation are separate commands.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import argparse
import importlib.util
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ALIGNED = ROOT / 'm3_aligned_training_20261026'
spec = importlib.util.spec_from_file_location('m3_aligned_reference', ALIGNED / 'run.py')
aligned = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aligned)
m3, tr, load, write = aligned.m3, aligned.tr, aligned.load, aligned.write
SEEDS = (17, 29, 43)
DIRECTIONS = (('oss1', 'oss2'), ('oss2', 'oss1'))
ARMS = ('both', 'learned_fixed', 'cosine_soft', 'cosine_fixed')
ENDPOINTS = ('bare', 'B_fixed', 'spatial', 'frequency')


class ReplacementMatcher(m3.Matcher):
    def __init__(self, features, refs, scaler, device, arm):
        super().__init__(features, refs, scaler, device)
        self.arm = arm

    def local(self, idx, model):
        if self.arm == 'both':
            return super().local(idx, model)
        idx = torch.as_tensor(idx, device=self.device)
        ref = self.refs[idx]
        q, s = self.x[idx], self.x[ref]
        a, b = self.mass[idx], self.mass[ref]
        learned = self.arm == 'learned_fixed'
        if learned:
            h = model.project(self.d)
            hq, hs = h[idx], h[ref]
        groups = []
        for r in m3.tokens.REGIONS:
            qi = q[:, :, r][:, None, None, :, :, None, :]
            sj = s[..., r, :][..., None, :, :]
            w = 1.
            if learned:
                hi = hq[:, :, r][:, None, None, :, :, None, :]
                hj = hs[..., r, :][..., None, :, :]
                w = model.weights(hi, hj)
            sim = (w*qi*sj).sum(-1)/((w*qi.square()).sum(-1)*(w*sj.square()).sum(-1)).clamp_min(1e-20).sqrt()
            qa, sa = a[:, None, None, :, r], b[..., r]
            if self.arm == 'cosine_soft':
                transport = m3.sinkhorn(sim, qa, sa)
            else:
                # Same spatial slot only; unmatched mass goes to the implicit dustbin.
                qa = qa/qa.sum(-1, keepdim=True).clamp_min(1e-20)
                sa = sa/sa.sum(-1, keepdim=True).clamp_min(1e-20)
                transport = torch.diag_embed(torch.minimum(qa, sa))
            transport = transport*((qa > 0)[..., :, None] & (sa > 0)[..., None, :])
            groups.append(torch.stack(((transport*sim).sum((-1, -2)), transport.sum((-1, -2))), -1))
        return torch.stack(groups, -2).mean(2)


def prepare(source, target):
    direction = f'{source}-to-{target}'
    table = pd.read_csv(ROOT/f'{source}_m123_replication_20261026/stack_table.csv', dtype={'mmsi': str})
    cross = tr.read_table(tr.HERE/direction/'target_table.csv')
    protocol = tr.read_json(tr.HERE/direction/'protocol.json')
    mroot = Path(protocol['source_m3'])
    feat = load(mroot/'local.npz')
    ext = tr.project_raw(tr.aligned(Path(protocol['target_cache'])/'m3_raw.npz', cross), load(mroot/'projection.npz'))
    features = {k: np.concatenate((feat[k], ext[k])) for k in tr.FEATURES}
    bank, bankrows, _ = tr.source_bank(mroot, table)
    er = tr.retrieve(ext, bank, cross.mmsi, table.iloc[bankrows].mmsi)
    refs = np.concatenate((load(mroot/'references.npz')['refs'], bankrows[er]))
    utility = np.flatnonzero(table.role.eq('utility_train'))
    selection = np.flatnonzero(table.role.eq('selection'))
    for ids in (utility, selection):
        assert np.isin(refs[ids], bankrows).all()
        for field in ('mmsi', 'pass_group'):
            assert not (table.iloc[ids][field].to_numpy()[:, None, None] == table[field].to_numpy()[refs[ids]]).any()
    for field in ('mmsi', 'pass_group'):
        assert not set(table.iloc[utility][field]) & set(table.iloc[selection][field])
        assert not set(table.loc[table.role.ne('evaluation'), field]) & set(cross[field])
    combined = pd.read_csv(aligned.REPAIR/direction/'table.csv')
    np.testing.assert_array_equal(combined.loc[combined.dataset_side.eq('source'), 'original_obs_id'], table.obs_id)
    y = table.class_id.to_numpy(int, copy=True)
    y[table.role.eq('evaluation')] = -1
    return table, cross, mroot, features, refs, y, utility, selection


def refit(local, base, cached, y, utility, selection, seed, out):
    scaler = m3.scale_fit(m3.local_raw(local, base), utility)
    x = m3.transform(m3.local_raw(local, base), scaler)
    null = m3.transform(np.zeros((1, 1, 15)), scaler).reshape(15)
    deltas, steps = [], []
    for p, name in enumerate(('vv', 'vh_centered')):
        torch.manual_seed(seed)
        head = m3.PolarHead(p, centered=bool(p), null=null)
        torch.nn.init.zeros_(head.net[-1].weight)
        torch.nn.init.zeros_(head.net[-1].bias)
        delta, step = m3.fit_head(name, head, x, base, y, utility, selection, out, steps=200)
        deltas.append(delta)
        steps.append(step)
    dv, dh = deltas
    av = cached['available']
    alpha = m3.choose_alpha(base, dv, dh, av, y, selection)
    weights = av*np.array([alpha, 1-alpha], np.float32)
    fixed = base+weights[:, :1]*dv+weights[:, 1:]*dh
    gr = aligned.old.previous.gate_raw(cached['gate_raw'], base, dv, dh, av)
    scores = dict(bare=base, B_fixed=fixed)
    extra = dict(delta_vv=dv, delta_vh=dh, weight_B_fixed=weights, gate_raw=gr,
                 available=av, local=local)
    receipt = dict(vv_step=steps[0], vh_step=steps[1], alpha=alpha)
    for gate in ('spatial', 'frequency'):
        scores[gate], gw, step = m3.fit_gate(gate, seed, gr, base, dv, dh, av, y, utility, selection, alpha, out)
        np.testing.assert_array_equal(scores[gate], base+gw[:, :1]*dv+gw[:, 1:]*dh)
        extra[f'weight_{gate}'] = gw
        receipt[f'{gate}_step'] = step
    torch.save(scaler, out/'M3_readout_scaler.pt')
    return scores, extra, receipt


def gate_activity(scores, extra, ids):
    result = {}
    for gate in ('spatial', 'frequency'):
        wd = np.abs(extra[f'weight_{gate}'][ids]-extra['weight_B_fixed'][ids])
        sd = np.abs(scores[gate][ids]-scores['B_fixed'][ids])
        result.update({f'{gate}_weight_max': float(wd.max()),
                       f'{gate}_weight_changed_n': int(np.any(wd > 0, axis=1).sum()),
                       f'{gate}_score_max': float(sd.max()),
                       f'{gate}_score_changed_n': int(np.any(sd > 0, axis=1).sum()),
                       f'{gate}_prediction_changed_n': int((scores[gate][ids].argmax(1) != scores['B_fixed'][ids].argmax(1)).sum()),
                       f'{gate}_equals_B_fixed': bool(np.array_equal(scores[gate][ids], scores['B_fixed'][ids]))})
    return result


def train(device):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    HERE.mkdir(parents=True, exist_ok=True)
    marker = HERE/'predictions_saved.json'
    marker.unlink(missing_ok=True)
    (HERE/'evaluation_complete.json').unlink(missing_ok=True)
    write(HERE/'PROTOCOL.json', dict(
        arms=ARMS, endpoints=ENDPOINTS, seeds=SEEDS, directions=DIRECTIONS,
        base='frozen retained raw_summary, no upstream refit', references=32, per_class_retrieved=2,
        learned_initialization='original source both_metric.pt, not retained aligned_metric.pt',
        metric_steps=100, metric_objective='actual frozen old raw_summary frequency decision',
        metric_selection_steps=[0, 5, 10, 15, 20, 30, 50, 75, 100],
        metric_optimizer='AdamW lr=.003 weight_decay=.01; full utility objective, chunks24',
        head_steps=200, gate_steps=100, head_gate_optimizer='AdamW lr=.003 weight_decay=.01',
        roles='utility_train optimization/scalers, selection exact J and alpha, evaluation held out',
        checkpoint='strict exact-J improvement, retain earliest including zero',
        fixed_metric='ordinary cosine on source-PCA spatial tokens, no PairMetric weights',
        fixed_matching='T_ij=1[i=j]*min(normalized_query_mass_i, normalized_reference_mass_i)',
        soft_matching='original dustbin Sinkhorn eps=.1 iterations60; no independent trainable solver',
        training_scope='learned_fixed and both adapt the same 640 PairMetric parameters; cosine arms have no metric optimizer',
        gate_scope='three readouts conditional on the same frequency-adapted metric; not independent matcher fits per gate',
        all_readouts_refit=True, target_labels='evaluate only after all24 configurations frozen',
        historical_target_development_exposure=True, device=device, raw_review_pending=True))
    receipts, checks, activities, coverage = [], [], [], []
    start = time.perf_counter()
    for source, target in DIRECTIONS:
        direction = f'{source}-to-{target}'
        table, cross, mroot, features, refs, y, utility, selection = prepare(source, target)
        for role, ids in (('utility_train', utility), ('selection', selection)):
            for c, part in table.iloc[ids].groupby('class_id'):
                coverage.append(dict(source=source, role=role, class_id=int(c), n=len(part), ships=part.mmsi.nunique(), passes=part.pass_group.nunique()))
        for seed in SEEDS:
            folder = aligned.OLD/direction/f'seed{seed}'
            zs, zt = [load(folder/f'{side}_predictions.npz') for side in ('source', 'cross')]
            for z, tab in ((zs, table), (zt, cross)):
                np.testing.assert_array_equal(z['obs_id'], tab.obs_id)
            cached = {k: np.concatenate((zs[k], zt[k])) for k in ('available', 'gate_raw', 'frequency')}
            base = np.concatenate((zs['bare'], zt['bare'])).astype(np.float32)
            ss = load(aligned.REPAIR/f'source_{source}/seed{seed}/predictions.npz')
            tt = load(aligned.REPAIR/direction/f'seed{seed}/predictions.npz')
            np.testing.assert_array_equal(tt['obs_id'], cross.obs_id)
            np.testing.assert_array_equal(tt['refs'], refs[len(table):])
            oldlocal = np.concatenate((ss['local'], tt['local']))
            retained = [load(ALIGNED/direction/f'seed{seed}/{side}_predictions.npz') for side in ('source', 'cross')]
            for z, tab in zip(retained, (table, cross)):
                np.testing.assert_array_equal(z['obs_id'], tab.obs_id)
            expected = {k: np.concatenate([z[k] for z in retained]) for k in ('bare', 'local', 'refs', 'B_fixed', 'frequency')}
            np.testing.assert_array_equal(base, expected['bare'])
            np.testing.assert_array_equal(refs, expected['refs'])
            metric_sc = tr.load_checkpoint(mroot/f'seed{seed}/scalers.pt')['metric_scaler']
            original_state = tr.load_checkpoint(mroot/f'seed{seed}/both_metric.pt')['state']
            for arm in ARMS:
                out = HERE/direction/f'seed{seed}'/arm
                out.mkdir(parents=True, exist_ok=True)
                tick = time.perf_counter()
                matcher = ReplacementMatcher(features, refs, metric_sc, device, arm)
                metric = m3.PairMetric().to(device)
                metric.load_state_dict(original_state)
                decision = aligned.FrozenDecision(folder, cached, base, device)
                if arm == 'both':
                    replay_local = matcher.predict(selection, metric)
                    np.testing.assert_allclose(replay_local, oldlocal[selection], atol=2e-5, rtol=0)
                    with torch.no_grad():
                        replay = decision(torch.tensor(oldlocal[selection], device=device), selection).cpu().numpy()
                    np.testing.assert_allclose(replay, cached['frequency'][selection], atol=2e-5, rtol=0)
                    checks.append(dict(source=source, seed=seed, original_local_max=float(abs(replay_local-oldlocal[selection]).max()), original_decision_max=float(abs(replay-cached['frequency'][selection]).max())))
                if arm in ('both', 'learned_fixed'):
                    step, gradient, seconds = aligned.fit_metric(matcher, metric, decision, y, utility, selection, out)
                else:
                    step, gradient, seconds = 0, 0., 0.
                    write(out/'fixed_metric.json', dict(metric='ordinary cosine', trainable_parameters=0, optimizer_steps=0))
                local = oldlocal.copy() if arm == 'both' and step == 0 else matcher.predict(np.arange(len(base)), metric)
                scores, extra, receipt = refit(local, base, cached, y, utility, selection, seed, out)
                receipt.update(source=source, seed=seed, arm=arm, metric_step=step,
                               metric_max_gradient=gradient, metric_seconds=seconds,
                               metric_parameter_change=(max(float(abs(v-original_state[k]).max()) for k, v in m3.state(metric).items()) if arm in ('both', 'learned_fixed') else 0.),
                               seconds=time.perf_counter()-tick)
                if arm == 'both':
                    old_selection = tr.read_json(ALIGNED/direction/f'seed{seed}/selection.json')
                    for field in ('metric_step', 'vv_step', 'vh_step', 'alpha', 'frequency_step'):
                        assert receipt[field] == old_selection[field], (source, seed, field, receipt[field], old_selection[field])
                    np.testing.assert_allclose(local, expected['local'], atol=2e-5, rtol=0)
                    for endpoint in ('B_fixed', 'frequency'):
                        np.testing.assert_array_equal(scores[endpoint], expected[endpoint])
                    checks.append(dict(source=source, seed=seed, retained_both_final_exact=True,
                                       retained_local_max=float(abs(local-expected['local']).max()),
                                       retained_frequency_equals_B_fixed=bool(np.array_equal(expected['frequency'], expected['B_fixed']))))
                extra['refs'] = refs
                for side, sl, tab in (('source', slice(0, len(table)), table), ('cross', slice(len(table), None), cross)):
                    np.savez_compressed(out/f'{side}_predictions.npz', obs_id=tab.obs_id.to_numpy(),
                                        **{k: v[sl] for k, v in scores.items()}, **{k: v[sl] for k, v in extra.items()})
                    activities.append(dict(source=source, seed=seed, arm=arm, side=side,
                                           spatial_step=receipt['spatial_step'], frequency_step=receipt['frequency_step'],
                                           **gate_activity(scores, extra, np.arange(len(base))[sl])))
                receipts.append(receipt)
                write(out/'selection.json', receipt)
                pd.DataFrame(receipts).to_csv(HERE/'selections.csv', index=False)
                pd.DataFrame(activities).to_csv(HERE/'gate_activity.csv', index=False)
                write(HERE/'checks.json', checks)
                print('FROZEN_M3', receipt, flush=True)
                del matcher, metric, decision
                if device == 'cuda':
                    torch.cuda.empty_cache()
    pd.DataFrame(coverage).to_csv(HERE/'source_coverage.csv', index=False)
    write(marker, dict(configurations=len(receipts), expected_configurations=24,
                       target_labels_read=False, all_both_replays_passed=True,
                       seconds=time.perf_counter()-start, raw_review_pending=True))


def evaluate():
    frozen = tr.read_json(HERE/'predictions_saved.json')
    assert frozen['configurations'] == 24 and frozen['all_both_replays_passed']
    rows, paired, classes, cases = [], [], [], []
    for source, target in DIRECTIONS:
        direction = f'{source}-to-{target}'
        for side in ('source', 'cross'):
            table = (pd.read_csv(ROOT/f'{source}_m123_replication_20261026/stack_table.csv', dtype={'mmsi': str})
                     if side == 'source' else tr.read_table(tr.HERE/direction/'target_table.csv', labels=True))
            for seed in SEEDS:
                z = {arm: load(HERE/direction/f'seed{seed}'/arm/f'{side}_predictions.npz') for arm in ARMS}
                for arm in ARMS:
                    np.testing.assert_array_equal(z[arm]['obs_id'], table.obs_id)
                    for role, ix in table.groupby('role').indices.items():
                        yy = table.iloc[ix].class_id.to_numpy(int)
                        meta = dict(source=source, side=side, role=role, seed=seed, arm=arm)
                        for endpoint in ENDPOINTS:
                            score = z[arm][endpoint][ix]
                            rows.append(dict(**meta, endpoint=endpoint, **m3.metrics(yy, score)))
                            for c in np.unique(yy):
                                use = yy == c
                                classes.append(dict(**meta, endpoint=endpoint, class_id=int(c), n=int(use.sum()), recall=float((score.argmax(1)[use] == c).mean())))
                        comparisons = [('both', 'frequency', 'frequency'),
                                       (arm, 'B_fixed', 'spatial'), (arm, 'B_fixed', 'frequency'),
                                       (arm, 'spatial', 'frequency')]
                        for before_arm, before, after in comparisons:
                            a = z[before_arm][before][ix].argmax(1) == yy
                            b = z[arm][after][ix].argmax(1) == yy
                            delta = b.astype(float)-a
                            paired.append(dict(**meta, before_arm=before_arm, before=before, after=after,
                                               rescue=int((b & ~a).sum()), harm=int((a & ~b).sum()),
                                               CC=int((a & b).sum()), WW=int((~a & ~b).sum()),
                                               OA_pp=100*delta.mean(), BA_pp=100*np.mean([delta[yy == c].mean() for c in np.unique(yy)])))
                    if side == 'cross':
                        yy = table.class_id.to_numpy(int)
                        before = z['both']['frequency'].argmax(1)
                        after = z[arm]['frequency'].argmax(1)
                        for i in range(len(table)):
                            a, b = before[i] == yy[i], after[i] == yy[i]
                            rival = int(np.argmax(np.where(np.arange(4) == yy[i], -np.inf, z[arm]['frequency'][i])))
                            cases.append(dict(source=source, seed=seed, arm=arm, row=i, obs_id=table.obs_id.iloc[i],
                                              mmsi=table.mmsi.iloc[i], pass_group=table.pass_group.iloc[i],
                                              class_id=int(yy[i]), before=int(before[i]), after=int(after[i]),
                                              outcome='CC' if a and b else 'WW' if not a and not b else 'rescue' if b else 'harm',
                                              margin=float(z[arm]['frequency'][i, yy[i]]-z[arm]['frequency'][i, rival]),
                                              local_change=float(abs(z[arm]['local'][i]-z['both']['local'][i]).max()),
                                              prediction_file=str(HERE/direction/f'seed{seed}'/arm/'cross_predictions.npz'),
                                              target_table=str(tr.HERE/direction/'target_table.csv')))
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE/'metrics.csv', index=False)
    frame.groupby(['source', 'side', 'role', 'arm', 'endpoint'])[['OA', 'BA', 'J', 'NLL', 'Brier']].agg(['mean', 'std']).to_csv(HERE/'summary.csv')
    pd.DataFrame(paired).to_csv(HERE/'paired.csv', index=False)
    pd.DataFrame(classes).to_csv(HERE/'per_class.csv', index=False)
    pd.DataFrame(cases).to_csv(HERE/'case_index.csv', index=False)
    write(HERE/'evaluation_complete.json', dict(predictions_saved_first=True, raw_review_pending=True,
                                               gate_activity='gate_activity.csv', cases='case_index.csv'))
    print(frame.query("side == 'cross'").groupby(['source', 'arm', 'endpoint'])[['OA', 'BA']].mean().to_string(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'evaluate'))
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args()
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if args.mode == 'train':
        train(device)
    else:
        evaluate()
