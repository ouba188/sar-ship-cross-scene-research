"""Align the existing PairMetric with the frozen raw-summary final decision."""
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
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT/'paper_evidence_completion_20261026'))
import run_classical as base_tools
tr = base_tools.tr
m3 = tr.m3
OLD = ROOT/'raw_region_bypass_20261026/downstream/raw_summary'
REPAIR = tr.HERE/'normalization_repair_20261026'
spec = importlib.util.spec_from_file_location('old_downstream', OLD.parent/'run.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
load, write = tr.load_npz, tr.write_json
SEEDS = (17, 29, 43)


class FrozenDecision:
    """Differentiate actual polar fusion, including the base-dependent gate fields."""
    def __init__(self, folder, cached, base, device):
        self.device = device
        self.b = torch.tensor(base, device=device)
        self.av = torch.tensor(cached['available'], device=device)
        self.tail = torch.tensor(cached['gate_raw'][..., 11:], device=device)
        sc = tr.load_checkpoint(folder/'M3_readout_scaler.pt')
        self.sc = {k: torch.as_tensor(v, device=device) for k, v in sc.items()}
        self.polar = []
        for p, name in enumerate(('vv', 'vh_centered')):
            ck = tr.load_checkpoint(folder/f'{name}.pt')
            h = m3.PolarHead(p, centered=bool(p), null=ck['state'].get('null'))
            self.polar.append(tr.frozen_model(h, ck, device))
        ck = tr.load_checkpoint(folder/'frequency.pt')
        self.gs = {k: torch.as_tensor(v, device=device) for k, v in ck['scaler'].items()}
        self.gate = tr.frozen_model(m3.Gate(185, ck['alpha']), ck, device)

    def __call__(self, local, ids):
        b, av = self.b[ids], self.av[ids]
        z = b-b.mean(1, keepdim=True)
        rival = torch.stack([torch.cat((z[:, :c], z[:, c+1:]), 1).max(1).values for c in range(4)], 1)
        raw = torch.cat((local.reshape(len(ids), 4, 12), torch.stack((z, rival, z-rival), -1)), -1)
        x = torch.where(self.sc['active'], (raw-self.sc['mu'])/self.sc['scale'], 0).float()
        dv, dh = (h(x) for h in self.polar)
        vals = [z, dv, dh]
        for s in (b, b+dv, b+dh):
            other = torch.stack([torch.cat((s[:, :c], s[:, c+1:]), 1).max(1).values for c in range(4)], 1)
            vals.append(s-other)
        vals.extend(torch.softmax(s, 1) for s in (b, b+dv, b+dh))
        vals.extend(av[:, p, None].expand_as(b) for p in range(2))
        gr = torch.cat((torch.stack(vals, -1), self.tail[ids]), -1)
        gx = torch.where(self.gs['active'], (gr-self.gs['mu'])/self.gs['scale'], 0).float()
        w = self.gate(gx, av)
        return b+w[:, :1]*dv+w[:, 1:]*dh


def fit_metric(matcher, metric, decision, y, fit, val, out):
    def predict(ids):
        with torch.no_grad():
            return np.concatenate([decision(matcher.local(ids[k:k+48], metric), ids[k:k+48]).cpu().numpy()
                                   for k in range(0, len(ids), 48)])
    initial = predict(val)
    best, selected, chosen = m3.exact_j(y[val], initial), 0, m3.state(metric)
    curve = [dict(step=0, **m3.metrics(y[val], initial))]
    wt = m3.balanced_weights(y, fit, matcher.device)
    yt = torch.tensor(y[fit], device=matcher.device)
    opt = torch.optim.AdamW(metric.parameters(), lr=.003, weight_decay=.01)
    tick = time.perf_counter()
    gradient_max = 0.
    for step in range(1, 101):
        opt.zero_grad(set_to_none=True)
        total = 0.
        for k in range(0, len(fit), 24):
            ix = fit[k:k+24]
            scores = decision(matcher.local(ix, metric), ix)
            loss = (F.cross_entropy(scores, yt[k:k+24], reduction='none')*wt[k:k+24]).sum()
            assert torch.isfinite(loss)
            loss.backward()
            total += float(loss.detach())
        grad = max(float(p.grad.abs().max()) for p in metric.parameters() if p.grad is not None)
        gradient_max = max(gradient_max, grad)
        # An exactly zero frozen correction has no gradient path to matching.
        if step == 1 and grad == 0:
            curve.append(dict(step=1, loss=total, gradient_max=grad, **m3.metrics(y[val], initial)))
            break
        opt.step()
        if step in (5, 10, 15, 20, 30, 50, 75, 100):
            scores = predict(val)
            j = m3.exact_j(y[val], scores)
            curve.append(dict(step=step, loss=total, gradient_max=grad, **m3.metrics(y[val], scores)))
            if j > best:
                best, selected, chosen = j, step, m3.state(metric)
            print('METRIC', out.parent.name, out.name, step, 'J', float(j), 'seconds', round(time.perf_counter()-tick), flush=True)
    metric.load_state_dict(chosen)
    metric.eval().requires_grad_(False)
    torch.save(dict(state=chosen, selected_step=selected), out/'aligned_metric.pt')
    pd.DataFrame(curve).to_csv(out/'metric_curve.csv', index=False)
    return selected, gradient_max, time.perf_counter()-tick


def train():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checks, receipts, coverage = [], [], []
    tick = time.perf_counter()
    for source, target in (('oss1', 'oss2'), ('oss2', 'oss1')):
        direction = f'{source}-to-{target}'
        source_root = ROOT/f'{source}_m123_replication_20261026'
        table = pd.read_csv(source_root/'stack_table.csv', dtype={'mmsi': str})
        crossroot = tr.HERE/direction
        cross = tr.read_table(crossroot/'target_table.csv')
        protocol = tr.read_json(crossroot/'protocol.json')
        mroot = Path(protocol['source_m3'])
        feat = load(mroot/'local.npz')
        projection = load(mroot/'projection.npz')
        raw = tr.aligned(Path(protocol['target_cache'])/'m3_raw.npz', cross)
        ext = tr.project_raw(raw, projection)
        features = {k: np.concatenate((feat[k], ext[k])) for k in tr.FEATURES}
        refs0 = load(mroot/'references.npz')
        bank, bankrows, covariance = tr.source_bank(mroot, table)
        er = tr.retrieve(ext, bank, cross.mmsi, table.iloc[bankrows].mmsi)
        refs = np.concatenate((refs0['refs'], bankrows[er]))
        utility = np.flatnonzero(table.role.eq('utility_train'))
        selection = np.flatnonzero(table.role.eq('selection'))
        y = table.class_id.to_numpy(int, copy=True)
        y[table.role.eq('evaluation')] = -1
        for role, ids in (('utility_train', utility), ('selection', selection)):
            for field in ('mmsi', 'pass_group'):
                assert not (table.iloc[ids][field].to_numpy()[:, None, None] == table[field].to_numpy()[refs[ids]]).any()
            for c in range(4):
                part = table.iloc[ids].query('class_id == @c')
                coverage.append(dict(source=source, role=role, class_id=c, n=len(part), ships=part.mmsi.nunique(), passes=part.pass_group.nunique()))
        for field in ('mmsi', 'pass_group'):
            assert not set(table.iloc[utility][field]) & set(table.iloc[selection][field])
            assert not set(table.loc[table.role.ne('evaluation'), field]) & set(cross[field])
        for seed in SEEDS:
            out = HERE/direction/f'seed{seed}'
            out.mkdir(parents=True, exist_ok=True)
            folder = OLD/direction/f'seed{seed}'
            zs, zt = (load(folder/f'{s}_predictions.npz') for s in ('source', 'cross'))
            np.testing.assert_array_equal(zs['obs_id'], table.obs_id)
            np.testing.assert_array_equal(zt['obs_id'], cross.obs_id)
            cached = {k: np.concatenate((zs[k], zt[k])) for k in ('available', 'gate_raw', 'frequency')}
            b = np.concatenate((zs['bare'], zt['bare'])).astype(np.float32)
            ss, tt = (load(REPAIR/p/f'seed{seed}/predictions.npz') for p in (f'source_{source}', direction))
            np.testing.assert_array_equal(tt['refs'], refs[len(table):])
            oldlocal = np.concatenate((ss['local'], tt['local']))
            sc = tr.load_checkpoint(mroot/f'seed{seed}/scalers.pt')['metric_scaler']
            matcher = m3.Matcher(features, refs, sc, device)
            metric = m3.PairMetric().to(device)
            metric.load_state_dict(tr.load_checkpoint(mroot/f'seed{seed}/both_metric.pt')['state'])
            decision = FrozenDecision(folder, cached, b, device)
            with torch.no_grad():
                replay = decision(torch.tensor(oldlocal[selection], device=device), selection).cpu().numpy()
            error = float(abs(replay-cached['frequency'][selection]).max())
            np.testing.assert_allclose(replay, cached['frequency'][selection], rtol=0, atol=2e-5)
            np.testing.assert_array_equal(replay.argmax(1), cached['frequency'][selection].argmax(1))
            replay_local = matcher.predict(selection, metric)
            np.testing.assert_allclose(replay_local, oldlocal[selection], rtol=0, atol=2e-5)
            initial_state = m3.state(metric)
            step, gradient, seconds = fit_metric(matcher, metric, decision, y, utility, selection, out)
            local = oldlocal.copy() if step == 0 else matcher.predict(np.arange(len(b)), metric)
            with torch.no_grad():
                metric_only = np.concatenate([decision(torch.tensor(local[k:k+48], device=device), np.arange(k, min(k+48, len(b)))).cpu().numpy()
                                              for k in range(0, len(b), 48)])
            if step == 0:
                metric_only = cached['frequency'].copy()
            raw = m3.local_raw(local, b)
            scaler = m3.scale_fit(raw, utility)
            x = m3.transform(raw, scaler)
            torch.manual_seed(seed)
            joint = m3.JointHead()
            dj, joint_step = m3.fit_head('joint_refit', joint, x, b, y, utility, selection, out)
            null = m3.transform(np.zeros((1, 1, 15)), scaler).reshape(15)
            deltas, steps = [], []
            for p, name in enumerate(('vv', 'vh_centered')):
                torch.manual_seed(seed)
                head = m3.PolarHead(p, centered=bool(p), null=null)
                torch.nn.init.zeros_(head.net[-1].weight)
                torch.nn.init.zeros_(head.net[-1].bias)
                d, hstep = m3.fit_head(name, head, x, b, y, utility, selection, out)
                deltas.append(d); steps.append(hstep)
            dv, dh = deltas
            av = cached['available']
            alpha = m3.choose_alpha(b, dv, dh, av, y, selection)
            weights = av*np.array([alpha, 1-alpha], np.float32)
            gr = old.previous.gate_raw(cached['gate_raw'], b, dv, dh, av)
            final, fw, gs = m3.fit_gate('frequency', seed, gr, b, dv, dh, av, y, utility, selection, alpha, out)
            np.testing.assert_allclose(final, b+fw[:, :1]*dv+fw[:, 1:]*dh, atol=0, rtol=0)
            if step == 0:
                np.testing.assert_array_equal(final, cached['frequency'])
            scores = dict(bare=b, old=cached['frequency'], metric_only=metric_only,
                          joint=b+dj, B_fixed=b+weights[:, :1]*dv+weights[:, 1:]*dh, frequency=final)
            extra = dict(delta_vv=dv, delta_vh=dh, weight_frequency=fw, local=local, refs=refs, gate_raw=gr, available=av)
            for side, sl, tab in (('source', slice(0, len(table)), table), ('cross', slice(len(table), None), cross)):
                np.savez_compressed(out/f'{side}_predictions.npz', obs_id=tab.obs_id.to_numpy(),
                                    **{k:v[sl] for k,v in scores.items()}, **{k:v[sl] for k,v in extra.items()})
            torch.save(scaler, out/'M3_readout_scaler.pt')
            receipt = dict(source=source, seed=seed, metric_step=step, max_gradient=gradient,
                           metric_seconds=seconds, metric_parameter_change=max(float(abs(v-initial_state[k]).max()) for k,v in m3.state(metric).items()),
                           joint_step=joint_step, vv_step=steps[0], vh_step=steps[1], alpha=alpha, frequency_step=gs,
                           local_change=float(abs(local-oldlocal).max()), new_base_training=False)
            receipts.append(receipt)
            checks.append(dict(source=source, seed=seed, old_final_replay_error=error,
                               old_local_replay_error=float(abs(replay_local-oldlocal[selection]).max()),
                               zero_metric_full_identity=step==0, source_query_ref_pass_ship_disjoint=True))
            write(out/'selection.json', receipt)
            print('FROZEN', receipt, flush=True)
            del matcher, metric, decision
            if device == 'cuda': torch.cuda.empty_cache()
    pd.DataFrame(coverage).to_csv(HERE/'source_coverage.csv', index=False)
    pd.DataFrame(receipts).to_csv(HERE/'selections.csv', index=False)
    write(HERE/'checks.json', checks)
    write(HERE/'predictions_saved.json', dict(configurations=6, target_labels_read=False, seconds=time.perf_counter()-tick,
          source_roles_unchanged=True, source_selection_reused=True, new_architecture=False))


def evaluate():
    assert (HERE/'predictions_saved.json').exists()
    rows, paired, classes, passes = [], [], [], []
    for source, target in (('oss1', 'oss2'), ('oss2', 'oss1')):
        direction=f'{source}-to-{target}'
        for side in ('source', 'cross'):
            table=pd.read_csv(ROOT/f'{source}_m123_replication_20261026/stack_table.csv', dtype={'mmsi':str}) if side=='source' else tr.read_table(tr.HERE/direction/'target_table.csv', labels=True)
            for seed in SEEDS:
                z=load(HERE/direction/f'seed{seed}/{side}_predictions.npz')
                np.testing.assert_array_equal(z['obs_id'],table.obs_id)
                for role,ix in table.groupby('role').indices.items():
                    yy=table.iloc[ix].class_id.to_numpy(int)
                    meta=dict(source=source,side=side,role=role,seed=seed)
                    for arm in ('bare','old','metric_only','joint','B_fixed','frequency'):
                        rows.append(dict(**meta,arm=arm,**m3.metrics(yy,z[arm][ix])))
                    for before,after in (('old','frequency'),('bare','frequency'),('old','metric_only')):
                        a=z[before][ix].argmax(1)==yy; b=z[after][ix].argmax(1)==yy
                        d=b.astype(float)-a
                        paired.append(dict(**meta,before=before,after=after,rescue=int((b&~a).sum()),harm=int((a&~b).sum()),OA_pp=100*d.mean(),BA_pp=100*np.mean([d[yy==c].mean() for c in np.unique(yy)])))
                        for c in np.unique(yy):
                            use=yy==c
                            classes.append(dict(**meta,before=before,after=after,class_id=int(c),n=int(use.sum()),before_recall=100*a[use].mean(),after_recall=100*b[use].mean(),rescue=int((b&~a&use).sum()),harm=int((a&~b&use).sum())))
                if side=='cross':
                    for name,ix in table.groupby('pass_group').indices.items():
                        yy=table.iloc[ix].class_id.to_numpy(int)
                        a=z['old'][ix].argmax(1)==yy; b=z['frequency'][ix].argmax(1)==yy
                        passes.append(dict(source=source,seed=seed,pass_group=name,n=len(ix),OA_pp=100*(b.astype(float)-a).mean()))
    frame=pd.DataFrame(rows)
    frame.to_csv(HERE/'metrics.csv',index=False)
    summary=frame.groupby(['source','side','role','arm']).agg(OA=('OA','mean'),BA=('BA','mean'),OA_std=('OA','std'),BA_std=('BA','std'),NLL=('NLL','mean'),Brier=('Brier','mean'),n=('n','first')).reset_index()
    summary.to_csv(HERE/'summary.csv',index=False)
    pd.DataFrame(paired).to_csv(HERE/'paired.csv',index=False)
    pd.DataFrame(classes).to_csv(HERE/'per_class.csv',index=False)
    pd.DataFrame(passes).to_csv(HERE/'per_pass.csv',index=False)
    write(HERE/'evaluation_complete.json',dict(predictions_saved_first=True,raw_review_pending=True))
    print(summary.query("side=='cross'").to_string(index=False),flush=True)


if __name__=='__main__':
    train()
    evaluate()
