"""Train the accepted architecture on eight classes, then freeze all predictions."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from prepare_bundle import HERE, load, write
from sar_m123 import train_aligned as ta, train_ops as op, tokens
from sar_m123.m3 import PairMetric, PolarHead, regional_descriptors

SEEDS = (17, 29, 43)


@torch.inference_mode()
def apply_readouts(folder, base, local, descriptors, refs):
    sc = op.load_checkpoint(folder/'M3_readout_scaler.pt')
    x = torch.tensor(op.transform(op.local_raw(local, base), sc))
    deltas = []
    for p, name in enumerate(('vv', 'vh_centered')):
        ck = op.load_checkpoint(folder/f'{name}.pt')
        head = PolarHead(p, bool(p), ck['state'].get('null'))
        head.load_state_dict(ck['state'])
        deltas.append(head.eval()(x).numpy())
    dv, dh = deltas
    raw, av = op.gate_raw(descriptors, descriptors, refs, base, dv, dh)
    output = dict(bare=base, delta_vv=dv, delta_vh=dh, raw_gate=raw, available=av)
    for arm in ('frequency', 'spatial'):
        ck = op.load_checkpoint(folder/f'{arm}.pt')
        dim = 185 if arm == 'frequency' else 41
        gate = op.Gate(dim, ck['alpha']).eval()
        gate.load_state_dict(ck['state'])
        inp = torch.tensor(op.transform(raw[..., :dim], ck['scaler']))
        w = gate(inp, torch.tensor(av)).numpy()
        output[arm] = base+w[:, :1]*dv+w[:, 1:]*dh
        output['weight_'+arm] = w
        a = ck['alpha']
        fixed = av*np.array([a, 1-a], np.float32)
        output['B_fixed'] = base+fixed[:, :1]*dv+fixed[:, 1:]*dh
        output['weight_B_fixed'] = fixed
    return output


class CosineFixed(op.Matcher):
    """Matched same-slot cosine control, with separately fitted downstream heads."""
    def local(self, idx, model):
        idx = torch.as_tensor(idx, device=self.device)
        ref = self.refs[idx]
        q, s = self.x[idx], self.x[ref]
        a, b = self.mass[idx], self.mass[ref]
        groups = []
        for region in tokens.REGIONS:
            qi = q[:, :, region][:, None, None]
            sj = s[..., region, :]
            sim = (qi*sj).sum(-1)/(qi.square().sum(-1)*sj.square().sum(-1)).clamp_min(1e-20).sqrt()
            qa, sa = a[:, None, None, :, region], b[..., region]
            qa = qa/qa.sum(-1, keepdim=True).clamp_min(1e-20)
            sa = sa/sa.sum(-1, keepdim=True).clamp_min(1e-20)
            mass = torch.minimum(qa, sa)*((qa > 0) & (sa > 0))
            groups.append(torch.stack(((mass*sim).sum(-1), mass.sum(-1)), -1))
        return torch.stack(groups, -2).mean(2)


def add_spatial(folder, cached, base, table, groups, seed):
    y = table.class_id.to_numpy(int)
    a = op.load_checkpoint(folder/'frequency.pt')['alpha']
    op.fit_gate('spatial', seed, cached['raw_gate'] if 'raw_gate' in cached else cached['gate_raw'],
        base, cached['delta_vv'], cached['delta_vh'], cached['available'], y,
        groups['utility_train'], groups['selection'], a, folder)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('train', 'predict', 'all'), default='all')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tick = time.perf_counter()
    if args.stage in ('train', 'all'):
        ta.train(HERE/'bundle', HERE/'initial_assets', HERE/'training', args.seeds, 'cuda')
    if args.stage == 'train':
        return
    source, ext = load(HERE/'bundle/features.npz'), load(HERE/'bundle/external.npz')
    table = pd.read_csv(HERE/'bundle/table.csv', dtype={'mmsi': str})
    groups = ta.validate_roles(table, source)
    n = len(table)
    covariance = {k: source['covariance_'+k] for k in ('mu', 'scale', 'active')}
    features = {k: np.concatenate((source[k], ext[k])) for k in ta.FEATURES}
    refs = np.concatenate((source['refs'], ext['refs']))
    descriptors = regional_descriptors(features, covariance)
    source_desc = regional_descriptors({k: source[k] for k in ta.FEATURES}, covariance)
    needed = np.r_[np.flatnonzero(source['local_available']), np.arange(n, n+len(ext['obs_id']))]
    for seed in args.seeds:
        folder = HERE/'training'/f'seed{seed}'
        dest = HERE/'results'/f'seed{seed}'
        dest.mkdir(parents=True, exist_ok=True)
        rawsc, auxsc = load(folder/'raw_scaler.npz'), load(folder/'summary_scaler.npz')
        x = np.c_[ta.scale_input(ext['raw'], rawsc), ta.scale_input(ext['aux'], auxsc)]
        ck = op.load_checkpoint(folder/'raw_summary.pt')
        head = nn.Sequential(nn.Linear(1714, 128), nn.GELU(), nn.Linear(128, 8)).eval()
        head.load_state_dict(ck['state'])
        with torch.inference_mode():
            eb = head(torch.tensor(x)).numpy()
        saved = load(folder/'predictions.npz')
        base = np.concatenate((saved['bare'], eb))
        add_spatial(folder, saved, base[:n], table, groups, seed)
        sc = op.load_checkpoint(HERE/'initial_assets/warm'/f'seed{seed}/metric_scaler.pt')
        matcher = op.Matcher(features, refs, sc, 'cuda')
        metric = PairMetric().cuda().eval()
        metric.load_state_dict(op.load_checkpoint(folder/'aligned_metric.pt')['state'])
        local = np.zeros((len(base), 8, 2, 3, 2), np.float32)
        local[:n] = saved['local']
        local[n:] = matcher.predict(np.arange(n, len(base)), metric)
        out = apply_readouts(folder, base, local, descriptors, refs)
        np.testing.assert_allclose(out['frequency'][:n], saved['frequency'], atol=2e-5, rtol=1e-5)
        np.testing.assert_array_equal(out['frequency'][:n].argmax(1), saved['frequency'].argmax(1))
        np.savez_compressed(dest/'both.npz', obs_id=np.r_[source['obs_id'], ext['obs_id']],
                            refs=refs, local=local, **out)
        # A frozen-metric control uses the exact warm heads on the same base.
        initial = PairMetric().cuda().eval()
        initial.load_state_dict(op.load_checkpoint(HERE/'initial_assets/warm'/f'seed{seed}/both_metric.pt')['state'])
        oldlocal = np.zeros_like(local)
        oldlocal[needed] = matcher.predict(needed, initial)
        warm = folder/'warm'
        ww = apply_warm(warm, base, oldlocal, descriptors, refs)
        add_spatial(warm, {k: v[:n] for k, v in ww.items()}, base[:n], table, groups, seed)
        wo = apply_readouts(warm, base, oldlocal, descriptors, refs)
        np.savez_compressed(dest/'frozen_metric.npz', refs=refs, **wo)
        simple = CosineFixed(features, refs, sc, 'cuda')
        sl = np.zeros_like(local)
        sl[needed] = simple.predict(needed, initial)
        sfolder = folder/'cosine_fixed'
        sfolder.mkdir(exist_ok=True)
        simple_cached, receipt = ta.fit_readouts(sl[:n], base[:n], {k: source[k] for k in ta.FEATURES},
            source['refs'], covariance, table, groups, seed, sfolder)
        add_spatial(sfolder, simple_cached, base[:n], table, groups, seed)
        so = apply_readouts(sfolder, base, sl, descriptors, refs)
        np.savez_compressed(dest/'cosine_fixed.npz', refs=refs, **so)
        write(sfolder/'selection.json', receipt)
        print('PREDICTIONS_FROZEN', seed, 'seconds', round(time.perf_counter()-tick, 1), flush=True)
    write(HERE/'prediction_complete.json', dict(seeds=args.seeds, elapsed_seconds=time.perf_counter()-tick,
        heldout_labels_used=False, n_source=n, n_coverage=len(ext['obs_id']),
        arms=['both', 'frozen_metric', 'cosine_fixed'], endpoints=['bare', 'B_fixed', 'spatial', 'frequency']))


@torch.inference_mode()
def apply_warm(folder, base, local, descriptors, refs):
    # Produce only the inputs needed to fit the spatial-only gate control.
    sc = op.load_checkpoint(folder/'M3_readout_scaler.pt')
    x = torch.tensor(op.transform(op.local_raw(local, base), sc))
    deltas = []
    for p, name in enumerate(('vv', 'vh_centered')):
        ck = op.load_checkpoint(folder/f'{name}.pt')
        h = PolarHead(p, bool(p), ck['state'].get('null'))
        h.load_state_dict(ck['state'])
        deltas.append(h.eval()(x).numpy())
    raw, av = op.gate_raw(descriptors, descriptors, refs, base, *deltas)
    return dict(raw_gate=raw, available=av, delta_vv=deltas[0], delta_vh=deltas[1])


if __name__ == '__main__':
    main()
