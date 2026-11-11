"""Matched M2 region-identity replacements; current model code is unchanged."""
from pathlib import Path
import argparse
import gc
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'staged_ablation_20261109'
sys.path.insert(0, str(SOURCE))
import staged as e

torch = e.torch
ARMS = ('merged_readout', 'merged_all')


def merge(masks):
    # Three equal shares preserve the sum of soft memberships, including edges.
    return masks.sum(1, keepdim=True).expand_as(masks) / masks.shape[1]


def merged_pool(module, args):
    return (args[0], merge(args[1]), *args[2:])


def merged_guide(module, args):
    b = dict(args[0])
    b['mask'] = b['mask'].sum(2, keepdim=True).expand_as(b['mask']) / 3
    return (b, *args[1:])


def install(net, arm):
    for pool in list(net.base.m2) + list(net.base.deep_pool):
        pool.register_forward_pre_hook(merged_pool)
    net.guide.register_forward_pre_hook(merged_guide)
    if arm == 'merged_all':
        for writer in net.base.internal_m2:
            writer.register_forward_pre_hook(merged_pool)


def smoke(protocol, rows, groups, a, c):
    ids = groups['small'][:4]
    rid = groups['references']
    seed = 17
    tests = {}
    clean = e.Network('full', seed, protocol['pretrain'])
    initial = {k: v.detach().cpu().clone() for k, v in clean.state_dict().items()}
    del clean
    for arm in ARMS:
        net = e.Network('full', seed, protocol['pretrain'])
        install(net, arm)
        assert all(torch.equal(v.cpu(), initial[k]) for k, v in net.state_dict().items())
        audits = []
        for name, slots in [('low', c['local_slots']), ('deep', a['slots'])]:
            for p in range(2):
                source = e.task.tensor(slots, ids)[:, p]
                r, en = net.base.region_integrals(source, source.new_zeros(len(ids), 20, 3, *source.shape[-2:]))
                merged = merge(r)
                assert float(r.sum(1).max()) <= 1.00001
                assert torch.allclose(r.sum(1), merged.sum(1), atol=1e-6)
                assert torch.equal(r.sum(1) > 0, merged.sum(1) > 0)
                assert torch.equal(merged[:, 0], merged[:, 1])
                feature = torch.arange(source.shape[-2]*source.shape[-1], device=source.device).float().reshape(1,1,*source.shape[-2:]).expand(len(ids),2,-1,-1)
                valid = e.task.tensor(a['valid'], ids)[:, p]
                pooled, _ = e.old.regional_mean(feature, merged, valid)
                union, _ = e.old.regional_mean(feature, r.sum(1, keepdim=True), valid)
                assert torch.allclose(pooled, union.expand_as(pooled), atol=1e-4)
                audits.append(dict(level=name,pol=p,max_membership=float(r.sum(1).max()),sum_error=float((r.sum(1)-merged.sum(1)).abs().max())))
        e.configure(net, 0)
        z, _ = net(a, c, rows, ids, rid)
        e.F.cross_entropy(z, torch.tensor([rows[i]['label'] for i in ids], device=e.task.DEVICE)).backward()
        gradients = {n: float(p.grad.norm()) for n,p in net.named_parameters() if p.requires_grad and p.grad is not None}
        assert all(np.isfinite(list(gradients.values())))
        assert any(v > 0 for n,v in gradients.items() if 'internal_m2' in n)
        tests[arm] = dict(initial_state_identical=True,parameters=sum(p.numel() for p in net.parameters()),geometry=audits,gradients=gradients)
        del net,z
        gc.collect(); torch.cuda.empty_cache()
    del initial
    net = e.Network('full', seed, protocol['pretrain'])
    parent = SOURCE / 'full_s17_lr1'
    net.load_state_dict(torch.load(parent/'readout/final.pt', map_location='cpu', weights_only=True))
    e.configure(net, 3)
    e.freeze_delta(net, a, rows, rid)
    score = e.predict(net,a,c,rows,groups['validation'][:12],rid)
    error = float(abs(score-np.load(parent/'validation.npz')['score'][:12]).max())
    assert error < 1e-4
    tests['full_replay_error'] = error
    e.task.write(ROOT/'smoke.json', tests)
    print('SMOKE_OK', error, flush=True)
    del net
    gc.collect(); torch.cuda.empty_cache()


def fit(arm, seed, protocol, rows, groups, a, c):
    folder = ROOT / f'{arm}_s{seed}'
    if (folder/'complete.json').exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    net = e.Network('full', seed, protocol['pretrain'])
    install(net, arm)
    parent = SOURCE / f'full_s{seed}_lr1'
    for cfg in protocol['phases']:
        dest = folder/cfg['name']
        e.task.write(ROOT/'status.json', dict(arm=arm, seed=seed, phase=cfg['name'], complete=False))
        if (dest/'receipt.json').exists():
            net.load_state_dict(torch.load(dest/'final.pt', map_location='cpu', weights_only=True))
            e.configure(net, cfg['phase'])
        else:
            e.train_phase(net,cfg,seed,folder,rows,groups,a,c,scale=1.)
        got = json.loads((dest/'orders.json').read_text())
        expected = json.loads((parent/cfg['name']/'orders.json').read_text())
        assert got == expected, (arm, seed, cfg['name'])
        if cfg['phase'] == 1:
            e.freeze_delta(net,a,rows,groups['references'])
    e.configure(net,3)
    net.requires_grad_(False)
    metrics = {}
    for split in ('validation','evaluation'):
        ids = groups[split]
        score = e.predict(net,a,c,rows,ids,groups['references'])
        np.savez_compressed(folder/f'{split}.npz',ids=ids,score=score)
        metrics[split] = e.task.metrics(np.array([rows[i]['label'] for i in ids]),score)
    e.task.write(folder/'complete.json',dict(arm=arm,seed=seed,metrics=metrics,orders_identical=True,formal_model_changed=False))
    print('COMPLETE',arm,seed,metrics,flush=True)
    del net
    gc.collect(); torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-only',action='store_true')
    args = parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    protocol = json.loads((SOURCE/'protocol.json').read_text(encoding='utf-8'))
    e.task.current.setup(e.task.DEVICE)
    rows,groups = e.prev.records()
    a,c = e.prev.arrays()
    a['write'] = np.load(e.PREVIOUS/'common_init/write.npy',mmap_mode='r')
    e.task.write(ROOT/'protocol.json',dict(parent=protocol,arms=ARMS,seeds=[17,29,43],data={k:len(v) for k,v in groups.items()},
        scope='matched full five-stage retraining from common initialization; Full reused',
        intervention='arm-specific teacher and source-hard weights under unchanged rules',
        geometry='M2 soft union preserved; equal thirds remove region identity; M3 original geometry unchanged',
        selection='inherited full scale1; no new tuning; test previously used in development'))
    e.resident(a,c)
    smoke(protocol,rows,groups,a,c)
    if args.smoke_only:
        return
    for seed in (17,29,43):
        for arm in ARMS:
            fit(arm,seed,protocol,rows,groups,a,c)
    e.task.write(ROOT/'status.json',dict(complete=True,arms=ARMS,seeds=[17,29,43]))


if __name__ == '__main__':
    main()
