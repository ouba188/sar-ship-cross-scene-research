"""Paired branch ablations from saved pre-treatment stages; formal code unchanged."""
from pathlib import Path
import argparse
import gc
import json
import sys
import types
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'staged_ablation_20261109'
sys.path.insert(0, str(SOURCE))
import staged as e

torch = e.torch
ARMS = {
    'm1_mean_only': ('m1', True, False),
    'm1_relation_only': ('m1', False, True),
    'm1_no_feedback': ('m1', False, False),
    'm3_metric_only': ('m3', True, False),
    'm3_bias_only': ('m3', False, True),
    'm3_neither': ('m3', False, False),
}


def suppress(module, args, output):
    return torch.zeros_like(output)


def no_competition(self, competition, arm):
    return torch.zeros_like(competition)


def install(net, arm):
    family, first, second = ARMS[arm]
    if family == 'm1':
        for branch in net.m1:
            if not first:
                branch.output.register_forward_hook(suppress)
            if not second:
                branch.relation_output.register_forward_hook(suppress)
    else:
        if not first:
            # The operator uses 1 + 0.5*tanh(metric); zero logits yield unit weights.
            net.matcher.metric.register_forward_hook(suppress)
        if not second:
            net.matcher.cost_competition = types.MethodType(no_competition, net.matcher)


def network(seed, checkpoint, protocol):
    net = e.Network('full', seed, protocol['pretrain'])
    net.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    return net


@torch.no_grad()
def smoke(protocol, rows, groups, a, c):
    seed = 17
    parent = SOURCE / f'full_s{seed}_lr1'
    net = network(seed, parent/'readout/final.pt', protocol)
    e.configure(net, 3)
    e.freeze_delta(net, a, rows, groups['references'])
    ids = groups['validation'][:12]
    actual = e.predict(net, a, c, rows, ids, groups['references'])
    expected = np.load(parent/'validation.npz')['score'][:len(ids)]
    error = float(np.abs(actual-expected).max())
    assert error < 1e-4, ('unchanged full replay', error)
    tests = {'full_replay_max_error': error}
    occupancy = torch.ones(2, 1, 20, 20, device=e.task.DEVICE)
    x = e.task.tensor(c['early_x'], groups['small'][:2])[:, 0]
    for arm in ('m1_mean_only','m1_relation_only','m1_no_feedback'):
        branch = e.old.RelationFeedback(128, 'class', .3).to(e.task.DEVICE)
        branch.load_state_dict(net.m1[0].state_dict())
        _, common, relation = ARMS[arm]
        components = {}
        handles = [branch.output.register_forward_hook(lambda m,a,o: components.update(common=o)),
                   branch.relation_output.register_forward_hook(lambda m,a,o: components.update(relation=o))]
        if not common: branch.output.register_forward_hook(suppress)
        if not relation: branch.relation_output.register_forward_hook(suppress)
        (out, _, _), trace = branch(x, occupancy, details=True)
        raw = (components['common'] if common else torch.zeros_like(components['common']))
        raw = raw + (components['relation'] if relation else torch.zeros_like(components['relation']))
        expected_out = x+.3*x*raw.tanh()*occupancy
        diff = float((out-expected_out).abs().max())
        assert diff < 1e-6
        if not common and not relation: assert torch.equal(out, x)
        tests[arm] = {'formula_error': diff}
    del net
    gc.collect()
    torch.cuda.empty_cache()
    rid = groups['references']
    ids = groups['small'][:2]
    for arm in ('m3_metric_only','m3_bias_only','m3_neither'):
        net = network(seed, parent/'reference/final.pt', protocol)
        install(net, arm)
        pair = e.task.legal(rows, ids, rid)[..., None] & e.task.tensor(a['available'],ids)[:,None].bool() & e.task.tensor(a['available'],rid)[None].bool()
        out = net.matcher(e.task.tensor(a['desc'],ids),e.task.tensor(a['desc'],rid),
            e.task.tensor(a['mass'],ids),e.task.tensor(a['mass'],rid),e.task.tensor(a['write'],ids),pair,
            torch.tensor([rows[i]['label'] for i in rid],device=e.task.DEVICE),arm='aggregate',details=True)
        _, metric, bias = ARMS[arm]
        traces = [r for pol in out[3] for r in pol['regions']]
        if not metric: assert all(torch.equal(r['metric'],torch.ones_like(r['metric'])) for r in traces)
        if not bias: assert all(torch.count_nonzero(r['competition']) == 0 for r in traces)
        assert torch.isfinite(out[0]).all()
        tests[arm] = {'unit_metric': not metric, 'zero_competition': not bias, 'finite_delta': True}
        del net,out,traces
        gc.collect()
        torch.cuda.empty_cache()
    e.task.write(ROOT/'smoke.json', tests)
    print('SMOKE_OK', tests, flush=True)


def fit(arm, seed, protocol, rows, groups, a, c):
    folder = ROOT/f'{arm}_s{seed}'
    if (folder/'complete.json').exists():
        return
    folder.mkdir(parents=True,exist_ok=True)
    family = ARMS[arm][0]
    parent = SOURCE/f'full_s{seed}_lr1'
    start = 'reference' if family == 'm1' else 'intervention'
    net = network(seed, parent/start/'final.pt', protocol)
    install(net, arm)
    phases = [v for v in protocol['phases'] if v['phase'] >= (2 if family == 'm1' else 1)]
    if family == 'm1':
        e.freeze_delta(net, a, rows, groups['references'])
    for cfg in phases:
        dest = folder/cfg['name']
        e.task.write(ROOT/'status.json', dict(arm=arm, seed=seed, stage=cfg['name'], complete=False))
        if (dest/'receipt.json').exists():
            net.load_state_dict(torch.load(dest/'final.pt', map_location='cpu',weights_only=True))
            e.configure(net,cfg['phase'])
        else:
            e.train_phase(net,cfg,seed,folder,rows,groups,a,c,scale=1.)
        # Source sampling orders must match the already frozen full arm.
        got = json.loads((dest/'orders.json').read_text())
        expected = json.loads((parent/cfg['name']/'orders.json').read_text())
        assert got == expected, (arm, seed, cfg['name'], 'orders differ')
        if cfg['phase'] == 1:
            e.freeze_delta(net,a,rows,groups['references'])
    e.configure(net,3)
    net.requires_grad_(False)
    metrics = {}
    for split in ('validation','evaluation'):
        ix = groups[split]
        score = e.predict(net,a,c,rows,ix,groups['references'])
        np.savez_compressed(folder/f'{split}.npz',ids=ix,score=score)
        metrics[split] = e.task.metrics(np.array([rows[i]['label'] for i in ix]),score)
    e.task.write(folder/'complete.json',dict(arm=arm,seed=seed,parent=str(parent/start/'final.pt'),
        phases=phases,metrics=metrics,orders_identical=True,formal_model_changed=False))
    print('COMPLETE',arm,seed,metrics,flush=True)
    del net
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',nargs='+',type=int,default=[17,29,43])
    parser.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    parser.add_argument('--smoke-only',action='store_true')
    args=parser.parse_args()
    protocol=json.loads((SOURCE/'protocol.json').read_text(encoding='utf-8'))
    e.task.current.setup(e.task.DEVICE)
    rows,groups=e.prev.records()
    a,c=e.prev.arrays()
    a['write']=np.load(e.PREVIOUS/'common_init/write.npy',mmap_mode='r')
    e.task.write(ROOT/'protocol.json',dict(parent_protocol=protocol,arms=ARMS,seeds=[17,29,43],
        data={k:len(v) for k,v in groups.items()},scope='paired suffix retraining from pre-treatment stage checkpoints',
        m1='only feedback paths ablated; reader, auxiliary supervision and M2 interface retained',
        m3='unit channel weights and/or zero competitive bias; aggregate value path unchanged',
        selection='fixed inherited scale 1 and step counts; no new tuning',full_reused=True))
    e.resident(a,c)
    smoke(protocol,rows,groups,a,c)
    if args.smoke_only:return
    for seed in args.seeds:
        for arm in args.arms:
            fit(arm,seed,protocol,rows,groups,a,c)
    e.task.write(ROOT/'status.json',dict(arms=args.arms,seeds=args.seeds,complete=True))


if __name__=='__main__':main()
