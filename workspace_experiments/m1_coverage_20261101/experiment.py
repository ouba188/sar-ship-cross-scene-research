"""Fixed-budget M1 coverage experiment; old experiments and formal models unchanged."""
from pathlib import Path
import os
import sys
import json
import argparse
import gc
import time

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'m1_covariance_20261101'
sys.path.insert(0, str(OLD))
import run as old
from run import torch, np, pd, task, e
from torch.nn import functional as F
from geometry import build_masks, pool_multiscale


def write(name, value):
    task.write(ROOT/name, value)


def records():
    parts = {k: task.data.read_manifest(task.PUBLIC/'manifests'/f'{k}.json')
             for k in ('source_fit', 'source_adapt', 'validation', 'selection', 'evaluation', 'references')}
    rows = parts['source_fit']+parts['source_adapt']+parts['validation']+parts['selection']+parts['evaluation']
    lookup = {r['id']: i for i, r in enumerate(rows)}
    groups = {k: np.array([lookup[r['id']] for r in v]) for k, v in parts.items()}
    groups['small'] = np.r_[groups['source_fit'], groups['source_adapt']]
    groups['expanded'] = np.r_[groups['small'], groups['selection']]
    assert len(lookup) == len(rows) == 1519
    assert len(groups['small']) == 719 and len(groups['expanded']) == 913
    for left, right in [('expanded', 'validation'), ('expanded', 'evaluation'), ('validation', 'evaluation')]:
        for key in ('id', 'mmsi', 'orbit'):
            def values(ids):
                return {task.orbit(rows[i]) if key == 'orbit' else rows[i][key] for i in ids}
            assert not values(groups[left]) & values(groups[right]), (left, right, key)
    assert set(groups['references']) <= set(groups['small'])
    return rows, groups


def coverage():
    rows, groups = records()
    summary = []
    for name in ('small', 'selection', 'expanded', 'validation', 'evaluation'):
        for label in (None, 0, 1, 2, 3):
            part = [rows[i] for i in groups[name] if label is None or rows[i]['label'] == label]
            summary.append(dict(cohort=name, label='all' if label is None else label, n=len(part),
                                ships=len({r['mmsi'] for r in part}), passes=len({task.orbit(r) for r in part}),
                                both=sum(r['vv'] is not None and r['vh'] is not None for r in part)))
    ROOT.mkdir(exist_ok=True)
    pd.DataFrame(summary).to_csv(ROOT/'coverage.csv', index=False)
    write('manifest.json', rows)
    write('groups.json', {k: v.tolist() for k, v in groups.items()})
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    return rows, groups


def arrays():
    a = {p.stem: np.load(p, mmap_mode='r') for p in (ROOT/'cache/a').glob('*.npy')}
    c = {p.stem: np.load(p, mmap_mode='r') for p in (ROOT/'cache/c').glob('*.npy')}
    return a, c


def gpu_cache(a, c, groups):
    """Keep the exact float32 frozen inputs resident, avoiding random disk paging."""
    original_tensor = task.tensor
    def resident_tensor(value, ids):
        if isinstance(value, torch.Tensor):
            index = torch.as_tensor(ids, device=value.device, dtype=torch.long)
            return value.index_select(0, index)
        return original_tensor(value, ids)
    task.tensor = resident_tensor
    net = Network('class')
    probe = np.r_[groups['small'][:2], groups['selection'][:2], groups['evaluation'][:2]]
    expected = e.predict(net, a, c, probe)['score']
    for table, keys in ((a, ('valid','available','slots')),
                        (c, ('early_x','early_identity','local_slots','delta'))):
        for key in keys:
            array = table[key]
            cached = torch.empty(array.shape, dtype=torch.float32, device=task.DEVICE)
            for start in range(0, len(array), 16):
                cached[start:start+16].copy_(torch.from_numpy(np.array(array[start:start+16], dtype=np.float32, copy=True)))
            table[key] = cached
            print('GPU_CACHE',key,round(torch.cuda.memory_allocated()/1024**3,3),flush=True)
    actual = e.predict(net, a, c, probe)['score']
    error = float(abs(actual-expected).max())
    assert error == 0, error
    write('gpu_cache.json',dict(replay_error=error,dtype='float32',n=len(a['valid']),
                              allocated_gib=torch.cuda.memory_allocated()/1024**3))
    del net
    gc.collect();torch.cuda.empty_cache()


@torch.no_grad()
def prepare():
    rows, groups = coverage()
    if (ROOT/'cache/complete.json').exists():
        return
    original_a = task.arrays()
    original_c = e.caches()
    original_c.update({p.stem: np.load(p, mmap_mode='r') for p in (OLD/'cache').glob('*.npy')})
    keys_a = ('valid', 'available', 'slots', 'desc', 'mass')
    keys_c = ('early_x', 'early_identity', 'local_slots', 'delta')
    outputs = {}
    for folder, source, keys in [('a', original_a, keys_a), ('c', original_c, keys_c)]:
        (ROOT/'cache'/folder).mkdir(parents=True, exist_ok=True)
        for key in keys:
            shape = (len(rows),)+source[key].shape[1:]
            out = np.lib.format.open_memmap(ROOT/'cache'/folder/(key+'.npy'), mode='w+', dtype=source[key].dtype, shape=shape)
            for first in range(0, 947, 16):
                out[first:min(first+16, 947)] = source[key][first:min(first+16, 947)]
            outputs[(folder, key)] = out
    carrier = e.previous.anchor()
    base = carrier.base
    bb = base.backbone.model
    block = bb.layer2[3]
    refids = groups['references']
    refdesc = task.tensor(original_a['desc'], refids)
    refmass = task.tensor(original_a['mass'], refids)
    refavailable = task.tensor(original_a['available'], refids).bool()
    reflabels = torch.tensor([rows[i]['label'] for i in refids], device=task.DEVICE)
    public = {key: np.load(task.PUBLIC/'cache'/f'{key}.npy', mmap_mode='r')
              for key in ('prefix', 'valid', 'available', 'slot_masks', 'local_slot_masks')}
    probe = np.array([0, 138, 478, 719])
    jobs = [probe]+[np.arange(i, min(i+8, len(rows))) for i in range(947, len(rows), 8)]
    errors = {}
    tick = time.perf_counter()
    for n, ids in enumerate(jobs):
        ix = [rows[i]['indices']['table_index'] for i in ids]
        b = {k: task.tensor(v, ix) for k, v in public.items()}
        images = np.zeros((len(ids), 2, 1, 160, 160), np.float32)
        geometry = []
        for j, i in enumerate(ids):
            gs = []
            for p, view in enumerate(task.data.prepare(rows[i])):
                present = view is not None and view['valid_target']
                if present:
                    images[j, p, 0] = view['pixels'].astype(np.float32)/255
                gs.append(build_masks(view['geometry'] if present else None,
                                      view['footprint'] if present else np.zeros((160,160)), 'coarse_fixed'))
            geometry.append(gs)
        f1 = base.prefix(torch.tensor(images, device=task.DEVICE), stop=1)
        z = f1.flatten(0, 1)
        for layer in list(bb.layer2.children())[:3]:
            z = layer(z)
        identity = z
        x = block.relu(block.bn1(block.conv1(z)))
        x = block.relu(block.bn2(block.conv2(x)))
        f2 = block.relu(block.bn3(block.conv3(x))+identity).reshape(len(ids), 2, 512, 20, 20)
        errors['prefix'] = max(errors.get('prefix', 0), float((f2-b['prefix']).abs().max()))
        read = torch.tensor(np.stack([[g['read'] for g in gs] for gs in geometry]), device=task.DEVICE)
        desc = pool_multiscale(f1.flatten(0,1), f2.flatten(0,1), read.flatten(0,1), 'coarse_fixed').reshape(len(ids),2,20,768)
        mass = torch.tensor(np.stack([[g['mass'] for g in gs] for gs in geometry]), device=task.DEVICE)
        writer = torch.tensor(np.stack([[g['write'] for g in gs] for gs in geometry]), device=task.DEVICE)
        pair = task.legal(rows, ids, refids)[...,None] & b['available'][:,None].bool() & refavailable[None]
        delta = carrier.matcher(desc, refdesc, mass, refmass, writer, pair, reflabels, arm='aggregate')[0]
        values = {('a','valid'): b['valid']*b['available'][...,None,None,None],
                  ('a','available'): b['available'], ('a','slots'): b['slot_masks'],
                  ('a','desc'): desc, ('a','mass'): mass,
                  ('c','early_x'): x.reshape(len(ids),2,128,20,20),
                  ('c','early_identity'): identity.reshape(len(ids),2,512,20,20),
                  ('c','local_slots'): b['local_slot_masks'], ('c','delta'): delta}
        for (folder, key), value in values.items():
            arr = value.cpu().numpy()
            if n == 0:
                src = original_a if folder == 'a' else original_c
                error = float(abs(arr-np.asarray(src[key][ids])).max())
                errors[key] = error
                assert error < 1e-4, (folder, key, error)
            else:
                outputs[(folder,key)][ids] = arr
        if n % 10 == 0:
            print('CACHE', n, len(jobs), errors, flush=True)
    assert errors['prefix'] < 1e-4
    for out in outputs.values():
        out.flush()
    write('cache/complete.json', dict(n=len(rows), reused=947, new=len(rows)-947,
                                     errors=errors, seconds=time.perf_counter()-tick))


class Network(old.Network):
    def __init__(self, arm):
        super().__init__('early', 'mean' if arm == 'mean' else 'class', .1 if arm == 'mean' else .3)
        self.arm = arm

    def forward(self, a, c, ids, mode=None):
        if self.arm == 'no_cov':
            if mode is not None:
                raise ValueError('The trained no-cov arm is evaluated on its training path.')
            mode = 'covariance_zero'
        return super().forward(a, c, ids, mode)


def sample_orders(train, seed, steps=180):
    rng = np.random.default_rng(seed)
    batches = []
    while len(batches) < steps:
        order = rng.permutation(train)
        batches.extend(order[i:i+24] for i in range(0, len(order), 24))
    return batches[:steps]


def fit(arm, cohort, seed, rows, groups, a, c):
    dest = ROOT/f'{cohort}_{arm}_s{seed}'
    if (dest/'receipt.json').exists():
        return
    dest.mkdir(exist_ok=True)
    ids_train = groups[cohort]
    y = np.array([r['label'] for r in rows])
    torch.manual_seed(seed)
    net = Network(arm)
    params = {n:p for n,p in net.named_parameters() if p.requires_grad}
    param_groups = {}
    for n,p in params.items():
        lr = 1e-5 if n.startswith('base.') else (.001 if '.output.' in n or '.relation_output.' in n else .0003)
        param_groups.setdefault(lr, []).append(p)
    opt = torch.optim.AdamW([dict(params=p, lr=lr) for lr,p in param_groups.items()], weight_decay=.01)
    source_initial = np.load(ROOT/'initial_source.npz')['score']
    fr = pd.DataFrame([rows[i] for i in ids_train])
    weight = np.zeros(len(rows), np.float32)
    weight[ids_train] = 1/(fr.groupby('label').mmsi.transform('nunique').to_numpy()*fr.groupby('mmsi').mmsi.transform('size').to_numpy())
    train_margin = source_initial[ids_train,y[ids_train]]-np.where(np.arange(4)[None]==y[ids_train,None],-np.inf,source_initial[ids_train]).max(1)
    weight[ids_train] *= np.where(train_margin<1, 2., 1.)
    weight[ids_train] /= weight[ids_train].mean()
    weight = torch.tensor(weight, device=task.DEVICE)
    orders = sample_orders(ids_train, seed)
    task.write(dest/'orders.json', [x.tolist() for x in orders])
    curve, gradients = [], []
    tick = time.perf_counter()
    losses, auxloss = [], []
    for step, ids in enumerate(orders, 1):
        labels = torch.tensor(y[ids], device=task.DEVICE)
        opt.zero_grad(set_to_none=True)
        score, aux, ratio = net(a,c,ids)
        ce = (F.cross_entropy(score,labels,reduction='none')*weight[ids]).mean()
        av = task.tensor(a['available'],ids)
        ace = F.cross_entropy(aux.flatten(0,1),labels[:,None].expand(-1,2).flatten(),reduction='none').reshape(len(ids),2)
        ace = ((ace*av).sum(1)/av.sum(1).clamp_min(1)*weight[ids]).mean()
        loss = ce+.02*ace
        assert torch.isfinite(loss), (arm,cohort,seed,step)
        loss.backward()
        if step in (1,2):
            gradients.append(dict(step=step,norms={n:float(p.grad.norm()) for n,p in params.items() if p.grad is not None}))
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params.values())
            assert all(float(net.m1[p].output.weight.grad.norm()) > 0 for p in range(2))
        torch.nn.utils.clip_grad_norm_(list(params.values()),5.)
        opt.step()
        losses.append(float(ce.detach())); auxloss.append(float(ace.detach()))
        if step%30 == 0:
            prediction = e.predict(net,a,c,groups['validation'])
            np.savez_compressed(dest/f'dev_step{step}.npz', **prediction)
            record = dict(step=step,ce=float(np.mean(losses)),auxiliary_ce=float(np.mean(auxloss)),
                          **task.metrics(y[groups['validation']],prediction['score']))
            curve.append(record)
            pd.DataFrame(curve).to_csv(dest/'curve.csv',index=False)
            print('STEP',dest.name,record,flush=True)
            losses, auxloss = [], []
    torch.save(net.live_state(),dest/'final.pt')
    metrics = {}
    for name in ('validation','evaluation'):
        p = e.predict(net,a,c,groups[name])
        np.savez_compressed(dest/(name+'.npz'), **p)
        metrics[name] = task.metrics(y[groups[name]],p['score'])
    probe = groups['validation'][:3]
    before = e.predict(net,a,c,probe)['score']
    net.restore(torch.load(dest/'final.pt',map_location='cpu',weights_only=True))
    reload_error = float(abs(before-e.predict(net,a,c,probe)['score']).max())
    assert reload_error == 0
    task.write(dest/'receipt.json',dict(arm=arm,cohort=cohort,seed=seed,n_train=len(ids_train),steps=180,
        examples=sum(map(len,orders)),unique_seen=len(set(np.concatenate(orders))),position='layer2[3] post-conv2',
        gain=.1 if arm=='mean' else .3,selected_step=180,selection='fixed, no target outcome selection',
        trainable=sum(p.numel() for p in params.values()),gradients=gradients,metrics=metrics,
        seconds=time.perf_counter()-tick,reload_error=reload_error))
    del net,opt
    gc.collect();torch.cuda.empty_cache()
    print('DONE',dest.name,metrics,flush=True)


def checks(rows, groups, a, c):
    if (ROOT/'checks.json').exists():
        return
    errors = {}
    probe = np.r_[groups['small'][:2],groups['selection'][:2],groups['validation'][:2]]
    torch.manual_seed(17)
    net = Network('class')
    initial = e.predict(net,a,c,probe)['score']
    op = net.m1[0]
    torch.nn.init.normal_(op.relation_output.weight,std=.01)
    xx = task.tensor(c['early_x'],probe)[:,0]
    occ = old.grid_occupancy(task.tensor(a['valid'],probe)[:,0], xx.shape[-2:])
    result, details = op(xx,occ,'covariance_zero',True)
    grad = torch.autograd.grad(result[0].sum()+result[1].sum(),op.transforms[0].weight,allow_unused=True)[0]
    errors['no_cov_transform_weight_gradient'] = float(grad.abs().max())
    assert errors['no_cov_transform_weight_gradient'] == 0
    del net
    gc.collect();torch.cuda.empty_cache()
    for arm in ('mean','no_cov'):
        torch.manual_seed(17)
        net = Network(arm)
        errors[arm+'_zero_start'] = float(abs(e.predict(net,a,c,probe)['score']-initial).max())
        assert errors[arm+'_zero_start'] == 0
        del net
        gc.collect();torch.cuda.empty_cache()
    # Recompute shared source difficulty with the same no-M1 carrier for both pools.
    net = Network('class')
    p = e.predict(net,a,c,groups['expanded'])
    scores = np.zeros((len(rows),4),np.float32)
    scores[groups['expanded']] = p['score']
    np.savez_compressed(ROOT/'initial_source.npz', score=scores)
    write('checks.json',errors)
    del net
    gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--seeds',nargs='+',type=int,default=[17,29,43])
    args=parser.parse_args()
    task.current.setup(task.DEVICE)
    if args.prepare:
        prepare()
        return
    rows,groups=records();a,c=arrays()
    assert (ROOT/'cache/complete.json').exists()
    checks(rows,groups,a,c)
    gpu_cache(a,c,groups)
    for seed in args.seeds:
        for cohort in ('small','expanded'):
            for arm in ('mean','class','no_cov'):
                fit(arm,cohort,seed,rows,groups,a,c)


if __name__ == '__main__':
    main()
