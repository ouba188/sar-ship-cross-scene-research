"""Fixed-model regional evidence diagnostics inspired by Li et al. GRSL 2023."""
import argparse
import contextlib
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
ROOT = Path(__file__).resolve().parent
PAPER, WORK = ROOT.parent, ROOT.parent.parent
spec = importlib.util.spec_from_file_location('dependency_live', PAPER/'full_finetune_20261111/train.py')
t = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = t
spec.loader.exec_module(t)
torch, np = t.torch, t.np
REGIONS = ('body', 'ends', 'side')
SEEDS = (17, 29, 43)
PATHS = ('full', 'm1_feedback_off', 'm2_write_off', 'm2_uniform', 'm3_off')
MICRO = 8


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


class Inputs:
    def __init__(self):
        self.rows = read(WORK/'m1_coverage_20261101/manifest.json')
        self.groups = {k: np.asarray(v) for k, v in read(WORK/'m1_coverage_20261101/groups.json').items()}
        self.ids = self.groups['validation']
        self.y = np.asarray([self.rows[i]['label'] for i in self.ids])
        cache = PAPER/'recent_baselines_20261111/native_cache'
        self.arrays = {k: np.load(cache/(k+'.npy'), mmap_mode='r') for k in ('pixels', 'footprint', 'available')}
        self.geometry, self.lookup = {}, {}
        for split in ('small', 'validation'):
            folder = PAPER/'full_finetune_20261111/geometry_cache'/split
            receipt = read(folder/'complete.json')
            assert np.array_equal(receipt['ids'], self.groups[split])
            self.lookup[split] = {int(i): j for j, i in enumerate(receipt['ids'])}
            self.geometry[split] = {k: np.load(folder/(k+'.npy'), mmap_mode='r')
                                   for k in ('slots', 'local_slots', 'read', 'write', 'mass')}
        self.rids = self.groups['references']
        self.rrows = [self.rows[i] for i in self.rids]
        self.masks = None

    def prepare_masks(self):
        path = ROOT/'native_regions.npy'
        if not path.exists():
            masks = np.zeros((len(self.ids), 2, 3, 160, 160), bool)
            meta = []
            for j, idx in enumerate(self.ids):
                views = t.staged.task.data.prepare(self.rows[idx])
                for p, view in enumerate(views):
                    if not self.arrays['available'][idx, p]:
                        continue
                    assert view is not None and view['valid_target']
                    assert np.array_equal(view['pixels'], self.arrays['pixels'][idx, p])
                    # Diagnostic ownership is disjoint; the model's original soft masks are unchanged.
                    soft = t.cost.raw.tokens.soft_regions(view['geometry'])
                    valid = self.arrays['footprint'][idx, p].astype(bool)
                    used = np.zeros((160, 160), bool)
                    for r in range(3):
                        masks[j, p, r] = (soft[r] >= .5) & valid & ~used
                        used |= masks[j, p, r]
                    meta.append(dict(index=int(idx), id=self.rows[idx]['id'], polarization=('VV','VH')[p],
                                     geometry=np.asarray(view['geometry']).tolist(),
                                     area=masks[j,p].sum((-2,-1)).tolist(),
                                     footprint=int(valid.sum())))
            np.save(path, masks)
            write(ROOT/'region_metadata.json', meta)
        self.masks = np.load(path, mmap_mode='r')
        assert self.masks.shape == (len(self.ids), 2, 3, 160, 160)
        assert self.masks.sum(2).max() <= 1

    def data(self, ids, split):
        ix = [self.lookup[split][int(i)] for i in ids]
        out = {k: torch.as_tensor(np.array(v[ix,0],copy=True), device='cuda')
               for k,v in self.geometry[split].items()}
        out.update(images=torch.as_tensor(np.array(self.arrays['pixels'][ids],copy=True),device='cuda').float().unsqueeze(2)/255,
                   valid=torch.as_tensor(np.array(self.arrays['footprint'][ids],copy=True),device='cuda').float().unsqueeze(2),
                   available=torch.as_tensor(np.array(self.arrays['available'][ids],copy=True),device='cuda').float())
        return out


def conditions():
    out = [dict(name='clean', kind='clean')]
    for p, pol in enumerate(('VV', 'VH')):
        for keep in range(7):
            out.append(dict(name=f'{pol}_mean50_keep{keep}', kind='coalition', p=p, keep=keep))
        for gain in (.8, 1.2):
            out.append(dict(name=f'{pol}_outside_gain{gain}', kind='gain', p=p, gain=gain))
        for r, region in enumerate(REGIONS):
            out.append(dict(name=f'{pol}_{region}_permute', kind='permute', p=p, r=r))
    return out


def perturb(images, masks, valid, ids, cond):
    out = images.clone()
    if cond['kind'] == 'clean':
        return out
    p = cond['p']
    m = torch.as_tensor(np.array(masks, copy=True), device=images.device)
    x = images[:,p,0]
    if cond['kind'] == 'coalition':
        for r in range(3):
            if cond['keep'] & (1 << r):
                continue
            region = m[:,p,r]
            mean = (x*region).sum((-2,-1))/region.sum((-2,-1)).clamp_min(1)
            out[:,p,0] += .5*region*(mean[:,None,None]-x)
    elif cond['kind'] == 'gain':
        outside = valid[:,p,0].bool() & ~m[:,p,0]
        out[:,p,0] = torch.where(outside, x*cond['gain'], x)
    elif cond['kind'] == 'permute':
        for j, idx in enumerate(ids):
            ix = torch.where(m[j,p,cond['r']].flatten())[0]
            order = np.random.default_rng(16000+int(idx)*6+p*3+cond['r']).permutation(len(ix))
            out[j,p,0].flatten()[ix] = x[j].flatten()[ix[torch.as_tensor(order,device=x.device)]]
    return out


@contextlib.contextmanager
def path_switch(net, arm):
    handles = []
    if arm == 'm1_feedback_off':
        for branch in net.net.m1:
            for layer in (branch.output, branch.relation_output):
                handles.append(layer.register_forward_hook(lambda m,a,o: torch.zeros_like(o)))
    elif arm == 'm2_write_off':
        for layer in net.net.base.internal_m2:
            handles.append(layer.register_forward_hook(lambda m,a,o: (a[0], o[1])))
    elif arm == 'm2_uniform':
        handles.append(net.net.guide.register_forward_pre_hook(
            lambda m,a,k: (a, dict(k, mode='uniform')), with_kwargs=True))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def model(seed, kind):
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    directory = PAPER/('full_finetune_20261111' if kind == 'full' else 'r50_tuned_20261111')/f'selected_s{seed}'
    net = t.LiveFull(seed,'full') if kind == 'full' else t.r50.PlainR50(seed,'full')
    net.load_state_dict(torch.load(directory/'best.pt',map_location='cpu',weights_only=True),strict=True)
    return net.eval().requires_grad_(False), directory


@torch.no_grad()
def predict(net, inputs, ref, kind, arm, cond, limit=None):
    scores = []
    ids = inputs.ids if limit is None else inputs.ids[:limit]
    manager = path_switch(net,arm) if kind == 'full' else contextlib.nullcontext()
    batch = 24 if kind == 'r50' else MICRO
    with manager:
        for first in range(0,len(ids),batch):
            part = ids[first:first+batch]
            data = inputs.data(part,'validation')
            data['images'] = perturb(data['images'],inputs.masks[first:first+len(part)],data['valid'],part,cond)
            if kind == 'full':
                a,c = net.extract(data,True)
                z,_ = net.finish(a,c,[inputs.rows[i] for i in part],ref,inputs.rrows,m3_on=arm!='m3_off')
            else:
                z = net(data['images'],data['valid'],data['available'])
            scores.append(z.cpu().numpy())
    return np.concatenate(scores)


def protocol(inputs):
    return dict(dataset='OSS2', split='validation; diagnostic, not held-out confirmation',
        n=len(inputs.ids), ships=len({inputs.rows[i]['mmsi'] for i in inputs.ids}),
        passes=len({inputs.rows[i]['pass_group'] for i in inputs.ids}),
        counts=np.bincount(inputs.y,minlength=4).tolist(),
        seeds=SEEDS, references=inputs.rids.tolist(), conditions=conditions(),
        regions='body, axis ends, lateral; existing OBB native masks threshold .5, disjoint ownership; far field unchanged in structure games',
        coalition='50% attenuation toward own original regional mean; NOT region absence or physical causal attribution',
        gain='entire valid OBB exterior times .8 or1.2; no clipping/no renormalization; display gray, NOT calibrated SCR',
        permutation='fixed within-region permutation; exact original histogram and mean; spatial arrangement disrupted, different dose from attenuation',
        switches='M1 feedback only (response retained), M2 internal write only, M2 guided pooling -> uniform, M3 write off; all keep full trained head',
        switch_conditions='clean and six single-region mean50 operations; three seeds',
        comparison='ordinary full-tuned R50 is a system comparator, not matched training-history module ablation',
        attribution='fixed original rival within each model/seed; additionally rerank strongest rival after operation; normalize Shapley by sum(abs) only within sample',
        frozen='weights, query cohort, OBB, availability, native160 crop/pad, legal source reference members and reference descriptors',
        training=False, formal_model_changed=False,
        source='Li et al. GRSL 2023 doi:10.1109/LGRS.2023.3266493, equations1-4/figures1-4; bounded adapted diagnosis, not its random-SCR training replication')


@torch.no_grad()
def smoke(inputs, selected_kind):
    report = {}
    data = inputs.data(inputs.ids[:MICRO], 'validation')
    original = data['images']
    masks = inputs.masks[:MICRO]
    for c in conditions():
        x = perturb(original,masks,data['valid'],inputs.ids[:MICRO],c)
        assert torch.isfinite(x).all()
        if c['kind'] == 'clean':
            assert torch.equal(x,original)
            continue
        p = c['p']
        assert torch.equal(x[:,1-p],original[:,1-p])
        assert torch.equal(x*(1-data['valid']),original*(1-data['valid']))
        if c['kind'] == 'coalition':
            for r in range(3):
                m = torch.as_tensor(np.array(masks[:,p,r],copy=True),device='cuda')
                err = ((x[:,p,0]-original[:,p,0])*m).sum((-2,-1)).abs()/m.sum((-2,-1)).clamp_min(1)
                assert err.max()<1e-6, (c,err)
        elif c['kind'] == 'permute':
            for j in range(len(x)):
                m = torch.as_tensor(np.array(masks[j,p,c['r']],copy=True),device='cuda')
                assert torch.equal(x[j,p,0][m].sort().values,original[j,p,0][m].sort().values)
    for kind in (selected_kind,):
        net,directory=model(17,kind)
        ref=net.reference(inputs.data(inputs.rids,'small')) if kind=='full' else None
        before=time.perf_counter()
        count=24 if kind=='r50' else MICRO
        z=predict(net,inputs,ref,kind,'full',conditions()[0],count)
        saved=np.load(directory/'validation.npz')
        where={int(i):j for j,i in enumerate(saved['ids'])}
        expected=saved['score'][[where[int(i)] for i in inputs.ids[:count]]]
        err=float(abs(z-expected).max())
        same=bool(np.array_equal(z.argmax(1),expected.argmax(1)))
        if kind=='full':
            assert err<1e-3 and same,(kind,err)
        report[kind]=dict(replay_error=err,predictions_identical=same,seconds=time.perf_counter()-before,
                         precision='FP32 no TF32; R50 historical cache used TF32; all effects use newly computed clean scores')
        if kind=='full':
            for arm in PATHS[1:]:
                other=predict(net,inputs,ref,kind,arm,conditions()[0],MICRO)
                delta=float(abs(other-z).max())
                assert delta>0,(arm,'inactive intervention')
                restored=predict(net,inputs,ref,kind,'full',conditions()[0],MICRO)
                assert np.allclose(restored,z,atol=1e-5)
                report[arm]=dict(max_logit_change=delta)
        del net,ref
        gc.collect();torch.cuda.empty_cache()
    report['pixel_controls']='means/histograms/opposite polarization/padding verified'
    write(ROOT/f'smoke_{selected_kind}.json',report)
    print(json.dumps(report),flush=True)


@torch.no_grad()
def run(inputs, selected_kind):
    write(ROOT/'protocol.json',protocol(inputs))
    all_conditions=conditions()
    limited=[c for c in all_conditions if c['kind']=='clean' or (c['kind']=='coalition' and c['keep'] in (3,5,6))]
    start=time.perf_counter()
    for seed in SEEDS:
        for kind in (selected_kind,):
            net,directory=model(seed,kind)
            ref=net.reference(inputs.data(inputs.rids,'small')) if kind=='full' else None
            for arm in (PATHS if kind=='full' else ('r50',)):
                chosen=all_conditions if arm in ('full','r50') else limited
                folder=ROOT/'predictions'/f'{arm}_s{seed}'
                folder.mkdir(parents=True,exist_ok=True)
                for c in chosen:
                    target=folder/(c['name']+'.npz')
                    if target.exists():
                        continue
                    z=predict(net,inputs,ref,kind,arm,c)
                    if c['kind']=='clean' and arm in ('full','r50'):
                        saved=np.load(directory/'validation.npz')
                        assert np.array_equal(saved['ids'],inputs.ids)
                        changed=saved['score'].argmax(1)!=z.argmax(1)
                        error=float(abs(saved['score']-z).max())
                        if kind=='full':
                            assert not changed.any() and error<1e-3
                        write(folder/'replay.json',dict(max_logit_error=error,changed_ids=inputs.ids[changed].tolist(),
                              current_precision='FP32 no TF32',baseline='new clean forward used for every intervention comparison'))
                    np.savez_compressed(target,ids=inputs.ids,y=inputs.y,score=z)
                    write(ROOT/f'status_{selected_kind}.json',dict(state='running',seed=seed,arm=arm,condition=c['name'],seconds=time.perf_counter()-start))
                    print('DONE',seed,arm,c['name'],'seconds',round(time.perf_counter()-start,1),flush=True)
            del net,ref
            gc.collect();torch.cuda.empty_cache()
    write(ROOT/f'status_{selected_kind}.json',dict(state='predictions_complete_analysis_pending',seconds=time.perf_counter()-start))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('smoke','run'))
    parser.add_argument('--kind',choices=('full','r50'),default='full')
    args=parser.parse_args()
    inputs=Inputs();inputs.prepare_masks()
    (smoke if args.action=='smoke' else run)(inputs,args.kind)
