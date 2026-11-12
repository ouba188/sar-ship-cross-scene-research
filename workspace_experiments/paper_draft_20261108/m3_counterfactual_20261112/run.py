"""Fixed-model computational counterfactual replay. CPU only; no fitting."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import csv
import gc
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent
WORK = PAPER.parent
spec = importlib.util.spec_from_file_location('cf_live_recipe', PAPER/'full_finetune_20261111/train.py')
t = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = t
spec.loader.exec_module(t)
# Device changes are confined to this diagnostic process.
for module in list(sys.modules.values()):
    if getattr(module, 'DEVICE', None) == 'cuda':
        module.DEVICE = 'cpu'
torch, np = t.torch, t.np
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
ARMS = ('factual', 'no_m3', 'no_vv', 'no_vh', 'no_body', 'no_side',
        'no_ends', 'unit_metric', 'no_competition', 'simple_mean')
REGIONS = t.cost.raw.tokens.REGIONS


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


class Inputs:
    def __init__(self):
        self.rows = read(WORK/'m1_coverage_20261101/manifest.json')
        self.groups = {k: np.asarray(v) for k,v in read(WORK/'m1_coverage_20261101/groups.json').items()}
        root = PAPER/'recent_baselines_20261111/native_cache'
        self.arrays = {k: np.load(root/(k+'.npy'), mmap_mode='r') for k in ('pixels','footprint','available')}
        self.geo, self.lookup = {}, {}
        for split in ('small','evaluation'):
            folder = PAPER/'full_finetune_20261111/geometry_cache'/split
            receipt = read(folder/'complete.json')
            assert np.array_equal(receipt['ids'], self.groups[split])
            self.lookup[split] = {int(i):j for j,i in enumerate(receipt['ids'])}
            self.geo[split] = {k:np.load(folder/(k+'.npy'), mmap_mode='r')
                               for k in ('slots','local_slots','read','write','mass')}

    def data(self, ids, split):
        ix = [self.lookup[split][int(i)] for i in ids]
        out = {k:torch.from_numpy(np.array(v[ix,0],copy=True)).float() for k,v in self.geo[split].items()}
        out.update(images=torch.from_numpy(np.array(self.arrays['pixels'][ids],copy=True)).float().unsqueeze(2)/255,
                   valid=torch.from_numpy(np.array(self.arrays['footprint'][ids],copy=True)).float().unsqueeze(2),
                   available=torch.from_numpy(np.array(self.arrays['available'][ids],copy=True)).float())
        return out


def simple_mean(matcher, q, r, qm, rm, writer, legal, labels):
    """Class-balanced reference means; same learned encoders/output, no learned metric or beta."""
    onehot = torch.nn.functional.one_hot(labels, matcher.classes).to(q)
    updates = []
    for p in range(2):
        _, vq = matcher.encode(q[:,p],p)
        _, vr = matcher.encode(r[:,p],p)
        fields = []
        for region in REGIONS:
            ix = region.tolist()
            qa, ra = qm[:,p,ix], rm[:,p,ix]
            live = legal[:,:,p] & (ra.sum(-1)>0)[None] & (qa.sum(-1)>0)[:,None]
            ra = ra/ra.sum(-1,keepdim=True).clamp_min(1e-20)
            ref = torch.einsum('rj,rjd->rd',ra,vr[:,ix])
            weights = live[:,:,None]*onehot[None]
            count = weights.sum(1)
            proto = torch.einsum('brc,rd->bcd',weights,ref)/count[...,None].clamp_min(1)
            active = (count>0)[...,None] & (qa>0)[:,None]
            evidence = -((vq[:,None,ix]-proto[:,:,None]).square().sum(-1))/.1*active
            centered = (evidence-evidence.sum(1,keepdim=True)/active.sum(1,keepdim=True).clamp_min(1))*active
            fields.append(centered.transpose(1,2))
        field = torch.cat(fields,1)
        mixed = torch.einsum('bshw,bsc->bchw',writer[:,p],field)
        updates.append(.1*matcher.output[p](mixed))
    return torch.stack(updates,1)


def region_components(matcher, traces, writer):
    parts = []
    for region in REGIONS:
        perpol = []
        for p in range(2):
            field = torch.zeros_like(traces[p]['message'])
            field[:,region] = traces[p]['message'][:,region]
            perpol.append(.1*matcher.output[p](torch.einsum('bshw,bsc->bchw',writer[:,p],field)))
        parts.append(torch.stack(perpol,1))
    return parts


class Replay:
    def __init__(self, seed):
        self.net = t.LiveFull(seed,'full').eval().requires_grad_(False)
        path = PAPER/f'full_finetune_20261111/selected_s{seed}'
        assert (path/'receipt.json').exists()
        self.net.load_state_dict(torch.load(path/'best.pt',map_location='cpu',weights_only=True),strict=True)
        self.seen = {}
        self.net.net.matcher.register_forward_hook(lambda m,a,o:self.seen.update(match_args=a,delta=o[0]))
        self.net.net.base.backbone.model.layer4.register_forward_pre_hook(lambda m,a:self.seen.update(x=a[0]))
        self.net.net.guide.register_forward_pre_hook(lambda m,a:self.seen.update(guide=a[0]))
        for p in range(2):
            self.net.net.base.m2[p].register_forward_hook(self.low_hook(p))

    def low_hook(self,p):
        def hook(m,a,o): self.seen['low',p]=o['pooled']
        return hook

    def downstream(self, x, data, context):
        net, base = self.net.net, self.net.net.base
        deep = base.backbone.model.layer4(x.flatten(0,1))
        valid, av = data['valid'],data['available']
        glob = t.staged.old.regional_mean(deep,valid.flatten(0,1),valid.flatten(0,1))[0][:,0].reshape(len(x),2,-1)
        pooled=[]
        for p in range(2):
            slots=data['slots'][:,p]
            reg,en=base.region_integrals(slots,slots.new_zeros(len(x),20,3,10,10))
            extra=base.deep_pool[p](x[:,p],reg,valid[:,p],en)['pooled']
            pooled.append((context['low',p]+base.deep_output[p](extra)).flatten(1))
        feat=torch.cat((glob,torch.stack(pooled,1)),-1)*av[...,None]
        b=dict(context['guide'])
        b['features']=torch.cat((feat.flatten(1),av),1)
        return net.guide(b)

    @torch.inference_mode()
    def batch(self, data, rows, ref, refrows, check=False):
        self.seen.clear()
        z,_=self.net(data,rows,ref,refrows)
        context=dict(self.seen)
        args=context['match_args']
        matcher=self.net.net.matcher
        delta,_,_,traces=matcher(*args,arm='aggregate',details=True)
        assert torch.allclose(delta,context['delta'],atol=1e-6,rtol=1e-6)
        xfact=context['x'].reshape(len(rows),2,1024,10,10)
        xbase=xfact-context['delta']
        parts=region_components(matcher,traces,args[4])
        reconstruction=float((sum(parts)-delta).abs().max())
        assert reconstruction<1e-4, reconstruction
        changed={'no_m3':torch.zeros_like(delta)}
        for p,name in enumerate(('no_vv','no_vh')):
            changed[name]=delta.clone();changed[name][:,p]=0
        for j,name in enumerate(('no_body','no_side','no_ends')):
            changed[name]=delta-parts[j]
        handle=matcher.metric.register_forward_hook(lambda m,a,o:torch.zeros_like(o))
        try: changed['unit_metric']=matcher(*args,arm='aggregate')[0]
        finally: handle.remove()
        original=matcher.cost_competition
        matcher.cost_competition=types.MethodType(lambda self,c,arm:torch.zeros_like(c),matcher)
        try: changed['no_competition']=matcher(*args,arm='aggregate')[0]
        finally: matcher.cost_competition=original
        changed['simple_mean']=simple_mean(matcher,*args)
        scores={'factual':z.cpu().numpy()}
        for arm,d in changed.items():
            scores[arm]=self.downstream(xbase+d,data,context).cpu().numpy()
        checks=dict(delta_region_reconstruction=reconstruction)
        if check:
            same=self.downstream(xfact,data,context)
            # LiveFull.finish retains the reference table even when M3 is bypassed.
            a,c=self.net.extract(data,True)
            off,_=self.net.finish(a,c,rows,ref,refrows,m3_on=False)
            checks.update(downstream_identity=float((same-z).abs().max()),
                          zero_vs_actual_switch=float(np.max(np.abs(off.numpy()-scores['no_m3']))))
            assert max(checks.values())<1e-4,checks
        # Per-class reference influence weights, not a physical explanation.
        beta=[]
        for p in range(2):
            beta.append(torch.stack([(r['beta']*args[2][:,p,ix][:,None]).sum(-1)/
                args[2][:,p,ix].sum(-1)[:,None].clamp_min(1e-20)
                for r,ix in zip(traces[p]['regions'],REGIONS)],1))
        beta=torch.stack(beta,1).cpu().numpy()
        self.seen.clear()
        return scores,checks,beta


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',nargs='+',type=int,default=[17,29,43])
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--batch',type=int,default=2)
    args=parser.parse_args()
    inputs=Inputs();rid=inputs.groups['references'];ids=inputs.groups['evaluation']
    assert len(ids)==378 and len(rid)==32
    if args.limit: ids=ids[:args.limit]
    folder=ROOT/('smoke' if args.limit else 'private')
    folder.mkdir(parents=True,exist_ok=True)
    for seed in args.seeds:
        dest=folder/f'seed{seed}.npz'
        if dest.exists() and (folder/f'seed{seed}_receipt.json').exists(): continue
        tick=time.perf_counter(); replay=Replay(seed)
        with torch.inference_mode():
            ref=replay.net.reference(inputs.data(rid,'small'))
        refrows=[inputs.rows[int(i)] for i in rid]
        allscores={arm:[] for arm in ARMS};checks=[];betas=[]
        for first in range(0,len(ids),args.batch):
            part=ids[first:first+args.batch]
            scores,check,beta=replay.batch(inputs.data(part,'evaluation'),
                [inputs.rows[int(i)] for i in part],ref,refrows,check=first==0)
            for arm in ARMS: allscores[arm].append(scores[arm])
            checks.append(check);betas.append(beta)
            if first%20==0:
                print('PROGRESS',seed,first+len(part),len(ids),round(time.perf_counter()-tick,1),flush=True)
                write(ROOT/'status.json',dict(state='running',seed=seed,done=first+len(part),n=len(ids),device='cpu'))
        results={arm:np.concatenate(v) for arm,v in allscores.items()}
        previous=np.load(PAPER/f'full_finetune_20261111/selected_s{seed}/evaluation.npz')
        lookup={int(i):j for j,i in enumerate(previous['ids'])}
        expected=previous['score'][[lookup[int(i)] for i in ids]]
        err=float(np.max(np.abs(results['factual']-expected)))
        assert err<1e-3 and np.array_equal(results['factual'].argmax(1),expected.argmax(1)),err
        labels=np.array([inputs.rows[int(i)]['label'] for i in ids])
        np.savez_compressed(dest,ids=ids,label=labels,reference_ids=rid,beta=np.concatenate(betas),**results)
        receipt=dict(seed=seed,n=len(ids),arms=list(ARMS),device='cpu',training=False,
                     seconds=time.perf_counter()-tick,saved_prediction_error=err,
                     checks=checks,formal_model_changed=False,actual_factual_arm='aggregate')
        write(folder/f'seed{seed}_receipt.json',receipt)
        print('DONE',seed,round(receipt['seconds'],1),err,flush=True)
        del replay,ref;gc.collect()
    write(ROOT/'status.json',dict(state='smoke_complete' if args.limit else 'replay_complete',
         seeds=args.seeds,device='cpu',analysis_pending=True,formal_model_changed=False))


if __name__=='__main__': main()
