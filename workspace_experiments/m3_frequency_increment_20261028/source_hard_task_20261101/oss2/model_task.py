"""Four-class OSS2 source task. No evaluation manifest or labels are loaded."""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
EXP=ROOT.parents[1]
PUBLIC=Path('E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/final_oss2_replication_20261029')
sys.path.insert(0,'E:/临时会话/sar_m123_standalone')
sys.path.insert(0,str(EXP/'local_scale_repair_20261101'))
sys.path.insert(0,str(EXP/'resolution_matching_20261031'))
from sar_m123.current import data, model as current
from sar_m123.selective_matching import SelectiveReferenceMatch
from sar_m123.pool_replacement import regional_mean
from scaled_operator import MaskedCostLocalMatch

spec=importlib.util.spec_from_file_location('oss2_api',PUBLIC.parent/'final_module_ablations_20261029/model.py')
api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
DEVICE='cuda'
CLASSES=('bulk_carrier','general_cargo','chemical_tanker','container')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def records():
    fit=data.read_manifest(PUBLIC/'manifests/source_fit.json')
    adapt=data.read_manifest(PUBLIC/'manifests/source_adapt.json')
    val=data.read_manifest(PUBLIC/'manifests/validation.json')
    refs=data.read_manifest(PUBLIC/'manifests/references.json')
    rows=fit+adapt+val; lookup={r['id']:i for i,r in enumerate(rows)}
    train=np.arange(len(fit)+len(adapt)); hold=np.arange(len(train),len(rows))
    rid=np.array([lookup[r['id']] for r in refs])
    assert set(rid)<=set(train)
    for key in ('id','mmsi'):
        assert not {rows[i][key] for i in train}&{rows[i][key] for i in hold}
    assert not {orbit(rows[i]) for i in train}&{orbit(rows[i]) for i in hold}
    return rows,train,hold,rid


def orbit(r):return '_'.join(r['pass_group'].split('_')[:2])


def legal(rows,ids,rid):
    return torch.tensor([[rows[i]['mmsi']!=rows[j]['mmsi'] and orbit(rows[i])!=orbit(rows[j]) for j in rid]
        for i in ids],device=DEVICE,dtype=torch.bool)


def tensor(a,ids):return torch.tensor(np.array(a[ids],copy=True),device=DEVICE,dtype=torch.float32)


def arrays():return {p.stem:np.load(p,mmap_mode='r') for p in (ROOT/'cache').glob('*.npy')}


def base_model():
    net=api.load(PUBLIC/'training/full_s17/adapt/model.pt',DEVICE)
    for ops in net.m3:
        for op in ops:op.frequency=False
    return net


class CachedOriginal(SelectiveReferenceMatch):
    def descriptors(self,anchor,local,*unused):
        deep,mass=anchor
        return [deep,local,local,local,local],mass


class Model(nn.Module):
    def __init__(self,arm,seed=17):
        super().__init__();self.arm=arm;self.base=base_model()
        if arm=='original':
            ops=[]
            for old in self.base.m3[0]:
                op=CachedOriginal(frequency=False,dynamic=old.dynamic,temperature=old.temperature).to(DEVICE)
                op.load_state_dict(old.state_dict());op.requires_grad_(True)
                op.reader2.requires_grad_(False);op.reader3.requires_grad_(False);ops.append(op)
            self.matcher=nn.ModuleList(ops)
        elif arm=='none':self.matcher=nn.ModuleList()
        else:self.matcher=MaskedCostLocalMatch(seed=seed,classes=4).to(DEVICE)
        self.base.m3=nn.ModuleList();self.base.head.requires_grad_(True);self.eval()

    def forward(self,a,rows,ids,rid,override=None):
        x=tensor(a['pre_m3'],ids);v=tensor(a['valid'],ids);av=tensor(a['available'],ids)
        slots=tensor(a['slots'],ids)
        if self.arm=='none' or override=='off':updated=x
        elif self.arm=='original':
            refs=tensor(a['pre_m3'],rid);good=legal(rows,ids,rid)
            rav=tensor(a['available'],rid).bool();out=[]
            for p,op in enumerate(self.matcher):
                ref=dict(features=refs[:,p],anchor=(tensor(a['key3'],rid)[:,p],tensor(a['original_mass'],rid)[:,p]),
                    local=tensor(a['key2'],rid)[:,p],slots=tensor(a['slots'],rid)[:,p],valid=tensor(a['valid'],rid)[:,p],
                    local_slots=None,energy=None,local_energy=None)
                z,_=op(x[:,p],(tensor(a['key3'],ids)[:,p],tensor(a['original_mass'],ids)[:,p]),
                    tensor(a['key2'],ids)[:,p],slots[:,p],None,v[:,p],None,None,ref,
                    good&av[:,p,None].bool()&rav[None,:,p])
                out.append(z)
            updated=torch.stack(out,1)
        else:
            inputs=[tensor(a[k],ix) for k,ix in [('desc',ids),('desc',rid),('mass',ids),('mass',rid),('write',ids)]]
            pair=legal(rows,ids,rid)[...,None]&av[:,None].bool()&tensor(a['available'],rid)[None].bool()
            delta,_,_,_=self.matcher(*inputs,pair,torch.tensor([rows[i]['label'] for i in rid],device=DEVICE),
                arm=override or self.arm)
            updated=x+delta
        deep=self.base.backbone.model.layer4(updated.flatten(0,1))
        glob=regional_mean(deep,v.flatten(0,1),v.flatten(0,1))[0][:,0].reshape(len(ids),2,-1)
        pooled=[]
        low=tensor(a['low'],ids)
        for p in range(2):
            reg,en=self.base.region_integrals(slots[:,p],slots.new_zeros(len(ids),20,3,10,10))
            ext=self.base.deep_pool[p](updated[:,p],reg,v[:,p],en)
            pooled.append((low[:,p]+self.base.deep_output[p](ext['pooled'])).flatten(1))
        features=torch.cat((glob,torch.stack(pooled,1)),-1)*av[...,None]
        score=self.base.head(torch.cat((features.flatten(1),av),-1))
        ratio=(updated-x).flatten(2).norm(dim=-1)/x.flatten(2).norm(dim=-1).clamp_min(1e-8)
        return score,ratio

    def live_state(self):return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if p.requires_grad}

    def restore(self,state):
        params=dict(self.named_parameters());assert set(state)=={n for n,p in params.items() if p.requires_grad}
        with torch.no_grad():
            for n,v in state.items():params[n].copy_(v)


@torch.no_grad()
def predict(model,a,rows,ids,rid,override=None):
    z=[];r=[]
    for first in range(0,len(ids),24):
        score,ratio=model(a,rows,ids[first:first+24],rid,override);z.append(score.cpu().numpy());r.append(ratio.cpu().numpy())
    return dict(ids=np.array(ids),score=np.concatenate(z),update_ratio=np.concatenate(r))


def metrics(y,score):
    correct=score.argmax(1)==y;prob=torch.tensor(score,dtype=torch.float64).softmax(-1).numpy()
    loss=-np.log(prob[np.arange(len(y)),y].clip(1e-20))
    return dict(oa=float(correct.mean()*100),ba=float(np.mean([correct[y==c].mean() for c in np.unique(y)])*100),
        nll=float(loss.mean()),balanced_nll=float(np.mean([loss[y==c].mean() for c in np.unique(y)])))
