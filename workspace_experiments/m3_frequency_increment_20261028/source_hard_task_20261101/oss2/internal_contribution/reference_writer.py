"""Class-preserving reference-feature bottleneck with source-only scale matching."""
import json
import sys
import numpy as np
import torch
from torch import nn
from ablate import HERE, AblationModel, IsolatedMatch, training
from model_task import records,arrays,tensor,legal,DEVICE,current


class ReferenceWriter(IsolatedMatch):
    def __init__(self,seed=17,classes=4):
        super().__init__(seed,classes,'full')
        torch.manual_seed(seed+10000)
        self.reference_projection=nn.ModuleList(nn.Linear(classes*32,classes,bias=False) for _ in range(2))
        self.register_buffer('message_gain',torch.ones(2))

    def write_message(self,centered,proto,active,pol):
        z=self.reference_projection[pol](proto.permute(0,2,1,3).flatten(2))
        live=active.transpose(1,2)
        z=z*live
        z=(z-z.sum(-1,keepdim=True)/live.sum(-1,keepdim=True).clamp_min(1))*live
        return z*self.message_gain[pol]


@torch.no_grad()
def calibrate(seed):
    current.setup(DEVICE)
    rows,train,_,rid=records();a=arrays()
    ids=np.concatenate([np.array([i for i in train if rows[i]['label']==c][:16]) for c in range(4)])
    op=ReferenceWriter(seed).to(DEVICE)
    energy=np.zeros((2,2),np.float64)
    for start in range(0,len(ids),16):
        ix=ids[start:start+16]
        inputs=[tensor(a[k],j) for k,j in [('desc',ix),('desc',rid),('mass',ix),('mass',rid),('write',ix)]]
        pair=legal(rows,ix,rid)[...,None]&tensor(a['available'],ix)[:,None].bool()&tensor(a['available'],rid)[None].bool()
        labels=torch.tensor([rows[i]['label'] for i in rid],device=DEVICE)
        _,_,_,traces=op(*inputs,pair,labels,arm='aggregate',details=True)
        for p in range(2):
            energy[p,0]+=sum(float(r['evidence'].square().sum()) for r in traces[p]['regions'])
            energy[p,1]+=sum(float(r['message'].square().sum()) for r in traces[p]['regions'])
    assert (energy>0).all()
    gain=np.sqrt(energy[:,0]/energy[:,1])
    result=dict(seed=seed,source_ids=ids.tolist(),gains=gain.tolist(),energy=energy.tolist(),
                basis='Fixed first16 source records per class; initial descriptor messages; no development scores or labels')
    (HERE/f'writer_calibration_{seed}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('CALIBRATION',seed,gain,flush=True)


class WriterModel(AblationModel):
    def __init__(self,arm,seed=17):
        super().__init__(arm,seed,'full')
        op=ReferenceWriter(seed).to(DEVICE)
        old=self.matcher.state_dict();fresh=op.state_dict()
        fresh.update(old);op.load_state_dict(fresh)
        cfg=json.loads((HERE/f'writer_calibration_{seed}.json').read_text())
        op.message_gain.copy_(torch.tensor(cfg['gains'],device=DEVICE))
        self.matcher=op


if __name__=='__main__':
    if sys.argv[1]=='calibrate':
        for seed in (17,29,43):calibrate(seed)
    else:
        training.Model=WriterModel
        training.main()
