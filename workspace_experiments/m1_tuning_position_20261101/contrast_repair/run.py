"""Separate common context from high/low deviations, with a matched mean control."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import tune as t
from tune import torch, np, e, task

ROOT=Path(__file__).resolve().parent


class ContrastNetwork(e.Network):
    def __init__(self,position,design,cfg):
        super().__init__(position+'_'+design,t.readers())
        for op in self.m1:
            op.gamma.data.fill_(cfg['gamma'])
            op.temperature=cfg['temperature']
            op.contrast_output=torch.nn.Conv2d(op.value.out_channels,op.channels,1,bias=False).to(op.value.weight.device)
            torch.nn.init.zeros_(op.contrast_output.weight)

    def apply_m1(self,x,occupancy,available,mode=None):
        outputs,aux,ratios=[],[],[]
        for p,op in enumerate(self.m1):
            _,d=op(x[:,p],occupancy[:,p],None,True)
            aux.append(d['aux_logits'])
            mean=d['tokens'][:,0]
            common=torch.stack([layer(mean) for layer in op.transforms],1).mean((1,2))
            common=common[...,None,None]*occupancy[:,p]
            if self.design=='mean_gate':
                differential=common
            else:
                difference=torch.stack([op.transforms[j](d['tokens'][:,j]-mean) for j in (1,2)],1)
                differential=torch.einsum('bskhw,bskd->bdhw',d['write_weights'][:,1:],difference)
                if mode=='uniform_context':
                    differential=differential*0
            raw=op.output(common)+op.contrast_output(differential)
            gate=raw.tanh()
            if mode=='spatial_mean':
                occ=occupancy[:,p]
                gate=(gate*occ).sum((2,3),keepdim=True)/occ.sum((2,3),keepdim=True).clamp_min(1e-8)
                gate=gate*occ
            update=op.gamma*x[:,p]*gate
            if mode=='off' or (mode=='vv_only' and p==1) or (mode=='vh_only' and p==0):
                update=update*0
            outputs.append(x[:,p]+update)
            ratios.append(update.flatten(1).norm(dim=1)/x[:,p].flatten(1).norm(dim=1).clamp_min(1e-8))
        return torch.stack(outputs,1),torch.stack(aux,1),torch.stack(ratios,1)


def main():
    task.current.setup(task.DEVICE)
    t.ROOT=ROOT
    t.network=ContrastNetwork
    rows,train,hold,rid=task.records()
    a,c=task.arrays(),e.caches()
    ids=np.r_[train[:2],hold[:2]]
    baseline=e.predict(e.Network('none'),a,c,ids)['score']
    errors={}
    for position in ('pre','post'):
        for design in ('class_gate','mean_gate'):
            net=ContrastNetwork(position,design,t.CONFIGS['task_lr'])
            errors[position+'_'+design]=float(abs(e.predict(net,a,c,ids)['score']-baseline).max())
            assert errors[position+'_'+design]<1e-5
            # A spatially constant feature must have no high/low deviation.
            op=net.m1[0]
            constant=torch.randn(2,256,1,1,device=task.DEVICE).expand(-1,-1,10,10)
            with torch.no_grad():
                _,d=op(constant,torch.ones(2,1,10,10,device=task.DEVICE),None,True)
                diff=float(abs(d['tokens'][:,1:]-d['tokens'][:,:1]).max())
                assert diff<1e-4,diff
            errors[position+'_'+design+'_constant_deviation']=diff
            del net
    task.write(ROOT/'checks.json',errors)
    for position in ('pre','post'):
        for design in ('class_gate','mean_gate'):
            t.fit(position,design,'task_lr',17,6,rows,train,hold,a,c)
            t.summarize()
    import importlib.util
    spec=importlib.util.spec_from_file_location('m1_tuning_analysis',ROOT.parent/'analyze.py')
    analyze=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyze)
    analyze.ROOT=ROOT
    analyze.main()


if __name__=='__main__':main()
