"""Frozen-weight query/reference recovery contrasts after fine-tuning selection."""
import argparse
import gc
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import train as t
spec=importlib.util.spec_from_file_location('ports24_recovery',ROOT/'polarization/recovery.py')
recovery=importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
np,torch=t.np,t.torch


class RecoveredInputs:
    def __init__(self,base,restore_references):
        self.base=base
        self.groups,self.y=base.groups,base.y
        self.rows=list(base.rows)
        self.restored={}
        active=list(self.groups['validation'])+list(self.groups['evaluation'])
        if restore_references:active+=list(self.groups['references'])
        for idx in active:
            row,audit=recovery.recover(self.rows[idx])
            if audit['status']=='recovered':
                self.rows[idx]=row
                self.restored[int(idx)]=audit
        self.reference_rows=[self.rows[i] for i in self.groups['references']]

    def data(self,ids,split=None,code=0):
        assert code==0,'Frozen evaluation only'
        data=self.base.data(ids,split,code)
        for j,idx in enumerate(ids):
            if int(idx) not in self.restored:continue
            view=t.ft.cost.prepare([self.rows[idx]])
            for k in data:data[k][j]=view[k][0]
        return data


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',nargs='+',type=int,default=list(t.SEEDS))
    args=parser.parse_args()
    for seed in args.seeds:
        assert (ROOT/f'selected_s{seed}/receipt.json').exists()
    base=t.Inputs();base.prepare()
    counts={}
    for refs in (False,True):
        inputs=RecoveredInputs(base,refs)
        suffix='query_reference' if refs else 'query'
        counts[suffix]={s:sum(int(i) in inputs.restored for i in inputs.groups[s])
                        for s in ('validation','evaluation','references')}
        for model in ('old','tuned'):
            for seed in args.seeds:
                dest=ROOT/f'{model}_{suffix}_s{seed}';dest.mkdir(exist_ok=True)
                if (dest/'complete.json').exists():continue
                net=t.LiveFull(seed)
                if model=='tuned':
                    net.load_state_dict(torch.load(ROOT/f'selected_s{seed}/best.pt',map_location='cpu',weights_only=True))
                for split in ('validation','evaluation'):
                    ids=inputs.groups[split]
                    score=t.ft.predict(net,inputs,ids,split)
                    np.savez_compressed(dest/(split+'.npz'),ids=ids,score=score,label=inputs.y[ids])
                    m=t.metrics(inputs.y[ids],score);t.write(dest/(split+'.json'),m)
                    if not refs:
                        path=(ROOT/f'selected_s{seed}'/(split+'.npz') if model=='tuned'
                              else t.STORE/f'full_s{seed}_lr1'/('readout/validation.npz' if split=='validation' else 'evaluation.npz'))
                        saved=np.load(path);ix={int(i):j for j,i in enumerate(saved['ids'])}
                        expected=saved['score'][[ix[int(i)] for i in ids]]
                        unchanged=np.array([int(i) not in inputs.restored for i in ids])
                        error=float(np.max(np.abs(score[unchanged]-expected[unchanged])))
                        assert error<1e-3 and np.array_equal(score[unchanged].argmax(1),expected[unchanged].argmax(1)),error
                        t.write(dest/(split+'_unchanged.json'),dict(n=int(unchanged.sum()),max_logit_error=error))
                    print('RECOVERY',model,suffix,seed,split,m['oa'],m['ba'],flush=True)
                t.write(dest/'complete.json',dict(training=False,weight_source=model,reference_recovery=refs,
                                                seed=seed,counts=counts[suffix]))
                del net;gc.collect();torch.cuda.empty_cache()
    t.write(ROOT/'recovery_counts.json',counts)


if __name__=='__main__':main()
