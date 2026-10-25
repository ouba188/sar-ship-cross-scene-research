"""Matched source-only correction: use the actual strongest post-action rival."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import importlib.util
import json
import time
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
spec=importlib.util.spec_from_file_location('dynamic_parent_training',PARENT/'train.py')
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
run.HERE=HERE


def margin(score,y):
    return score[np.arange(len(y)),y]-np.where(np.arange(8)[None]==y[:,None],-np.inf,score).max(1)


def main(training=True):
    tick=time.perf_counter();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for name in ('models','results'):(HERE/name).mkdir(exist_ok=True)
    data=run.Data();receipts=[];labels={};utility=[]
    for seed in run.SEEDS:
        reader=run.Reader(data,seed)
        with np.load(PARENT/'results'/f'source_interventions_{seed}.npz') as z:
            ids=z['ids'];scores=z['score'];fixed=z['effect']
        y=data.table.iloc[ids].class_id.to_numpy(int)
        baseline=reader.baseline[ids]
        target=np.stack([margin(scores[:,t],y)-margin(baseline,y) for t in range(18)],1)
        full=np.zeros((data.n,18),np.float32);full[ids]=target
        labels[seed]=(ids,target,fixed)
        np.savez_compressed(HERE/'results'/f'source_dynamic_effect_{seed}.npz',ids=ids,effect=target,
                            fixed_rival_effect=fixed)
        if training:
            receipts.append(run.fit(reader,'dynamic_effect',seed,full))
            pd.DataFrame(receipts).to_csv(HERE/'training_receipts.csv',index=False)
        del reader;torch.cuda.empty_cache()
    # No development-guided checkpoint choice; only source J is used above.
    rows=[]
    for seed in run.SEEDS:
        reader=run.Reader(data,seed);model,receipt=run.restore(reader,'dynamic_effect',seed)
        score,local,gate=reader.predict(np.arange(data.n),model,batch=64,return_local=True)
        np.savez_compressed(HERE/'results'/f'dynamic_effect_{seed}.npz',score=score,local=local,gate=gate,
                            up_index=data.up[:data.n])
        ids,target,fixed=labels[seed]
        active=(reader.valid[ids,...,0]&(reader.valid[ids,...,1]|reader.valid[ids,...,2])).cpu().numpy().reshape(len(ids),18)
        tr=np.isin(ids,data.tr);va=np.isin(ids,data.va)
        scale=receipt['source_effect_rms'];normalized=np.tanh(target/scale)
        mean=np.divide((normalized[tr]*active[tr]).sum(0),active[tr].sum(0),out=np.zeros(18),where=active[tr].sum(0)>0)
        # Evaluate both fixed and dynamic teachers' final learners on the same dynamic target.
        for arm,path in [('dynamic_effect',HERE),('effect',PARENT),('shuffled',PARENT),('ce_only',PARENT)]:
            final=run.make_model(seed)
            ck=torch.load(path/'models'/f'final_{arm}_{seed}.pt',map_location='cpu',weights_only=False)
            final.load_state_dict(ck['state']);final.eval()
            with torch.no_grad(): _,u=final(reader.x[ids],reader.valid[ids],reader.context[ids])
            u=u.cpu().numpy().reshape(len(ids),18)
            raw_prediction=np.arctanh(np.clip(u,-.999999,.999999))*ck['receipt']['source_effect_rms']
            u=np.tanh(raw_prediction/scale)
            for role,take in [('utility_train',tr),('selection',va)]:
                mask=active[take]
                utility.append(dict(seed=seed,arm=arm,role=role,target='dynamic_tanh_source_rms',
                    mse=float(((u[take]-normalized[take])**2)[mask].mean()),
                    type_mean_mse=float(((mean-normalized[take])**2)[mask].mean()),
                    sign_accuracy=float((np.sign(u[take])==np.sign(target[take]))[mask&(abs(target[take])>1e-6)].mean())))
        for role,take in [('utility_train',data.tr),('selection',data.va),('evaluation',data.te)]:
            yy=data.table.iloc[take].class_id.to_numpy(int)
            b=reader.baseline[take];a=score[take]
            good=a.argmax(1)==yy;old=b.argmax(1)==yy
            rows.append(dict(seed=seed,arm='dynamic_effect',role=role,**run.metrics(yy,a),
                rescued=int((good&~old).sum()),harmed=int((~good&old).sum()),
                max_gate=float(gate[take].max()),mean_gate=float(gate[take].mean()),
                max_score_change=float(abs(a-b).max())))
    pd.DataFrame(rows).to_csv(HERE/'metrics.csv',index=False)
    pd.DataFrame(utility).to_csv(HERE/'source_utility.csv',index=False)
    if training:
        run.write_json(HERE/'complete.json',dict(fits=3,seconds=time.perf_counter()-tick,
            reason='Source selection obs1822 and three harmful seed29 actions had positive fixed-rival teacher effects; retain original experiment.'))
    else:
        receipt=json.loads((HERE/'complete.json').read_text())
        receipt['utility_analysis_replayed_seconds']=time.perf_counter()-tick
        receipt['utility_units']='Predicted tanh utility decoded using its own source RMS and re-encoded with the common dynamic source RMS.'
        run.write_json(HERE/'complete.json',receipt)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print(pd.DataFrame(utility).query("role=='selection'").to_string(index=False),flush=True)


if __name__=='__main__':main('--analyze-only' not in sys.argv)
