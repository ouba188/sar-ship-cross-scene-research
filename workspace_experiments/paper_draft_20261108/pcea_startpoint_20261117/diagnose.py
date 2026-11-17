"""Replay selected and final validation checkpoints; isolate the live PCEA branch."""
from pathlib import Path
import gc
import importlib.util
import sys

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('startpoint_diagnosis_run',ROOT/'run.py')
run=importlib.util.module_from_spec(spec);sys.modules[spec.name]=run;spec.loader.exec_module(run)
model,inputs=run.runtime();torch,np,full=model.torch,model.np,model.full
ids=inputs.groups['validation'];y=inputs.y[ids]
rows=[]
for strength in run.STRENGTHS:
    for arm in run.ARMS:
        dest=run.destination(arm,17,strength)
        receipt=run.read(dest/'receipt.json')
        net=model.Model(17,arm);net.accelerate();net.requires_grad_(False).eval()
        state=torch.load(dest/'best.pt',map_location='cpu',weights_only=True)
        net.load_state_dict(state,strict=True)
        best=full.predict(net,inputs,ids,'validation')
        replay=run.h.replay(model,ids,best,dest/'validation.npz')
        net.load_state_dict(torch.load(dest/'last.pt',map_location='cpu',weights_only=True),strict=True)
        last=full.predict(net,inputs,ids,'validation')
        np.savez_compressed(dest/'lastvalidation.npz',ids=ids,label=y,score=last)
        assert abs(full.metrics(y,last)['ba']-receipt['last']['ba'])<1e-6
        record=dict(arm=arm,strength=strength,best_replay=replay,last=full.metrics(y,last))
        if arm=='cross_mean':
            net.exchange.disabled=True
            off=full.predict(net,inputs,ids,'validation')
            np.savez_compressed(dest/'lastvalidation_off.npz',ids=ids,label=y,score=off)
            good=last.argmax(1)==y;old=off.argmax(1)==y
            record['last_pcea_on_minus_off']=dict(rescued=int((good&~old).sum()),
                harmed=int((old&~good).sum()),changed=int((last.argmax(1)!=off.argmax(1)).sum()),
                max_logit_difference=float(abs(last-off).max()))
        rows.append(record);print(record,flush=True)
        del net,state;gc.collect();torch.cuda.empty_cache()
run.write(ROOT/'diagnostics.json',dict(rows=rows,split='validation only; no evaluation loaded'))
