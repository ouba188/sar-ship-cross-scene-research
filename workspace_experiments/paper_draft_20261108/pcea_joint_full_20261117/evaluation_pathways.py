"""Post-result diagnostics only; never select a replacement from evaluation."""
import gc
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('joint_eval_diagnostic',ROOT/'analyze.py')
a=importlib.util.module_from_spec(spec);sys.modules[spec.name]=a;spec.loader.exec_module(a)
model,inputs=a.run.runtime()
torch,np,full=model.torch,model.np,model.full
inputs.load_split('evaluation');inputs.prepare_geometry('evaluation')
ids=inputs.groups['evaluation'];y=inputs.y[ids]
out=[]
for seed in (17,29,43):
    selected=a.entries(seed)
    for arm in ('cross_mean','cross_local'):
        folder=ROOT/'evaluation'/f'{arm}_s{seed}'
        net=model.Model(seed,arm)
        net.load_state_dict(torch.load(selected[arm]['checkpoint'],map_location='cpu',weights_only=True))
        net.eval().requires_grad_(False);net.accelerate()
        actual=full.predict(net,inputs,ids,'evaluation')
        replay=a.h.load_score(folder/'predictions.npz',ids,y)
        error=float(abs(actual-replay).max())
        assert error<1e-3 and np.array_equal(actual.argmax(1),replay.argmax(1))
        variants={}
        for flag in (('disabled','uniform_attention') if arm=='cross_local' else ('disabled',)):
            setattr(net.exchange,flag,True)
            z=full.predict(net,inputs,ids,'evaluation')
            setattr(net.exchange,flag,False)
            variants[flag]=z
            record=dict(seed=seed,arm=arm,flag=flag,metrics=full.metrics(y,z),
                        paired=a.h.paired(y,z,actual),replay_error=error,
                        max_logit_change=float(abs(z-actual).max()))
            out.append(record);print(record,flush=True)
        np.savez_compressed(folder/'pathways.npz',ids=ids,label=y,actual=actual,**variants)
        del net;gc.collect();torch.cuda.empty_cache()
a.write(ROOT/'evaluation_pathways.json',dict(rows=out,
    note='Same selected checkpoints. Diagnostic replacements after evaluation; no tuning, no deployment selection. Disabling after joint training does not undo co-adapted weights.'))
