"""Frozen choices on the existing exposed evaluation split; no tuning or updates."""
import gc
from pathlib import Path
import importlib.util
import sys

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('joint_full_evaluation',ROOT/'analyze.py')
a=importlib.util.module_from_spec(spec);sys.modules[spec.name]=a;spec.loader.exec_module(a)
assert a.read(ROOT/'confirmation.json')['complete']
model,inputs=a.run.runtime()
torch,np,full=model.torch,model.np,model.full
inputs.load_split('evaluation');inputs.prepare_geometry('evaluation')
ids=inputs.groups['evaluation'];y=inputs.y[ids]
results=[]
for seed in (17,29,43):
    chosen=a.entries(seed);scores={}
    for arm in a.run.ARMS:
        dest=ROOT/'evaluation'/f'{arm}_s{seed}';dest.mkdir(parents=True,exist_ok=True)
        if (dest/'receipt.json').exists():
            record=a.read(dest/'receipt.json')
            assert record['checkpoint']==chosen[arm]['checkpoint']
            scores[arm]=a.h.load_score(dest/'predictions.npz',ids,y)
            continue
        net=model.Model(seed,arm)
        net.load_state_dict(torch.load(chosen[arm]['checkpoint'],map_location='cpu',weights_only=True))
        net.eval().requires_grad_(False);net.accelerate()
        val=inputs.groups['validation']
        validation=full.predict(net,inputs,val,'validation')
        replay=a.run.h.replay(model,val,validation,chosen[arm]['validation_predictions'])
        scores[arm]=full.predict(net,inputs,ids,'evaluation')
        np.savez_compressed(dest/'predictions.npz',ids=ids,label=y,score=scores[arm])
        a.write(dest/'receipt.json',dict(seed=seed,arm=arm,checkpoint=chosen[arm]['checkpoint'],
            config=chosen[arm]['config'],epoch=chosen[arm]['best']['epoch'],
            validation_replay=replay,metrics=full.metrics(y,scores[arm]),
            note='Previously exposed evaluation378; frozen validation choices, not a new blind test.'))
        del net;gc.collect();torch.cuda.empty_cache()
    for arm,z in scores.items():
        results.append(dict(seed=seed,arm=arm,metrics=full.metrics(y,z),
            vs_full=a.h.paired(y,z,scores['full']),vs_mean=a.h.paired(y,z,scores['cross_mean'])))
        print('EVALUATION',seed,arm,results[-1],flush=True)
a.write(ROOT/'evaluation_summary.json',dict(rows=results,
    means={arm:{k:float(np.mean([r['metrics'][k] for r in results if r['arm']==arm]))
                for k in ('oa','ba','macro_f1','nll')} for arm in a.run.ARMS},
    note='No evaluation-based checkpoint/hyperparameter selection; historically exposed 378 observations.'))
