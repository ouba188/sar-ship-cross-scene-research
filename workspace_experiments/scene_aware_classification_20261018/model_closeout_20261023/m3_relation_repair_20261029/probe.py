"""Source-only M3 path diagnosis; no held-query model selection."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE.parents[2]/'sar_m123_standalone'))
from sar_m123.current.model import load_model, training_policy, setup
from sar_m123.current.training import fitting_loss

spec = importlib.util.spec_from_file_location('repair_refit', ROOT/'spatial_m12_refit_20261029/run.py')
refit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refit)


def run():
    setup()
    rows, arrays, indices, bank = refit.load()
    labels = np.array([r['label'] for r in rows['fit']])
    chosen = np.concatenate([np.flatnonzero(labels == c)[:8] for c in range(8)])
    result = dict(scope='first up-to-eight source-fit samples per class; no target labels',
                  count=len(chosen), samples=[rows['fit'][i]['id'] for i in chosen], states={})
    refs = torch.tensor([r['label'] for r in rows['references']], device='cuda')
    for state, path in [('initial', refit.RELEASE/'weights/init.pt'),
                        ('fitted', refit.HERE/'training/spatial_s17/model.pt')]:
        net = load_model(path, 'cuda')
        training_policy(net, 'fit')
        totals = [[], []]
        for begin in range(0, len(chosen), 8):
            ix = chosen[begin:begin+8]
            q = [rows['fit'][i] for i in ix]
            y = torch.tensor(labels[ix], device='cuda')
            score, details = refit.score(net, arrays, indices['fit'][ix], q, rows['references'], bank, True)
            loss, terms = fitting_loss(score, details, y, torch.ones(len(ix), device='cuda'), refs)
            if begin == 0:
                names, params = zip(*[(n,p) for n,p in net.named_parameters() if n.startswith('m3.') and p.requires_grad])
                gradients = {}
                for term in ('ce', 'relation'):
                    grads = torch.autograd.grad(terms[term], params, allow_unused=True, retain_graph=True)
                    gradients[term] = {n: None if g is None else float(g.norm()) for n,g in zip(names,grads)}
            for pol, d in enumerate(details['operators']['m3_0']):
                live = d['reference_valid']
                same = refs[None].eq(y[:,None]) & live
                active = same.any(1)
                prob = d['reference_logits'].masked_fill(~live, -1e4).softmax(1)
                weights = d['reference_weights']
                ev = torch.cat([v.reshape(-1,5) for v in d['branch_similarity']])
                bw = torch.cat([v.reshape(-1,5) for v in d['branch_weights']])
                totals[pol].append(dict(n=int(active.sum()),
                    top1_correct=int(((refs[prob.argmax(1)] == y)&active).sum()),
                    same_mass=prob.mul(same).sum(1)[active].detach().cpu().tolist(),
                    reference_entropy=(-(prob*prob.clamp_min(1e-20).log()).sum(1))[active].detach().cpu().tolist(),
                    logit_span=(d['reference_logits'].max(1).values-d['reference_logits'].masked_fill(~live,1e4).min(1).values)[active].detach().cpu().tolist(),
                    branch_mean=ev.mean(0).detach().cpu().tolist(),
                    branch_spread=float((ev.max(1).values-ev.min(1).values).mean()),
                    branch_weights=bw.mean(0).detach().cpu().tolist(),
                    write_same_mass=(weights*same[:,:,None]).sum(1)[active].mean().item() if active.any() else None,
                    update_rms=float(d['update'].square().mean().sqrt())))
            del score,details,loss,terms
        summary=[]
        for pol,items in enumerate(totals):
            entry=dict(polarization=['VV','VH'][pol], n=sum(t['n'] for t in items))
            entry['reference_top1_accuracy']=sum(t['top1_correct'] for t in items)/max(entry['n'],1)
            for key in ('same_mass','reference_entropy','logit_span'):
                entry[key]=float(np.mean([v for t in items for v in t[key]]))
            for key in ('branch_mean','branch_spread','branch_weights','write_same_mass','update_rms'):
                entry[key]=np.mean([t[key] for t in items if t[key] is not None],axis=0).tolist()
            entry['gain']=float(net.m3[0][pol].gain)
            summary.append(entry)
        result['states'][state]=dict(summary=summary, gradients=gradients)
        print(state,json.dumps(summary),flush=True)
        del net
        torch.cuda.empty_cache()
    (HERE/'source_diagnosis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__ == '__main__':
    run()
