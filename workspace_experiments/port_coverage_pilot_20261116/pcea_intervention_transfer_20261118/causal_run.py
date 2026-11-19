"""Source-only training entry for the current causal-subfeature candidate."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import time
import torch
import numpy as np
ROOT = Path(__file__).resolve().parent
if __name__ == '__main__':
    sys.modules['causal_run'] = sys.modules[__name__]
spec = importlib.util.spec_from_file_location('causal_training_base',ROOT/'run.py')
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
from causal_model import Model
from causal_split import (donor_permutation, classification, swap_consistency,
    bias_prior_loss, effective_prior)

OUT = ROOT/'causal_subfeatures_20261119'
ARMS = ('baseline', 'split_ce', 'causal')


def make_model(keep_ratio=.5):
    model = Model(keep_ratio)
    model.accelerate()
    return model


def set_source_prior(model, inputs, fit):
    prior = effective_prior(torch.as_tensor(inputs.y[fit]),
        torch.as_tensor(inputs.weights[fit]),model.net.matcher.classes)
    model.source_prior.copy_(prior.to(model.source_prior))


def forward(net, inputs, ids, code, mode='causal', donor=None):
    data = inputs.data(ids,'small',code)
    reference = net.reference(inputs.data(inputs.groups['references'],'small'))
    z,aux = net.forward_variant(data,[inputs.rows[i] for i in ids],reference,
        inputs.reference_rows,mode,donor)
    return z,aux,data


def backward(net, inputs, ids, code, generator, strength=1., swap_weight=1., bias_weight=.5):
    """Three fresh graphs keep live source references and bounded GPU memory."""
    net.train()
    z,aux,data = forward(net,inputs,ids,code)
    labels = torch.as_tensor(inputs.y[ids],device=z.device)
    weights = torch.as_tensor(inputs.weights[ids],device=z.device)
    ce = classification(z,aux,labels,weights,data['available'])
    clean = z.detach()
    donors,eligible = donor_permutation(data['available'],generator)
    ce.backward()
    stats = dict(ce=float(ce.detach()),swap=0.,bias=0.,swapped=int(eligible.sum()),strength=strength)
    del ce,z,aux,data
    if strength > 0:
        changed,aux,data = forward(net,inputs,ids,code,'swap',donors)
        swap = swap_consistency(changed,clean,eligible.to(changed.device))
        (strength*swap_weight*swap).backward()
        stats['swap'] = float(swap.detach())
        del changed,aux,data,swap
        biased,aux,data = forward(net,inputs,ids,code,'bias')
        prior_loss = bias_prior_loss(biased,net.source_prior,data['available'].bool())
        (strength*bias_weight*prior_loss).backward()
        stats['bias'] = float(prior_loss.detach())
        del biased,aux,data,prior_loss
    stats['loss'] = stats['ce']+strength*(swap_weight*stats['swap']+bias_weight*stats['bias'])
    return clean,stats


@torch.no_grad()
def predict(net, inputs, ids):
    net.eval()
    return base.m.full.predict(net,inputs,ids,'small')


def train(args):
    inputs,fit,inner = base.runtime('unified')
    fit = np.setdiff1d(fit,inputs.groups['references'])
    count = args.steps*args.batch
    if count > len(fit):
        raise ValueError('Requested distinct source queries exceed source-fit scope')
    dest = OUT/f'{args.arm}_s{args.seed}'
    dest.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    net = make_model(1. if args.arm == 'baseline' else args.keep_ratio)
    set_source_prior(net,inputs,fit)
    groups = net.optimizer_groups()
    if args.arm == 'baseline':
        net.disentangler.requires_grad_(False)
        groups = [g for g in groups if g['name'] != 'disentangler']
    optimizer = torch.optim.AdamW(groups,weight_decay=.01)
    order = np.random.default_rng(args.seed+20261119).permutation(fit)[:count]
    codes = np.random.default_rng(args.seed+20261120).integers(8,size=args.steps)
    generator = torch.Generator().manual_seed(args.seed+20261121)
    np.savez_compressed(dest/'schedule_local.npz',ids=order,codes=codes,references=inputs.groups['references'])
    config = dict(vars(args),prior=net.source_prior.tolist(),rates={g['name']:g['lr'] for g in groups},
        source_fit=len(fit),source_validation=len(inner),reference_count=len(inputs.groups['references']),
        initialization=str(base.CHECKPOINT),selected_by='source validation BA, OA, negative NLL',
        split='per-polarization shared channel mask after layer1; deterministic hard train/eval',
        old_teacher_bank_used=False,old_intervention_margin_loss_used=False,
        note='Feature substitution is not physically simulated SAR data or proof of causal identification.')
    (dest/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    initial = predict(net,inputs,inner)
    best = base.m.full.metrics(inputs.y[inner],initial)
    torch.save(net.state_dict(),dest/'best.pt')
    np.savez_compressed(dest/'validation_best_local.npz',ids=inner,label=inputs.y[inner],scores=initial)
    (dest/'step_0.json').write_text(json.dumps(dict(metrics=best)),encoding='utf-8')
    history=[]
    tick=time.perf_counter()
    for step in range(args.steps):
        batch=order[step*args.batch:(step+1)*args.batch]
        ramp = min(1.,max(0.,(step-args.warmup+1)/max(args.ramp,1))) if args.arm == 'causal' else 0.
        optimizer.zero_grad(set_to_none=True)
        _,stats=backward(net,inputs,batch,int(codes[step]),generator,ramp,args.swap_weight,args.bias_weight)
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']],5.,error_if_nonfinite=True)
        optimizer.step()
        history.append(dict(step=step+1,**stats))
        if (step+1)%args.eval_every == 0 or step+1 == args.steps:
            scores=predict(net,inputs,inner)
            metrics=base.m.full.metrics(inputs.y[inner],scores)
            if base.recipe.previous.key(metrics) > base.recipe.previous.key(best):
                best=metrics
                torch.save(net.state_dict(),dest/'best.pt')
                np.savez_compressed(dest/'validation_best_local.npz',ids=inner,label=inputs.y[inner],scores=scores)
            (dest/f'step_{step+1}.json').write_text(json.dumps(dict(metrics=metrics,loss=stats)),encoding='utf-8')
            print(args.arm,step+1,metrics,flush=True)
    torch.save(net.state_dict(),dest/'last.pt')
    (dest/'training_log.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    (dest/'receipt.json').write_text(json.dumps(dict(completed=True,updates=args.steps,
        seconds=time.perf_counter()-tick,source_only=True,external_evaluation=False)),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('verify','train'),default='verify')
    parser.add_argument('--arm',choices=ARMS,default='causal')
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--steps',type=int,default=120)
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--keep-ratio',type=float,default=.5)
    parser.add_argument('--warmup',type=int,default=30)
    parser.add_argument('--ramp',type=int,default=30)
    parser.add_argument('--swap-weight',type=float,default=1.)
    parser.add_argument('--bias-weight',type=float,default=.5)
    parser.add_argument('--eval-every',type=int,default=30)
    args=parser.parse_args()
    if args.phase == 'verify':
        from verify_causal import main as verify
        verify()
    else:
        if args.steps < 1 or args.batch < 2 or args.eval_every < 1 or args.warmup < 0 or args.ramp < 0:
            parser.error('positive steps/eval-every, batch >=2, nonnegative warmup/ramp required')
        train(args)


if __name__ == '__main__':
    main()
