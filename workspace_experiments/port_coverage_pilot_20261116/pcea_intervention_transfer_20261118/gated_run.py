"""Source-only GDOS-inspired training; historical hard split remains separate."""
import argparse
import json
import time
import numpy as np
import torch
from causal_run import base, ROOT, predict
from causal_split import donor_permutation, classification
from gated_split import weighted_consistency
from gated_model import Model

OUT = ROOT/'gated_disentanglement_20261119'
ARMS = ('gate_ce', 'gate_uwcl')


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def make_model(initial_gate=.95):
    model = Model(initial_gate)
    model.accelerate()
    return model


def forward(net, inputs, ids, code, mode='content', donor=None):
    data = inputs.data(ids, 'small', code)
    reference = net.reference(inputs.data(inputs.groups['references'], 'small'))
    z, aux = net.forward_variant(data, [inputs.rows[i] for i in ids], reference,
                                inputs.reference_rows, mode, donor)
    return z, aux, data


def backward(net, inputs, ids, code, generator, consistency_weight=1.):
    net.train()
    z, aux, data = forward(net, inputs, ids, code)
    labels = torch.as_tensor(inputs.y[ids], device=z.device)
    weights = torch.as_tensor(inputs.weights[ids], device=z.device)
    ce = classification(z, aux, labels, weights, data['available'])
    clean = z.detach()
    donor, eligible = donor_permutation(data['available'], generator)
    ce.backward()
    rows = [inputs.rows[i] for i in ids]
    domains = ['_'.join(r['pass_group'].split('_')[:2]) for r in rows]
    cross = sum(bool(eligible[i]) and domains[i] != domains[int(donor[i])] for i in range(len(ids)))
    stats = dict(ce=float(ce.detach()), consistency=0., swapped=int(eligible.sum()),
                 cross_passage_swaps=cross, kl=0., reliability=0., probability_mse=0.)
    del z, aux, data, ce
    if consistency_weight > 0:
        changed, aux, data = forward(net, inputs, ids, code, 'swap', donor)
        consistency, details = weighted_consistency(changed, clean, eligible.to(changed.device))
        (consistency_weight*consistency).backward()
        stats.update(consistency=float(consistency.detach()), **details)
    stats['loss'] = stats['ce']+consistency_weight*stats['consistency']
    return clean, stats


def train(args):
    inputs, fit, inner = base.runtime('unified')
    fit = np.setdiff1d(fit, inputs.groups['references'])
    count = args.steps*args.batch
    if count > len(fit):
        raise ValueError('Requested source queries exceed distinct fit scope')
    # Reuse the predeclared source validation subset and exact previous order.
    selection = ROOT/'causal_work_check_20261119/selection_local.npz'
    validation = np.load(selection)['validation']
    assert np.isin(validation, inner).all() and not np.isin(validation, fit).any()
    dest = OUT/f'{args.arm}_s{args.seed}'
    dest.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    net = make_model(args.initial_gate)
    if args.arm == 'gate_ce':
        net.disentangler.fusions.requires_grad_(False)
    groups = net.optimizer_groups()
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    order = np.random.default_rng(args.seed+20261119).permutation(fit)[:count]
    codes = np.random.default_rng(args.seed+20261120).integers(8, size=args.steps)
    generator = torch.Generator().manual_seed(args.seed+20261121)
    np.savez_compressed(dest/'schedule_local.npz', ids=order, codes=codes,
                        references=inputs.groups['references'], validation=validation)
    write(dest/'config.json', dict(vars(args), source_queries=len(fit), validation=len(validation),
          initialization=str(base.CHECKPOINT), reference_count=len(inputs.groups['references']),
          rates={g['name']:g['lr'] for g in groups}, source_only_selection=True,
          hard_split=False, prior_loss=False, clutter_loss=False, box_regression_loss=False,
          old_intervention_margin_loss=False, default_inference='content only'))
    initial = predict(net, inputs, validation)
    best = base.m.full.metrics(inputs.y[validation], initial)
    best_step = 0
    torch.save(net.state_dict(), dest/'best.pt')
    np.savez_compressed(dest/'validation_best_local.npz', ids=validation, labels=inputs.y[validation], scores=initial)
    write(dest/'step_0.json', dict(metrics=best))
    print('INITIAL', args.arm, best, flush=True)
    history = []
    tick = time.perf_counter()
    for step in range(args.steps):
        ids = order[step*args.batch:(step+1)*args.batch]
        optimizer.zero_grad(set_to_none=True)
        _, stats = backward(net, inputs, ids, int(codes[step]), generator,
                            args.consistency_weight if args.arm == 'gate_uwcl' else 0.)
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']], 5., error_if_nonfinite=True)
        optimizer.step()
        history.append(dict(step=step+1, **stats))
        if (step+1)%10 == 0:
            print('PROGRESS', args.arm, step+1, stats, flush=True)
        if (step+1)%args.eval_every == 0 or step+1 == args.steps:
            scores = predict(net, inputs, validation)
            metrics = base.m.full.metrics(inputs.y[validation], scores)
            if base.recipe.previous.key(metrics) > base.recipe.previous.key(best):
                best, best_step = metrics, step+1
                torch.save(net.state_dict(), dest/'best.pt')
                np.savez_compressed(dest/'validation_best_local.npz', ids=validation, labels=inputs.y[validation], scores=scores)
            write(dest/f'step_{step+1}.json', dict(metrics=metrics, loss=stats))
            write(dest/'training_log.json', history)
            print('EVAL', args.arm, step+1, metrics, flush=True)
    torch.save(net.state_dict(), dest/'last.pt')
    write(dest/'receipt.json', dict(completed=True, updates=args.steps, best_step=best_step,
          seconds=time.perf_counter()-tick, source_only=True, peak_cuda_gib=torch.cuda.max_memory_allocated()/1024**3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('verify', 'train'), default='verify')
    parser.add_argument('--arm', choices=ARMS, default='gate_uwcl')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--initial-gate', type=float, default=.95)
    parser.add_argument('--consistency-weight', type=float, default=1.)
    parser.add_argument('--eval-every', type=int, default=30)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.phase == 'verify':
        from verify_gated import main as verify
        verify()
    else:
        if args.steps < 1 or args.batch < 2 or args.eval_every < 1 or args.consistency_weight < 0:
            parser.error('positive steps/eval-every, batch>=2, nonnegative consistency weight required')
        train(args)


if __name__ == '__main__':
    main()
