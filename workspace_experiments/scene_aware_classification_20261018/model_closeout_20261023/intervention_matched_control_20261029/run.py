"""Matched CE+KL continuation without intervention-selected margin protection."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = HERE.parents[2]
FULL = ROOT/'spatial_m12_full_20261029'
FIT = ROOT/'spatial_m12_refit_20261029'
sys.path.insert(0, str(WORK/'sar_m123_standalone'))
from sar_m123.current.data import batch, pair_mask, prepare_record
from sar_m123.current.model import LIVE, forward_batch, load_model, reference_bank, setup
from sar_m123.current.training import adaptation_loss, train

SEEDS = (17, 29, 43)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def loss_check():
    score = torch.tensor([[.2, -.4, 1.1], [.8, -.1, -.5]], requires_grad=True)
    teacher = torch.tensor([[.9, -.3, .2], [.4, .7, -.6]])
    labels = torch.tensor([0, 2])
    weights = torch.tensor([.5, 2.])
    ce = (weights*F.cross_entropy(score, labels, reduction='none')).sum()/weights.sum()
    kl = F.kl_div(score.log_softmax(1), teacher.log_softmax(1),
                  reduction='none', log_target=True).sum(1)
    expected = ce+.1*(weights*teacher.argmax(1).eq(labels)*kl).sum()/weights.sum()
    expected_grad = torch.autograd.grad(expected, score, retain_graph=True)[0]
    errors = []
    for flags in (torch.zeros(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool)):
        actual, _ = adaptation_loss(score, labels, teacher, weights, flags, 0.)
        grad = torch.autograd.grad(actual, score, retain_graph=True)[0]
        errors.append(max(float(abs(actual-expected).detach()), float(abs(grad-expected_grad).max())))
    assert max(errors) < 1e-7, errors
    write(HERE/'loss_check.json', dict(value_and_gradient_max_errors=errors,
        comparison='CE + .1 KL on teacher-correct rows, independent of useful flags'))


def checkpoint(seed):
    return HERE/'training'/f'control_s{seed}'/'model.pt'


def fit_controls(rows):
    for seed in SEEDS:
        output = checkpoint(seed).parent
        assert not checkpoint(seed).exists(), f'Completed checkpoint exists: {output}'
        source = FIT/'training'/f'spatial_s{seed}'/'model.pt'
        net = load_model(source, 'cuda')
        records = json.loads((FULL/'selection'/f'spatial_s{seed}.json').read_text(encoding='utf-8'))
        # Reuse only unaltered teacher logits; no intervention identity or outcome.
        targets = [dict(id=r['id'], label=r['label'], original=r['original'], useful=False) for r in records]
        fixed = {k: v.detach().cpu().clone() for k, v in net.state_dict().items() if k not in LIVE}
        initial = {k: v.detach().cpu().clone() for k, v in net.state_dict().items() if k in LIVE}
        tick = time.perf_counter()
        receipt = train(net, rows['adapt'], rows['references'], output, 'cuda', stage='adapt',
                        selection=targets, seed=seed, epochs=6, batch_size=16, coefficient=0.)
        assert receipt['steps'] == 54 and receipt['selected_count'] == 0
        assert all(torch.equal(net.state_dict()[k].cpu(), v) for k, v in fixed.items())
        updates = {k: float((net.state_dict()[k].cpu()-v).norm()) for k, v in initial.items()}
        assert all(value > 0 for value in updates.values())
        assert all(epoch['floor'] == 0. for epoch in receipt['curve'])
        del fixed, initial
        query = rows['development'][:16]
        values = batch([prepare_record(r) for r in query], 'cuda')
        bank = reference_bank(net, rows['references'], 'cuda')
        legal = pair_mask(query, rows['references'], 'cuda')
        restored = load_model(checkpoint(seed), 'cuda')
        with torch.no_grad():
            error = float(abs(forward_batch(net, values, bank, legal)-
                              forward_batch(restored, values, bank, legal)).max())
        assert error < 1e-5, error
        receipt.update(seconds=time.perf_counter()-tick, frozen_unchanged=True, live_updates=updates,
                       save_replay_max_error=error, source_checkpoint=str(source),
                       intervention_outcomes_used=False, original_teacher_reused=True,
                       contrast='guided spatial complete minus matched CE+KL continuation')
        write(output/'training.json', receipt)
        print('CONTROL_COMPLETE', seed, round(receipt['seconds'], 1), flush=True)
        del net, restored, values, bank
        torch.cuda.empty_cache()


@torch.no_grad()
def evaluate(rows):
    models = {s: load_model(checkpoint(s), 'cuda') for s in SEEDS}
    first = models[SEEDS[0]]
    prefix_names = [k for k in first.state_dict() if k.startswith('backbone.model.') and
                    not k.startswith(('backbone.model.layer3.', 'backbone.model.layer4.', 'backbone.model.fc.'))]
    for net in models.values():
        assert all(torch.equal(first.state_dict()[k], net.state_dict()[k]) for k in prefix_names)
    bank = reference_bank(first, rows['references'], 'cuda')
    replay = {}
    tick = time.perf_counter()
    for scope in ('development', 'expanded'):
        scores = {s: [] for s in SEEDS}
        for start in range(0, len(rows[scope]), 16):
            group = rows[scope][start:start+16]
            values = batch([prepare_record(r) for r in group], 'cuda')
            prefix = first.prefix(values['images'], stop=2)
            legal = pair_mask(group, rows['references'], 'cuda')
            for seed, net in models.items():
                result = net.forward_from(prefix, **{k: v for k, v in values.items() if k != 'images'},
                                          reference=bank, pair_valid=legal)
                if scope == 'development' and start == 0:
                    error = float(abs(result-forward_batch(net, values, bank, legal)).max())
                    assert error < 1e-5, error
                    replay[str(seed)] = error
                scores[seed].append(result.cpu().numpy())
            if start % 512 == 0 or start+16 >= len(rows[scope]):
                print('EVAL', scope, min(start+16, len(rows[scope])), len(rows[scope]),
                      round(time.perf_counter()-tick, 1), flush=True)
        for seed, parts in scores.items():
            score = np.concatenate(parts)
            assert np.isfinite(score).all()
            path = HERE/'predictions'/f'{scope}_control_s{seed}.npz'
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, score=score, ids=np.array([str(r['id']) for r in rows[scope]]),
                                y=np.array([r['label'] for r in rows[scope]]))
    write(HERE/'evaluation_check.json', dict(native_replay_errors=replay, development=160,
        expanded=3075, models=3, seconds=time.perf_counter()-tick, default_changed=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('check', 'train', 'evaluate', 'all'))
    args = parser.parse_args()
    setup('cpu' if args.command == 'check' else 'cuda')
    loss_check()
    rows = json.loads((FULL/'cohorts.json').read_text(encoding='utf-8'))
    if args.command in ('train', 'all'):
        fit_controls(rows)
    if args.command in ('evaluate', 'all'):
        evaluate(rows)
    print('COMPLETE', args.command, flush=True)


if __name__ == '__main__':
    main()
