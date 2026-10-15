"""CUDA training through the deployed support solve and full-episode handoff."""

import json
from pathlib import Path

import numpy as np
import torch

from budget_conditioned_metric import channel_weights, metric_scores, gate_policy, training_loss


OUT = Path('E:/Hermes/paper_tracking/sensitivity/budget_conditioned_metric_followup')
ARMS = ('classification_metric', 'budget_metric')
DEVICE = torch.device('cuda')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def load_episode(path):
    with np.load(path) as saved:
        ep = {k: torch.from_numpy(saved[k]).to(DEVICE) for k in saved.files}
    ep['labels'] = ep['labels'].long()
    ep['support'] = ep['support'].long()
    mask = torch.zeros(len(ep['labels']), dtype=torch.bool, device=DEVICE)
    mask[ep['query'].long()] = True
    ep['query'] = mask
    return ep


def adapted(ep, theta):
    return metric_scores(ep['prior'], ep['retained'], ep['complement'], ep['support'],
        ep['labels'][ep['support']], theta)


def summary(ep, scores, gate):
    selected, _ = gate_policy(ep['base'], scores, gate, ep['labels'][ep['support']], ep['query'], ep['rows'])
    query, y = ep['query'], ep['labels'][ep['query']]
    prediction = torch.where(selected, scores.argmax(1), ep['base'].argmax(1))[query]
    correct = prediction == y
    ec = scores[query].argmax(1) == y
    ba = torch.stack([correct[y == c].double().mean() for c in y.unique()]).mean()
    expert_ba = torch.stack([ec[y == c].double().mean() for c in y.unique()]).mean()
    ce = training_loss(scores, ep['labels'], query, selected, False)
    budget_loss = training_loss(scores, ep['labels'], query, selected, True)
    return dict(oa=float(correct.double().mean()), ba=float(ba), expert_ba=float(expert_ba),
        full_ce=float(ce), budget_loss=float(budget_loss), admissions=int(selected.sum()),
        rescues=int(((ep['base'].argmax(1)[query] != y) & correct).sum()),
        harms=int(((ep['base'].argmax(1)[query] == y) & ~correct).sum()))


def source_readout(port, descriptors, theta, gate, arm, stage):
    records = []
    with torch.no_grad():
        for descriptor in descriptors:
            ep = load_episode(descriptor['path'])
            scores = ep['expert'] if stage == 'initial' else adapted(ep, theta)
            record = summary(ep, scores, gate)
            record.update(port=port, held_source=descriptor['port'], split=descriptor['split'], arm=arm, stage=stage)
            records.append(record)
    return records


def main():
    protocol = read(OUT / 'PROTOCOL.json')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(271828)
    all_records, all_histories, verification = [], [], []
    for port in protocol['ports']:
        folder = OUT / 'ports' / port
        if (folder / 'TRAINED.json').exists():
            all_records.extend(read(folder / 'source_metrics.json'))
            all_histories.extend(read(folder / 'training_history.json'))
            verification.append(read(folder / 'TRAINED.json'))
            continue
        descriptors = read(folder / 'PREPARED.json')['episodes']
        train = [d for d in descriptors if d['split'] == 'train']
        with np.load(folder / 'gate.npz') as saved:
            gate = {k: torch.tensor(saved[k], dtype=torch.double, device=DEVICE) for k in saved.files}
        initial = torch.zeros(2560, device=DEVICE)
        ep = load_episode(train[0]['path'])
        with torch.no_grad():
            replay = adapted(ep, initial)
            error = float((replay - ep['expert']).abs().max())
            if error > 5e-5:
                raise ValueError(f'CUDA/archived expert mismatch {port}: {error}')
        records = source_readout(port, descriptors, initial, gate, 'matched_gate', 'initial')
        weights, histories, diagnostics = {}, [], []
        for arm in ARMS:
            theta = torch.zeros(2560, device=DEVICE, requires_grad=True)
            optimizer = torch.optim.AdamW([theta], lr=.01, weight_decay=.001)
            max_gradient = 0.
            for epoch in range(6):
                losses, ce_values, admitted = [], [], []
                for index in np.random.default_rng(271828 + epoch).permutation(len(train)):
                    ep = load_episode(train[index]['path'])
                    optimizer.zero_grad()
                    scores = adapted(ep, theta)
                    selected, _ = gate_policy(ep['base'], scores, gate, ep['labels'][ep['support']], ep['query'], ep['rows'])
                    loss = training_loss(scores, ep['labels'], ep['query'], selected, arm == 'budget_metric')
                    loss.backward()
                    gradient = float(torch.nn.utils.clip_grad_norm_([theta], 5.))
                    if not np.isfinite(gradient) or not torch.isfinite(loss):
                        raise ValueError('Nonfinite metric training')
                    max_gradient = max(max_gradient, gradient)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                    ce_values.append(float(training_loss(scores.detach(), ep['labels'], ep['query'], selected, False)))
                    admitted.append(int(selected.sum()))
                record = dict(port=port, arm=arm, epoch=epoch + 1, objective=float(np.mean(losses)),
                    full_ce=float(np.mean(ce_values)), mean_admissions=float(np.mean(admitted)))
                histories.append(record)
                write(folder / 'training_history.json', histories)
                print(f'TRAIN {port} {arm} {epoch+1}/6 loss={record["objective"]:.6f} CE={record["full_ce"]:.6f}', flush=True)
            with torch.no_grad():
                learned = channel_weights(theta)
                weights[arm] = learned.cpu().numpy()
                weights[arm + '_theta'] = theta.detach().cpu().numpy()
                diagnostics.append(dict(arm=arm, weight_min=float(learned.min()), weight_max=float(learned.max()),
                    weight_rms_change=float((learned - 1).square().mean().sqrt()), maximum_gradient=max_gradient))
            records.extend(source_readout(port, descriptors, theta, gate, arm, 'final'))
        np.savez(folder / 'learned_weights.npz', **weights)
        write(folder / 'source_metrics.json', records)
        receipt = dict(port=port, status='complete', actual_cuda_replay_error=error,
            epochs=6, checkpoint='last only', diagnostics=diagnostics, target_labels_used=0)
        write(folder / 'TRAINED.json', receipt)
        all_records.extend(records)
        all_histories.extend(histories)
        verification.append(receipt)
        print('TRAINED ' + port, flush=True)
    write(OUT / 'source_metrics.json', all_records)
    write(OUT / 'training_history.json', all_histories)
    write(OUT / 'TRAINING_COMPLETE.json', dict(ports=protocol['ports'], verification=verification, target_labels_used=0))


if __name__ == '__main__':
    main()
