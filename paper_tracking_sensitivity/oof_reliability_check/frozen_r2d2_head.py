"""R2D2-style meta-ridge head on frozen C4 coordinates, not full R2D2 reproduction."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
import math
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import f1_score
from frozen_comparison import OUT, EXP, HEADS, SOURCE, ROOT, DATASETS, folders, load_npz, write, ba

DEST = OUT/'r2d2_frozen_head'
NORMALIZED = False


def meta_scores(x, support, labels, log_lambda, log_scale):
    if NORMALIZED:
        x = torch.cat([x, torch.ones((len(x), 1), dtype=x.dtype)], dim=1)
    xs = x[support]
    targets = F.one_hot(labels, 8).to(x.dtype)
    strength = log_lambda.exp()*(len(xs) if NORMALIZED else 1)
    dual = torch.linalg.solve(xs@xs.T + strength*torch.eye(len(xs), dtype=x.dtype), targets)
    return log_scale.exp()*(x@xs.T)@dual


def train(folder):
    info = json.loads((folder/'PREPARED.json').read_text())
    suffix = f"{info['dataset']}/{info['task']}"
    dest = DEST/'source'/suffix
    if (dest/'MODEL.json').exists():
        return
    paths = sorted((folder/'episodes').glob('*.npz'))
    episodes = []
    for path in sorted((HEADS/'source'/suffix).glob('episode_*.npz')):
        z = load_npz(path)
        ep = load_npz(paths[int(path.stem.split('_')[-1])])
        ix = z['indices']
        episodes.append((torch.tensor(ep['mean'][ix], dtype=torch.float64),
                         torch.tensor(ep['labels'][ix], dtype=torch.long)))
    log_lambda = torch.nn.Parameter(torch.tensor(-math.log(64) if NORMALIZED else 0., dtype=torch.float64))
    log_scale = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))
    optimizer = torch.optim.Adam([log_lambda, log_scale], lr=.01)
    rng = np.random.default_rng(20261010)
    logs = []
    for epoch in range(30):
        losses = []
        for ei in rng.permutation(len(episodes)):
            x, labels = episodes[ei]
            optimizer.zero_grad()
            scores = meta_scores(x, torch.arange(64), labels[:64], log_lambda, log_scale)
            loss = F.cross_entropy(scores[64:], labels[64:])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite meta training loss')
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        logs.append(dict(epoch=epoch+1, loss=float(np.mean(losses))))
    write(dest/'MODEL.json', dict(**info, ridge=float(log_lambda.exp().detach()),
        scale=float(log_scale.exp().detach()), epochs=30, learning_rate=.01, parameters=2,
        training=logs, target_selection=False, source_episodes=len(episodes)))
    print(f'META {suffix}: lambda={log_lambda.exp().item():.5f} scale={log_scale.exp().item():.5f}', flush=True)


def target(folder):
    info = json.loads((folder/'PREPARED.json').read_text())
    suffix = f"{info['dataset']}/{info['task']}"
    dest = DEST/'targets'/suffix
    if (dest/'COMPLETE.json').exists():
        return
    fitted = json.loads((DEST/'source'/suffix/'MODEL.json').read_text())
    data = load_npz(folder/'target.npz')
    x = data['mean'].astype(float)
    if NORMALIZED:
        x = np.c_[x, np.ones(len(x))]
    predictions = []
    for s, y in zip(data['support'], data['support_labels']):
        dual = np.linalg.solve(x[s]@x[s].T+fitted['ridge']*(64 if NORMALIZED else 1)*np.eye(64), np.eye(8)[y])
        pred = (fitted['scale']*x@(x[s].T@dual)).argmax(1)
        predictions.append(pred)
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest/'policies.npz', predictions=np.array(predictions, dtype=np.int8),
        ids=data['ids'], rows=data['rows'], support=data['support'], eval_masks=data['eval_masks'])
    write(dest/'COMPLETE.json', dict(dataset=info['dataset'], task=info['task'], draws=5,
        query_labels_read=False, full_r2d2_reproduction=False))
    print('META TARGET '+suffix, flush=True)


def report():
    tasks = folders()
    markers = list((DEST/'targets').glob('*/*/COMPLETE.json'))
    if len(markers) != 25:
        raise ValueError('All25 target predictions must be frozen first')
    rows, classes = [], []
    for folder in tasks:
        info = json.loads((folder/'PREPARED.json').read_text())
        p = load_npz(DEST/'targets'/info['dataset']/info['task']/'policies.npz')
        original = load_npz(folder/'policies.npz')
        manifest, idcol = ((SOURCE/'manifest.csv', 'instance_id') if info['dataset'] == 'self24ports'
            else (ROOT/DATASETS[info['dataset']][0]/'target_manifest.csv', 'sample_id'))
        frame = pd.read_csv(manifest, usecols=[idcol, 'class_id'], keep_default_na=False).iloc[p['rows']]
        np.testing.assert_array_equal(frame[idcol].to_numpy(str), p['ids'])
        slot = list(original['modes']).index('selective')
        for draw, ev in enumerate(p['eval_masks']):
            y, pred = frame.class_id.to_numpy()[ev], p['predictions'][draw, ev]
            ref = original['predictions'][draw, slot, ev]
            rows.append(dict(dataset=info['dataset'], task=info['task'], draw=draw,
                mode='r2d2_normalized_head' if NORMALIZED else 'r2d2_frozen_head', oa=100*np.mean(pred == y), ba=ba(y, pred),
                macro_f1_8=100*f1_score(y, pred, labels=np.arange(8), average='macro', zero_division=0),
                delta_oa_vs_ours=100*(np.mean(pred == y)-np.mean(ref == y)), delta_ba_vs_ours=ba(y,pred)-ba(y,ref)))
            sy = frame.class_id.to_numpy()[p['support'][draw]]
            for c in np.unique(y):
                classes.append(dict(dataset=info['dataset'], task=info['task'], draw=draw, class_id=int(c),
                    support_count=int(np.sum(sy == c)), query_count=int(np.sum(y == c)),
                    prediction_fraction=100*np.mean(pred == c), recall=100*np.mean(pred[y == c] == c),
                    ours_recall=100*np.mean(ref[y == c] == c)))
    frame = pd.DataFrame(rows)
    frame.to_csv(DEST/'metrics.csv', index=False)
    pd.DataFrame(classes).to_csv(DEST/'class_metrics.csv', index=False)
    pd.DataFrame(classes).groupby(['dataset','task','class_id'])[['prediction_fraction','recall','ours_recall']].mean().groupby(
        ['dataset','class_id']).mean().to_csv(DEST/'class_summary.csv')
    fields = ['oa','ba','macro_f1_8','delta_oa_vs_ours','delta_ba_vs_ours']
    task = frame.groupby(['dataset','task','mode'])[fields].mean().reset_index()
    task.to_csv(DEST/'task_means.csv', index=False)
    summary = task.groupby(['dataset','mode'])[fields].mean().reset_index()
    summary.to_csv(DEST/'summary.csv', index=False)
    write(DEST/'COMPLETE.json', dict(tasks=25, draws=125, full_r2d2_reproduction=False,
        label='R2D2-style meta-ridge head, frozen-feature protocol adaptation'))
    print(summary.to_string(index=False), flush=True)


def main():
    global NORMALIZED, DEST
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('source','target','report','all'), default='all')
    parser.add_argument('--normalized', action='store_true')
    args = parser.parse_args()
    NORMALIZED = args.normalized
    if NORMALIZED:
        DEST = OUT/'r2d2_normalized_head'
    torch.set_num_threads(2)
    if not (DEST/'PROTOCOL.json').exists():
        write(DEST/'PROTOCOL.json', dict(date='2026-10-10', name='R2D2-style frozen-feature meta-ridge head',
            reference='Bertinetto et al. ICLR2019, arXiv:1805.08136',
            formula=('scale * Xq_tilde Xs_tilde.T (Xs_tilde Xs_tilde.T + n*lambda I)^-1 onehot(Ys)'
                     if NORMALIZED else 'scale * Xq Xs.T (Xs Xs.T + lambda I)^-1 onehot(Ys)'),
            regularized_class_intercept=NORMALIZED, mean_square_normalization=NORMALIZED,
            initial_lambda=(1/64 if NORMALIZED else 1), initial_scale=1,
            official_default_differences='Learned positive lambda, scale-aware initial effective sum-loss lambda1, alpha1,30epochs; not official default recipe',
            feature='Existing source-standardized normalized C4 PCA128 mean coordinates',
            training='Learn positive lambda and positive output scale on source episodes, Adam .01,30epochs,final epoch',
            target='Same five64label draws, no neural target fitting, no source score anchor, eight outputs',
            source_loss='Ordinary cross entropy, no class balancing',
            adaptations=['Frozen backbone instead of end-to-end feature meta-learning',
                         'Fixed eight classes with naturally incomplete support coverage, not balanced N-way K-shot',
                         'No common scalar output bias: it cancels in softmax and argmax'],
            not_full_reproduction=True, target_query_selection=False, independent_confirmation=False))
    if args.stage in ('source','all'):
        for folder in folders():
            train(folder)
    if args.stage in ('target','all'):
        for folder in folders():
            target(folder)
    if args.stage in ('report','all'):
        report()


if __name__ == '__main__':
    main()
