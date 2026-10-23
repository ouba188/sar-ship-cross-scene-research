"""Support-only evidence weighting with matched image/pass exclusion controls."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
UP = ROOT.parent / 'regional_utility_pilot/upstream'
sys.path.insert(0, str(ROOT))
from match_readout import unit
from train_readout import metrics, exact_j

REGIONS = ((0, 8), (8, 16), (16, 20))
TAU = 1 / np.sqrt(32)
ARMS = ('reference_image', 'reference_pass', 'token_image', 'token_pass')
CHECKPOINTS = (0, 15, 50, 100, 200)


def load():
    table = pd.read_csv(UP / 'table.csv', dtype={'mmsi': str, 'instance_id': str})
    data = np.load(ROOT / 'tokens.npz')
    projection = np.load(ROOT / 'projection_layer1.npz')
    np.testing.assert_array_equal(data['instance_ids'].astype(str), table.instance_id.to_numpy())
    x = unit((data['layer1'] - projection['mean']) @ projection['components'].T).astype(np.float32)
    mass = data['mass'].astype(np.float32)
    x[mass <= 0] = 0
    bank = projection['bank'].astype(int)
    assert bank.shape == (8, 8) and len(np.unique(bank)) == 64
    assert np.array_equal(table.iloc[bank.ravel()].class_id.to_numpy().reshape(8, 8),
                          np.broadcast_to(np.arange(8)[:, None], (8, 8)))
    return table, x, mass, bank


def kernels(x, mass, bank, ids):
    """Cache query-averaged log kernels, keeping support token identities."""
    output = []
    with torch.inference_mode():
        for first in range(0, len(ids), 128):
            qids = ids[first:first + 128]
            pieces = []
            for p in range(2):
                for lo, hi in REGIONS:
                    q = torch.tensor(x[qids, p, lo:hi], device='cuda')
                    s = torch.tensor(x[bank, p, lo:hi], device='cuda')
                    a = torch.tensor(mass[qids, p, lo:hi], device='cuda')
                    a = a / a.sum(-1, keepdim=True).clamp_min(1e-20)
                    sim = torch.einsum('bid,ckjd->bckij', q, s) / TAU
                    loga = a.masked_fill(a <= 0, 1).log().masked_fill(a <= 0, -1e4)
                    k = torch.logsumexp(sim + loga[:, None, None, :, None], -2)
                    pieces.append(F.pad(k, (0, 8 - (hi - lo)), value=-1e4).cpu().numpy())
            output.append(np.stack(pieces, 1))
    return np.concatenate(output).astype(np.float32)


class Readout:
    def __init__(self, kernel, mass, bank, ids, allowed):
        self.kernel = torch.as_tensor(kernel)
        bs, qs = [], []
        for p in range(2):
            for lo, hi in REGIONS:
                b = mass[bank, p, lo:hi].copy()
                b /= np.maximum(b.sum(-1, keepdims=True), 1e-20)
                b = np.pad(b, ((0, 0), (0, 0), (0, 8 - hi + lo)))
                bs.append(b)
                qs.append(mass[ids, p, lo:hi].sum(-1) > 0)
        b = torch.tensor(np.stack(bs), dtype=torch.float32)
        allowed = torch.as_tensor(allowed, dtype=torch.bool)
        # Each available reference has unit mass before task-dependent reweighting.
        base = b[None] * allowed[:, None, :, :, None]
        self.base = base
        self.logbase = base.masked_fill(base <= 0, 1).log().masked_fill(base <= 0, -1e4)
        self.valid = (base.sum((-1, -2)) > 0) & torch.tensor(np.stack(qs, 1))[:, :, None]

    @staticmethod
    def expand(v, kind):
        if kind == 'reference':
            return v.reshape(1, 8, 8, 1).expand(6, -1, -1, 8)
        pieces = []
        for p in range(2):
            for lo, hi in REGIONS:
                pieces.append(F.pad(v[:, :, p, lo:hi], (0, 8 - hi + lo)))
        return torch.stack(pieces)

    def __call__(self, v, kind, return_reference=False):
        weight = self.expand(v, kind)[None]
        raw_mass = self.logbase + weight
        lognorm = torch.logsumexp(raw_mass.flatten(-2), -1)
        logmass = raw_mass - lognorm[..., None, None]
        ref = torch.logsumexp(self.kernel + logmass, -1)
        score = torch.logsumexp(ref, -1)
        score = torch.where(self.valid, score, 0)
        total = score.sum(1) / self.valid.sum(1).clamp_min(1)
        total = total - total.mean(-1, keepdim=True)
        if return_reference:
            contribution = (ref - torch.logsumexp(ref, -1, keepdim=True)).exp()
            contribution = torch.where(self.valid[..., None], contribution, 0)
            return total, score, contribution
        return total


def main(arms=ARMS, cached=None):
    tick = time.perf_counter()
    torch.set_num_threads(2)
    table, x, mass, bank = load()
    idx = np.flatnonzero(table.role.isin(['utility_train', 'selection', 'evaluation']))
    ids = np.r_[bank.ravel(), idx]
    if cached is None:
        k = kernels(x, mass, bank, ids)
    else:
        prior = np.load(cached)
        np.testing.assert_array_equal(prior['ids'], ids)
        np.testing.assert_array_equal(prior['bank'], bank)
        k = prior['kernel']
    np.savez_compressed(HERE / 'kernel.npz', kernel=k, ids=ids, bank=bank)
    allowed = np.ones((len(idx), 8, 8), bool)
    reader = Readout(k[64:], mass, bank, idx, allowed)
    labels = table.class_id.to_numpy()
    y = torch.tensor(labels[bank.ravel()])
    role = table.iloc[idx].role.to_numpy()
    tr, va, te = [np.flatnonzero(role == r) for r in ('utility_train', 'selection', 'evaluation')]
    zero = torch.zeros(8, 8, 2, 20)
    uniform = reader(zero, 'token').detach().numpy()
    scale = max(float(np.sqrt(np.mean(uniform[tr] ** 2))), 1e-6)
    local = {'uniform': uniform}
    weights = {'uniform': zero.numpy()}
    curves, chosen, checks = [], [], []
    for arm in arms:
        kind, exclusion = arm.split('_')
        key = 'mmsi' if exclusion == 'image' else 'pass_group'
        groups = table.iloc[bank.ravel()][key].to_numpy()
        mask = (groups[:, None] != groups[None]).reshape(64, 8, 8)
        if exclusion in ('crosspos', 'crossneg'):
            same_pass = ~mask.reshape(64, 64)
            same_class = y.numpy()[:, None] == y.numpy()[None]
            remove = same_pass & (same_class if exclusion == 'crosspos' else ~same_class)
            # These are supervised loss-edge masks; query labels never enter inference.
            mask = (~(remove | np.eye(64, dtype=bool))).reshape(64, 8, 8)
        train_reader = Readout(k[:64], mass, bank, bank.ravel(), mask)
        assert not np.any(mask.reshape(64, 64).diagonal())
        assert train_reader.valid[:, :3].all(), 'Every VV query must retain every class.'
        shape = (8, 8) if kind == 'reference' else (8, 8, 2, 20)
        v = torch.nn.Parameter(torch.zeros(shape))
        opt = torch.optim.Adam([v], lr=.03)
        best = exact_j(labels[idx[tr]], uniform[tr].argmax(-1))
        selected = 0
        selected_v = v.detach().clone()
        selected_scores = uniform.copy()
        initial_train = None
        for step in range(201):
            if step in CHECKPOINTS:
                with torch.no_grad():
                    train_scores = train_reader(v, kind).numpy()
                    all_scores = reader(v, kind).numpy()
                if step == 0:
                    np.testing.assert_allclose(all_scores, uniform, atol=2e-6)
                    initial_train = metrics(y.numpy(), train_scores)
                for name, yv, values in [('support_train', y.numpy(), train_scores),
                                         ('utility_train', labels[idx[tr]], all_scores[tr])]:
                    curves.append(dict(arm=arm, step=step, role=name, **metrics(yv, values)))
                objective = exact_j(labels[idx[tr]], all_scores[tr].argmax(-1))
                if objective > best:
                    best = objective
                    selected, selected_v, selected_scores = step, v.detach().clone(), all_scores.copy()
            if step == 200:
                break
            opt.zero_grad(set_to_none=True)
            scores = train_reader(v, kind)
            loss = F.cross_entropy(scores, y) + .01 * v.square().mean()
            loss.backward()
            opt.step()
        local[arm] = selected_scores
        weights[arm] = selected_v.numpy()
        chosen.append(dict(arm=arm, step=selected, mask_excluded_pairs=int((~mask).sum()),
                           weight_rms=float(selected_v.square().mean().sqrt()),
                           terminal_weight_rms=float(v.square().mean().sqrt()),
                           **metrics(labels[idx[tr]], selected_scores[tr])))
        checks.append(dict(arm=arm, initial_train=initial_train,
                           terminal_train=metrics(y.numpy(), train_reader(v, kind).detach().numpy())))
        print('TRAIN', chosen[-1], flush=True)
    np.savez_compressed(HERE / 'weights.npz', **weights)
    np.savez_compressed(HERE / 'local_scores.npz', idx=idx, bank=bank, **local)
    pd.DataFrame(curves).to_csv(HERE / 'training_curve.csv', index=False)
    pd.DataFrame(chosen).to_csv(HERE / 'checkpoints.csv', index=False)
    direct = []
    for arm, value in local.items():
        for name, select in [('utility_train', tr), ('selection', va), ('evaluation', te)]:
            direct.append(dict(arm=arm, role=name, **metrics(labels[idx[select]], value[select])))
    pd.DataFrame(direct).to_csv(HERE / 'direct_metrics.csv', index=False)
    rows, selections, paired = [], [], []
    for seed in (17, 29, 43):
        base = np.load(UP / f'strategies_seed{seed}.npz')['logits'][:, 2].astype(np.float32)
        predictions = {'full': base}
        selected_weights = {}
        for arm, value in local.items():
            best = exact_j(labels[idx[va]], base[idx[va]].argmax(-1))
            gain = 0.
            for w in (0., .1, .25, .5, 1., 2.):
                score = base[idx[va]] + w * value[va] / scale
                objective = exact_j(labels[idx[va]], score.argmax(-1))
                selections.append(dict(seed=seed, arm=arm, gain=w, **metrics(labels[idx[va]], score)))
                if objective > best:
                    best, gain = objective, w
            selected_weights[arm] = gain
            s = base.copy()
            s[idx] += gain * value / scale
            predictions[arm] = s
            for name, select in [('selection', va), ('evaluation', te)]:
                rows.append(dict(seed=seed, arm=arm, gain=gain, role=name,
                                 **metrics(labels[idx[select]], s[idx[select]])))
        for arm, value in local.items():
            s = base.copy()
            gain = selected_weights['uniform']
            s[idx] += gain * value / scale
            predictions[arm + '_common'] = s
            rows.append(dict(seed=seed, arm=arm + '_common', gain=gain, role='evaluation',
                             **metrics(labels[idx[te]], s[idx[te]])))
        rows.append(dict(seed=seed, arm='full', gain=0., role='evaluation',
                         **metrics(labels[idx[te]], base[idx[te]])))
        for arm, scores in predictions.items():
            correct = scores[idx[te]].argmax(-1) == labels[idx[te]]
            old = predictions['uniform'][idx[te]].argmax(-1) == labels[idx[te]]
            paired.append(dict(seed=seed, arm=arm, rescued=int((correct & ~old).sum()),
                               harmed=int((~correct & old).sum()), net=int(correct.sum() - old.sum())))
        np.savez_compressed(HERE / f'predictions_{seed}.npz', **predictions)
    pd.DataFrame(rows).to_csv(HERE / 'source_metrics.csv', index=False)
    pd.DataFrame(selections).to_csv(HERE / 'selection_curve.csv', index=False)
    pd.DataFrame(paired).to_csv(HERE / 'paired_results.csv', index=False)
    report = dict(seconds=time.perf_counter() - tick, scale=scale, support_n=64,
                  source_query_n=len(idx), tau=TAU, source_only=True, checks=checks,
                  no_backbone_training=True, method='normalized FewTURE-inspired control, not official reproduction')
    (HERE / 'receipt.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(pd.DataFrame(rows).query("role == 'evaluation'").groupby('arm')[['OA', 'BA']].mean().to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--factorial', action='store_true')
    args = parser.parse_args()
    if args.factorial:
        cached = HERE / 'kernel.npz'
        HERE = HERE / 'mask_factorial'
        HERE.mkdir(exist_ok=True)
        main(('token_crosspos', 'token_crossneg'), cached)
    else:
        main()
