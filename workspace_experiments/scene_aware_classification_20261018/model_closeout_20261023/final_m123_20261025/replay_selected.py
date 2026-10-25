"""Replay selected M1/M2 heads and M3 fusion on frozen representation caches.

This is not raw-chip deployment or training. Original operators and checkpoints
remain authoritative; the entry point joins them without rewriting their math.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REDESIGN = BASE / 'evidence_preserving_redesign'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


anchor = load('selected_anchor', REDESIGN / 'anchored_polar_adjustment_20261025/train.py')
polar = load('selected_polar', REDESIGN / 'polar_readout_diagnosis_20261025/train.py')
center = load('selected_center', REDESIGN / 'null_centered_readout_20261025/train.py')
upstream = load('selected_upstream', BASE / 'regional_utility_pilot/upstream/pipeline.py')


def replay_upstream(seed, data):
    root = upstream.HERE
    with np.load(root / 'summaries.npz') as z:
        means, hb, summary, quality, geometry, valid = (
            z[k] for k in ('mean3', 'hb3', 'summary1', 'quality', 'geometry', 'valid'))
    with np.load(root / f'base_seed{seed}.npz') as z:
        score = z['B'].copy()
    with np.load(root / f'strategies_seed{seed}.npz') as z:
        expected = z['logits'].copy()
    result = {}
    for stage, model, col in (
        ('attention', upstream.AttentionHead(), 0),
        ('full', upstream.RegionHead(), 2),
    ):
        ck = torch.load(root / 'models' / f'{stage}_seed{seed}.pt',
                        map_location='cpu', weights_only=False)
        raw = (upstream.attention_raw(score, means, hb, valid) if stage == 'attention'
               else upstream.regional_raw(score, summary, quality, geometry))
        x = ((raw - ck['mu']) / ck['scale']).astype(np.float32)
        model.load_state_dict(ck['state'])
        model.eval()
        delta = upstream.predict(model, x)
        score = score + delta
        error = float(abs(score - expected[:, col]).max())
        np.testing.assert_allclose(score, expected[:, col], atol=3e-6, rtol=0)
        result[stage] = dict(selected_step=int(ck['epoch']), max_error=error,
                            max_delta=float(abs(delta[data.up[data.te]]).max()))
    return torch.tensor(score[data.up[:data.n]].astype(np.float32)), result


class SelectedCachedReadout(nn.Module):
    """Frozen original VV, zero-evidence-centered VH, and anchored fusion."""

    def __init__(self, seed, null):
        super().__init__()
        self.vv = polar.Head('vv')
        self.vh = center.Head('center_vh', null)
        for model, folder, arm in (
            (self.vv, 'polar_readout_diagnosis_20261025', 'vv'),
            (self.vh, 'null_centered_readout_20261025', 'center_vh'),
        ):
            checkpoint = torch.load(REDESIGN / folder / 'models' / f'{arm}_{seed}.pt',
                                    map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['state'])
        checkpoint = torch.load(
            REDESIGN / 'anchored_polar_adjustment_20261025/models' / f'frequency_{seed}.pt',
            map_location='cpu', weights_only=False)
        self.gate = anchor.Gate('frequency', checkpoint['receipt']['alpha_anchor'])
        self.gate.load_state_dict(checkpoint['state'])
        self.receipt = checkpoint['receipt']
        self.eval().requires_grad_(False)

    def forward(self, base, local_features, gate_features, available):
        dv = self.vv(local_features)
        dh = self.vh(local_features, available)
        weights, alpha, _ = self.gate(gate_features, available)
        scores = base + weights[:, :1] * dv + weights[:, 1:] * dh
        return scores, weights, alpha, dv, dh


def main():
    torch.set_num_threads(2)
    data = polar.Data()
    y = data.table.iloc[:data.n].class_id.to_numpy(int)
    receipts, predictions = [], []
    for seed in (17, 29, 43):
        base, upstream_receipt = replay_upstream(seed, data)
        raw_gate, z = anchor.inputs(data, seed)
        np.testing.assert_allclose(base.numpy(), z['base'].numpy(), atol=3e-6, rtol=0)
        _, gate_x, _ = anchor.restore(seed, 'frequency', raw_gate)
        with np.load(REDESIGN / 'polar_readout_diagnosis_20261025/results' /
                     f'inputs_{seed}.npz') as cache:
            np.testing.assert_array_equal(cache['up_index'], data.up[:data.n])
            raw_local = polar.raw_features(cache['local'], cache['base'])
            local_x = torch.tensor(np.where(cache['active'],
                (raw_local - cache['mu']) / cache['scale'], 0).astype(np.float32))
            null = np.where(cache['active'], -cache['mu'] / cache['scale'], 0).astype(np.float32)
            np.testing.assert_allclose(cache['base'], z['base'].numpy(), atol=3e-6, rtol=0)
        model = SelectedCachedReadout(seed, null)
        with torch.inference_mode():
            values = model(base, local_x, gate_x, z['available'])
        scores, weights, alpha, dv, dh = (value.numpy() for value in values)
        vv_error = float(abs(dv - z['delta_vv'].numpy()).max())
        vh_error = float(abs(dh - z['delta_vh'].numpy()).max())
        with np.load(REDESIGN / 'anchored_polar_adjustment_20261025/results' /
                     f'frequency_{seed}.npz') as expected:
            np.testing.assert_array_equal(expected['up_index'], data.up[:data.n])
            score_error = float(abs(scores - expected['score']).max())
            np.testing.assert_allclose(scores, expected['score'], atol=3e-6, rtol=0)
            np.testing.assert_array_equal(scores.argmax(1), expected['score'].argmax(1))
            np.testing.assert_allclose(weights, expected['weight'], atol=1e-6, rtol=0)
        assert max(vv_error, vh_error) < 3e-6
        rows = data.te
        receipt = dict(seed=seed, alpha_anchor=model.receipt['alpha_anchor'],
            selected_step=model.receipt['selected_step'], n_replayed=data.n,
            m1_selected_step=upstream_receipt['attention']['selected_step'],
            m2_selected_step=upstream_receipt['full']['selected_step'],
            m1_score_error=upstream_receipt['attention']['max_error'],
            m2_score_error=upstream_receipt['full']['max_error'],
            n_evaluation=len(rows), vv_max_error=vv_error, vh_max_error=vh_error,
            score_max_error=score_error, **anchor.metrics(y[rows], scores[rows]))
        receipts.append(receipt)
        frame = data.table.iloc[rows][['obs_id', 'up_index', 'class_id']].copy()
        frame['seed'] = seed
        frame['prediction'] = scores[rows].argmax(1)
        frame['vv_weight'] = weights[rows, 0]
        frame['vh_weight'] = weights[rows, 1]
        predictions.append(frame)
        print(json.dumps(receipt), flush=True)
    pd.DataFrame(receipts).to_csv(HERE / 'REPLAY_METRICS.csv', index=False)
    pd.concat(predictions).to_csv(HERE / 'REPLAY_PREDICTIONS.csv', index=False)
    report = dict(status='passed', new_training=False, raw_chip_extraction=False,
        scope='Original M1/M2 heads from summaries, VV/VH heads and selected frequency gate replayed; image extraction and both matching cached.',
        configurations=receipts,
        mean_metrics=pd.DataFrame(receipts)[['OA', 'BA', 'NLL', 'Brier']].mean().to_dict())
    (HERE / 'REPLAY_RECEIPT.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['mean_metrics']), flush=True)


if __name__ == '__main__':
    main()
