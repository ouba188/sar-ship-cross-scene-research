"""Source-only reconstruction of the deployed SSL view from existing R50 views."""

import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from compact_fusion_source import ROOT, CACHE, EPISODES
from unified_head_expert import ExpertContext

OUT = ROOT/'dual_reconstruction_20261010'
KINDS = ('linear', 'mlp')
BATCH = 512


def folders():
    return [p.parent for p in sorted((ROOT/'learned_frequency_residual_20261009/tasks').rglob('PREPARED.json'))]


def info_for(folder):
    info = json.loads((folder/'PREPARED.json').read_text())
    return info, f"{info['dataset']}/{info['task']}"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def freeze_protocol():
    path = OUT/'PROTOCOL.json'
    if path.exists():
        return
    write(path, dict(date='2026-10-10', tasks=[info_for(f)[1] for f in folders()],
        inputs='Original-view R50 pooled2048 plus C4 R50 pooled2048, already available in full base',
        teacher='Actual task-specific joint-tail SSL18 pooled512; training on permitted source_rows only',
        standardization='Per-coordinate source mean/std, std floor 1e-4, for both input and teacher',
        linear='Centered standardized MSE plus .01 Frobenius penalty, fixed closed-form solution',
        mlp='4096-512-512, exact GELU; AdamW lr=.001 weight_decay=.0001; 8 epochs batch512; seed20261010; final checkpoint',
        target='No mapper fitting on target features or labels; original five shared64-label support draws only for residual classifier',
        source_head='Refit source balanced ridge1 with original geometry and source rows, as matched dual/r50 controls',
        metric='Original retained/complement divisors and port stable/drift/incidence operators preserved',
        readouts=['expert', 'late', 'unified'], tuning='None; both mapper arms fixed before query evaluation',
        feature_diagnostics='Target teacher errors accumulated during frozen inference and reported after predictions are saved; never used for fitting or selection',
        boundary='Source-trained teacher is still needed during training; success would remove only the second inference backbone. Failure does not establish information-theoretic absence from R50.',
        evaluation='All25 tasks saved before classification query labels are evaluated',
        independent_confirmation=False))


class Inputs:
    def __init__(self, folder):
        self.ctx = ExpertContext(folder)
        self.info, self.suffix = info_for(folder)
        self.dataset = self.info['dataset']
        self.raw = self.ctx.raw
        self.c4 = np.load(ROOT/'c4_base_trial_20261009/pooled_c4.npy', mmap_mode='r')
        with np.load(folder/'target.npz') as z:
            self.rows, self.source_rows = z['rows'], z['source_rows']
        c = self.ctx
        if self.dataset == 'self24ports':
            np.testing.assert_array_equal(np.load(c.parent/'source_rows.npy'), self.source_rows)
            d = ROOT/'evidence_aware_handoff_20261008/source_features'/c.task
            np.testing.assert_array_equal(np.load(d/'rows.npy'), self.source_rows)
            self.source_extra = np.load(d/'features.npy', mmap_mode='r')
            self.target_raw, self.target_c4 = self.raw, self.c4
            self.target_extra = np.load(EPISODES/c.task/'matched_joint/joint_extra.npy', mmap_mode='r')
            self.model = c.geometry['model']
        else:
            split = np.load(c.source_dir/'source_split.npz')
            np.testing.assert_array_equal(split['source_rows'], self.source_rows)
            if np.intersect1d(self.source_rows, split['excluded_source_rows']).size:
                raise ValueError('Excluded identity in mapper source')
            path = c.joint/'source_extra.npy'
            if path.exists():
                self.source_extra = np.load(path, mmap_mode='r')[self.source_rows]
            else:
                with np.load(CACHE/f'{c.joint.parent.name}_source_extra.npz') as z:
                    np.testing.assert_array_equal(z['rows'], self.source_rows)
                    self.source_extra = z['features']
            target = Path(json.loads((c.joint.parent/'protocol.json').read_text())['target'])
            if self.dataset == 'oss2':
                with np.load(target/'features.npz') as z:
                    self.target_raw = z['paired_r50']
            else:
                self.target_raw = np.load(target/'paired_r50.npy', mmap_mode='r')
            self.target_c4 = np.load(ROOT/'cross_dataset_c4_20261009'/self.dataset/'pooled_c4.npy', mmap_mode='r')
            self.target_extra = np.load(c.joint/'target_extra.npy', mmap_mode='r')
            self.model = c.model
            np.testing.assert_array_equal(self.rows, np.arange(len(self.target_raw)))
        if len(self.source_extra) != len(self.source_rows) or len(self.target_extra) != len(self.rows):
            raise ValueError('Teacher row alignment differs')

    def source_x(self, index):
        rows = self.source_rows[index]
        return np.concatenate((self.raw[rows], self.c4[rows]), axis=1).astype(np.float32)

    def target_x(self, index):
        rows = self.rows[index]
        return np.concatenate((self.target_raw[rows], self.target_c4[rows]), axis=1).astype(np.float32)

    def target_raw_block(self, index):
        return self.target_raw[self.rows[index]]


def reconstruct(x, mapper, kind):
    x = (x-mapper['x_mean'])/mapper['x_std']
    if kind == 'linear':
        z = x@mapper['linear_weight']
    else:
        h = x@mapper['mlp_weight1']+mapper['mlp_bias1']
        h *= ndtr(h)
        z = h@mapper['mlp_weight2']+mapper['mlp_bias2']
    return (z*mapper['y_std']+mapper['y_mean']).astype(np.float32)
