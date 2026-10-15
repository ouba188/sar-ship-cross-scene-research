"""Frozen R50 expert ablation, with newly fitted source priors."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from compact_fusion_source import ROOT, SOURCE, CACHE, EPISODES, load_npz, _expert_inputs
from unified_head_expert import ExpertContext, _source_frame, _conditions, _pickle, PORT_REFERENCE
from protected_drift_activation import fit_source_heads
from projection_residual_expert import discarded_coordinates

OUT = ROOT/'backbone_ablation_20261010'
TASK_ROOT = ROOT/'learned_frequency_residual_20261009/tasks'
OLD = ROOT/'unified_head_20261010'
BATCH = 2048
MODES = ('base', 'dual_original_expert', 'dual_original_late', 'dual_original_unified',
         'dual_refit_expert', 'dual_refit_late', 'dual_refit_unified',
         'r50_expert', 'r50_late', 'r50_unified')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def folders():
    return [p.parent for p in sorted(TASK_ROOT.rglob('PREPARED.json'))]


def info_for(folder):
    info = json.loads((folder/'PREPARED.json').read_text())
    return info, f"{info['dataset']}/{info['task']}"


def single_transform(transform):
    # Keep the original metric for every remaining direction, including stable ones.
    columns = np.r_[np.arange(128), np.arange(192, transform.shape[1])]
    return transform[:, columns]


def r50_coordinates(raw, model):
    xs = ((model['pca'].transform(raw)-model['center'])/model['scale']).astype(np.float32)
    return (xs-model['expert_center'][:128])/model['expert_scale'][:128]


def r50_views(raw, model, stats, bank=None, geometry=None):
    residual = discarded_coordinates(raw, model['pca'])
    residual = (residual-stats['discarded_center'][:2048])/stats['discarded_scale'][:2048]
    if bank is None:
        retained = r50_coordinates(raw, model)
    else:
        from run_protected_drift_activation import DRIFT_STRENGTH
        rank = geometry['stable_rank']
        drift = np.asarray(bank[:, 192+rank:])
        core = np.asarray(bank[:, :128])-DRIFT_STRENGTH*(drift@geometry['drift_to_xs'])
        retained = (core-model['expert_center'][:128])/model['expert_scale'][:128]
        residual -= DRIFT_STRENGTH*(drift@geometry['drift_to_complement'])
    # No post-removal renormalization: this isolates removal from kernel reweighting.
    return np.asarray(retained/np.sqrt(192), float), residual/np.sqrt(2560)


class AblationContext(ExpertContext):
    """Read existing geometry; fit both heads on the exact permitted source rows."""

    def fit_heads(self, data):
        started = time.perf_counter()
        frame = _source_frame()
        if self.dataset == 'self24ports':
            self._prepare_source()
            transforms = self.transforms['joint_expert']
            fitted = fit_source_heads(self.augmented, self.source_labels,
                {'dual': transforms, 'r50': single_transform(transforms)})
            self.fitted = fitted
            self.model, self.stats = self.geometry['model'], self.geometry['stats']
            if np.any(self.source_ports == self.task):
                raise ValueError('Outer target port enters source fit')
            self.original_head = _pickle(PORT_REFERENCE/self.task/'direct_heads.pkl')['joint_expert']
            probe = self.augmented[:64]
            np.testing.assert_allclose(probe@fitted['dual']['coef']+fitted['dual']['intercept'],
                probe@self.original_head['coef']+self.original_head['intercept'], atol=1e-9, rtol=1e-9)
            if np.any(fitted['r50']['coef'][128:192] != 0):
                raise AssertionError('SSL columns enter R50-only source head')
            self.fit_details = dict(dual_width=int(transforms.shape[1]),
                r50_width=int(single_transform(transforms).shape[1]),
                source_feature_recipe='Frozen matched source bank; drift/stable/incidence retained',
                dual_refit_matches_original=True)
        else:
            split = load_npz(self.source_dir/'source_split.npz')
            self.source_rows = split['source_rows']
            if np.intersect1d(self.source_rows, split['excluded_source_rows']).size:
                raise ValueError('Excluded identity rows enter source head fitting')
            self.source_labels = frame.class_id.to_numpy()[self.source_rows]
            self.source_ports = frame.port.to_numpy()[self.source_rows]
            path = self.joint/'source_extra.npy'
            if path.exists():
                extra = np.load(path, mmap_mode='r')
                take = lambda rows, lo, hi: extra[rows]
            else:
                cache = load_npz(CACHE/f'{self.joint.parent.name}_source_extra.npz')
                np.testing.assert_array_equal(cache['rows'], self.source_rows)
                extra = cache['features']
                take = lambda rows, lo, hi: extra[lo:hi]
            inputs = np.empty((len(self.source_rows), 192), dtype=float)
            for lo in range(0, len(inputs), BATCH):
                hi = min(lo+BATCH, len(inputs))
                rows = self.source_rows[lo:hi]
                inputs[lo:hi] = _expert_inputs(self.raw[rows], take(rows, lo, hi), self.model)
            self.fitted = {}
            for name, width in [('dual', 192), ('r50', 128)]:
                head = clone(self.model['expert']).fit(inputs[:, :width], self.source_labels)
                np.testing.assert_array_equal(head.classes_, np.arange(8))
                if head.class_weight != 'balanced' or head.alpha != 1.:
                    raise AssertionError('Unexpected source ridge recipe')
                self.fitted[name] = dict(coef=head.coef_.T, intercept=head.intercept_)
            self.fit_details = dict(dual_width=192, r50_width=128,
                source_feature_recipe='Frozen joint-tail deployment coordinates for both refitted heads',
                original_source_feature_recipe='Pre-tail source prior retained only in original controls',
                excluded_source_rows=int(len(split['excluded_source_rows'])))
        np.testing.assert_array_equal(self.source_rows, data['source_rows'])
        self.fit_details.update(source_n=int(len(self.source_rows)),
            source_classes={str(c): int(n) for c, n in zip(*np.unique(self.source_labels, return_counts=True))},
            source_fit_seconds=time.perf_counter()-started,
            helper_reads_full_source_manifest_labels=True,
            fit_labels_sliced_to_permitted_source_rows=True,
            target_query_labels_used_for_fit=False)
        return self.fit_details

    def build(self, data):
        """Build both support-union kernels in memory, never save feature matrices."""
        started = time.perf_counter()
        reference = np.unique(data['support'])
        n = len(data['rows'])
        kernels = {name: np.empty((n, len(reference)), float) for name in ('dual', 'r50')}
        priors = {name: np.empty((n, 8), float) for name in ('original', 'dual', 'r50')}
        if self.dataset == 'self24ports':
            from run_protected_drift_activation import make_views
            bank = np.load(self.parent/'target_bank.npy', mmap_mode='r')
            extra = np.load(EPISODES/self.task/'matched_joint/joint_extra.npy', mmap_mode='r')
            np.testing.assert_array_equal(np.load(self.parent/'target_rows.npy'), data['rows'])
            u = _conditions()[data['rows']]
            rv = make_views(bank[reference], self.raw[data['rows'][reference]], extra[reference], self.geometry)
            rr, rd = r50_views(self.raw[data['rows'][reference]], self.model, self.stats,
                              bank[reference], self.geometry)
            dr, dd = rv['common'][True]['retained'], rv['common'][True]['complement']
            for lo in range(0, n, BATCH):
                hi = min(lo+BATCH, n)
                raw = self.raw[data['rows'][lo:hi]]
                views = make_views(bank[lo:hi], raw, extra[lo:hi], self.geometry)
                ret, comp = views['common'][True]['retained'], views['common'][True]['complement']
                sr, sd = r50_views(raw, self.model, self.stats, bank[lo:hi], self.geometry)
                common = 1.+.25*(views['stable']@rv['stable'].T)+.125*(u[lo:hi]@u[reference].T)
                kernels['dual'][lo:hi] = .5*(ret@dr.T+comp@dd.T)+common
                kernels['r50'][lo:hi] = .5*(sr@rr.T+sd@rd.T)+common
                augmented = np.c_[bank[lo:hi], u[lo:hi]]
                for name, head in [('original', self.original_head), *self.fitted.items()]:
                    if name == 'r50':
                        keep = np.r_[np.arange(128), np.arange(192, augmented.shape[1])]
                        priors[name][lo:hi] = augmented[:, keep]@head['coef'][keep]+head['intercept']
                    else:
                        priors[name][lo:hi] = augmented@head['coef']+head['intercept']
        else:
            from evaluate_joint_residual import views
            protocol = json.loads((self.joint.parent/'protocol.json').read_text())
            target = Path(protocol['target'])
            if self.dataset == 'oss2':
                with np.load(target/'features.npz') as z:
                    raw = z['paired_r50']
            else:
                raw = np.load(target/'paired_r50.npy', mmap_mode='r')
            np.testing.assert_array_equal(data['rows'], np.arange(len(raw)))
            extra = np.load(self.joint/'target_extra.npy', mmap_mode='r')
            _, dr, dd = views(raw[reference], extra[reference], self.model, self.stats, True)
            rr, rd = r50_views(raw[reference], self.model, self.stats)
            for lo in range(0, n, BATCH):
                hi = min(lo+BATCH, n)
                rb, re = raw[lo:hi], extra[lo:hi]
                original, ret, comp = views(rb, re, self.model, self.stats, True)
                sr, sd = r50_views(rb, self.model, self.stats)
                kernels['dual'][lo:hi] = .5*(ret@dr.T)+.5*(comp@dd.T)+1.
                kernels['r50'][lo:hi] = .5*(sr@rr.T)+.5*(sd@rd.T)+1.
                priors['original'][lo:hi] = original
                for name, inputs in [('dual', _expert_inputs(rb, re, self.model)),
                                     ('r50', r50_coordinates(rb, self.model))]:
                    head = self.fitted[name]
                    priors[name][lo:hi] = (inputs@head['coef']+head['intercept']).astype(np.float32).astype(float)
                if lo % (BATCH*12) == 0:
                    print(f'KERNEL {self.dataset}/{self.task}: {hi}/{n}', flush=True)
        return reference, priors, kernels, time.perf_counter()-started
