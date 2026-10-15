"""Exact frozen expert priors and support kernels for unified adaptation.

ExpertContext(folder).source_components() yields dictionaries with episode,
indices (support first), prior [N,8], kernel [N,64], rows and expected (query
scores only). target_components(data, support) returns (prior, kernel) in the
order of data['rows']. Scores are replay references, never kernel features.
All returned priors/kernels are float64. Only source linear heads are fitted;
geometry, feature caches and neural weights remain frozen. No files are written.
"""

from functools import lru_cache
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from compact_fusion_source import (
    ROOT, SOURCE, PRESERVE, CACHE, EXTERNAL, EPISODES, load_npz,
    _expert_inputs, _expert_fit_inputs, _expert_kernel,
)

BATCH_SIZE = 2048
PORT_PARENT = ROOT/'protected_drift_activation_20261008/ports'
PORT_REFERENCE = ROOT/'direct_incidence_comparison_20261009/ports'


def _pickle(path):
    with path.open('rb') as stream:
        return pickle.load(stream)


@lru_cache(maxsize=1)
def _source_frame():
    return pd.read_csv(SOURCE/'manifest.csv',
                       usecols=['instance_id', 'product', 'port', 'class_id'])


@lru_cache(maxsize=1)
def _conditions():
    from direct_incidence_conditioning import condition_features
    from run_conditioned_vessel_drift import SIDECAR

    side = pd.read_csv(SIDECAR,
                      usecols=['instance_id', 'product', 'status', 'incidence_deg'])
    frame = _source_frame().merge(side, on=['instance_id', 'product'], how='left',
                                 sort=False, validate='one_to_one')
    frame.loc[frame.status.ne('ok'), 'incidence_deg'] = np.nan
    return condition_features(frame.incidence_deg.to_numpy())


class ExpertContext:
    """One task's lazy source state and target kernel-to-support-union cache."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.info = json.loads((self.folder/'PREPARED.json').read_text())
        self.dataset, self.task = self.info['dataset'], self.info['task']
        if self.dataset not in ('self24ports', 'oss2', 'f0921'):
            raise ValueError(f'Unsupported expert dataset: {self.dataset}')
        self.raw = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
        self._source_ready = False
        self._heads = {}
        self._target_cache = None
        self._expected = None
        if self.dataset == 'self24ports':
            self.parent = PORT_PARENT/self.task
            self.geometry = _pickle(self.parent/'geometry.pkl')
        else:
            source_dir, joint_dir = EXTERNAL[self.dataset]
            self.source_dir = ROOT/source_dir
            self.joint = ROOT/joint_dir/'joint'
            self.model = _pickle(self.joint.parent/'source_model.pkl')
            self.stats = load_npz(self.joint.parent/'deployment_complement.npz')

    def _prepare_source(self):
        if self._source_ready:
            return
        frame = _source_frame()
        if self.dataset == 'self24ports':
            from direct_incidence_conditioning import extend_transforms
            from run_protected_drift_activation import make_transforms

            self.source_rows = np.load(self.parent/'source_rows.npy')
            self.bank = np.load(self.parent/'source_bank.npy', mmap_mode='r')
            self.extra = np.load(ROOT/'evidence_aware_handoff_20261008/source_features'
                                 /self.task/'features.npy', mmap_mode='r')
            self.augmented = np.c_[self.bank, _conditions()[self.source_rows]]
            self.transforms = extend_transforms(
                {'joint_expert': make_transforms(self.geometry)['joint_expert']})
            self.items = json.loads((EPISODES/self.task/'EPISODES_COMPLETE.json')
                                    .read_text())['episodes']
        else:
            self.source_rows = load_npz(self.source_dir/'source_split.npz')['source_rows']
            cached = self.joint/'source_extra.npy'
            if cached.exists():
                self.extra = np.asarray(np.load(cached, mmap_mode='r')[self.source_rows])
            else:
                saved = load_npz(CACHE/f'{self.joint.parent.name}_source_extra.npz')
                np.testing.assert_array_equal(saved['rows'], self.source_rows)
                self.extra = saved['features']
            if self.dataset == 'oss2':
                with np.load(SOURCE.parents[1]/'b2_ssl_feats_38091.npz') as saved:
                    fit_extra = saved['F'].astype(np.float32)[self.source_rows]
            else:
                path = ROOT/'neural_complement_pilot_20261007/frozen_replay/source_extra.npy'
                fit_extra = np.asarray(np.load(path, mmap_mode='r')[self.source_rows])
            self.fit_inputs = np.concatenate([
                _expert_fit_inputs(self.raw[self.source_rows[lo:lo+BATCH_SIZE]],
                                   fit_extra[lo:lo+BATCH_SIZE], self.model)
                for lo in range(0, len(self.source_rows), BATCH_SIZE)])
        self.locations = pd.Index(self.source_rows)
        self.source_labels = frame.class_id.to_numpy()[self.source_rows]
        self.source_ports = frame.port.to_numpy()[self.source_rows]
        self._source_ready = True

    def _expected_query(self, episode, rows):
        if self._expected is None:
            self._expected = load_npz(ROOT/'compact_fusion_20261010/source'
                                      /self.dataset/self.task/'source_pairs_final.npz')
        saved = self._expected
        mask = saved['episode'] == episode
        positions = pd.Index(saved['rows'][mask]).get_indexer(rows)
        if (positions < 0).any():
            raise ValueError(f'Missing expert replay rows: {self.task}/{episode}')
        return saved['expert'][mask][positions].astype(np.float64)

    def source_components(self):
        """Yield exactly the retained frequency_preserve episodes, in file order."""
        paths = sorted((PRESERVE/self.dataset/self.task).glob('episode_*.npz'))
        if not paths:
            raise FileNotFoundError(f'No retained source episodes for {self.folder}')
        for path in paths:
            episode = int(path.stem.split('_')[-1])
            with np.load(path) as saved:
                indices = saved['indices'].astype(np.int64)
            yield self.source(episode, indices)

    def source(self, episode, indices):
        """Build one retained episode; episode is its integer file number."""
        with threadpool_limits(limits=2):
            self._prepare_source()
            ep = load_npz(self.folder/'episodes'/f'{episode:02d}.npz')
            ix = np.asarray(indices, dtype=np.int64)
            np.testing.assert_array_equal(ix[:64], ep['support'])
            if len(np.unique(ix)) != len(ix) or not np.isin(ix[64:], ep['query']).all():
                raise ValueError('Source indices must be support followed by distinct queries')
            rows = ep['rows'][ix]
            positions = self.locations.get_indexer(rows)
            if (positions < 0).any():
                raise ValueError('Source episode outside permitted source rows')
            held = str(ep['held_port'])
            if self.dataset == 'self24ports':
                from protected_drift_activation import fit_source_heads

                original = load_npz(Path(self.items[episode]['path']))
                np.testing.assert_array_equal(original['rows'], ep['rows'])
                np.testing.assert_array_equal(original['support'], ix[:64])
                train = self.locations.get_indexer(original['train_rows'])
                if (train < 0).any() or np.isin(self.source_ports[train], [held, self.task]).any():
                    raise ValueError('Source expert fit includes an excluded port')
                if held not in self._heads:
                    self._heads[held] = fit_source_heads(
                        self.augmented[train], self.source_labels[train],
                        self.transforms)['joint_expert']
                head = self._heads[held]
                prior = self.augmented[positions]@head['coef']+head['intercept']
                kernel = self._port_kernel(self.bank[positions], self.raw[rows],
                                           self.extra[positions], _conditions()[rows],
                                           np.arange(64))
            else:
                train = self.source_ports != held
                if train[positions].any():
                    raise ValueError('External source head train/episode overlap')
                if held not in self._heads:
                    self._heads[held] = clone(self.model['expert']).fit(
                        self.fit_inputs[train], self.source_labels[train])
                    np.testing.assert_array_equal(self._heads[held].classes_, np.arange(8))
                raw, extra = self.raw[rows], self.extra[positions]
                xe = _expert_inputs(raw, extra, self.model)
                prior = self._heads[held].decision_function(xe).astype(np.float32).astype(float)
                kernel = _expert_kernel(raw, extra, xe, self.model, self.stats)
            return dict(episode=int(episode), indices=ix, rows=rows,
                        prior=np.asarray(prior, dtype=np.float64),
                        kernel=np.asarray(kernel, dtype=np.float64),
                        expected=self._expected_query(episode, rows[64:]))

    def _port_kernel(self, bank, raw, extra, u, reference):
        from run_protected_drift_activation import make_views

        ref = make_views(bank[reference], raw[reference], extra[reference], self.geometry)
        rc, rs = ref['common'][True], ref['stable']
        kernel = np.empty((len(bank), len(reference)), dtype=np.float64)
        for lo in range(0, len(bank), BATCH_SIZE):
            hi = min(lo+BATCH_SIZE, len(bank))
            views = make_views(bank[lo:hi], raw[lo:hi], extra[lo:hi], self.geometry)
            common = views['common'][True]
            block = .5*(common['retained']@rc['retained'].T
                        + common['complement']@rc['complement'].T)+1.
            block += .25*views['stable']@rs.T
            block += .25*(u[lo:hi]@u[reference].T)/2.
            kernel[lo:hi] = block
        return kernel

    def _build_target(self, data):
        rows = np.asarray(data['rows'])
        reference = np.unique(data['support'])
        if self.dataset == 'self24ports':
            np.testing.assert_array_equal(np.load(self.parent/'target_rows.npy'), rows)
            bank = np.load(self.parent/'target_bank.npy', mmap_mode='r')
            extra = np.load(EPISODES/self.task/'matched_joint/joint_extra.npy', mmap_mode='r')
            u = _conditions()[rows]
            head = _pickle(PORT_REFERENCE/self.task/'direct_heads.pkl')['joint_expert']
            prior = np.c_[bank, u]@head['coef']+head['intercept']
            kernel = self._port_kernel(bank, self.raw[rows], extra, u, reference)
        else:
            # Same operation order and separately projected reference as build_kernels.
            from evaluate_joint_residual import views

            protocol = json.loads((self.joint.parent/'protocol.json').read_text())
            target = Path(protocol['target'])
            if self.dataset == 'oss2':
                with np.load(target/'features.npz') as saved:
                    raw = saved['paired_r50']
            else:
                raw = np.load(target/'paired_r50.npy', mmap_mode='r')
            extra = np.load(self.joint/'target_extra.npy', mmap_mode='r')
            np.testing.assert_array_equal(rows, np.arange(len(raw)))
            _, retained_ref, discarded_ref = views(
                raw[reference], extra[reference], self.model, self.stats, True)
            prior = np.empty((len(raw), 8), dtype=np.float64)
            kernel = np.empty((len(raw), len(reference)), dtype=np.float64)
            for lo in range(0, len(raw), BATCH_SIZE):
                hi = min(lo+BATCH_SIZE, len(raw))
                p, retained, discarded = views(raw[lo:hi], extra[lo:hi],
                                               self.model, self.stats, True)
                prior[lo:hi] = p
                product = retained@retained_ref.T
                kernel[lo:hi] = .5*product+.5*(discarded@discarded_ref.T)+1.
        self._target_cache = (rows.copy(), reference, prior, kernel)

    def target_components(self, data, support):
        """Cache the support-union kernel once, then slice its ordered 64 columns."""
        support = np.asarray(support, dtype=np.int64)
        if support.shape != (64,) or len(np.unique(support)) != 64:
            raise ValueError('Expected the original 64 distinct support indices')
        with threadpool_limits(limits=2):
            if self._target_cache is None:
                self._build_target(data)
        rows, reference, prior, kernel = self._target_cache
        np.testing.assert_array_equal(rows, data['rows'])
        columns = pd.Index(reference).get_indexer(support)
        if (columns < 0).any():
            raise ValueError('Support outside the cached target support union')
        return prior, kernel[:, columns]

    target = target_components


@lru_cache(maxsize=1)
def get_context(folder):
    """Convenience cache for serial task traversal; explicit contexts are preferred."""
    return ExpertContext(folder)


def source_components(folder):
    return get_context(folder).source_components()


def target_components(folder, data, support):
    return get_context(folder).target_components(data, support)
