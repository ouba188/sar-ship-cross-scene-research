"""Chunked exact kernels for the matched dual/R50 evidence diagnosis."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'

import json
import numpy as np

from backbone_ablation_core import (
    AblationContext, OUT as PREVIOUS, ROOT, EPISODES, r50_views,
    r50_coordinates, _expert_inputs, load_npz, info_for,
)
from unified_head_expert import _conditions

OUT = ROOT/'dual_evidence_20261010'
BATCH = 512
ARMS = ('RR', 'DR', 'RD', 'DD')  # First letter: prior; second: kernel.
HEADS = ('expert', 'late', 'unified')
MODES = ('base',) + tuple(f'{arm}_{head}' for arm in ARMS for head in HEADS)


def subset_archive(path, keys):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in keys}


def load_task(folder):
    data = subset_archive(folder/'target.npz',
        ('rows', 'ids', 'phi', 'prior', 'mean', 'conditions', 'support',
         'support_labels', 'eval_masks', 'source_rows'))
    saved = subset_archive(folder/'selective_embedding.npz', ('rows', 'embedding'))
    np.testing.assert_array_equal(saved['rows'], data['rows'])
    data['embedding'] = saved['embedding']
    return data


def base_cross(data, rows, support, eta):
    cond = np.exp(-np.sum((data['conditions'][rows, None]
        - data['conditions'][support][None])**2, axis=2)/.5)
    kernel = data['phi'][rows]@data['phi'][support].T
    kernel += eta*(data['mean'][rows]@data['mean'][support].T)*cond
    return kernel + .25*(data['embedding'][rows].astype(float)
        @ data['embedding'][support].astype(float).T)*cond


class EvidenceContext:
    """Use saved matched source coefficients, not a new source-head fit."""

    def __init__(self, folder, data):
        self.ctx = ctx = AblationContext(folder)
        self.data = data
        _, suffix = info_for(folder)
        self.saved = load_npz(PREVIOUS/'targets'/suffix/'coefficients.npz')
        self.fitted = {name: {k: self.saved[name+'_'+k]
            for k in ('coef', 'intercept')} for name in ('dual', 'r50')}
        self.model, self.stats = (ctx.geometry['model'], ctx.geometry['stats']) \
            if ctx.dataset == 'self24ports' else (ctx.model, ctx.stats)
        if ctx.dataset == 'self24ports':
            self.bank = np.load(ctx.parent/'target_bank.npy', mmap_mode='r')
            self.extra = np.load(EPISODES/ctx.task/'matched_joint/joint_extra.npy', mmap_mode='r')
            self.u = _conditions()[data['rows']]
            np.testing.assert_array_equal(np.load(ctx.parent/'target_rows.npy'), data['rows'])
            np.testing.assert_array_equal(np.load(ctx.parent/'source_rows.npy'), data['source_rows'])
            self.raw = ctx.raw
        else:
            from pathlib import Path
            protocol = json.loads((ctx.joint.parent/'protocol.json').read_text())
            target = Path(protocol['target'])
            if ctx.dataset == 'oss2':
                self.raw = subset_archive(target/'features.npz', ('paired_r50',))['paired_r50']
            else:
                self.raw = np.load(target/'paired_r50.npy', mmap_mode='r')
            self.extra = np.load(ctx.joint/'target_extra.npy', mmap_mode='r')
            np.testing.assert_array_equal(data['rows'], np.arange(len(self.raw)))
            np.testing.assert_array_equal(load_npz(ctx.source_dir/'source_split.npz')['source_rows'],
                                          data['source_rows'])

    def features(self, indices):
        ctx, model, stats = self.ctx, self.model, self.stats
        raw = self.raw[self.data['rows'][indices]]
        extra = self.extra[indices]
        if ctx.dataset == 'self24ports':
            from run_protected_drift_activation import make_views
            bank = self.bank[indices]
            view = make_views(bank, raw, extra, ctx.geometry)
            rr, rd = r50_views(raw, model, stats, bank, ctx.geometry)
            dr, dd = view['common'][True]['retained'], view['common'][True]['complement']
            augmented = np.c_[bank, self.u[indices]]
            keep = np.r_[np.arange(128), np.arange(192, augmented.shape[1])]
            p = {}
            for name, head in self.fitted.items():
                inputs, coef = (augmented[:, keep], head['coef'][keep]) if name == 'r50' \
                    else (augmented, head['coef'])
                p[name] = inputs@coef+head['intercept']
            return dict(rr=rr, rd=rd, dr=dr, dd=dd, stable=view['stable'],
                        u=self.u[indices], priors=p)
        from evaluate_joint_residual import views
        _, dr, dd = views(raw, extra, model, stats, True)
        rr, rd = r50_views(raw, model, stats)
        p = {}
        for name, inputs in [('dual', _expert_inputs(raw, extra, model)),
                             ('r50', r50_coordinates(raw, model))]:
            head = self.fitted[name]
            p[name] = (inputs@head['coef']+head['intercept']).astype(np.float32).astype(float)
        return dict(rr=rr, rd=rd, dr=dr, dd=dd, priors=p)

    def cross(self, query, reference):
        common = 1.
        if self.ctx.dataset == 'self24ports':
            common = 1.+.25*(query['stable']@reference['stable'].T) \
                + .125*(query['u']@reference['u'].T)
        # Preserve the historical operation order separately for the two datasets.
        if self.ctx.dataset == 'self24ports':
            dual = .5*(query['dr']@reference['dr'].T+query['dd']@reference['dd'].T)+common
            single = .5*(query['rr']@reference['rr'].T+query['rd']@reference['rd'].T)+common
        else:
            dual = .5*(query['dr']@reference['dr'].T)+.5*(query['dd']@reference['dd'].T)+common
            single = .5*(query['rr']@reference['rr'].T+query['rd']@reference['rd'].T)+common
        return dict(R=single, D=dual)
