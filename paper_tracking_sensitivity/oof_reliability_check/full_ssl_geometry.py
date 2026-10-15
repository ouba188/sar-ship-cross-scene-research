"""Source-only geometry for the matched frozen G50/D50 and clean D18 readout.

All row arguments index the full source manifest. No excluded rows, target
labels, historical heads, or Torch models are accessed by this module.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier


def _project(values, pca):
    centered = np.asarray(values, dtype=np.float64)-pca.mean_
    return centered@np.asarray(pca.components_, dtype=np.float64).T


def _xs(values, state):
    return (_project(values, state['pca'])-state['center'])/state['scale']


def _discarded(values, pca):
    centered = np.asarray(values, dtype=np.float64)-pca.mean_
    components = np.asarray(pca.components_, dtype=np.float64)
    return centered-(centered@components.T)@components


def _discarded_stats(values, rows, pca, batch_size):
    total = np.zeros(values.shape[1], dtype=np.float64)
    squares = np.zeros_like(total)
    for lo in range(0, len(rows), batch_size):
        residual = _discarded(values[rows[lo:lo+batch_size]], pca)
        total += residual.sum(0)
        squares += np.square(residual).sum(0)
    center = total/len(rows)
    scale = np.sqrt(np.maximum(squares/len(rows)-center**2, 0.))+1e-6
    return center, scale


def _fit_head(values, labels):
    head = RidgeClassifier(alpha=1., class_weight='balanced').fit(values, labels)
    if not np.array_equal(head.classes_, np.arange(8)):
        raise ValueError('Source head requires class IDs 0 through 7 in metadata order')
    return head


def fit_geometry(raw_full, c4_full, extra_full, labels_full, source_rows, *,
                 raw_rank=128, extra_rank=64, batch_size=2048):
    """Fit pickle-able PCA, statistics and four balanced ridge1 source heads.

    PCA fitting uses float32; subsequent coordinates and heads use float64.
    The original-view R50 PCA and its source scaling also define C4 coordinates.
    Only source_rows are read, so other rows may contain NaN. Lower ranks are
    available for small synthetic tests. Discarded statistics are blockwise.
    """
    rows = np.asarray(source_rows, dtype=np.int64)
    labels = np.asarray(labels_full)[rows]
    if not np.array_equal(np.unique(labels), np.arange(8)):
        raise ValueError('Source head requires class IDs 0 through 7 in metadata order')
    state = dict(source_rows=rows.copy(), classes=np.arange(8),
                 raw_width=raw_full.shape[1], extra_width=extra_full.shape[1])
    state['pca'] = PCA(n_components=raw_rank, random_state=0).fit(
        np.asarray(raw_full[rows], dtype=np.float32))
    projected = _project(raw_full[rows], state['pca'])
    state['center'], state['scale'] = projected.mean(0), projected.std(0)+1e-6
    xs = (projected-state['center'])/state['scale']
    state['residual_center'] = xs.mean(0)
    state['residual_scale'] = xs.std(0)+1e-6
    state['pca_ssl'] = PCA(n_components=extra_rank, random_state=0).fit(
        np.asarray(extra_full[rows], dtype=np.float32))
    joint = np.c_[xs, _project(extra_full[rows], state['pca_ssl'])]
    state['expert_center'], state['expert_scale'] = joint.mean(0), joint.std(0)+1e-6
    expert = (joint-state['expert_center'])/state['expert_scale']
    state['base'] = _fit_head(_xs(c4_full[rows], state), labels)
    state['raw_base'] = _fit_head(xs, labels)
    state['single'] = _fit_head(expert[:, :raw_rank], labels)
    state['dual'] = _fit_head(expert, labels)
    raw_center, raw_scale = _discarded_stats(raw_full, rows, state['pca'], batch_size)
    extra_center, extra_scale = _discarded_stats(extra_full, rows, state['pca_ssl'], batch_size)
    state['discarded_center'] = np.r_[raw_center, extra_center]
    state['discarded_scale'] = np.r_[raw_scale, extra_scale]
    return state


def _parts(values, state, head):
    xs = _xs(values, state)
    normalized = (xs-state['residual_center'])/state['residual_scale']
    mean = normalized/np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-12)
    return dict(xs=xs, phi=np.c_[normalized/np.sqrt(xs.shape[1]), np.ones(len(xs))],
                mean=mean, prior=head.decision_function(xs))


def base_parts(c4_batch, state):
    """C4 xs, phi, unit mean and prior; no R18 inputs or state are consulted.

    xs is the input to RidgeClassifier for held-port source episode fitting.
    Its prior must be replaced by that episode's independently fitted head.
    """
    return _parts(c4_batch, state, state['base'])


def transform(raw_batch, c4_batch, extra_batch, state):
    """Transform one batch; single expert and raw/base parts never use extra.

    Single/dual experts contain half retained and half discarded kernels plus
    exactly one constant. Each uses its actual retained and raw feature widths.
    """
    base = base_parts(c4_batch, state)
    raw = _parts(raw_batch, state, state['raw_base'])
    rank = raw['xs'].shape[1]
    width = state['raw_width']
    retained = (raw['xs']-state['expert_center'][:rank])/state['expert_scale'][:rank]
    raw_discarded = (_discarded(raw_batch, state['pca'])-state['discarded_center'][:width])
    raw_discarded /= state['discarded_scale'][:width]
    extra_retained = (_project(extra_batch, state['pca_ssl'])-state['expert_center'][rank:])
    extra_retained /= state['expert_scale'][rank:]
    joint = np.c_[retained, extra_retained]
    extra_discarded = (_discarded(extra_batch, state['pca_ssl'])-state['discarded_center'][width:])
    extra_discarded /= state['discarded_scale'][width:]
    discarded = np.c_[raw_discarded, extra_discarded]
    bias = np.ones(len(raw_batch))
    return dict(phi=base['phi'], mean=base['mean'], prior=base['prior'],
                raw_phi=raw['phi'], raw_prior=raw['prior'],
                single_expert=np.c_[retained/np.sqrt(2*rank),
                                    raw_discarded/np.sqrt(2*width), bias],
                dual_expert=np.c_[joint/np.sqrt(2*joint.shape[1]),
                                  discarded/np.sqrt(2*discarded.shape[1]), bias],
                single_prior=state['single'].decision_function(retained),
                dual_prior=state['dual'].decision_function(joint))
