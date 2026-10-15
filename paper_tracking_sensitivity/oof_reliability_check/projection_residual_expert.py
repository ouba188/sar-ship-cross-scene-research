"""Support-fitted experts with an explicit source-PCA-complement branch."""

import numpy as np
from sklearn.linear_model import Ridge

from budgeted_residual_handoff import ranked_mask
from receiver_consistent_handoff import gate_scores
from run_opensarship2_transfer import target_geometry


def discarded_coordinates(x, pca):
    centered = np.asarray(x, dtype=float)-pca.mean_
    components = np.asarray(pca.components_, dtype=float)
    return centered-(centered@components.T)@components


def fit_geometry_statistics(base, extra, source_rows, model, batch_size=2048):
    """Fit coordinate scales on the already identity-purged source cohort."""
    dimension = base.shape[1]+extra.shape[1]
    sums = {name: np.zeros(dimension) for name in ('raw', 'discarded')}
    squares = {name: np.zeros(dimension) for name in sums}
    for lo in range(0, len(source_rows), batch_size):
        rows = source_rows[lo:lo+batch_size]
        b, e = np.asarray(base[rows], dtype=float), np.asarray(extra[rows], dtype=float)
        values = dict(raw=np.column_stack((b, e)), discarded=np.column_stack((
            discarded_coordinates(b, model['pca']),
            discarded_coordinates(e, model['pca_ssl']))))
        for name, value in values.items():
            sums[name] += value.sum(0)
            squares[name] += np.square(value).sum(0)
    stats = {}
    for name in sums:
        mean = sums[name]/len(source_rows)
        scale = np.sqrt(np.maximum(squares[name]/len(source_rows)-mean**2, 0.))+1e-6
        stats[name+'_center'], stats[name+'_scale'] = mean, scale
    return stats


def feature_views(base, extra, model, stats):
    """Bias-free views; the ridge kernel adds exactly one shared bias."""
    _, _, _, phi_e = target_geometry(model, base, extra)
    raw = np.column_stack((base, extra)).astype(float)
    discarded = np.column_stack((discarded_coordinates(base, model['pca']),
                                 discarded_coordinates(extra, model['pca_ssl'])))
    dimension = raw.shape[1]
    return dict(pca=phi_e[:, :-1],
        raw=(raw-stats['raw_center'])/stats['raw_scale']/np.sqrt(dimension),
        discarded=(discarded-stats['discarded_center'])/stats['discarded_scale']/np.sqrt(dimension))


def build_support_kernels(base, extra, model, stats, reference_rows, batch_size=2048):
    reference = feature_views(base[reference_rows], extra[reference_rows], model, stats)
    kernels = {name: np.empty((len(base), len(reference_rows))) for name in reference}
    for lo in range(0, len(base), batch_size):
        hi = min(lo+batch_size, len(base))
        views = feature_views(base[lo:hi], extra[lo:hi], model, stats)
        for name, value in views.items():
            kernels[name][lo:hi] = value@reference[name].T
    return kernels


def fit_source_discriminant(base, extra, source_rows, source_labels, model, stats, batch_size=2048):
    """Compress the PCA complement using source classification residuals only."""
    dimension = base.shape[1]+extra.shape[1]
    phi = np.ones((len(source_rows), dimension+1))
    prior = np.empty((len(source_rows), 8))
    for lo in range(0, len(source_rows), batch_size):
        hi = min(lo+batch_size, len(source_rows))
        rows = source_rows[lo:hi]
        b, e = base[rows], extra[rows]
        discarded = np.column_stack((discarded_coordinates(b, model['pca']),
                                     discarded_coordinates(e, model['pca_ssl'])))
        phi[lo:hi, :-1] = ((discarded-stats['discarded_center'])/
                          stats['discarded_scale']/np.sqrt(dimension))
        _, prior[lo:hi], _, _ = target_geometry(model, b, e)
    labels = np.asarray(source_labels)[source_rows]
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    weights = len(labels)/(len(counts)*counts[inverse])
    desired = 2*np.eye(8)[labels]-1.-prior
    # This is a supervised source representation, not source-OOF evaluation.
    fitted = Ridge(alpha=1., fit_intercept=False, solver='cholesky').fit(phi, desired, sample_weight=weights)
    coordinates = fitted.predict(phi)
    return dict(coef=fitted.coef_.T, center=coordinates.mean(0), scale=coordinates.std(0)+1e-6)


def source_discriminant_view(discarded_view, projection):
    values = discarded_view@projection['coef'][:-1]+projection['coef'][-1]
    return (values-projection['center'])/projection['scale']/np.sqrt(values.shape[1])


def build_guided_kernel(base, extra, model, stats, projection, support_rows, batch_size=2048):
    reference = feature_views(base[support_rows], extra[support_rows], model, stats)
    projected_reference = source_discriminant_view(reference['discarded'], projection)
    kernel = np.empty((len(base), len(support_rows)))
    for lo in range(0, len(base), batch_size):
        hi = min(lo+batch_size, len(base))
        views = feature_views(base[lo:hi], extra[lo:hi], model, stats)
        guided = source_discriminant_view(views['discarded'], projection)
        kernel[lo:hi] = .5*(views['pca']@reference['pca'].T)+.5*(guided@projected_reference.T)+1.
    return kernel


def fit_kernel_correction(kss, support_labels, support_prior, strength=8.):
    labels = np.asarray(support_labels)
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    root = np.sqrt(len(labels)/(len(counts)*counts[inverse]))
    desired = 2*np.eye(support_prior.shape[1])[labels]-1.
    weighted_kernel = root[:, None]*kss*root[None, :]
    dual = np.linalg.solve(weighted_kernel+strength*np.eye(len(labels)),
                           root[:, None]*(desired-support_prior))
    return root[:, None]*dual


def freeze_kernel_draw(base_scores, prior_expert, kernel_to_support, support_rows,
                       support_labels, gate, row_ids):
    """Accept support truth only; base and source expert prior stay frozen."""
    if len(support_rows) != 64 or len(np.unique(support_rows)) != 64 or len(support_labels) != 64:
        raise ValueError('requires exactly 64 unique support annotations')
    beta = fit_kernel_correction(kernel_to_support[support_rows], support_labels,
                                 prior_expert[support_rows])
    expert_scores = prior_expert+kernel_to_support@beta
    evaluation = ~np.isin(np.arange(len(row_ids)), support_rows)
    cap = int(np.floor(.05*evaluation.sum()))
    b, e = base_scores.argmax(1), expert_scores.argmax(1)
    score = gate_scores(base_scores, expert_scores, gate)
    observed = np.bincount(support_labels, minlength=prior_expert.shape[1]) > 0
    eligible = evaluation & (b != e) & observed[e] & (score > 0)
    selected = ranked_mask(score, eligible, row_ids, cap)
    return dict(expert_scores=expert_scores, expert_predictions=e,
        predictions=np.where(selected, e, b), selected=selected,
        eval_mask=evaluation, cap=cap, coefficients=beta, gate_scores=score)
