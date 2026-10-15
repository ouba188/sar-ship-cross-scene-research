"""Source-fitted directional score mixtures for eight aligned classes.

Scores are logits, labels are integer IDs 0..7, and context columns are
inc_t, inc_t2, inc_known, relative_known, cos2relative, sin2relative.
Context missing entries (including nonfinite entries) become zero before
source-only standardization. Source-constant features stay disabled at query.
Models contain JSON-compatible parameters and optimizer diagnostics, not data.
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logsumexp


K = 8
QUALITY_NAMES = ('original_margin', 'c4_margin', 'centered_delta_rms', 'disagree')
CONTEXT_NAMES = ('inc_t', 'inc_t2', 'inc_known', 'relative_known',
                 'cos2relative', 'sin2relative')


def _scores(original, c4):
    original, c4 = np.asarray(original, dtype=np.float64), np.asarray(c4, dtype=np.float64)
    if original.ndim != 2 or original.shape[1] != K or c4.shape != original.shape:
        raise ValueError('original and c4 must both have shape (N, 8)')
    if not np.isfinite(original).all() or not np.isfinite(c4).all():
        raise ValueError('branch scores must be finite')
    return original, c4


def quality_features(original, c4):
    """Return float64 (N, 4) margins, centered-difference RMS, disagreement."""
    original, c4 = _scores(original, c4)
    original_top = np.partition(original, -2, axis=1)[:, -2:]
    c4_top = np.partition(c4, -2, axis=1)[:, -2:]
    centered = original - c4
    centered -= centered.mean(axis=1, keepdims=True)
    return np.column_stack((original_top[:, 1] - original_top[:, 0],
                            c4_top[:, 1] - c4_top[:, 0],
                            np.sqrt(np.mean(centered ** 2, axis=1)),
                            (original.argmax(1) != c4.argmax(1)).astype(np.float64)))


def _context(context, n):
    if context is None:
        return np.zeros((n, 6), dtype=np.float64)
    context = np.asarray(context, dtype=np.float64)
    if context.shape != (n, 6):
        raise ValueError('context must have shape (N, 6) or be None')
    return np.nan_to_num(context, nan=0., posinf=0., neginf=0.)


def _fit_normalization(features):
    # Unweighted source moments; loss weights do not change feature units.
    center = features.mean(axis=0)
    std = features.std(axis=0)
    active = std > 1e-6
    return dict(feature_center=center.tolist(),
                feature_scale=np.where(active, std, 1.).tolist(),
                feature_active=active.tolist())


def _standardize(features, model):
    return ((features - np.asarray(model['feature_center']))
            / np.asarray(model['feature_scale'])
            * np.asarray(model['feature_active']))


def _loss_gradient(scores, labels, weights):
    """Weighted CE and score gradient; weights here already sum to one."""
    shifted = scores - scores.max(axis=1, keepdims=True)
    log_normalizer = logsumexp(shifted, axis=1)
    rows = np.arange(len(labels))
    loss = np.dot(weights, log_normalizer - shifted[rows, labels])
    gradient = np.exp(shifted - log_normalizer[:, None])
    gradient[rows, labels] -= 1.
    gradient *= weights[:, None]
    return float(loss), gradient


def _class_objective(alpha, c4, delta, labels, weights, global_alpha):
    loss, gradient = _loss_gradient(c4 + alpha * delta, labels, weights)
    deviation = alpha - global_alpha
    return (loss + .01 * np.mean(deviation ** 2),
            np.sum(gradient * delta, axis=0) + .02 * deviation / K)


def _bias_objective(bias, c4, labels, weights):
    loss, gradient = _loss_gradient(c4 + bias, labels, weights)
    return loss + .01 * np.mean(bias ** 2), gradient.sum(0) + .02 * bias / K


def _sigmoid_objective(parameters, x, c4, delta, labels, weights, anchor, shared):
    groups = 1 if shared else K
    intercepts = parameters[:groups]
    slopes = parameters[groups:].reshape(groups, x.shape[1])
    alpha = expit(x @ slopes.T + intercepts)
    loss, score_gradient = _loss_gradient(c4 + alpha * delta, labels, weights)
    gate_gradient = score_gradient * delta * alpha * (1. - alpha)
    if shared:
        gate_gradient = gate_gradient.sum(axis=1, keepdims=True)
    deviation = intercepts - anchor
    loss += .01 * np.mean(slopes ** 2) + .001 * np.mean(deviation ** 2)
    intercept_gradient = gate_gradient.sum(0) + .002 * deviation / groups
    slope_gradient = gate_gradient.T @ x + .02 * slopes / slopes.size
    return float(loss), np.r_[intercept_gradient, slope_gradient.ravel()]


def _diagnostics(result):
    return dict(success=bool(result.success), status=int(result.status),
                message=str(result.message), nit=int(result.nit),
                training_objective=float(result.fun))


def fit_fusions(original, c4, labels, weights, context):
    """Fit six models on source held-port predictions, with normalized weights.

    Model keys: global_mix, class_mix, quality_mix, class_quality_mix,
    class_geometry_mix, class_bias. Each stores kind, success/status/message,
    nit, training_objective (including penalties), training_logloss, and
    feature normalization lists (empty for models without feature inputs).
    Sigmoid models store intercepts and slopes (groups x features); mixtures
    store alpha (scalar or length eight), and class_bias stores eight biases.
    L-BFGS-B uses analytic gradients, maxiter=150, and O(N*(8+10)) memory.
    """
    original, c4 = _scores(original, c4)
    n = len(original)
    labels = np.asarray(labels)
    weights = np.asarray(weights, dtype=np.float64)
    if (n == 0 or labels.shape != (n,)
            or not np.issubdtype(labels.dtype, np.integer)
            or np.any((labels < 0) | (labels >= K))):
        raise ValueError('labels must be nonempty integer class IDs in [0, 8)')
    if (weights.shape != (n,) or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.any(weights > 0)):
        raise ValueError('weights must be finite, nonnegative, with positive total')
    weights = weights / weights.max()
    weights /= weights.sum()
    context = _context(context, n)
    delta = original - c4
    quality = quality_features(original, c4)

    def global_objective(alpha):
        return _loss_gradient(c4 + alpha * delta, labels, weights)[0]

    result = minimize_scalar(global_objective, bounds=(0., 1.), method='bounded',
                             options=dict(maxiter=150, xatol=1e-8))
    # A bounded scalar search approaches, but does not evaluate, the endpoints.
    global_alpha, global_loss = min(
        ((float(result.x), float(result.fun)), (0., global_objective(0.)),
         (1., global_objective(1.))), key=lambda item: item[1])
    models = {'global_mix': dict(kind='global_mix', alpha=global_alpha,
                                 **_diagnostics(result))}
    models['global_mix']['training_objective'] = global_loss
    clipped = np.clip(global_alpha, .001, .999)
    anchor = float(np.log(clipped / (1. - clipped)))
    options = dict(maxiter=150, ftol=1e-10, gtol=1e-7)

    result = minimize(_class_objective, np.full(K, global_alpha),
                      args=(c4, delta, labels, weights, global_alpha),
                      method='L-BFGS-B', jac=True, bounds=[(0., 1.)] * K,
                      options=options)
    models['class_mix'] = dict(kind='class_mix', alpha=result.x.tolist(),
                               global_alpha=global_alpha, **_diagnostics(result))

    for kind in ('quality_mix', 'class_quality_mix', 'class_geometry_mix'):
        geometry = kind == 'class_geometry_mix'
        features = np.column_stack((quality, context)) if geometry else quality
        normalization = _fit_normalization(features)
        x = _standardize(features, normalization)
        shared = kind == 'quality_mix'
        groups = 1 if shared else K
        initial = np.r_[np.full(groups, anchor), np.zeros(groups * x.shape[1])]
        result = minimize(_sigmoid_objective, initial,
                          args=(x, c4, delta, labels, weights, anchor, shared),
                          method='L-BFGS-B', jac=True, options=options)
        models[kind] = dict(kind=kind, global_alpha=global_alpha,
                            logit_global=anchor,
                            intercepts=result.x[:groups].tolist(),
                            slopes=result.x[groups:].reshape(groups, -1).tolist(),
                            feature_names=list(QUALITY_NAMES + CONTEXT_NAMES
                                               if geometry else QUALITY_NAMES),
                            **normalization, **_diagnostics(result))

    result = minimize(_bias_objective, np.zeros(K), args=(c4, labels, weights),
                      method='L-BFGS-B', jac=True, bounds=[(-.5, .5)] * K,
                      options=options)
    models['class_bias'] = dict(kind='class_bias', bias=result.x.tolist(),
                                **_diagnostics(result))
    for model in models.values():
        for key in ('feature_names', 'feature_center', 'feature_scale', 'feature_active'):
            model.setdefault(key, [])
        scores, _ = predict_fusion(model, original, c4, context)
        model['training_logloss'] = _loss_gradient(scores, labels, weights)[0]
    return models


def predict_fusion(model, original, c4, context):
    """Return float64 scores (N, 8), alpha (N, 8); bias returns alpha (N, 0).

    This interface never accepts labels or updates feature normalization.
    None context means all six raw context features are missing (zero).
    """
    original, c4 = _scores(original, c4)
    kind = model['kind']
    if kind == 'class_bias':
        return c4 + np.asarray(model['bias']), np.empty((len(c4), 0), dtype=np.float64)
    if kind in ('global_mix', 'class_mix'):
        alpha = np.broadcast_to(np.asarray(model['alpha']), c4.shape).copy()
    elif kind in ('quality_mix', 'class_quality_mix', 'class_geometry_mix'):
        features = quality_features(original, c4)
        if kind == 'class_geometry_mix':
            features = np.column_stack((features, _context(context, len(c4))))
        x = _standardize(features, model)
        alpha = expit(x @ np.asarray(model['slopes']).T + np.asarray(model['intercepts']))
        alpha = np.broadcast_to(alpha, c4.shape).copy()
    else:
        raise ValueError('unknown fusion kind: ' + str(kind))
    return c4 + alpha * (original - c4), alpha
