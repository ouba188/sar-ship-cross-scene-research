"""Small source-trained mixtures of two frozen eight-class predictors."""

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, softmax

FEATURES = ('base_margin', 'expert_margin', 'base_entropy', 'expert_entropy', 'disagreement')
GRID = tuple(sorted(np.linspace(0., 1., 11), key=lambda w: (abs(w-.5), w)))
PENALTY = .01


def probabilities(base, expert):
    return softmax(base, axis=1), softmax(expert, axis=1)


def features(pb, pe):
    values = []
    for p in (pb, pe):
        top = np.sort(p, axis=1)[:, -2:]
        values.append(top[:, 1]-top[:, 0])
    for p in (pb, pe):
        values.append(-np.sum(p*np.log(np.maximum(p, 1e-15)), axis=1)/np.log(p.shape[1]))
    values.append((pb.argmax(1) != pe.argmax(1)).astype(float))
    return np.column_stack(values)


def balanced_weights(y, episode):
    """Each source episode, then each present class, receives equal total mass."""
    out = np.zeros(len(y), dtype=float)
    groups = np.unique(episode)
    for group in groups:
        ix = np.flatnonzero(episode == group)
        _, inv, count = np.unique(y[ix], return_inverse=True, return_counts=True)
        out[ix] = 1./(len(groups)*len(count)*count[inv])
    return out


def mix(pb, pe, w):
    w = np.asarray(w)
    if w.ndim:
        w = w[:, None]
    return (1-w)*pb+w*pe


def select_fixed(pb, pe, y, sample_weight):
    accuracy = [float(sample_weight @ (mix(pb, pe, w).argmax(1) == y)) for w in GRID]
    return float(GRID[int(np.argmax(accuracy))]), accuracy


def constant_nll(pb, pe, y, sample_weight):
    i = np.arange(len(y))
    b, e = pb[i, y], pe[i, y]
    def objective(w):
        return float(-sample_weight @ np.log(np.maximum((1-w)*b+w*e, 1e-15)))
    result = minimize_scalar(objective, bounds=(0., 1.), method='bounded', options={'xatol': 1e-10})
    candidates = (float(result.x), 0., 1.)
    return min(candidates, key=objective)


def objective(theta, design, b, e, sample_weight, penalty=PENALTY):
    w = expit(design @ theta)
    p = np.maximum(b+w*(e-b), 1e-15)
    value = -sample_weight @ np.log(p)+.5*penalty*np.sum(theta[1:]**2)
    derivative = -sample_weight*(e-b)/p*w*(1-w)
    gradient = design.T @ derivative
    gradient[1:] += penalty*theta[1:]
    return float(value), gradient


def fit_nll(pb, pe, y, episode):
    weight = balanced_weights(y, episode)
    x = features(pb, pe)
    center = weight @ x
    scale = np.sqrt(weight @ ((x-center)**2))
    scale[scale < 1e-6] = 1.
    design = np.column_stack((np.ones(len(x)), (x-center)/scale))
    constant = constant_nll(pb, pe, y, weight)
    initial = np.zeros(design.shape[1])
    c = np.clip(constant, 1e-3, 1-1e-3)
    initial[0] = np.log(c/(1-c))
    i = np.arange(len(y))
    args = (design, pb[i, y], pe[i, y], weight)
    result = minimize(objective, initial, args=args, method='L-BFGS-B', jac=True,
                      bounds=[(-8., 8.)]*len(initial),
                      options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8})
    if not result.success:
        raise RuntimeError('Mixture fitting failed: '+str(result.message))
    fixed, accuracies = select_fixed(pb, pe, y, weight)
    return dict(theta=result.x.tolist(), center=center.tolist(), scale=scale.tolist(),
                fixed_weight=fixed, constant_nll_weight=constant,
                source_grid_ba=(100*np.array(accuracies)).tolist(),
                initial_objective=objective(initial, *args)[0], final_objective=float(result.fun),
                iterations=int(result.nit), convergence=str(result.message),
                penalty=PENALTY, features=FEATURES)


def utility_objective(theta, design, target, weight, penalty=PENALTY):
    logit = design @ theta
    value = weight @ (np.logaddexp(0., logit)-target*logit)+.5*penalty*np.sum(theta[1:]**2)
    gradient = design.T @ (weight*(expit(logit)-target))
    gradient[1:] += penalty*theta[1:]
    return float(value), gradient


def fit(pb, pe, y, episode):
    model = fit_nll(pb, pe, y, episode)
    model['nll_theta'] = model['theta']
    model['nll_initial_objective'] = model['initial_objective']
    model['nll_final_objective'] = model['final_objective']
    correct_b, correct_e = pb.argmax(1) == y, pe.argmax(1) == y
    informative = correct_b != correct_e
    if not informative.any():
        raise ValueError('No exclusive correctness events in source fusion fit')
    weight = balanced_weights(y, episode)[informative]
    weight /= weight.sum()
    x = (features(pb, pe)-np.asarray(model['center']))/np.asarray(model['scale'])
    design = np.column_stack((np.ones(len(x)), x))[informative]
    target = correct_e[informative].astype(float)
    constant = float(weight @ target)
    initial = np.zeros(design.shape[1])
    clipped = np.clip(constant, 1e-3, 1-1e-3)
    initial[0] = np.log(clipped/(1-clipped))
    args = (design, target, weight)
    result = minimize(utility_objective, initial, args=args, jac=True, method='L-BFGS-B',
                      bounds=[(-8., 8.)]*len(initial),
                      options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8})
    if not result.success:
        raise RuntimeError('Relative correctness fitting failed: '+str(result.message))
    model.update(theta=result.x.tolist(), utility_constant=constant,
                 exclusive_events=int(informative.sum()),
                 initial_objective=utility_objective(initial, *args)[0],
                 final_objective=float(result.fun), iterations=int(result.nit),
                 convergence=str(result.message),
                 objective='Weighted binary log loss of expert-only vs base-only correctness')
    return model


def adaptive_weight(model, pb, pe):
    x = (features(pb, pe)-np.asarray(model['center']))/np.asarray(model['scale'])
    theta = np.asarray(model['theta'])
    return expit(theta[0]+x @ theta[1:])
