"""Classwise residual utility measured after the actual probability fusion."""

import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedGroupKFold
import warnings

SEEDS = (20261012, 20261013, 20261014)
TAUS = (.003, .01, .03)
MODES = ('global', 'self', 'joint')


def class_weights(y):
    _, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    return 1./(len(counts)*counts[inverse])


def loss_grad(gamma, prior, residual, base_probability, y, tau=0.):
    pe = softmax(prior+residual*gamma, axis=1)
    error = .5*(base_probability+pe)-np.eye(prior.shape[1])[y]
    w = class_weights(y)
    risk = np.sum(w*np.square(error).sum(1))
    logit_grad = pe*(error-np.sum(error*pe, axis=1, keepdims=True))
    grad = np.sum(w[:, None]*logit_grad*residual, axis=0)
    return (risk+tau*np.mean(np.square(gamma-1)),
            grad+2*tau*(gamma-1)/len(gamma))


def fit_gate(prior, residual, base_probability, y, tau, mode):
    width = prior.shape[1]
    records = []
    def solve(fun, dimension):
        result = minimize(fun, np.ones(dimension), jac=True, method='L-BFGS-B',
                          bounds=[(0.,2.)]*dimension,
                          options=dict(maxiter=100, ftol=1e-12, gtol=1e-8))
        records.append(dict(success=bool(result.success), iterations=int(result.nit), message=str(result.message)))
        if not result.success:
            raise RuntimeError(str(result.message))
        return result.x
    if mode=='joint':
        gate = solve(lambda g:loss_grad(g, prior, residual, base_probability, y, tau), width)
    elif mode=='global':
        def objective(value):
            loss, grad = loss_grad(np.repeat(value, width), prior, residual, base_probability, y, tau)
            return loss, np.array([grad.sum()])
        gate = np.repeat(solve(objective,1), width)
    elif mode=='self':
        gate = np.ones(width)
        for cls in np.unique(y):
            mask = y==cls
            def objective(value):
                g = np.ones(width)
                g[cls] = value[0]
                loss, grad = loss_grad(g, prior[mask], residual[mask], base_probability[mask], y[mask], 0.)
                return loss+tau*(value[0]-1)**2, np.array([grad[cls]+2*tau*(value[0]-1)])
            gate[cls] = solve(objective,1)[0]
    else:
        raise ValueError(mode)
    return gate, records


def support_oof(ke, prior, kb, base_prior, y, groups, coefficients):
    n, width = prior.shape
    outputs = np.full((len(SEEDS),3,n,width), np.nan)
    ids, notes = [], []
    for repeat, seed in enumerate(SEEDS):
        if len(np.unique(groups))<4:
            raise ValueError('Fewer than four support identities: fixed grouped protocol not feasible')
        splitter = StratifiedGroupKFold(4,shuffle=True,random_state=seed)
        with warnings.catch_warnings(record=True) as captured:
            parts = list(splitter.split(np.zeros((n,1)),y,groups))
        notes.extend(str(v.message) for v in captured)
        seen = np.zeros(n,int)
        for fold,(train,val) in enumerate(parts):
            assert not set(groups[train]) & set(groups[val])
            strength = len(train)/8.
            ae = coefficients(ke[np.ix_(train,train)],prior[train],y[train],strength)
            ab = coefficients(kb[np.ix_(train,train)],base_prior[train],y[train],strength)
            outputs[repeat,0,val] = prior[val]
            outputs[repeat,1,val] = ke[np.ix_(val,train)]@ae
            outputs[repeat,2,val] = softmax(base_prior[val]+kb[np.ix_(val,train)]@ab,axis=1)
            seen[val] += 1
            ids.append(dict(repeat=repeat,fold=fold,train=train.tolist(),validation=val.tolist()))
        np.testing.assert_array_equal(seen,np.ones(n,int))
    assert np.isfinite(outputs).all()
    return tuple(outputs[:,i].reshape(-1,width) for i in range(3)), np.tile(y,len(SEEDS)), ids, notes
