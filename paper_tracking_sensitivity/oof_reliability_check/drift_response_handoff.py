"""Class-balanced net-benefit routing from paired latent perturbation responses."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from receiver_consistent_handoff import confidence_parts


FEATURE_NAMES = ('expert_margin', 'base_margin', 'expert_entropy', 'base_entropy',
                 'base_flip', 'expert_flip', 'base_to_expert', 'expert_to_base',
                 'perturbed_agreement', 'base_relative_margin_sd', 'expert_relative_margin_sd')


def confidence_features(base, expert):
    bm, bh = confidence_parts(base)
    em, eh = confidence_parts(expert)
    return np.column_stack((em,bm,eh,bh))


def response_features(base, expert, base_gradient, expert_gradient, raw_scale, factors):
    """Perturb query R50 only; hold supports, R18 and adapted coefficients fixed.

    Factors in source-standardized coordinates satisfy F F.T = truncated
    source class-conditional port-mean covariance. Symmetric sqrt(rank) axes
    have this covariance exactly; they are not observed or physical images.
    """
    base, expert = np.asarray(base,dtype=float), np.asarray(expert,dtype=float)
    factors = np.asarray(factors,dtype=float)
    rank = factors.shape[1]
    basic = confidence_features(base,expert)
    if rank == 0:
        extras = np.zeros((len(base),7))
        extras[:,4] = base.argmax(1)==expert.argmax(1)
        return np.column_stack((basic,extras))
    raw_factors = np.asarray(raw_scale)[:,None]*factors
    db = raw_factors.T@np.asarray(base_gradient,dtype=float)
    de = raw_factors.T@np.asarray(expert_gradient,dtype=float)
    offsets_b = np.concatenate((db,-db))*np.sqrt(rank)
    offsets_e = np.concatenate((de,-de))*np.sqrt(rank)
    bp, ep = base.argmax(1), expert.argmax(1)
    rows = np.arange(len(base))
    bs = np.argsort(base,axis=1)[:,-2]
    es = np.argsort(expert,axis=1)[:,-2]
    bsd = np.linalg.norm(db[:,bp]-db[:,bs],axis=0)
    esd = np.linalg.norm(de[:,ep]-de[:,es],axis=0)
    bmargin, emargin = base[rows,bp]-base[rows,bs], expert[rows,ep]-expert[rows,es]
    extras = np.empty((len(base),7))
    for lo in range(0,len(base),512):
        hi = min(lo+512,len(base))
        pb = (base[lo:hi,None,:]+offsets_b[None,:,:]).argmax(2)
        pe = (expert[lo:hi,None,:]+offsets_e[None,:,:]).argmax(2)
        extras[lo:hi,:5] = np.column_stack(((pb!=bp[lo:hi,None]).mean(1),
            (pe!=ep[lo:hi,None]).mean(1), (pb==ep[lo:hi,None]).mean(1),
            (pe==bp[lo:hi,None]).mean(1), (pb==pe).mean(1)))
    extras[:,5] = np.log1p(bsd/(bmargin+1e-6))
    extras[:,6] = np.log1p(esd/(emargin+1e-6))
    return np.column_stack((basic,extras))


def fit_net_gate(features, base, expert, labels, eligible, weights):
    """Keep neutral outcomes: rank P(rescue)-P(harm), not rescue|nonneutral."""
    use = np.asarray(eligible,dtype=bool)
    x = np.asarray(features,dtype=float)[use]
    utility = ((np.asarray(expert).argmax(1)==labels).astype(int)-
               (np.asarray(base).argmax(1)==labels).astype(int))[use]
    weights = np.asarray(weights,dtype=float)[use]
    if not len(x):
        return dict(constant=0., rows=0)
    weights = weights/weights.sum()*len(weights)
    center = np.average(x,axis=0,weights=weights)
    scale = np.maximum(np.sqrt(np.average((x-center)**2,axis=0,weights=weights)),1e-6)
    counts = {str(c):int(np.sum(utility==c)) for c in (-1,0,1)}
    if len(np.unique(utility))==1:
        return dict(constant=float(utility[0]), rows=len(x), counts=counts)
    lr = LogisticRegression(C=1.,max_iter=1000,tol=1e-7,solver='lbfgs')
    lr.fit((x-center)/scale,utility,sample_weight=weights)
    return dict(estimator=lr,center=center,scale=scale,rows=len(x),counts=counts,
                feature_sd=np.std(x,axis=0))


def predict_net_gate(model, features):
    if 'constant' in model:
        return np.full(len(features),model['constant'])
    probabilities = model['estimator'].predict_proba((features-model['center'])/model['scale'])
    return probabilities@model['estimator'].classes_.astype(float)
