"""Source-reference calibration and prior-anchored few-label inference."""
import numpy as np
from scipy.special import softmax
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def unit_response(z, centers, tau2):
    log_r = -cdist(z, centers, 'sqeuclidean') / tau2
    # The common radial factor cancels in L2 normalization; remove it before exp.
    r = np.exp(log_r-log_r.max(axis=1, keepdims=True))
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def reference_map(z, bank):
    return (unit_response(z, bank['centers'], bank['tau2'])-bank['center'])*bank['scale']


def cluster_bank(z, template, seed):
    km = KMeans(n_clusters=8, n_init=5, random_state=seed).fit(z)
    r = unit_response(z, km.cluster_centers_, template['tau2'])
    center = r.mean(0)
    scale = np.sqrt(np.mean(np.sum(z*z, axis=1)+1) /
                    np.mean(np.sum((r-center)**2, axis=1)))
    return dict(centers=km.cluster_centers_, tau2=template['tau2'],
                center=center, scale=scale, seed=seed), km.labels_


class ReferenceSupportModel:
    """Complete cached-feature reader; no query labels or batch fitting."""

    def __init__(self, coefficient_solver, bank=None, rho=0., ridge=8., weight=.5):
        self.solve = coefficient_solver
        self.bank = bank
        self.rho = rho
        self.ridge = ridge
        self.weight = weight

    def kernel(self, left, right):
        k = (1-self.rho) * (left['z'] @ right['z'].T + 1)
        if self.bank is not None:
            k = k + self.rho * (reference_map(left['z'], self.bank) @
                                reference_map(right['z'], self.bank).T)
        return k

    def fit_support(self, support, labels):
        self.support = support
        kb = support['b'] @ support['b'].T + 1
        self.base_coef = self.solve(kb, support['pb'], labels, self.ridge)
        self.coef = self.solve(self.kernel(support, support), support['pd'],
                               labels, self.ridge)
        return self

    def predict(self, query):
        kb = query['b'] @ self.support['b'].T + 1
        k = self.kernel(query, self.support)
        base = softmax(query['pb'] + kb @ self.base_coef, axis=1)
        scores = query['pd'] + k @ self.coef
        expert = softmax(scores, axis=1)
        return dict(base=base, expert=expert, scores=scores, kernel=k,
                    full=(1-self.weight)*base+self.weight*expert)
