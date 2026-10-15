"""Frozen pre-repair reader for reproducing the initial experiment."""
import numpy as np
from scipy.special import softmax
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def reference_map(z, bank):
    r = np.exp(-cdist(z, bank['centers'], 'sqeuclidean') / bank['tau2'])
    r /= np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-15)
    return (r - bank['center']) * bank['scale']


def cluster_bank(z, template, seed):
    km = KMeans(n_clusters=8, n_init=5, random_state=seed).fit(z)
    r = np.exp(-cdist(z, km.cluster_centers_, 'sqeuclidean') / template['tau2'])
    r /= np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-15)
    center = r.mean(0)
    scale = np.sqrt(np.mean(np.sum(z*z, axis=1)+1) /
                    np.mean(np.sum((r-center)**2, axis=1)))
    return dict(centers=km.cluster_centers_, tau2=template['tau2'],
                center=center, scale=scale, seed=seed), km.labels_


class ReferenceSupportModel:
    def __init__(self, coefficient_solver, bank=None, rho=0., ridge=8., weight=.5):
        self.solve=coefficient_solver
        self.bank=bank
        self.rho=rho
        self.ridge=ridge
        self.weight=weight

    def kernel(self, left, right):
        k=(1-self.rho)*(left['z']@right['z'].T+1)
        if self.bank is not None:
            k=k+self.rho*(reference_map(left['z'],self.bank)@reference_map(right['z'],self.bank).T)
        return k

    def fit_support(self, support, labels):
        self.support=support
        self.base_coef=self.solve(support['b']@support['b'].T+1,support['pb'],labels,self.ridge)
        self.coef=self.solve(self.kernel(support,support),support['pd'],labels,self.ridge)
        return self

    def predict(self, query):
        kb=query['b']@self.support['b'].T+1
        k=self.kernel(query,self.support)
        base=softmax(query['pb']+kb@self.base_coef,axis=1)
        scores=query['pd']+k@self.coef
        expert=softmax(scores,axis=1)
        return dict(base=base,expert=expert,scores=scores,kernel=k,
                    full=(1-self.weight)*base+self.weight*expert)
