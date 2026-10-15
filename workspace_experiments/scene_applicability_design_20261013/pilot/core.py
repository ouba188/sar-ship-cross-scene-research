"""Fixed source-reference maps for a small scene-applicability experiment."""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

ARMS = ('original', 'class_mean', 'class_cluster', 'scene_memory')
RHO = .25


def build_memories(z, labels, ships, scenes, seed):
    rng = np.random.default_rng(seed)
    # One seeded observation per ship avoids weighting frequent revisits by count.
    chosen = np.array([rng.choice(np.flatnonzero(ships == s)) for s in np.unique(ships)])
    means, clusters, scene_modes, lab, mean_lab, coverage = [], [], [], [], [], []
    for c in range(8):
        idx = chosen[labels[chosen] == c]
        scene_centers = []
        for a in np.unique(scenes[labels == c]):
            aa = np.flatnonzero((labels == c) & (scenes == a))
            uu = np.array([rng.choice(aa[ships[aa] == s]) for s in np.unique(ships[aa])])
            scene_centers.append(z[uu].mean(0))
        sc = np.stack(scene_centers)
        k = min(8, len(idx), len(sc))
        means.append(z[idx].mean(0)); mean_lab.append(c)
        clusters.extend(KMeans(n_clusters=k, random_state=seed+c, n_init=5).fit(z[idx]).cluster_centers_)
        scene_modes.extend(KMeans(n_clusters=k, random_state=seed+c, n_init=5).fit(sc).cluster_centers_)
        lab.extend([c]*k)
        coverage.append(dict(class_id=c, ships=len(idx), scenes=len(sc), prototypes=k))
    # A common bandwidth comes from source-ship distances, not arm-specific outcomes.
    left = rng.choice(chosen, 2048, replace=True)
    right = rng.choice(chosen, 2048, replace=True)
    d2 = np.sum((z[left]-z[right])**2, axis=1)
    tau2 = float(np.median(d2[d2 > 0]))
    energy = float(np.mean(np.sum(z[chosen]**2, axis=1)+1))
    memories = {}
    for name, centers, classes in [('class_mean', means, mean_lab),
                                    ('class_cluster', clusters, lab),
                                    ('scene_memory', scene_modes, lab)]:
        centers = np.asarray(centers); classes = np.asarray(classes)
        counts = np.bincount(classes, minlength=8)[classes]
        raw = np.exp(-cdist(z[chosen], centers, 'sqeuclidean')/tau2)/np.sqrt(counts)
        scale = np.sqrt(energy/np.mean(np.sum(raw**2, axis=1)))
        memories[name] = dict(centers=centers, classes=classes, tau2=tau2, scale=float(scale))
    return memories, coverage, chosen


def relation(z, memory):
    counts = np.bincount(memory['classes'], minlength=8)[memory['classes']]
    return np.exp(-cdist(z, memory['centers'], 'sqeuclidean')/memory['tau2']) / np.sqrt(counts) * memory['scale']


def kernel(zq, zs, rq=None, rs=None, rho=RHO):
    original = zq@zs.T + 1.
    if rq is None:
        return original
    return (1-rho)*original + rho*(rq@rs.T)


def map_at_r(similarities, qlabels, glabels):
    values = np.full(len(qlabels), np.nan)
    for i, y in enumerate(qlabels):
        relevant = glabels == y
        r = int(relevant.sum())
        if not r:
            continue
        order = np.argsort(-similarities[i], kind='stable')[:r]
        hit = relevant[order]
        values[i] = np.sum(np.cumsum(hit)/np.arange(1, r+1)*hit)/r
    return values
