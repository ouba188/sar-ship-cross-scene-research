"""Support-only bounded CE control of the incremental frequency score vector."""

import numpy as np


def balanced_weights(labels):
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    return len(labels) / (len(counts) * counts[inverse])


def coefficients(gram, labels, prior):
    root = np.sqrt(balanced_weights(labels))
    desired = 2 * np.eye(prior.shape[1])[labels] - 1
    return root[:, None] * np.linalg.solve(
        root[:, None] * gram * root[None, :] + 8 * np.eye(len(labels)),
        root[:, None] * (desired - prior))


def kernels(data, embedding, support, eta):
    condition = np.exp(-np.sum((data['conditions'][:, None]
        - data['conditions'][support][None]) ** 2, axis=2) / .5)
    old = data['phi'] @ data['phi'][support].T
    old += eta * (data['mean'] @ data['mean'][support].T) * condition
    extra = .25 * (embedding @ embedding[support].T) * condition
    return old, old + extra


def strict_loo(old_gram, new_gram, prior, labels):
    """Each fit recomputes its class weights from the remaining 63 labels."""
    rows = np.arange(len(labels))
    old, new = np.empty_like(prior), np.empty_like(prior)
    for held in rows:
        keep = rows[rows != held]
        for gram, out in ((old_gram, old), (new_gram, new)):
            alpha = coefficients(gram[np.ix_(keep, keep)], labels[keep], prior[keep])
            out[held] = prior[held] + gram[held, keep] @ alpha
    return old, new


def balanced_ce(score, labels):
    return float(np.average(point_ce(score, labels), weights=balanced_weights(labels)))


def point_ce(score, labels):
    z = score - score.max(axis=1, keepdims=True)
    return np.log(np.exp(z).sum(axis=1)) - z[np.arange(len(labels)), labels]


def local_strength(similarity, utility):
    """Fixed top8 positive frequency-condition neighbors propagate LOO utility."""
    gamma = np.empty(len(similarity))
    for lo in range(0, len(similarity), 4096):
        s = np.maximum(similarity[lo:lo+4096], 0.)
        top = np.argsort(-s, axis=1, kind='stable')[:, :8]
        w = np.take_along_axis(s, top, axis=1)
        u = utility[top]
        numerator = np.sum(w * np.maximum(u, 0.), axis=1)
        denominator = np.sum(w * np.abs(u), axis=1)
        gamma[lo:lo+len(s)] = np.divide(numerator, denominator,
            out=np.zeros_like(numerator), where=denominator > 0.)
    return gamma


def balanced_accuracy(score, labels):
    correct = score.argmax(axis=1) == labels
    return 100 * float(np.average(correct, weights=balanced_weights(labels)))


def ce_strength(old, new, labels):
    """Convex one-dimensional optimization, no temperature/threshold tuning."""
    delta = new - old
    weights = balanced_weights(labels)

    def derivative(gamma):
        z = old + gamma * delta
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        d = (p * delta).sum(axis=1) - delta[np.arange(len(labels)), labels]
        return float(np.average(d, weights=weights))

    d0, d1 = derivative(0.), derivative(1.)
    if d0 == 0. and d1 == 0.:
        return 1.
    if d0 >= 0.:
        return 0.
    if d1 <= 0.:
        return 1.
    low, high = 0., 1.
    for _ in range(48):
        mid = (low + high) / 2
        if derivative(mid) > 0.:
            high = mid
        else:
            low = mid
    return (low + high) / 2


def select_rule(records):
    modes = ['learned', 'loo_ce']
    if 'local_ba' in records[0]:
        modes.append('local')
    means = {m: float(np.mean([r[m+'_ba'] for r in records])) for m in modes}
    rule = max(modes, key=lambda m: means[m])
    return rule, means[rule] - means['learned']
