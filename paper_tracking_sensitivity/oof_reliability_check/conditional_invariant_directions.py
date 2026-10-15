"""Penalize source within-class domain means, not distributional invariance."""

import numpy as np


def nuisance_penalized_basis(x, y, domains, pairs, pooled_basis,
                             min_count=5, strength=1.0):
    """Shrink class contrasts along within-class source-domain mean variation.

    Features must already use source-fitted standardization. Classes contribute
    equally; eligible domains within each class contribute equally regardless
    of sample counts. At least three sufficiently populated domains are needed
    per class. All eligible source classes contribute, not just classes in pairs.

    Solve (I + strength * U.T @ U) W = pooled_basis, then L2-normalize
    columns. retained_norm records each column's norm before normalization.
    This constrains first moments only and identifies no causal invariants.
    """
    x, y, domains = np.asarray(x), np.asarray(y), np.asarray(domains)
    pairs = np.asarray(pairs)
    pooled = np.asarray(pooled_basis, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or domains.shape != (len(x),):
        raise ValueError('Expected features (N,D) and source labels/domains (N,)')
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pooled.shape != (x.shape[1], len(pairs)):
        raise ValueError('Expected pairs (P,2) and pooled_basis (D,P)')
    if min_count < 1 or not np.isfinite(strength) or strength < 0:
        raise ValueError('Need positive min_count and finite nonnegative strength')
    if not np.all(np.isfinite(pooled)) or np.any(np.linalg.norm(pooled, axis=0) == 0):
        raise ValueError('Pooled directions must be finite and nonzero')

    class_cells = []
    for label in np.unique(y):
        rows = np.flatnonzero(y == label)
        class_domains = domains[rows]
        means, names = [], []
        for domain, count in zip(*np.unique(class_domains, return_counts=True)):
            if count < min_count:
                continue
            cell_rows = rows[class_domains == domain]
            total = np.zeros(x.shape[1], dtype=np.float64)
            # Keep only a bounded feature block in memory, not a float64 bank.
            for lo in range(0, len(cell_rows), 1024):
                total += x[cell_rows[lo:lo+1024]].sum(axis=0, dtype=np.float64)
            means.append(total/count)
            names.append(str(domain))
        if len(means) >= 3:
            class_cells.append((label, np.stack(means), names))

    nuisance, cell_classes, cell_domains = [], [], []
    for label, means, names in class_cells:
        nuisance.append((means-means.mean(axis=0))/
                        np.sqrt(len(class_cells)*len(means)))
        cell_classes.extend([label]*len(means))
        cell_domains.extend(names)
    u = (np.concatenate(nuisance, axis=0) if nuisance else
         np.empty((0, x.shape[1]), dtype=np.float64))

    # Woodbury solves only over class-domain cells (at most 8 x 24 here).
    w = pooled.copy()
    if len(u) and strength:
        system = np.eye(len(u))+strength*(u@u.T)
        w -= strength*u.T@np.linalg.solve(system, u@pooled)
    norms = np.linalg.norm(w, axis=0)
    if np.any(norms == 0) or not np.all(np.isfinite(norms)):
        raise ValueError('Degenerate numerical solution cannot be normalized')
    return dict(basis=(w/norms).astype(np.float32),
                nuisance_rows=u.astype(np.float32),
                cell_classes=np.asarray(cell_classes, dtype=np.int64),
                cell_domains=np.asarray(cell_domains, dtype=str),
                retained_norm=norms)
