"""Frozen-feature local baselines; see LOCAL_METHODS_ZH.md for conventions.

Inputs: query [N,2,20,D], query_valid [N,2,20], bank [B,2,20,D],
bank_valid [B,2,20], bank_y [B] with integer class IDs 0..C-1.
Returns scores [N,C], or (scores, available [N,C]) when requested.
NumPy query -> NumPy output; torch query -> detached torch output on its device.
"""
from functools import lru_cache

import numpy as np
import torch

REGIONS = ((0, 8), (8, 16), (16, 20))


def _numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _prepare(query, query_valid, bank, bank_valid, bank_y, query_ids,
             bank_ids, num_classes, batch_size):
    device = query.device if isinstance(query, torch.Tensor) else torch.device('cpu')
    dtype = torch.float64 if query.dtype in (torch.float64, np.dtype('float64')) else torch.float32
    q = torch.as_tensor(query, dtype=dtype, device=device).detach()
    b = torch.as_tensor(bank, dtype=dtype, device=device).detach()
    if q.ndim != 4 or b.ndim != 4 or q.shape[1:3] != (2, 20) or b.shape[1:] != q.shape[1:]:
        raise ValueError('query and bank must be [N/B,2,20,D] with the same D')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')

    def clean(x, valid):
        valid = torch.as_tensor(valid, dtype=torch.bool, device=device)
        if valid.shape != x.shape[:-1]:
            raise ValueError('valid must have shape [N/B,2,20]; use mass > 0')
        norm = torch.linalg.vector_norm(x, dim=-1)
        valid = valid & torch.isfinite(x).all(-1) & torch.isfinite(norm) & (norm > 1e-8)
        x = torch.where(valid[..., None], x, 0)
        return x / torch.where(valid, norm, 1)[..., None], valid

    q, qv = clean(q, query_valid)
    b, bv = clean(b, bank_valid)
    y = _numpy(bank_y)
    if y.shape != (len(b),) or not np.issubdtype(y.dtype, np.integer) or np.any(y < 0):
        raise ValueError('bank_y must be a nonnegative integer vector of length B')
    c = int(y.max()) + 1 if len(y) else 0
    if num_classes is not None:
        if num_classes < c or num_classes < 1:
            raise ValueError('num_classes must cover bank_y and be positive')
        c = int(num_classes)
    if not c:
        raise ValueError('empty bank requires num_classes')
    if (query_ids is None) != (bank_ids is None):
        raise ValueError('query_ids and bank_ids must be supplied together')
    eligible = torch.ones((len(q), len(b)), dtype=torch.bool, device=device)
    if query_ids is not None:
        qi, bi = _numpy(query_ids), _numpy(bank_ids)
        if qi.shape != (len(q),) or bi.shape != (len(b),):
            raise ValueError('IDs must have one entry per query/reference image')
        eligible = torch.as_tensor(qi[:, None] != bi[None, :], device=device)
    return q, qv, b, bv, torch.as_tensor(y, device=device), eligible, c


def _finish(query, scores, available, return_available):
    if not isinstance(query, torch.Tensor):
        scores, available = _numpy(scores), _numpy(available)
    return (scores, available) if return_available else scores


@torch.inference_mode()
def dn4_scores(query, query_valid, bank, bank_valid, bank_y, *, k=3,
               query_ids=None, bank_ids=None, num_classes=None, batch_size=128,
               return_available=False):
    """Class-pool top-k cosine SUM, mean over valid query tokens and polarizations.

    All references participate. If fewer than k valid bank tokens remain, sum
    the available neighbors without padding or rescaling. No evidence -> zero.
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError('k must be a positive integer')
    q, qv, b, bv, y, eligible, c = _prepare(
        query, query_valid, bank, bank_valid, bank_y, query_ids, bank_ids, num_classes, batch_size)
    scores = q.new_zeros((len(q), c))
    available = torch.zeros_like(scores, dtype=torch.bool)
    for cls in range(c):
        select = y == cls
        if not select.any():
            continue
        for start in range(0, len(q), batch_size):
            sl = slice(start, start + batch_size)
            polar, has = [], []
            for p in range(2):
                support = b[select, p].reshape(-1, b.shape[-1])
                valid = (eligible[sl, select, None] & bv[select, p][None]).flatten(1)
                sim = (q[sl, p] @ support.T).clamp(-1, 1)
                sim.masked_fill_(~valid[:, None], -torch.inf)
                top = sim.topk(min(k, support.shape[0]), dim=-1).values
                top = torch.where(torch.isfinite(top), top, 0).sum(-1)
                count = qv[sl, p].sum(-1)
                polar.append((top * qv[sl, p]).sum(-1) / count.clamp_min(1))
                has.append((count > 0) & valid.any(-1))
            polar, has = torch.stack(polar, -1), torch.stack(has, -1)
            scores[sl, cls] = (polar * has).sum(-1) / has.sum(-1).clamp_min(1)
            available[sl, cls] = has.any(-1)
    return _finish(query, scores, available, return_available)


@torch.inference_mode()
def regional_prototype_scores(query, query_valid, bank, bank_valid, bank_y, *,
                              merge_regions=False, query_ids=None, bank_ids=None,
                              num_classes=None, batch_size=128, return_available=False):
    """Cosine of query and class means of unit tokens, with optional 20-token pool.

    Valid tokens have equal weight within a region, including across references.
    Available regions are averaged within each polarization, then polarizations.
    """
    q, qv, b, bv, y, eligible, c = _prepare(
        query, query_valid, bank, bank_valid, bank_y, query_ids, bank_ids, num_classes, batch_size)
    scores = q.new_zeros((len(q), c))
    available = torch.zeros_like(scores, dtype=torch.bool)
    regions = ((0, 20),) if merge_regions else REGIONS
    for cls in range(c):
        select = y == cls
        for start in range(0, len(q), batch_size):
            sl = slice(start, start + batch_size)
            polar, polar_has = [], []
            for p in range(2):
                values, has = [], []
                for lo, hi in regions:
                    qm = q[sl, p, lo:hi].sum(1) / qv[sl, p, lo:hi].sum(1).clamp_min(1)[:, None]
                    ref_sum = b[select, p, lo:hi].sum(1)
                    ref_count = bv[select, p, lo:hi].sum(1)
                    allow = eligible[sl, select].to(q.dtype)
                    bm = (allow @ ref_sum) / (allow @ ref_count.to(q.dtype)).clamp_min(1)[:, None]
                    qn, bn = qm.norm(dim=-1), bm.norm(dim=-1)
                    ok = (qn > 1e-8) & (bn > 1e-8)
                    values.append(((qm * bm).sum(-1) / (qn * bn).clamp_min(1e-20)).clamp(-1, 1))
                    has.append(ok)
                values, has = torch.stack(values, -1), torch.stack(has, -1)
                polar.append((values * has).sum(-1) / has.sum(-1).clamp_min(1))
                polar_has.append(has.any(-1))
            polar, has = torch.stack(polar, -1), torch.stack(polar_has, -1)
            scores[sl, cls] = (polar * has).sum(-1) / has.sum(-1).clamp_min(1)
            available[sl, cls] = has.any(-1)
    return _finish(query, scores, available, return_available)


@lru_cache(maxsize=400)
def _transport_constraints(n, m):
    from scipy.sparse import eye, kron, vstack
    a = vstack((kron(eye(n), np.ones((1, m))),
                kron(np.ones((1, n)), eye(m)))).tocsr()
    return a, np.r_[np.full(n, 1 / n), np.full(m, 1 / m)]


@torch.inference_mode()
def balanced_emd_scores(query, query_valid, bank, bank_valid, bank_y, *,
                        query_ids=None, bank_ids=None, num_classes=None,
                        batch_size=128, return_available=False):
    """CPU exact balanced OT per image pair; NOT a DeepEMD reproduction.

    Uniform token masses, cost 1-cos, score 1-min_cost. Average available
    references per class/polarization, then available polarizations equally.
    batch_size is accepted for API parity; LPs run one pair at a time.
    """
    from scipy.optimize import linprog
    qt, qvt, bt, bvt, yt, et, c = _prepare(
        query, query_valid, bank, bank_valid, bank_y, query_ids, bank_ids, num_classes, batch_size)
    q, qv, b, bv, y, eligible = map(_numpy, (qt, qvt, bt, bvt, yt, et))
    sums, counts = np.zeros((len(q), c, 2)), np.zeros((len(q), c, 2), int)
    for i in range(len(q)):
        for p in range(2):
            a = q[i, p, qv[i, p]].astype(np.float64)
            if not len(a):
                continue
            for j in np.flatnonzero(eligible[i]):
                s = b[j, p, bv[j, p]].astype(np.float64)
                if not len(s):
                    continue
                cost = 1 - np.clip(a @ s.T, -1, 1)
                eq, mass = _transport_constraints(len(a), len(s))
                result = linprog(cost.ravel(), A_eq=eq, b_eq=mass,
                                 bounds=(0, None), method='highs')
                if not result.success:
                    raise RuntimeError(f'EMD failed for query {i}, bank {j}, polar {p}: {result.message}')
                sums[i, y[j], p] += 1 - result.fun
                counts[i, y[j], p] += 1
    has = counts > 0
    scores = (sums / np.maximum(counts, 1)).sum(-1) / np.maximum(has.sum(-1), 1)
    scores = torch.as_tensor(scores, dtype=qt.dtype, device=qt.device)
    available = torch.as_tensor(has.any(-1), device=qt.device)
    return _finish(query, scores, available, return_available)
