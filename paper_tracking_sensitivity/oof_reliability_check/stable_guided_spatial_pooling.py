"""Source-scaled pair attention and a coordinated residual expert."""

import numpy as np
from scipy.linalg import solve

from projection_residual_expert import fit_kernel_correction


class TokenCache:
    """Read the two historical four-token caches in manifest row order."""

    def __init__(self, old, new, old_indices, boundary=25656):
        self.old, self.new = old, new
        self.old_indices = np.asarray(old_indices, dtype=int)
        self.boundary = boundary

    def read(self, rows):
        rows = np.asarray(rows, dtype=int)
        result = np.empty((len(rows), 4, self.old.shape[-1]), dtype=np.float32)
        legacy = rows < self.boundary
        result[legacy] = self.old[self.old_indices[rows[legacy]]]
        result[~legacy] = self.new[rows[~legacy] - self.boundary]
        return result


def source_rms(values):
    """Caller supplies source values only; zero inputs stay zero after scaling."""
    values = np.asarray(values, dtype=float)
    axes = tuple(range(values.ndim - 1))
    return np.maximum(np.sqrt(np.mean(values * values, axis=axes)), 1e-6)


def predicted_pair_indices(base_prior, pairs):
    top = np.sort(np.argsort(base_prior, axis=1, kind='stable')[:, -2:], axis=1)
    pairs = np.sort(np.asarray(pairs), axis=1)
    matches = (top[:, None, :] == pairs[None, :, :]).all(2)
    if not np.all(matches.sum(1) == 1):
        raise ValueError('Every predicted top-two class pair needs one direction')
    return matches.argmax(1)


def spatial_residual(projected_tokens, evidence, base_prior, pairs, evidence_scale):
    """No truth argument: inference uses only source-base predicted top-two."""
    selected = predicted_pair_indices(base_prior, pairs)
    logits = np.abs(evidence[np.arange(len(evidence)), :, selected])
    logits = logits / np.asarray(evidence_scale)[selected, None]
    logits -= logits.max(1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(1, keepdims=True)
    residual = np.einsum('nt,ntd->nd', weights - .25, projected_tokens)
    return residual, weights


def fit_residual_head(features, labels, original_prior, strength=1.):
    """Balanced ridge on desired-minus-original-prior, with no extra bias."""
    x, labels = np.asarray(features, dtype=float), np.asarray(labels)
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    weights = len(labels) / (len(counts) * counts[inverse])
    desired = 2 * np.eye(original_prior.shape[1])[labels] - 1 - original_prior
    gram = x.T @ (weights[:, None] * x)
    gram.flat[::len(gram) + 1] += strength
    return solve(gram, x.T @ (weights[:, None] * desired), assume_a='pos')


def coordinated_expert(original_prior, original_kernel, features, head,
                       support, support_labels, kernel_strength=.25):
    """Refit the original correction against the augmented source prior/kernel."""
    prior = original_prior + features @ head
    kernel = original_kernel + kernel_strength * (
        features @ features[support].T) / features.shape[1]
    dual = fit_kernel_correction(kernel[support], support_labels, prior[support])
    return prior + kernel @ dual, dual
