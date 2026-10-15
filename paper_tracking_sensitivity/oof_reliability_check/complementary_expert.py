"""Fit an expert's correction relative to the actual adapted base."""

from residual_expert import fit_correction


def fit_predict_complement(features, support, labels, training_base,
                           deployed_base, source_contrast):
    """Training base scores are held-out predictions, except in the ablation."""
    training_prior = training_base + source_contrast[support]
    coefficient = fit_correction(features[support], labels, training_prior)
    scores = deployed_base + source_contrast + features @ coefficient
    return scores, coefficient
