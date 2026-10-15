"""Matched incidence information for source heads and target residual fitting."""

import numpy as np

from residual_expert import fit_correction
from projection_residual_expert import fit_kernel_correction


def condition_features(angles):
    angles = np.asarray(angles, dtype=float)
    u = np.zeros((len(angles), 2))
    valid = np.isfinite(angles)
    t = (angles[valid]-38.)/5.
    u[valid] = np.column_stack((t, t**2))
    return u


def extend_transforms(transforms, condition_width=2):
    answer = {}
    for name, original in transforms.items():
        d, m = original.shape
        extended = np.zeros((d+condition_width, m+condition_width))
        extended[:d,:m] = original
        extended[d:,m:] = np.eye(condition_width)
        answer[name] = extended
    return answer


def predict_conditioned(bank, views, heads, support, labels, u, condition_scales=None):
    augmented = np.column_stack((bank, u))
    bp = augmented @ heads['joint_base']['coef'] + heads['joint_base']['intercept']
    ep = augmented @ heads['joint_expert']['coef'] + heads['joint_expert']['intercept']
    original_phi = views['phis'][True,True]
    weighted = u/np.sqrt(2) if condition_scales is None else u*np.asarray(condition_scales)
    phi = np.column_stack((original_phi[:,:-1], weighted, original_phi[:,-1]))
    bcoef = fit_correction(phi[support], labels, bp[support])
    common = views['common'][True]
    kernel = .5*(common['retained'] @ common['retained'][support].T
                  + common['complement'] @ common['complement'][support].T)+1.
    kernel += .25*views['stable'] @ views['stable'][support].T
    if condition_scales is None:
        kernel += .25*(u @ u[support].T)/2.
    else:
        kernel += .25*(weighted @ weighted[support].T)
    ecoef = fit_kernel_correction(kernel[support], labels, ep[support])
    return bp+phi @ bcoef, ep+kernel @ ecoef
