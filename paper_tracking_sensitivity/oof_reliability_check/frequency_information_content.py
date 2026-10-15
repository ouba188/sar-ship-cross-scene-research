"""Fixed interventions on information content, not new model candidates."""

import numpy as np


def pooled_intervention(embedding, weights, mode):
    blocks = embedding.reshape(-1, 4, 16, 4).astype(float)
    if mode == 'no_quadrant_layout':
        blocks = np.broadcast_to(blocks.mean(-1, keepdims=True), blocks.shape).copy()
    elif mode == 'no_channel_identity':
        blocks = np.broadcast_to(blocks.mean(-2, keepdims=True), blocks.shape).copy()
    else:
        raise ValueError(mode)
    blocks /= np.maximum(np.linalg.norm(blocks.reshape(-1, 4, 64), axis=-1)[:, :, None, None], 1e-12)
    return (blocks*np.sqrt(weights)[:, :, None, None]).reshape(-1, 256)


def band_intervention(bands, mode):
    if mode == 'unsigned_bands':
        return bands.abs()
    if mode == 'no_local_band_layout':
        return bands.mean((-1, -2), keepdim=True).expand_as(bands)
    raise ValueError(mode)
