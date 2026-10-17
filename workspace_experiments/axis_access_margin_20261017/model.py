"""Shared local reader with separate availability masks and final-margin loss."""
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent/'local_evidence_reliability_20261017'
sys.path.insert(0, str(PREVIOUS))
import train_reliability as prior
from local_evidence import EvidenceReader, retrieval_loss

COMMON_SHARE = .5
MARGIN_WEIGHT = .05


def read(model, features, ids, common=False):
    prefix = 'common_' if common else ''
    body = features[prefix+'body'][ids]
    tokens = features[prefix+'tokens'][ids]
    positions = features[prefix+'positions'][ids]
    quality = positions.new_zeros((len(ids), 12, 10))
    return model(body, tokens, positions, quality)


def predict(axis_raw, axis_valid, task, common_raw=None, common_valid=None, *,
            axis_weight=1-COMMON_SHARE):
    if common_raw is None:
        p, h = prior.old.predict(axis_raw, axis_valid, task, 1.)
        return p, h, axis_valid
    ns = task['ns']
    axis_h = prior.old.support_features(axis_raw, axis_valid, ns).double()
    common_h = prior.old.support_features(common_raw, common_valid, ns).double()
    h = torch.cat((np.sqrt(axis_weight)*axis_h, np.sqrt(COMMON_SHARE)*common_h), dim=1)
    ss = task['ss']+h[:ns]@h[:ns].T
    qs = task['qs']+h[ns:]@h[:ns].T
    root = task['root']
    coefficient = root[:, None]*torch.linalg.solve(
        root[:, None]*ss*root[None, :]+8*torch.eye(ns, dtype=ss.dtype, device=ss.device),
        root[:, None]*(task['desired']-task['prior_s']))
    probability = .5*(task['base']+(task['prior_q']+qs@coefficient).softmax(-1))
    active_axis = axis_valid[ns:] & (axis_valid[:ns].sum() >= 2)
    active_common = common_valid[ns:] & (common_valid[:ns].sum() >= 2)
    probability = torch.where((active_axis | active_common)[:, None], probability, task['old'])
    return probability, h, axis_valid | common_valid


def decision_margin_loss(probability, truth):
    logp = probability.clamp_min(1e-12).log()
    own = logp.gather(1, truth[:, None]).squeeze(1)
    rivals = logp.masked_fill(F.one_hot(truth, probability.shape[1]).bool(), -torch.inf).amax(1)
    per = F.softplus(rivals-own)
    return torch.stack([per[truth == c].mean() for c in truth.unique()]).mean()


def loss_parts(model, features, task, labels, use_common, use_margin, *,
               axis_weight=1-COMMON_SHARE):
    ids = task['ids']
    raw, _, _ = read(model, features, ids)
    common = read(model, features, ids, True)[0] if use_common else None
    common_valid = features['common_valid'][ids] if use_common else None
    p, h, valid = predict(raw, features['valid'][ids], task, common, common_valid,
                         axis_weight=axis_weight)
    truth = labels[task['q']]
    brier = prior.old.risk(p, truth)
    auxiliary = retrieval_loss(h, labels[ids], task['ns'], valid)
    margin = decision_margin_loss(p, truth)
    loss = brier+.05*auxiliary+(MARGIN_WEIGHT*margin if use_margin else 0.)
    return loss, brier, auxiliary, margin, p, h


@torch.no_grad()
def all_readouts(model, features, use_common):
    output = []
    for common in ([False, True] if use_common else [False]):
        chunks = [read(model, features, np.arange(i, min(i+256, len(features['body']))), common)
                  for i in range(0, len(features['body']), 256)]
        output.append(tuple(torch.cat([part[k] for part in chunks]) for k in range(3)))
    return output
