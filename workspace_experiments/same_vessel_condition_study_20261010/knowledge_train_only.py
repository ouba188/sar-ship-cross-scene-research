"""Training-only AIS supervision; shared interfaces for the compact A/B/C/D trial.

prepare_attributes(values, valid, fit_meta, probability) accepts ONLY fit rows.
load_upstream(cache, seed, raw, meta, fitting, validation) returns FULL-normalized
inputs/fixed case kernel and native vessel-fold OOF other-product teachers.
SAROnlyPredictor takes frozen SAR features; metadata and auxiliary heads are absent.
"""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from explicit_candidate_projection import CandidateProjector, identity_episode, balanced_mse
from crossfit_evidence import ship_folds, assert_excluded
from relative_content_condition import vessel_targets
from scene_training import sample_weights, metrics
from target_response_adapter import residual_head

ARMS = ('A', 'B', 'C', 'D')
SEEDS = (31415, 27182)
STEPS = 400


def phi(z):
    return torch.cat((z, z.new_ones((len(z), 1))), dim=1)


def prepare_attributes(values, valid, fit_meta, probability):
    """Balanced log statistics and D class means from valid source ships only.

importance corrects the COMMON classification sampler to the distribution of
valid ships, equal within each class. Invalid entries never enter log or loss.
"""
    values, valid = np.asarray(values, float), np.asarray(valid, bool)
    if values.shape != (len(fit_meta), 2) or valid.shape != (len(fit_meta),):
        raise ValueError('Attribute arrays must follow fit_obs_ids, with two dimensions')
    if not valid.any() or not (np.isfinite(values[valid]) & (values[valid] > 0)).all():
        raise ValueError('Valid AIS dimensions must be finite and positive')
    vm = fit_meta.loc[valid].reset_index(drop=True)
    logs = np.log(values[valid])
    for _, ids in vm.groupby('mmsi').groups.items():
        if not np.all(logs[list(ids)] == logs[list(ids)[0]]):
            raise ValueError('Conflicting valid dimensions within a vessel')
    weights = sample_weights(vm)
    weights /= weights.sum()
    mean = weights @ logs
    scale = np.sqrt(weights @ (logs-mean)**2).clip(1e-6)
    class_log_means = np.full((8, 2), np.nan)
    for c in np.unique(vm.class_id):
        rows = vm.class_id.to_numpy() == c
        class_log_means[c] = np.average(logs[rows], axis=0, weights=weights[rows])
    real, control = np.zeros_like(values), np.zeros_like(values)
    real[valid] = (logs-mean)/scale
    control[valid] = (class_log_means[vm.class_id.to_numpy()]-mean)/scale
    importance = np.zeros(len(values))
    importance[valid] = weights/np.asarray(probability)[valid]
    return dict(real=real.astype(np.float32), control=control.astype(np.float32),
                valid=valid.copy(), importance=importance.astype(np.float32),
                log_mean=mean, log_scale=scale, class_log_means=class_log_means)


class KnowledgeStudent(CandidateProjector):
    def __init__(self, source, labels, weights, dimensions=32):
        super().__init__(source, labels, weights, dimensions)
        # Nonzero identical initialization permits the initial real-gradient check.
        self.attribute_head = nn.Linear(dimensions, 2)


def state_geometry(state):
    q, r = torch.linalg.qr(state['frame'], mode='reduced')
    q = q*torch.where(r.diagonal() < 0, -1., 1.)
    scale = ((state['covariance']@q)*q).sum(0).clamp_min(1e-6).sqrt()
    return q, scale


def state_forward(state, x):
    q, scale = state_geometry(state)
    z = ((x-state['center'])@q)/scale/np.sqrt(q.shape[1])
    return z, F.linear(z, state['head.weight'], state['head.bias'])


class SAROnlyPredictor(nn.Module):
    """Frozen feature-to-score deployment, with no attribute head or identity API."""
    def __init__(self, buffers):
        super().__init__()
        for key, value in buffers.items():
            self.register_buffer(key, torch.as_tensor(value).detach().clone())

    @classmethod
    def from_student(cls, student, case_state, center, scale):
        q, zs = student.geometry()
        cq, cs = state_geometry(case_state)
        return cls(dict(raw_center=center, raw_scale=scale, center=student.center,
                        basis=q, scale=zs, head_weight=student.head.weight,
                        head_bias=student.head.bias, case_center=case_state['center'],
                        case_basis=cq, case_scale=cs))

    def forward(self, raw):
        x = (raw-self.raw_center)/self.raw_scale
        z = ((x-self.center)@self.basis)/self.scale/np.sqrt(self.basis.shape[1])
        kernel = ((x-self.case_center)@self.case_basis)/self.case_scale/np.sqrt(self.case_basis.shape[1])
        return F.linear(z, self.head_weight, self.head_bias), kernel

    def establish_support(self, sar_features, labels):
        prior, kernel = self(sar_features)
        return residual_head(phi(kernel), prior, labels, balanced=True)[0]

    def predict(self, sar_features, support_coefficient=None):
        prior, kernel = self(sar_features)
        return prior if support_coefficient is None else prior+phi(kernel)@support_coefficient


def validate_native_cache(meta, raw, fitting, excluded, selection, normalization, outputs):
    """Detect actual fold/cache mismatch before a cached teacher can train a student."""
    for blob in (normalization, outputs):
        np.testing.assert_array_equal(blob['fit_obs_ids'], fitting)
        np.testing.assert_array_equal(blob['excluded_obs_ids'], excluded)
    np.testing.assert_array_equal(outputs['obs_ids'], np.arange(len(meta)))
    np.testing.assert_array_equal(normalization['selection_obs_ids'], selection)
    assert_excluded(meta, fitting, selection, excluded)
    if set(meta.iloc[fitting].mmsi) & set(meta.iloc[selection].mmsi):
        raise ValueError('New-ship validation overlaps upstream fitting vessels')
    np.testing.assert_array_equal(normalization['raw_center'], raw[fitting].mean(0))
    np.testing.assert_array_equal(normalization['raw_scale'], np.maximum(raw[fitting].std(0), 1e-5))


def load_upstream(cache, seed, raw, meta, fitting, validation):
    fm = meta.iloc[fitting].reset_index(drop=True)
    with np.load(cache/'source_folds.npz') as f:
        np.testing.assert_array_equal(f['fit_obs_ids'], fitting)
        assignment = f['assignment'].copy()
    np.testing.assert_array_equal(assignment, ship_folds(fm))
    selection = np.unique(np.concatenate([np.r_[s, q] for s, q in validation]))
    oof = np.empty((len(fitting), 8), np.float32)
    audit = []
    full = None
    for name, fold in [('full', None)]+[(f'fold{k}', k) for k in range(3)]:
        fit = fitting if fold is None else fitting[assignment != fold]
        excluded = np.array([], dtype=int) if fold is None else fitting[assignment == fold]
        with np.load(cache/f'{name}_{seed}_normalization.npz') as f:
            norm = {k: f[k] for k in f.files}
        with np.load(cache/f'{name}_{seed}_outputs.npz') as f:
            out = {k: f[k] for k in ('fit_obs_ids', 'excluded_obs_ids', 'obs_ids', 'prior', 'kernel')}
        validate_native_cache(meta, raw, fit, excluded, selection, norm, out)
        # Replay in the fold's OWN normalization. Never feed FULL inputs to a fold.
        prior_state = torch.load(cache/f'{name}_{seed}_prior.pt', map_location='cpu', weights_only=True)['selected']
        ids = fitting if fold is None else excluded
        native = torch.from_numpy(((raw[ids]-norm['raw_center'])/norm['raw_scale']).astype(np.float32))
        with torch.no_grad():
            _, replay = state_forward(prior_state, native)
        np.testing.assert_allclose(replay.numpy(), out['prior'][ids], rtol=1e-5, atol=2e-6)
        if fold is None:
            case = torch.load(cache/f'{name}_{seed}_case.pt', map_location='cpu', weights_only=True)['selected']
            x = torch.from_numpy(((raw-norm['raw_center'])/norm['raw_scale']).astype(np.float32))
            with torch.no_grad():
                kernel, _ = state_forward(case, x)
            np.testing.assert_array_equal(kernel.numpy(), out['kernel'])
            full = dict(x=x, kernel=kernel, raw_center=norm['raw_center'], raw_scale=norm['raw_scale'],
                        case_state=case, reference_prior=out['prior'])
        else:
            oof[assignment == fold] = out['prior'][excluded]
        audit.append(dict(seed=seed, bundle=name, fit_observations=len(fit),
                          fit_ships=int(meta.iloc[fit].mmsi.nunique()), excluded_observations=len(excluded),
                          excluded_ships=int(meta.iloc[excluded].mmsi.nunique()),
                          native_normalization_verified=True, cache_prior_replayed=True))
    _, teacher, valid = vessel_targets(oof, fm)
    full.update(teacher=torch.from_numpy(teacher), teacher_valid=torch.from_numpy(valid),
                oof_prior=oof, assignment=assignment, audit=audit)
    return full


def make_schedule(fit_meta, probability, seed, steps=STEPS):
    batches, supports, queries = [], [], []
    for step in range(1, steps+1):
        batches.append(np.random.default_rng(seed+step).choice(len(fit_meta), 128, replace=True, p=probability))
        s, q = identity_episode(fit_meta, np.random.default_rng(seed+30000+step))
        supports.append(s)
        queries.append(q)
    return dict(batch=np.stack(batches), support=np.stack(supports), query=np.stack(queries))


def attribute_loss(prediction, targets, valid, importance):
    if not valid.any():
        return prediction.sum()*0
    return ((prediction[valid]-targets[valid]).square().mean(1)*importance[valid]).sum()/len(prediction)


def batch_losses(model, x, y, kernel, teacher, teacher_valid, attributes, schedule, step):
    ids, s, q = (schedule[k][step] for k in ('batch', 'support', 'query'))
    z, score = model(x[np.r_[ids, s, q]])
    n, ns = len(ids), len(s)
    p, ps, pq = score[:n], score[n:n+ns], score[n+ns:]
    beta = residual_head(phi(kernel[s]), ps, y[s], balanced=True)[0]
    adapted = pq+phi(kernel[q])@beta
    supervised = F.mse_loss(p, 2*F.one_hot(y[ids], 8).to(p.dtype)-1)
    episode = balanced_mse(adapted, y[q])
    mask = teacher_valid[ids]
    if mask.any():
        target, student = teacher[ids][mask], p[mask]
        teaching = F.mse_loss(student-student.mean(1, keepdim=True), target-target.mean(1, keepdim=True))
    else:
        teaching = p.sum()*0
    pred = model.attribute_head(z[:n])
    real = attribute_loss(pred, attributes['real'][ids], attributes['valid'][ids], attributes['importance'][ids])
    control = attribute_loss(pred, attributes['control'][ids], attributes['valid'][ids], attributes['importance'][ids])
    return dict(supervised=supervised, episode=episode, teacher=teaching, real=real, control=control)


def arm_loss(losses, arm, alpha):
    result = losses['supervised']+.5*losses['episode']
    if arm != 'A':
        result = result+.5*losses['teacher']
    if arm in ('C', 'D'):
        result = result+alpha*losses['real' if arm == 'C' else 'control']
    return result


def calibrate_alpha(model, x, y, kernel, teacher, teacher_valid, attributes, schedule):
    """One real-attribute calibration; cap EACH fixed-batch frame-gradient ratio."""
    rows = []
    for step in range(8):
        losses = batch_losses(model, x, y, kernel, teacher, teacher_valid, attributes, schedule, step)
        main = torch.autograd.grad(arm_loss(losses, 'B', 0), model.frame, retain_graph=True)[0]
        aux = torch.autograd.grad(losses['real'], model.frame)[0]
        gm, ga = float(main.norm()), float(aux.norm())
        rows.append(dict(batch=step, main_frame_gradient=gm, attribute_frame_gradient=ga,
                         bound=.1 if ga == 0 else min(.1, .1*gm/ga)))
    alpha = min(row['bound'] for row in rows)
    for row in rows:
        row['alpha'] = alpha
        row['scaled_ratio'] = (alpha*row['attribute_frame_gradient']/row['main_frame_gradient']
                               if row['main_frame_gradient'] else 0.)
    return alpha, rows


def balanced_ce(scores, labels):
    losses = F.cross_entropy(scores, labels, reduction='none')
    return torch.stack([losses[labels == c].mean() for c in labels.unique()]).mean()


def selection_key(ba, ce, step):
    return float(ba), -float(ce), -int(step)


def train_arm(initial, x, y, kernel, fitting, teacher, teacher_valid, attributes,
              schedule, validation, seed, arm, alpha):
    model = copy.deepcopy(initial)
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    best, best_key, choice = None, None, None
    traces = []

    def evaluate(step, losses=None):
        nonlocal best, best_key, choice
        bas, ces = [], []
        model.eval()
        with torch.no_grad():
            for s, q in validation:
                _, scores = model(x[np.r_[s, q]])
                ps, pq = scores[:len(s)], scores[len(s):]
                beta = residual_head(phi(kernel[s]), ps, y[s], balanced=True)[0]
                score = pq+phi(kernel[q])@beta
                bas.append(metrics(score.argmax(1).numpy(), y[q].numpy())['BA'])
                ces.append(float(balanced_ce(score, y[q])))
        ba, ce = float(np.mean(bas)), float(np.mean(ces))
        row = dict(seed=seed, arm=arm, step=step, validation_BA=ba, validation_CE=ce, alpha=alpha,
                   **(losses or {}))
        traces.append(row)
        key = selection_key(ba, ce, step)
        if best_key is None or key > best_key:
            best, best_key = copy.deepcopy(model.state_dict()), key
            choice = dict(seed=seed, arm=arm, step=step, validation_BA=ba, validation_CE=ce, alpha=alpha)
        model.train()

    evaluate(0)
    xf, yf, kf = x[fitting], y[fitting], kernel[fitting].detach()
    for step in range(STEPS):
        losses = batch_losses(model, xf, yf, kf, teacher, teacher_valid, attributes, schedule, step)
        loss = arm_loss(losses, arm, alpha)
        optimizer.zero_grad()
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.))
        optimizer.step()
        if (step+1)%20 == 0:
            evaluate(step+1, dict(loss=float(loss.detach()), gradient_norm=gradient,
                                  **{k: float(v.detach()) for k, v in losses.items()}))
            print('VALIDATION', traces[-1], flush=True)
    states = dict(selected=best, last=copy.deepcopy(model.state_dict()))
    model.load_state_dict(best)
    return model.eval(), states, traces, choice
