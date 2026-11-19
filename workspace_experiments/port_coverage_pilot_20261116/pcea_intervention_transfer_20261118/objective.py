"""Teacher-measured intervention targets, not a causal/noncausal segmentation."""
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
import torch
import torch.nn.functional as F

REGIONS = ('body', 'ends', 'side')
CONDITIONS = [(p, r, band) for p in range(2) for r in range(3)
              for band in ('fine', 'middle')]


def relations(z, y):
    return z.gather(1, y[:, None]) - z


def margin(z, y):
    return relations(z, y).masked_fill(F.one_hot(y, z.shape[1]).bool(), torch.inf).min(1).values


def make_interventions(images, valid, masks, available, strength=.5):
    """Native-pixel bands; valid chip cropped before filtering, padding untouched.

    Returns spatial-scale intervention and a regional intensity-level control
    with identical signed pixel-sum change. The control is NOT energy matched.
    """
    images = np.asarray(images, dtype=np.float32)
    proposals, levels, eligible, diagnostics = [], [], [], []
    bands = np.zeros((len(images), 2, 2, *images.shape[-2:]), np.float32)
    for i in range(len(images)):
        for p in range(2):
            if not available[i, p]:
                continue
            yy, xx = np.where(valid[i, p, 0] > 0)
            if not len(yy):
                continue
            sy, sx = slice(yy.min(), yy.max()+1), slice(xx.min(), xx.max()+1)
            crop = images[i, p, 0, sy, sx]
            g1, g2 = [gaussian_filter(crop, sigma, mode='reflect') for sigma in (1, 2)]
            bands[i, p, 0, sy, sx] = crop-g1
            bands[i, p, 1, sy, sx] = g1-g2
    for p, r, band in CONDITIONS:
        altered, level = images.copy(), images.copy()
        ok, rows = [], []
        for i in range(len(images)):
            mask = masks[i, p, r].astype(bool) & (valid[i, p, 0] > 0)
            taper = np.minimum(distance_transform_edt(mask)/2, 1).astype(np.float32)*mask
            x = images[i, p, 0]
            delta = np.clip(x-strength*taper*bands[i, p, int(band == 'middle')], 0, 1)-x
            total = float(delta.sum(dtype=np.float64))
            basis = taper*((1-x) if total >= 0 else x)
            denominator = float(basis.sum(dtype=np.float64))
            correction = total/denominator if denominator > 0 else 0.
            if abs(correction) > 1+1e-5:
                raise ValueError('Level control leaves valid intensity interval')
            altered[i, p, 0] = x+delta
            level[i, p, 0] = x+correction*basis
            ok.append(bool(available[i, p] and mask.sum() >= 4 and np.linalg.norm(delta) > 1e-6))
            rows.append(dict(pixels=int(mask.sum()), l2=float(np.linalg.norm(delta)),
                             mean_control_error=float(abs(total-(level[i,p,0]-x).sum(dtype=np.float64)))))
        proposals.append(altered)
        levels.append(level)
        eligible.append(ok)
        diagnostics.append(rows)
    return np.stack(proposals), np.stack(levels), np.asarray(eligible), diagnostics


@torch.no_grad()
def select_targets(clean, interventions, controls, y, eligible, eps=1e-4, cap=.1):
    """All logits from ONE frozen teacher, identical reference bank and geometry."""
    clean, interventions, controls = clean.detach(), interventions.detach(), controls.detach()
    n, k = clean.shape
    yy = y[None].expand(len(interventions), -1).reshape(-1)
    base_margin = margin(clean, y)
    im = margin(interventions.reshape(-1, k), yy).reshape(-1, n)
    cm = margin(controls.reshape(-1, k), yy).reshape(-1, n)
    gather = y[None, :, None].expand(len(interventions), -1, 1)
    ip = interventions.softmax(-1).gather(2, gather).squeeze(2)
    cp = controls.softmax(-1).gather(2, gather).squeeze(2)
    bp = clean.softmax(-1).gather(1, y[:, None]).squeeze(1)
    good = eligible.bool() & (im > base_margin+eps) & (ip > bp+eps)
    good &= (im > cm+eps) & (ip > cp+eps)
    ranked = (im-base_margin).masked_fill(~good, -torch.inf)
    selected = ranked.argmax(0)
    use = good.any(0)
    picked = interventions[selected, torch.arange(n, device=clean.device)]
    delta = relations(picked, y)-relations(clean, y)
    positive = delta.clamp(0, cap)*use[:, None]
    wrong = ~F.one_hot(y, k).bool()
    competitor_weights = clean.masked_fill(~wrong, -torch.inf).softmax(1)
    return dict(use=use, selected=selected, clean=clean, chosen=picked,
                relation_gain=positive, competitor_weights=competitor_weights,
                margin_gain=(margin(picked,y)-base_margin).clamp(0,cap)*use,
                good=good, raw_margin_gain=im-base_margin)


MATCHED_MODES = ('none', 'protect', 'fixed', 'measured', 'shuffled')


@torch.no_grad()
def matched_plan(targets, y, active, seed=17, fixed_gain=.1):
    """Build donors over the whole source schedule, never just a minibatch.

    Singleton classes are excluded from improvement in ALL three gain arms.
    Shuffling moves the complete gain vector; recipient weights stay fixed.
    """
    eligible = targets['use'].bool() & active.bool() & (margin(targets['clean'], y) < .2)
    donor = torch.full_like(y, -1)
    rng = np.random.default_rng(seed)
    for label in y.unique().tolist():
        members = torch.where(eligible & (y == label))[0].cpu().numpy()
        if len(members) < 2:
            continue
        members = rng.permutation(members)
        source = np.roll(members, 1)
        donor[torch.as_tensor(members, device=y.device)] = torch.as_tensor(source, device=y.device)
    matched = donor >= 0
    measured = targets['relation_gain'].detach()*matched[:, None]
    shuffled = measured[donor.clamp_min(0)]*matched[:, None]
    fixed = (measured > 0).to(measured.dtype)*fixed_gain
    return dict(matched=matched, eligible=eligible, donor=donor,
                measured=measured, shuffled=shuffled, fixed=fixed)


def matched_auxiliary(z_on, z_off, y, weights, targets, active, mode, weight=.2):
    if mode not in MATCHED_MODES:
        raise ValueError(mode)
    anchor = margin(targets['clean'].detach(), y)
    eligible = targets['use'].bool() & active.bool() & (anchor < .2)
    use = eligible & targets['matched'].bool()
    gain = targets[mode].detach() if mode in ('fixed', 'measured', 'shuffled') else torch.zeros_like(z_on)
    gap = relations(z_off.detach(), y)+gain-relations(z_on, y)
    improve = (F.relu(gap)*(gain > 0)*targets['competitor_weights'].detach()).sum(1)*use
    protect = F.relu(anchor.clamp(max=.2)-margin(z_on,y))*(anchor > 0)*active
    if mode == 'none':
        protect = protect*0
    loss = weight*((improve+protect)*weights).mean()
    return loss, dict(selected=int(use.sum()) if mode in ('fixed','measured','shuffled') else 0,
        eligible=int(eligible.sum()), active=int(active.sum()),
        unpaired=int((eligible & ~targets['matched'].bool()).sum()),
        active_gain_relations=int(((gap > 0) & (gain > 0) & use[:,None]).sum()),
        improve=float(improve.mean().detach()), protect=float(protect.mean().detach()))


@torch.no_grad()
def fulfillment(z_on, z_off, y, targets, active, mode='measured'):
    """Log absolute clean-image progress as well as the on/off gap."""
    real_gain = targets['relation_gain'].detach()
    relation_delta = relations(z_on,y)-relations(z_off,y)
    wanted = (real_gain > 0) & targets['use'][:,None] & active[:,None] & (margin(targets['clean'],y) < .2)[:,None]
    satisfied = ((relation_delta >= real_gain-1e-6) & wanted).sum(1)
    gain = targets[mode] if mode in ('fixed','measured','shuffled') else torch.zeros_like(real_gain)
    requested = (gain > 0) & targets['use'][:,None] & active[:,None] & targets['matched'][:,None]
    requested &= (margin(targets['clean'],y) < .2)[:,None]
    m_on, m_off = margin(z_on,y), margin(z_off,y)
    teacher = margin(targets['clean'],y)
    return dict(on_margin=m_on, off_margin=m_off, teacher_margin=teacher,
        on_minus_teacher=m_on-teacher, on_minus_off=m_on-m_off,
        measured_probe_relations=wanted.sum(1), measured_probe_satisfied=satisfied,
        requested_relations=requested.sum(1),
        satisfied_relations=((relation_delta >= gain-1e-6) & requested).sum(1),
        active_gain_relations=((relation_delta < gain) & requested).sum(1),
        rescued=(z_on.argmax(1)==y) & (z_off.argmax(1)!=y),
        harmed=(z_on.argmax(1)!=y) & (z_off.argmax(1)==y),
        on_prediction=z_on.argmax(1), off_prediction=z_off.argmax(1))


def auxiliary(z_on, z_off, y, weights, targets, active, mode='relations', weight=.2):
    """Only positive measured gains are transferred; no feature-equality loss."""
    if mode not in ('margin', 'relations'):
        raise ValueError(mode)
    anchor = margin(targets['clean'].detach(), y)
    use = targets['use'].detach() & active.bool() & (anchor < .2)
    before = z_off.detach()
    if mode == 'margin':
        goal = margin(before, y)+targets['margin_gain'].detach()
        improve = F.relu(goal-margin(z_on,y))*use
    else:
        gain = targets['relation_gain'].detach()
        gap = relations(before,y)+gain-relations(z_on,y)
        # An unhelped competitor does not become an invented improvement target.
        improve = (F.relu(gap)*(gain > 0)*targets['competitor_weights'].detach()).sum(1)*use
    protect = F.relu(anchor.clamp(max=.2)-margin(z_on,y))*(anchor > 0)*active
    loss = weight*((improve+protect)*weights).mean()
    return loss, dict(selected=int(use.sum()), eligible=int(active.sum()),
                      improve=float(improve.mean().detach()), protect=float(protect.mean().detach()))
