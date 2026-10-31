"""Training-only source selection by paired, equal-dose image interventions."""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, distance_transform_edt
from PIL import Image, ImageDraw

from .. import frequency, tokens
from .data import FREQUENCY_KEYS, batch, energy_for, pair_mask, prepare_record, assert_source
from .model import forward_batch, reference_bank


def masks_for(view, spec):
    raw = view['raw']
    sh = spec.get('ship_axis') or view['ship_axis']
    yy, xx = np.indices(raw.shape)
    dx, dy = xx+.5-sh['center'][0], yy+.5-sh['center'][1]
    c, s = np.cos(sh['angle']), np.sin(sh['angle'])
    u, v = dx*c+dy*s, -dx*s+dy*c
    canvas = Image.new('L', (raw.shape[1], raw.shape[0]))
    target = spec['detections'][int(spec['target_index'])]
    ImageDraw.Draw(canvas).polygon([tuple(x) for x in target['corners']], fill=1)
    body = np.asarray(canvas).astype(bool)
    side = ((abs(u) <= sh['length_px']/2) & (abs(v) > sh['width_px']/2)
            & (abs(v) <= sh['width_px']/2+8) & ~body)
    return dict(body=body, side=side)


def variants(record):
    original, views = prepare_record(record, True)
    proposals = []
    for p, pol in enumerate(('vv', 'vh')):
        view = views[p]
        if view is None or not original['available'][p] or not record[pol].get('ship_axis'):
            continue
        im = view['raw'].astype(float)/255
        g1, g2 = [gaussian_filter(im, sigma, mode='reflect') for sigma in (1, 2)]
        components = dict(fine=im-g1, middle=g1-g2)
        place = view['placement']
        visible = np.zeros(im.shape, bool)
        x, y, w, h = [place[k] for k in ('crop_x', 'crop_y', 'crop_width', 'crop_height')]
        visible[y:y+h, x:x+w] = True
        for region, mask in masks_for(view, record[pol]).items():
            mask = mask & visible
            if mask.sum() < 8:
                continue
            taper = np.minimum(distance_transform_edt(mask)/2, 1)*mask
            for band, component in components.items():
                delta = np.clip(im-.5*taper*component, 0, 1)-im
                norm = np.linalg.norm(delta)
                if norm >= 1e-8:
                    proposals.append((p, region, band, im, delta, taper, norm))
    items = [original]
    meta = [dict(kind='original', pol='none', region='none', band='none')]
    if not proposals:
        return items, meta
    dose = .5*min(a[-1] for a in proposals)
    for p, region, band, im, delta, taper, norm in proposals:
        altered = im+dose*delta/norm
        change = altered-im
        basis = taper*((1-im) if change.sum() >= 0 else im)
        level = im+change.sum()/basis.sum()*basis if basis.sum() else im.copy()
        for kind, raw in (('structure', altered), ('mean_control', level)):
            item = {k: v.copy() for k, v in original.items()}
            item['images'][p, 0] = frequency.place(raw, views[p]['placement'])
            slots = tokens.slots(views[p]['geometry'], views[p]['footprint'])
            for key, value in energy_for(raw*255, views[p], slots).items():
                item[key][p] = value
            items.append(item)
            meta.append(dict(kind=kind, pol=('vv', 'vh')[p], region=region, band=band,
                             l2=float(np.linalg.norm(raw-im)), mean_delta=float((raw-im).mean())))
    return items, meta


def select_teacher(scores, metadata, label):
    scores = np.asarray(scores, dtype=np.float64)
    other = scores[0].copy()
    other[label] = -np.inf
    rival = int(other.argmax())
    margins = scores[:, label]-scores[:, rival]
    shifted = scores-scores.max(1, keepdims=True)
    ce = np.log(np.exp(shifted).sum(1))-shifted[:, label]
    keys = ('pol', 'region', 'band')
    controls = {tuple(m[k] for k in keys): i for i, m in enumerate(metadata) if m['kind'] == 'mean_control'}
    structures = [i for i, m in enumerate(metadata) if m['kind'] == 'structure']
    eligible = []
    for i in structures:
        j = controls.get(tuple(metadata[i][k] for k in keys))
        if j is None:
            continue
        gain, residual = margins[i]-margins[0], margins[i]-margins[j]
        if ce[0]-ce[i] >= .01 and gain >= .01 and residual >= .01:
            eligible.append((i, j, residual))
    chosen, control, residual = max(eligible, key=lambda x: x[2]) if eligible else (
        structures[0] if structures else 0, None, 0.)
    return dict(chosen_index=chosen, control_index=control, rival=rival,
                useful=bool(eligible), structural_margin_gain=float(residual))


@torch.no_grad()
def select_records(model, records, references, device, progress=None):
    assert_source(records)
    bank = reference_bank(model, references, device)
    result = []
    for n, record in enumerate(records):
        items, meta = variants(record)
        scores = []
        for first in range(0, len(items), 8):
            group = items[first:first+8]
            scores.append(forward_batch(model, batch(group, device), bank,
                pair_mask([record]*len(group), references, device)).cpu().numpy())
        scores = np.concatenate(scores)
        y = record['label']
        choice = select_teacher(scores, meta, y)
        altered = {k: v.copy() for k, v in items[choice['chosen_index']].items()}
        # Confirm the useful change through image features, keeping explicit
        # Gabor inputs at their original values (the measured E1/K0/F0 path).
        for key in FREQUENCY_KEYS:
            altered[key] = items[0][key]
        score = forward_batch(model, batch([altered], device), bank,
                              pair_mask([record], references, device))[0].cpu()
        clean = torch.tensor(scores[0])
        def margin(s):
            competitors = s.clone()
            competitors[y] = -torch.inf
            return s[y]-competitors.max()
        useful = choice['useful'] and bool(
            (score.log_softmax(0)[y] > clean.log_softmax(0)[y]) & (margin(score) > margin(clean)))
        result.append(dict(id=str(record['id']), label=y, useful=useful,
            original=clean.tolist(), altered=score.tolist(), choice=choice,
            selected_intervention=meta[choice['chosen_index']]))
        if progress is not None:
            progress(n+1, len(records), result[-1])
    return result
