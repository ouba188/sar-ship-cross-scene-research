"""Source-teacher measurement of native-scale interventions and weak M1 priors.

The frequency attenuation and signed-intensity-sum controls are ported from
pcea_intervention_transfer_20261118/objective.py without changing the operation.
These controls are not energy matched. A batch uses 25 teacher forwards:
one original, 12 interventions and 12 brightness controls.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
import torch
from torch.nn import functional as F

from .priors import OPERATIONS, REGIONS, attention_from_response
from .region_ops import grid_occupancy
from .supervision import effects_from_legacy


CONDITIONS = tuple((p, r, band) for p in range(2) for r in range(3) for band in OPERATIONS)


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def make_interventions(data, strength=.5):
    """Generate the existing frequency/brightness pairs on the native pixel grid.

    data images/valid are [B,2,1,H,W], regions_native [B,2,3,H,W]
    in body/ends/side order, and available [B,2]. Images use the loader's [0,1]
    display scale. Filtering is confined to each valid crop before tapering;
    padding, other regions and the other polarization remain untouched.
    """
    images = np.asarray(_numpy(data["images"]), dtype=np.float32)
    valid = _numpy(data["valid"])
    masks = _numpy(data["regions_native"])
    available = _numpy(data["available"])
    proposals, levels, eligible, diagnostics = [], [], [], []
    bands = np.zeros((len(images), 2, 2, *images.shape[-2:]), np.float32)
    for i in range(len(images)):
        for p in range(2):
            if not available[i, p]:
                continue
            yy, xx = np.where(valid[i, p, 0] > 0)
            if not len(yy):
                continue
            sy, sx = slice(yy.min(), yy.max() + 1), slice(xx.min(), xx.max() + 1)
            crop = images[i, p, 0, sy, sx]
            g1, g2 = [gaussian_filter(crop, sigma, mode="reflect") for sigma in (1, 2)]
            bands[i, p, 0, sy, sx] = crop - g1
            bands[i, p, 1, sy, sx] = g1 - g2
    for p, r, band in CONDITIONS:
        altered, level = images.copy(), images.copy()
        ok, rows = [], []
        for i in range(len(images)):
            mask = masks[i, p, r].astype(bool) & (valid[i, p, 0] > 0)
            taper = np.minimum(distance_transform_edt(mask) / 2, 1).astype(np.float32) * mask
            x = images[i, p, 0]
            delta = np.clip(x - strength * taper * bands[i, p, int(band == "middle")], 0, 1) - x
            total = float(delta.sum(dtype=np.float64))
            basis = taper * ((1 - x) if total >= 0 else x)
            denominator = float(basis.sum(dtype=np.float64))
            correction = total / denominator if denominator > 0 else 0.
            if abs(correction) > 1 + 1e-5:
                raise ValueError("Level control leaves valid intensity interval")
            altered[i, p, 0] = x + delta
            level[i, p, 0] = x + correction * basis
            ok.append(bool(available[i, p] and mask.sum() >= 4 and np.linalg.norm(delta) > 1e-6))
            rows.append(dict(pixels=int(mask.sum()), l2=float(np.linalg.norm(delta)),
                             mean_control_error=float(abs(total - (level[i, p, 0] - x).sum(dtype=np.float64)))))
        proposals.append(altered)
        levels.append(level)
        eligible.append(ok)
        diagnostics.append(rows)
    return dict(interventions=np.stack(proposals), controls=np.stack(levels),
                eligible=np.asarray(eligible), conditions=CONDITIONS,
                region_names=REGIONS, strength=float(strength), diagnostics=diagnostics)


@torch.no_grad()
def measure_teacher(teacher, data, reference, reference_labels, legal=None, *, labels, strength=.5):
    """Freeze the supplied source teacher and measure all candidate responses.

    teacher(data, reference, reference_labels, legal=legal) returns logits and
    pre-existing M1 responses. Only its correctly predicted true-class weak
    spatial maps supervise attention. All valid candidate-class effects remain
    eligible even when the teacher is wrong. Neither masks nor references change
    across the original/intervention/control forwards.
    """
    teacher.requires_grad_(False).eval()
    original = teacher(data, reference, reference_labels, legal=legal)
    clean = original["logits"]
    responses = original["responses"]
    labels = torch.as_tensor(labels, device=clean.device, dtype=torch.long)
    valid = grid_occupancy(data["valid"].flatten(0, 1).float(), responses.shape[-2:])
    valid = valid.reshape(len(clean), 2, 1, *responses.shape[-2:])
    available = data["available"]
    attention = attention_from_response(responses, valid, available)
    correct = clean.argmax(-1) == labels
    present = (valid.sum((-1, -2, -3)) > 0) & available.bool()
    attention_mask = F.one_hot(labels, clean.shape[-1])[:, None] * correct[:, None, None] * present[:, :, None]

    operations = make_interventions(data, strength)
    changed_scores, control_scores = [], []
    for intervention, control in zip(operations["interventions"], operations["controls"]):
        for images, scores in ((intervention, changed_scores), (control, control_scores)):
            changed = dict(data, images=torch.as_tensor(images, device=data["images"].device,
                                                       dtype=data["images"].dtype))
            scores.append(teacher(changed, reference, reference_labels, legal=legal)["logits"])
    eligible = torch.as_tensor(operations["eligible"], device=clean.device)
    effects = effects_from_legacy(clean, torch.stack(changed_scores), CONDITIONS, eligible)
    controls = effects_from_legacy(clean, torch.stack(control_scores), CONDITIONS, eligible)
    return dict(attention=attention.detach(), effects=effects["effects"],
                attention_mask=attention_mask.to(clean.dtype), effect_mask=effects["effect_mask"],
                clean_logits=clean.detach(), control_effects=controls["effects"],
                conditions=CONDITIONS, region_names=REGIONS, strength=float(strength),
                diagnostics=operations["diagnostics"], teacher_forwards=25)
