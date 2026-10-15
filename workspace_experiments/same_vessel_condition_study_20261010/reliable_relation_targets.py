"""Scene-balanced margin targets from metadata-matched teacher groups."""

import numpy as np
import pandas as pd
import torch


ARMS = ('mean', 'same', 'joint')
AUDIT_COLUMNS = ['anchor', 'class_id', 'n_same_scenes', 'n_other_identities',
                 'eligible'] + [name for arm in ARMS for name in
                                (arm + '_edges', arm + '_initial_violated_edges')]


def build_targets(prior: np.ndarray, meta: pd.DataFrame,
                  matches: pd.DataFrame) -> tuple[dict[str, dict[str, np.ndarray]],
                                                  pd.DataFrame]:
    """Build full-size targets and one audit row for every local anchor.

    Matches follow ``relation_teacher_matching.build_matches``: every group
    contains chips from one scene. Repeated same products and other (MMSI,
    product) scenes are merged, with each chip counted once. Scene means get
    equal weight within each side; the mean arm gives each side half the weight.
    Eligibility uses distinct same products and other MMSIs, never scores.
    Audit violations count only masked edges, using the clipped intervals.
    """
    prior = np.asarray(prior, dtype=np.float64)
    targets = {arm: dict(lower=np.zeros(prior.shape, dtype=np.float32),
                         upper=np.zeros(prior.shape, dtype=np.float32),
                         mask=np.zeros(prior.shape, dtype=bool),
                         eligible=np.zeros(len(prior), dtype=bool))
               for arm in ARMS}
    matched = {int(anchor): rows for anchor, rows in matches.groupby('anchor')}
    audit = []
    for anchor, label in enumerate(meta.class_id.to_numpy(dtype=int)):
        same_groups, other_groups = {}, {}
        if anchor in matched:
            for row in matched[anchor].itertuples(index=False):
                same = meta.iloc[row.same_indices[0]]
                other = meta.iloc[row.other_indices[0]]
                same_groups.setdefault(same['product'], set()).update(row.same_indices)
                other_groups.setdefault((other.mmsi, other['product']), set()).update(
                    row.other_indices)
        n_same = len(same_groups)
        n_other = len({mmsi for mmsi, product in other_groups})
        eligible = n_same >= 2 and n_other >= 2
        record = dict(anchor=anchor, class_id=int(label), n_same_scenes=n_same,
                      n_other_identities=n_other, eligible=eligible)
        if eligible:
            # Sort chip tuples so duplicate/reordered match rows give identical sums.
            sides = []
            for groups in (same_groups, other_groups):
                indices = sorted(tuple(sorted(ids)) for ids in groups.values())
                scores = np.stack([prior[list(ids)].mean(axis=0) for ids in indices])
                sides.append(scores[:, label, None] - scores)
            same_margins, other_margins = sides
            mean = (same_margins.mean(axis=0) + other_margins.mean(axis=0)) / 2
            pooled = np.concatenate(sides, axis=0)
            intervals = dict(mean=(mean, mean, mean > 0),
                             same=(same_margins.min(axis=0), same_margins.max(axis=0),
                                   (same_margins > 0).all(axis=0)),
                             joint=(pooled.min(axis=0), pooled.max(axis=0),
                                    (pooled > 0).all(axis=0)))
            for arm, (lower, upper, mask) in intervals.items():
                mask[label] = False
                targets[arm]['lower'][anchor] = np.clip(lower, 0, 2)
                targets[arm]['upper'][anchor] = np.clip(upper, 0, 2)
                targets[arm]['mask'][anchor] = mask
        margin = prior[anchor, label] - prior[anchor]
        for arm, arrays in targets.items():
            arrays['eligible'][anchor] = eligible
            mask = arrays['mask'][anchor]
            outside = ((margin < arrays['lower'][anchor])
                       | (margin > arrays['upper'][anchor]))
            record[arm + '_edges'] = int(mask.sum())
            record[arm + '_initial_violated_edges'] = int((mask & outside).sum())
        audit.append(record)
    return targets, pd.DataFrame(audit, columns=AUDIT_COLUMNS)


def interval_rows(scores: torch.Tensor, labels: torch.Tensor,
                  lower, upper, mask) -> torch.Tensor:
    """Return mean squared two-sided margin violations for each image.

    Bounds and masks may be target NumPy arrays or tensors on the score device.
    Rows without masked edges contribute zero.
    """
    lower = torch.as_tensor(lower, dtype=scores.dtype, device=scores.device)
    upper = torch.as_tensor(upper, dtype=scores.dtype, device=scores.device)
    mask = torch.as_tensor(mask, dtype=torch.bool, device=scores.device)
    margins = scores.gather(1, labels[:, None]) - scores
    violations = (lower - margins).clamp_min(0).square()
    violations = violations + (margins - upper).clamp_min(0).square()
    return (violations * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
