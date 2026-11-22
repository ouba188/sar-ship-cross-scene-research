"""Source-only real-ship representatives and query/reference isolation."""
from collections import defaultdict
from collections.abc import Sequence
import numpy as np
import torch
from torch.nn import functional as F

from .contracts import EncodedShip, weighted_pool


@torch.no_grad()
def whole_ship_descriptors(ship: EncodedShip):
    """Return six normalized Fc target descriptors and their observation mask."""
    values, masks = [], []
    for p in range(2):
        branch, active = [], []
        for l in range(3):
            geometry = ship.geometry.levels[l]
            vector = weighted_pool(ship.content[l][p:p + 1], geometry.slots[p:p + 1, 0])[0]
            valid = geometry.slot_valid[p, 0] & (vector.norm() > 1e-12)
            branch.append(F.normalize(vector, dim=0) * valid)
            active.append(valid)
        values.append(torch.stack(branch))
        masks.append(torch.stack(active))
    return torch.stack(values), torch.stack(masks)


def branch_distance(left, right):
    """Mean cosine distance on common (polarization, scale) branches.

    No common observation gives infinity, not an artificial perfect match.
    The overlap count is returned explicitly; bank selection also tracks masks.
    """
    x, xm = left
    y, ym = right
    common = xm & ym
    count = common.sum()
    if not bool(count):
        return x.new_tensor(float("inf")), count
    distance = (1 - (x * y).sum(-1).clamp(-1, 1))[common].mean()
    return distance, count


def _distances(vectors, masks, rows, columns):
    """Compute only the requested cell, without an all-source quadratic matrix."""
    result = np.empty((len(rows), len(columns)), dtype=np.float32)
    y, ym = vectors[columns], masks[columns]
    for start in range(0, len(rows), 128):
        batch = rows[start:start + 128]
        x, xm = vectors[batch], masks[batch]
        similarity = torch.einsum("ipld,jpld->ijpl", x, y).clamp(-1, 1)
        common = xm[:, None] & ym[None, :]
        count = common.sum((-1, -2))
        distance = ((1 - similarity) * common).sum((-1, -2)) / count.clamp_min(1)
        distance[count == 0] = torch.inf
        result[start:start + len(batch)] = distance.numpy()
    return result


def _medoid(indices, vectors, masks):
    """Prefer a measured representative covering most observations, then medoid."""
    scores = []
    for start in range(0, len(indices), 128):
        batch = indices[start:start + 128]
        distances = _distances(vectors, masks, batch, indices)
        for index, row in zip(batch, distances):
            finite = np.isfinite(row)
            scores.append(((int(finite.sum()), -float(row[finite].mean()) if finite.any() else -np.inf,
                            int(masks[index].sum()), -index), index))
    return max(scores)[1]


@torch.no_grad()
def select_reference_indices(ships: Sequence[EncodedShip], labels, ports, vessel_ids,
                             group_ids, budget_per_class: int, source_mask=None):
    """Choose real source observations, at most budget_per_class per class.

    Pass an already source-training-only list (the default), or an explicit
    source_mask. No held-out feature is read when source_mask is supplied.
    Vessel identities are globally deduplicated before port coverage selection.
    Sparse cells are never padded by copying a vessel.
    """
    n = len(ships)
    if any(len(items) != n for items in (labels, ports, vessel_ids, group_ids)):
        raise ValueError("source metadata lengths differ")
    if budget_per_class < 1:
        raise ValueError("budget_per_class must be positive")
    if source_mask is None:
        candidates = list(range(n))
    else:
        if len(source_mask) != n:
            raise ValueError("source_mask length differs")
        candidates = [i for i in range(n) if bool(source_mask[i])]
    if not candidates:
        return []
    by_vessel = defaultdict(list)
    for i in candidates:
        if vessel_ids[i] is None or str(vessel_ids[i]).strip() == "":
            raise ValueError("reference selection requires a nonempty vessel identity")
        by_vessel[str(vessel_ids[i])].append(i)
    for indices in by_vessel.values():
        if len({int(labels[i]) for i in indices}) != 1:
            raise ValueError("one source vessel has conflicting class labels")
    descriptors = [whole_ship_descriptors(ships[i]) for i in candidates]
    if any(not bool(mask.any()) for _, mask in descriptors):
        raise ValueError("source reference has no valid whole-ship descriptor")
    vectors = torch.stack([item[0].detach().cpu() for item in descriptors])
    masks = torch.stack([item[1].detach().cpu() for item in descriptors])
    local = {original: i for i, original in enumerate(candidates)}
    representatives = []
    for indices in by_vessel.values():
        best = _medoid([local[i] for i in indices], vectors, masks)
        representatives.append(best)
    cells = defaultdict(list)
    for i in representatives:
        original = candidates[i]
        cells[int(labels[original]), str(ports[original])].append(i)
    selected = []
    for category in sorted({key[0] for key in cells}):
        category_cells = {port: indices for (label, port), indices in cells.items() if label == category}
        # If the budget cannot cover every port, record the returned port coverage
        # upstream; the deterministic choice favors more populated source cells.
        ordered_ports = sorted(category_cells, key=lambda port: (-len(category_cells[port]), port))
        chosen = []
        for port in ordered_ports[:budget_per_class]:
            chosen.append(_medoid(category_cells[port], vectors, masks))
        while len(chosen) < budget_per_class:
            options = []
            for port, members in category_cells.items():
                available = [i for i in members if i not in chosen]
                if not available:
                    continue
                covered = [i for i in chosen if i in members]
                if not covered:
                    options.append(((1., 1., np.inf, len(members), port),
                                    _medoid(available, vectors, masks)))
                    continue
                nearest = np.min(_distances(vectors, masks, members, covered), axis=1)
                nearest_by_index = dict(zip(members, nearest))
                patterns = {tuple(masks[i].flatten().tolist()) for i in covered}
                uncovered_pattern = [i for i in available if tuple(masks[i].flatten().tolist()) not in patterns]
                pool = uncovered_pattern or available
                # Farthest-first within the port; incomparable observation modes
                # are explicitly covered instead of winning via a fake distance.
                pick = max(pool, key=lambda i: (float(nearest_by_index[i]),
                                                int(masks[i].sum()), -i))
                finite = np.isfinite(nearest)
                priority = (float((~finite).mean()), float(bool(uncovered_pattern)),
                            float(nearest[finite].mean()) if finite.any() else np.inf,
                            len(members), port)
                options.append((priority, pick))
            if not options:
                break
            chosen.append(max(options, key=lambda item: item[0])[1])
        selected.extend(candidates[i] for i in chosen)
    return selected


def legal_reference_mask(query_vessel_ids, query_group_ids, ref_vessel_ids,
                         ref_group_ids, device=None):
    """Exclude the same physical vessel OR acquisition group before matching.

    Empty group IDs are unknown and do not form a shared group. A known vessel
    identity is still sufficient for the vessel exclusion.
    """
    if len(query_vessel_ids) != len(query_group_ids) or len(ref_vessel_ids) != len(ref_group_ids):
        raise ValueError("identity metadata lengths differ")
    mask = torch.ones(len(query_vessel_ids), len(ref_vessel_ids), dtype=torch.bool, device=device)
    def known(value):
        return value is not None and str(value).strip() != ""
    for i, (vessel, group) in enumerate(zip(query_vessel_ids, query_group_ids)):
        for j, (ref_vessel, ref_group) in enumerate(zip(ref_vessel_ids, ref_group_ids)):
            same_vessel = known(vessel) and known(ref_vessel) and str(vessel) == str(ref_vessel)
            same_group = known(group) and known(ref_group) and str(group) == str(ref_group)
            mask[i, j] = not (same_vessel or same_group)
    return mask
