"""Select real source observations; never average local tokens across ships."""
import warnings

import numpy as np

from .protocol import _validate_records


def _squared_distances(x, y):
    return np.maximum((x * x).sum(1)[:, None] + (y * y).sum(1)[None]
                      - 2 * x @ y.T, 0)


def select_prototypes(records, descriptors, per_class, method="coverage", seed=17,
                      port_balance=True, observation="medoid"):
    """Return indices into source-only ``records``, ordered by class then choice.

    Each vessel supplies one vote and one real observation nearest its normalized
    descriptor mean. Observation IDs only break ties; observation='first' is the
    historical control. Conflicting vessel labels are rejected. Descriptors are
    normalized for selection only; selected ships are re-encoded by the live model.

    Coverage starts at the real observation nearest the class mean and uses
    farthest-first coverage. With port_balance, each next choice comes from a
    currently least represented available port. One assignment/medoid update
    then chooses the real point nearest each cluster mean, preserving selected
    ports. These are squared-Euclidean medoids, not free learned centroids.

    Random uses the same unique-vessel pool and optional port quotas. A class
    with fewer ships than per_class returns all available ships and warns with
    its actual count; observations are never replicated to fill the budget.
    """
    _validate_records(records)
    if not isinstance(per_class, (int, np.integer)) or per_class < 1:
        raise ValueError("per_class must be a positive integer")
    if method not in ("coverage", "random"):
        raise ValueError("method must be coverage or random")
    features = np.asarray(descriptors, dtype=np.float64)
    if features.ndim != 2 or len(features) != len(records) or features.shape[1] == 0:
        raise ValueError("descriptors must have shape [len(records), D] with D > 0")
    if not np.isfinite(features).all():
        raise ValueError("descriptors contain nonfinite values")
    features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)

    if observation not in ("medoid", "first"):
        raise ValueError("observation must be medoid or first")
    vessels = {}
    for i in sorted(range(len(records)), key=lambda j: str(records[j]["id"])):
        row = records[i]
        members = vessels.setdefault(row["vessel_id"], [])
        if members and records[members[0]]["label"] != row["label"]:
            raise ValueError(f"Conflicting source labels for vessel: {row['vessel_id']}")
        members.append(i)
    unique = {}
    for vessel, members in vessels.items():
        center = features[members].mean(0, keepdims=True)
        nearest = _squared_distances(features[members], center)[:, 0].argmin()
        unique[vessel] = members[int(nearest)] if observation == "medoid" else members[0]

    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted({row["label"] for row in records}):
        indices = [i for i in unique.values() if records[i]["label"] == label]
        size = min(per_class, len(indices))
        if size < per_class:
            warnings.warn(f"Class {label}: requested {per_class} prototypes, only {size} distinct source vessels",
                          UserWarning, stacklevel=2)
        x = features[indices]
        ports = np.asarray([records[i]["port"] for i in indices], dtype=object)
        chosen = []
        counts = {p: 0 for p in ports}
        nearest = np.full(len(indices), np.inf)
        for _ in range(size):
            candidates = np.asarray([i for i in range(len(indices)) if i not in chosen], dtype=np.int64)
            if port_balance:
                least = min(counts[ports[i]] for i in candidates)
                candidates = np.asarray([i for i in candidates if counts[ports[i]] == least], dtype=np.int64)
            if method == "random":
                picked = int(rng.choice(candidates))
            elif not chosen:
                center = x.mean(0, keepdims=True)
                picked = int(candidates[_squared_distances(x[candidates], center)[:, 0].argmin()])
            else:
                picked = int(candidates[nearest[candidates].argmax()])
            chosen.append(picked)
            counts[ports[picked]] += 1
            nearest = np.minimum(nearest, _squared_distances(x, x[[picked]])[:, 0])

        if method == "coverage" and chosen:
            assignment = _squared_distances(x, x[chosen]).argmin(1)
            medoids = []
            for cluster, anchor in enumerate(chosen):
                members = np.flatnonzero(assignment == cluster)
                eligible = [i for i in members
                            if i not in set(chosen) - {anchor}
                            and (not port_balance or ports[i] == ports[anchor])]
                if eligible:
                    center = x[members].mean(0, keepdims=True)
                    anchor = eligible[_squared_distances(x[eligible], center)[:, 0].argmin()]
                medoids.append(int(anchor))
            chosen = medoids
        selected.extend(indices[i] for i in chosen)
    return np.asarray(selected, dtype=np.int64)

