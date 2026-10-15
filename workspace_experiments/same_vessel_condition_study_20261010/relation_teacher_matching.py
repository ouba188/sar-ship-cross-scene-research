"""Metadata-only teacher matching within the supplied source training frame."""

import numpy as np
import pandas as pd


MATCH_COLUMNS = ['anchor', 'same_indices', 'other_indices', 'match_kind',
                 'incidence_gap']


def _observed(values):
    return (values.notna() & ~values.astype(str).str.strip().str.lower().isin(
        ['', 'unknown'])).all()


def build_matches(meta: pd.DataFrame) -> pd.DataFrame:
    """Return one matched scene pair per anchor and eligible same-ship product.

    Indices are row positions in ``meta``, never its index labels. Each tuple
    contains all chips of one (mmsi, product) group; average within each tuple
    before averaging matched rows to give scenes equal weight. Other-ship
    groups may be reused by different matches. No unmatched rows are returned.

    Scene incidence is the chip mean, available only when all chips have finite
    incidence. Fallback also requires observed, consistent pass and platform
    within both groups. Exact scene matches may lack angle metadata, but known
    angle gaps over two degrees are excluded even within the same product.
    """
    frame = meta.reset_index(drop=True)
    scenes = []
    for (mmsi, product), group in frame.groupby(['mmsi', 'product'], sort=True):
        first = group.iloc[0]
        angles = group.incidence_deg.to_numpy(dtype=float, na_value=np.nan)
        scenes.append(dict(
            mmsi=mmsi, product=product, port=first.port, class_id=first.class_id,
            indices=tuple(int(i) for i in group.index),
            incidence_deg=float(angles.mean()) if np.isfinite(angles).all() else np.nan,
            pass_name=(first.pass_name if _observed(group.pass_name)
                       and group.pass_name.nunique() == 1 else None),
            platform=(first.platform if _observed(group.platform)
                      and group.platform.nunique() == 1 else None)))
    scene_frame = pd.DataFrame(scenes)
    matches = []
    if len(scene_frame):
        for anchor, row in frame.iterrows():
            same = scene_frame[(scene_frame.mmsi == row.mmsi)
                               & (scene_frame['product'] != row['product'])]
            other = scene_frame[(scene_frame.mmsi != row.mmsi)
                                & (scene_frame.class_id == row.class_id)
                                & (scene_frame['product'] != row['product'])]
            for donor in same.itertuples(index=False):
                candidates = other[other.port == donor.port].copy()
                candidates['gap'] = (candidates.incidence_deg - donor.incidence_deg).abs()
                selected = candidates[(candidates['product'] == donor.product)
                                      & (candidates.gap.isna() | (candidates.gap <= 2.))]
                kind = 'same_scene'
                if selected.empty:
                    if (not np.isfinite(donor.incidence_deg)
                            or donor.pass_name is None or donor.platform is None):
                        continue
                    selected = candidates[(candidates.pass_name == donor.pass_name)
                                          & (candidates.platform == donor.platform)
                                          & (candidates.gap <= 2.)]
                    kind = 'fallback'
                if selected.empty:
                    continue
                order = ['mmsi', 'product'] if kind == 'same_scene' else ['gap', 'mmsi', 'product']
                candidate = selected.sort_values(order).iloc[0]
                matches.append((int(anchor), donor.indices, candidate['indices'],
                                kind, float(candidate.gap)))
    return pd.DataFrame(matches, columns=MATCH_COLUMNS).astype(
        {'anchor': 'int64', 'incidence_gap': 'float64'})


def summarize_matches(meta: pd.DataFrame, matches: pd.DataFrame) -> dict:
    """Return metadata coverage and pair-level match fractions, without I/O.

    Coverage counts unique matched anchors, their ships and classes. Per-class
    denominators include every input row/ship, including unmatched classes.
    Gap statistics omit unobserved angles and use degrees; empty statistics
    are None and empty match fractions are zero.
    """
    anchors = meta.iloc[matches.anchor.unique()]
    per_class = {}
    for class_id, group in meta.groupby('class_id', sort=True):
        covered = anchors[anchors.class_id == class_id]
        per_class[str(class_id)] = dict(
            total_anchors=int(len(group)), unique_anchors=int(len(covered)),
            anchor_coverage=float(len(covered) / len(group)),
            total_ships=int(group.mmsi.nunique()),
            unique_ships=int(covered.mmsi.nunique()),
            ship_coverage=float(covered.mmsi.nunique() / group.mmsi.nunique()))
    gaps = matches.loc[np.isfinite(matches.incidence_gap), 'incidence_gap']
    return dict(
        match_count=int(len(matches)), unique_anchors=int(len(anchors)),
        unique_ships=int(anchors.mmsi.nunique()),
        unique_classes=int(anchors.class_id.nunique()), per_class=per_class,
        same_scene_fraction=float((matches.match_kind == 'same_scene').mean()) if len(matches) else 0.,
        fallback_fraction=float((matches.match_kind == 'fallback').mean()) if len(matches) else 0.,
        incidence_gap=dict(count=int(len(gaps)),
                           mean=float(gaps.mean()) if len(gaps) else None,
                           median=float(gaps.median()) if len(gaps) else None,
                           max=float(gaps.max()) if len(gaps) else None))
