"""Held-source identity controls; multi-observation diagnostics, not deployment."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from relation_teacher_matching import build_matches


BASE = Path(__file__).resolve().parent
RUN = BASE / 'paired_evidence_factors_20261011'
STRATA = ('all', 'known_angle', 'unknown_angle')
METRICS = ('same_distance', 'other_distance', 'same_rescue_pct', 'same_harm_pct',
           'same_net_pct', 'other_rescue_pct', 'other_harm_pct', 'other_net_pct',
           'anchor_positive_pct', 'same_positive_pct', 'other_positive_pct')


def held_cohort(meta, held_ids):
    """All returned indices are LOCAL positions in the held-only frame."""
    held_ids = np.asarray(held_ids, dtype=int)
    frame = meta.iloc[held_ids].reset_index(drop=True).copy()
    frame['obs_id'] = held_ids
    matches = build_matches(frame)
    keep = []
    for row in matches.itertuples(index=False):
        anchor = frame.iloc[row.anchor]
        same = frame.iloc[list(row.same_indices)]
        other = frame.iloc[list(row.other_indices)]
        keep.append(row.match_kind == 'same_scene'
                    and (same.port != anchor.port).all()
                    and (same['product'] != anchor['product']).all()
                    and (same.mmsi == anchor.mmsi).all()
                    and (same.class_id == anchor.class_id).all()
                    and (other.mmsi != anchor.mmsi).all()
                    and (other.class_id == anchor.class_id).all()
                    and (other.port == same.port.iloc[0]).all()
                    and (other['product'] == same['product'].iloc[0]).all()
                    and (np.isnan(row.incidence_gap) or row.incidence_gap <= 2.))
    matches = matches.loc[np.asarray(keep, dtype=bool)].copy()
    matches['angle_status'] = np.where(np.isfinite(matches.incidence_gap),
                                       'known_angle', 'unknown_angle')
    return frame, matches.reset_index(drop=True)


def aligned_held(archive, held_ids, key):
    """Honor archive obs_ids even when exports are reordered or held-only."""
    ids = pd.Index(archive['obs_ids'])
    if not ids.is_unique:
        raise ValueError('Export obs_ids must be unique')
    positions = ids.get_indexer(held_ids)
    if (positions < 0).any():
        raise ValueError('Export is missing held observations')
    return np.asarray(archive[key][positions], dtype=np.float64)


def coverage(frame, matches):
    rows = []
    for stratum in STRATA:
        selected = matches if stratum == 'all' else matches[matches.angle_status == stratum]
        anchors = frame.iloc[selected.anchor.unique()]
        for label, group in frame.groupby('class_id', sort=True):
            covered = anchors[anchors.class_id == label]
            edges = selected[selected.anchor.isin(covered.index)]
            rows.append(dict(stratum=stratum, class_id=int(label),
                total_chips=len(group), total_ships=group.mmsi.nunique(),
                anchors=len(covered), ships=covered.mmsi.nunique(),
                acquisitions=len(covered.drop_duplicates(['mmsi', 'product', 'port'])),
                matched_pairs=len(edges), known_pairs=int(np.isfinite(edges.incidence_gap).sum()),
                unknown_pairs=int(edges.incidence_gap.isna().sum()),
                anchor_fraction=len(covered)/len(group),
                ship_fraction=covered.mmsi.nunique()/group.mmsi.nunique()))
    return pd.DataFrame(rows)


def anchor_metrics(values, scores, frame, matches):
    """Scene-mean donors, equal matched-scene teachers, no correctness filtering.

    Distances average squared Euclidean distance to individual donor scene means.
    Scores average those scene teachers before testing the seven rival orderings.
    Other donors retain matched-pair weights if reused, keeping the two sides paired.
    """
    rows = []
    for stratum in STRATA:
        selected = matches if stratum == 'all' else matches[matches.angle_status == stratum]
        for anchor, group in selected.groupby('anchor', sort=True):
            record = frame.iloc[anchor]
            label = int(record.class_id)
            rivals = np.arange(scores.shape[1]) != label
            base_positive = (scores[anchor, label] - scores[anchor])[rivals] > 1e-8
            result = dict(stratum=stratum, obs_id=int(record.obs_id),
                          mmsi=record.mmsi, class_id=label, product=record['product'],
                          port=record.port, pairs=len(group),
                          anchor_positive_pct=100*base_positive.mean())
            for side, column in [('same', 'same_indices'), ('other', 'other_indices')]:
                indices = group[column].tolist()
                donor_z = np.stack([values[list(ids)].mean(0) for ids in indices])
                teacher = np.mean([scores[list(ids)].mean(0) for ids in indices], axis=0)
                positive = (teacher[label]-teacher)[rivals] > 1e-8
                rescue = float(100*np.mean(~base_positive & positive))
                harm = float(100*np.mean(base_positive & ~positive))
                result.update({side+'_distance': float(np.square(values[anchor]-donor_z).sum(1).mean()),
                               side+'_rescue_pct': rescue, side+'_harm_pct': harm,
                               side+'_net_pct': rescue-harm,
                               side+'_positive_pct': float(100*positive.mean())})
            rows.append(result)
    return pd.DataFrame(rows, columns=['stratum', 'obs_id', 'mmsi', 'class_id',
                                      'product', 'port', 'pairs', *METRICS])


def summarize(records):
    """Equal chips within acquisition, acquisitions within ship, then ships/classes.

    Report both equal-ship and equal-covered-class means. Distance ratios are
    ratios of these weighted mean distances, never means of per-pair ratios.
    """
    summary, classes = [], []
    for stratum, part in records.groupby('stratum', sort=False):
        acquisitions = part.groupby(['class_id', 'mmsi', 'product', 'port'])[list(METRICS)].mean()
        ships = acquisitions.groupby(['class_id', 'mmsi']).mean()
        class_means = ships.groupby('class_id').mean()
        for label, row in class_means.iterrows():
            classes.append(dict(stratum=stratum, class_id=int(label), **row.to_dict(),
                distance_ratio=row.same_distance/row.other_distance if row.other_distance > 0 else np.nan))
        for weighting, row in [('equal_ship', ships.mean()), ('equal_class', class_means.mean())]:
            summary.append(dict(stratum=stratum, weighting=weighting,
                anchors=part.obs_id.nunique(), ships=len(ships), classes=len(class_means),
                **row.to_dict(),
                distance_ratio=row.same_distance/row.other_distance if row.other_distance > 0 else np.nan))
    return pd.DataFrame(summary), pd.DataFrame(classes)


def main():
    paths = [RUN/f'{branch}_{factor}_{seed}_vectors.npz'
             for branch in ('split_task', 'split_relation')
             for factor in ('paired', 'class_only', 'stable_only', 'pca', 'head')
             for seed in (31415, 27182)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError('\n'.join(missing))
    # Read only metadata for the held role; no target scores or labels enter metrics.
    meta = pd.read_csv(BASE/'observations.csv')
    with np.load(BASE/'grouped_condition_anchor_20261011/source_split.npz') as split:
        held_ids = split['test_source'].copy()
    frame, matches = held_cohort(meta, held_ids)
    del meta
    out = RUN/'identity_controls'
    out.mkdir(exist_ok=False)
    coverage(frame, matches).to_csv(out/'coverage.csv', index=False)
    exported = matches.copy()
    exported['anchor_obs_id'] = frame.iloc[matches.anchor].obs_id.to_numpy()
    for side in ('same', 'other'):
        exported[side+'_obs_ids'] = [json.dumps(frame.iloc[list(ids)].obs_id.tolist())
                                    for ids in matches[side+'_indices']]
    exported.to_csv(out/'matches.csv', index=False)
    summaries, class_rows, details = [], [], []
    for path in paths:
        name = path.stem.removesuffix('_vectors')
        branch = 'split_relation' if name.startswith('split_relation_') else 'split_task'
        factor, seed = name[len(branch)+1:].rsplit('_', 1)
        with np.load(path) as archive:
            for space in ('candidate', 'remainder'):
                values = aligned_held(archive, held_ids, 'z_'+space)
                scores = aligned_held(archive, held_ids, space+'_scores')
                records = anchor_metrics(values, scores, frame, matches)
                summary, by_class = summarize(records)
                tags = dict(branch=branch, factor=factor, seed=int(seed), space=space)
                summaries.append(summary.assign(**tags))
                class_rows.append(by_class.assign(**tags))
                details.append(records.assign(**tags))
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(out/'summary.csv', index=False)
    pd.concat(class_rows, ignore_index=True).to_csv(out/'class_metrics.csv', index=False)
    pd.concat(details, ignore_index=True).to_csv(out/'anchor_metrics.csv', index=False)
    protocol = dict(held_chips=len(frame), held_ships=int(frame.mmsi.nunique()),
        common_anchors=int(matches.anchor.nunique()),
        common_ships=int(frame.iloc[matches.anchor.unique()].mmsi.nunique()),
        matched_pairs=len(matches), known_pairs=int(np.isfinite(matches.incidence_gap).sum()),
        unknown_pairs=int(matches.incidence_gap.isna().sum()), exports=len(paths),
        cohort='Metadata-only held-source cohort, identical across models and spaces; no correctness filter.',
        matching='Same MMSI, different port/product. Other MMSI, same class, donor port/product exact; known scene-mean angle gap <=2 degrees. No fallback.',
        distances='Squared Euclidean distances in supplied z/sqrt(d); no additional division by dimension. Ratio is weighted same/other distance.',
        teachers='Average chips per donor scene, then equal matched scenes. Reused other scenes retain pair weights. Multi-observation diagnostic, not deployment.',
        relations='All seven true-vs-rival relations; margin>1e-8 is positive. Rescue: nonpositive to positive; harm: positive to nonpositive. Percentages use all seven relations.',
        aggregation='Equal chips/acquisition, acquisitions/vessel, then either vessels or covered classes. Strata have separate cohorts; never compare strata as causal effects.',
        limitations='Natural repeats, shared scenes and reused held development data; unknown angles retained separately; no causal identification or target evaluation.')
    (out/'PROTOCOL.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    selected = summary[(summary.stratum == 'all') & (summary.weighting == 'equal_class')]
    means = selected.groupby(['branch', 'factor', 'space'])[
        ['distance_ratio', 'same_net_pct', 'other_net_pct', 'anchor_positive_pct']].mean()
    report = ['# Held Source Identity Controls', '',
              json.dumps(protocol, indent=2), '', 'Two-seed descriptive means:', '',
              means.to_string(), '',
              'Positive net correction is teacher ordering rescue minus harm, not eight-class BA.',
              'Candidate/remainder distance magnitudes have different supplied dimension scaling.',
              'No training, target predictions, target metrics, or model selection performed.']
    (out/'REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps(protocol, indent=2))
    print(means.to_string())


if __name__ == '__main__':
    main()
