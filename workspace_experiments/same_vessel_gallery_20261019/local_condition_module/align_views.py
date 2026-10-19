"""Flag cross-view target conflicts without changing images, masks, or caches."""
import csv
import json
import os
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
MIN_CONFIDENCE = 0.25


def usable(detection):
    return detection.get('shape_usable', False) and detection['confidence'] >= MIN_CONFIDENCE


def contains(detection, point):
    corners = np.asarray(detection['corners'], dtype=np.float32)
    return cv2.pointPolygonTest(corners, tuple(map(float, point)), False) >= 0


def align(snapshot, cohort, index):
    vv, vh = (snapshot['images'][pol] for pol in ('vv', 'vh'))
    p, q = vv['priors'], vh['priors']
    s, t = p.get('scale') or {}, q.get('scale') or {}
    i, j = p.get('target_index', -1), q.get('target_index', -1)
    row = dict(snapshot_index=index, obs_id=snapshot['obs_id'],
               instance_id=cohort['instance_id'], product=cohort['product'],
               status='unknown', reason='', transform_usable=0, crs_basis='',
               scale_source_vv=s.get('source', ''), scale_source_vh=t.get('source', ''),
               crs_vv=s.get('crs', ''), crs_vh=t.get('crs', ''),
               vv_target_index=i, vh_target_index=j,
               vv_match_status=p.get('match_status', ''),
               vh_match_status=q.get('match_status', ''),
               vv_center_x='', vv_center_y='', mapped_vh_x='', mapped_vh_y='',
               inside_vh_target='', vh_other_indices='[]', disable_new_vh=0)

    for pol, view in (('vv', vv), ('vh', vh)):
        url = urlsplit(view['url'])
        image = unquote(url.path).lstrip('/')
        if (url.scheme != 'file'
                or os.path.normcase(os.path.normpath(image))
                != os.path.normcase(os.path.normpath(cohort['image_' + pol]))
                or not Path(image).name.startswith(cohort['product'] + '_' + pol.upper() + '_')):
            row['reason'] = 'image_product_or_path_mismatch'
            return row
    if any(scale.get('status') != 'valid_native_png_original_crop_grid' for scale in (s, t)):
        row['reason'] = 'missing_or_invalid_scale'
        return row
    if not s.get('source') or s['source'] != t.get('source'):
        row['reason'] = 'scale_source_mismatch'
        return row
    crs_vv, crs_vh = s.get('crs'), t.get('crs')
    if crs_vv is None and crs_vh is None:
        row['crs_basis'] = 'assumed_same_crs_same_product_same_scale_source'
    elif crs_vv and crs_vv == crs_vh:
        row['crs_basis'] = 'explicit_matching_crs'
    else:
        row['reason'] = 'crs_missing_one_view_or_mismatch'
        return row
    transforms = [np.asarray(scale.get('transform', []), dtype=float) for scale in (s, t)]
    if any(a.shape != (6,) or not np.isfinite(a).all() for a in transforms):
        row['reason'] = 'invalid_transform'
        return row
    a, b = transforms
    av, ah = a[[1, 2, 4, 5]].reshape(2, 2), b[[1, 2, 4, 5]].reshape(2, 2)
    if np.linalg.det(av) == 0 or np.linalg.det(ah) == 0:
        row['reason'] = 'singular_transform'
        return row
    row['transform_usable'] = 1
    detections_vv, detections_vh = p.get('detections', []), q.get('detections', [])
    if not 0 <= i < len(detections_vv) or not usable(detections_vv[i]):
        row['reason'] = 'missing_or_unusable_vv_target'
        return row
    center = np.asarray(detections_vv[i]['xywhr'][:2], dtype=float)
    # Transform continuous native coordinates directly; no resize or half-pixel shift.
    mapped = np.linalg.solve(ah, a[[0, 3]] + av @ center - b[[0, 3]])
    row.update(vv_center_x=float(center[0]), vv_center_y=float(center[1]),
               mapped_vh_x=float(mapped[0]), mapped_vh_y=float(mapped[1]))
    if not (0 <= mapped[0] < vh['w'] and 0 <= mapped[1] < vh['h']):
        row.update(status='outside', reason='mapped_center_outside_vh_image')
        return row
    if not 0 <= j < len(detections_vh) or not usable(detections_vh[j]):
        row['reason'] = 'missing_or_unusable_vh_target'
        return row
    in_target = contains(detections_vh[j], mapped)
    others = [k for k, det in enumerate(detections_vh)
              if k != j and usable(det) and contains(det, mapped)]
    row.update(inside_vh_target=int(in_target), vh_other_indices=json.dumps(others))
    if in_target:
        row.update(status='target', reason='mapped_center_in_vh_target')
    elif others:
        row.update(status='other', reason='mapped_center_in_other_not_target', disable_new_vh=1)
    else:
        row['reason'] = 'mapped_center_in_image_without_qualified_box'
    return row


def main():
    snapshot_path = ROOT / 'view_snapshot.json'
    cohort_path = ROOT.parent / 'conditional_evidence_pilot/cohort.csv'
    snapshots = json.loads(snapshot_path.read_text(encoding='utf-8'))
    with cohort_path.open(encoding='utf-8-sig', newline='') as stream:
        cohort_rows = list(csv.DictReader(stream))
    cohort = {int(row['obs_id']): row for row in cohort_rows}
    ids = [row['obs_id'] for row in snapshots]
    if (len(cohort) != len(cohort_rows) or len(set(ids)) != len(ids)
            or set(ids) != set(cohort)):
        raise ValueError('Snapshot and cohort must contain the same unique observation IDs')
    rows = [align(snapshot, cohort[snapshot['obs_id']], index)
            for index, snapshot in enumerate(snapshots)]
    counts = Counter(row['status'] for row in rows)
    report = dict(
        status='complete', snapshot=str(snapshot_path), cohort=str(cohort_path),
        n=len(rows), status_counts={key: counts[key] for key in ('target', 'other', 'outside', 'unknown')},
        reason_counts=dict(Counter(row['reason'] for row in rows)),
        transform_usable=sum(row['transform_usable'] for row in rows),
        crs_basis_counts=dict(Counter(row['crs_basis'] or 'unavailable' for row in rows)),
        disable_new_vh=sum(row['disable_new_vh'] for row in rows),
        candidate_rule=dict(min_confidence=MIN_CONFIDENCE, shape_usable=True),
        direction='VV target center -> VH native image; boundary counts as inside box',
        crs_assumption='Snapshot scale has no CRS. Matching original product, exact cohort image paths, '
                       'valid native-crop scales and identical scale source are assumed to share a CRS. '
                       'This assumption is not independent georeferencing verification.',
        scope='Entire fixed snapshot cohort, not the full 24-port image inventory. '
              'Point-in-box correspondence is not proof of ship identity. '
              'Only other sets disable_new_vh=1; unknown/outside are not confirmed alignment. '
              'Flags are for the new VH channel only; no masks, caches or old-model changes.',
        cases={str(row['obs_id']): row for row in rows if row['obs_id'] in (31237, 5155, 21571, 21572)},
    )
    with (ROOT / 'crossview_alignment.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / 'ALIGNMENT.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('n', 'status_counts', 'transform_usable', 'disable_new_vh')}, indent=2))
    print(json.dumps(report['cases']['31237'], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
