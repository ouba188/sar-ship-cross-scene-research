"""Create a LOCAL, NOT TRAINING-QUALIFIED manifest from verified OSS2 pairs.

Run after audit_alignment.py. The original manifest and imagery are read only.
Unknown pixel validity is deliberately left unknown; no masks are generated.
The current two_stage.data loader understands the image/ship_axis interface but
does not enforce training_qualified. This output is a qualification draft only.
"""
from pathlib import Path
import copy
import csv
import hashlib
import json
import math
import sys
from collections import Counter
import numpy as np
from PIL import Image
import tifffile

ROOT = Path(__file__).resolve().parent
MANIFEST = Path('E:/临时会话/m1_coverage_20261101/manifest.json')
SOURCE_IDS = Path('E:/临时会话/oss2_port_prototypes_20261121/source_observations_local.csv')
EVIDENCE = ROOT/'row_evidence_local.csv'
OUTPUT = ROOT/'manifest_alignment_restored_local.json'
SUMMARY = ROOT/'restore_summary.json'
REQUIRED_FLAGS = (
    'same_pixel_index_derivation_verified', 'product_metadata_valid',
    'product_band_grid_equal', 'same_filename_center', 'same_product_folder',
    'xml_center_unique', 'xml_identity_match', 'raw_joint_pixel_interleaved',
    'raw_all_finite', 'native_shape_equal', 'vv_raw_to_uint8_exact',
    'vh_raw_to_uint8_exact', 'vv_png_equal_uint8', 'vh_png_equal_uint8',
)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_quantizer_replay(raw, visual, evidence, pol):
    """Recheck saved two-parameter evidence, without re-fitting or moving pixels."""
    a = float(evidence[f'{pol}_raw_to_uint8_slope'])
    b = float(evidence[f'{pol}_raw_to_uint8_offset'])
    if not math.isfinite(a) or not math.isfinite(b) or a <= 0:
        raise ValueError('Invalid saved radiometric mapping parameters')
    channel = int(evidence[f'{pol}_joint_channel'])
    if channel != {'vv': 1, 'vh': 0}[pol]:
        raise ValueError('Unexpected verified joint channel mapping')
    replay = np.clip(np.floor(raw[...,channel].astype(np.float64)*a+b+.5),0,255).astype(np.uint8)
    if not np.array_equal(replay, visual):
        raise ValueError('Raw -> Uint8 evidence no longer reproduces the current files')


def main():
    hashes_before = {str(p): sha256(p) for p in (MANIFEST, SOURCE_IDS, EVIDENCE)}
    original = json.loads(MANIFEST.read_text(encoding='utf8'))
    source = read_csv(SOURCE_IDS)
    source_ids = {r['sample_id'] for r in source}
    evidence_rows = read_csv(EVIDENCE)
    evidence = {r['sample_id']: r for r in evidence_rows}
    if len(source_ids) != len(source) or len(evidence) != len(evidence_rows):
        raise ValueError('Duplicate source or evidence identity')
    if source_ids != set(evidence) or len(source_ids) != 719:
        raise ValueError('Expected exactly the audited 719 source identities')
    originals = [r for r in original if r['id'] in source_ids]
    if len(originals) != 719 or len({r['id'] for r in originals}) != 719:
        raise ValueError('Missing or duplicate original source identity')
    if any(r['role'] not in ('fit_train','utility_train') for r in originals):
        raise ValueError('Non-source role in restoration scope')
    outputs = []; restored = Counter(); shapes = Counter()
    for old in originals:
        ev = evidence[old['id']]
        if not all(ev[k] == 'True' for k in REQUIRED_FLAGS):
            raise ValueError('An identity lacks the complete verified common-grid chain')
        if ev['role'] != old['role'] or int(ev['label']) != old['label'] or ev['product'] != old['scene_id']:
            raise ValueError('Evidence no longer matches original metadata')
        rawpath = Path(ev['raw_path'])
        raw = tifffile.imread(rawpath)
        with tifffile.TiffFile(rawpath) as tf:
            page = tf.pages[0]
            if len(tf.pages)!=1 or page.samplesperpixel!=2 or page.planarconfig!=1:
                raise ValueError('Original joint raster format changed')
        if raw.ndim != 3 or raw.shape[2] != 2 or not np.isfinite(raw).all():
            raise ValueError('Invalid joint raster')
        shape = tuple(raw.shape[:2]); shapes[f'{shape[0]}x{shape[1]}'] += 1
        if tuple(old['native_size']) != shape:
            raise ValueError('Original manifest native size disagrees with verified raster')
        r = copy.deepcopy(old)
        available = [p for p in ('vv','vh') if old[p] is not None]
        if not available:
            raise ValueError('No original target geometry to reference')
        for pol in ('vv','vh'):
            pngpath, tifpath = Path(ev[f'{pol}_png_path']), Path(ev[f'{pol}_path'])
            if not pngpath.is_absolute() or not tifpath.is_absolute():
                raise ValueError('Expected absolute verified image paths')
            if Path(old['source_paths'][pol]).resolve() != tifpath.resolve():
                raise ValueError('Original source path differs from verified evidence')
            visual = tifffile.imread(tifpath)
            with Image.open(pngpath) as im:
                if im.mode != 'L':
                    raise ValueError('Expected scalar 8-bit PNG')
                pixels = np.array(im)
            if visual.shape != shape or pixels.shape != shape or not np.array_equal(visual,pixels):
                raise ValueError('Verified source / PNG coordinate or intensity mismatch')
            verified_quantizer_replay(raw, visual, ev, pol)
            was_missing = old[pol] is None
            if was_missing:
                reference = available[0]
                axis = copy.deepcopy(old[reference]['ship_axis'])
                r[pol] = dict(image=str(pngpath), ship_axis=axis, detections=[], target_index=None,
                              geometry_source='paired_reference',
                              geometry_reference_polarization=reference,
                              geometry_reference_sample_id=old['id'],
                              detector_available=False, original_branch_missing=True)
                restored[pol] += 1
            else:
                if Path(old[pol]['image']).resolve() != pngpath.resolve():
                    raise ValueError('Original detector image is not the verified image')
                r[pol].update(geometry_source='original_detector',
                              detector_available=True, original_branch_missing=False)
            branch = r[pol]
            branch.update(
                alignment_valid=True,
                alignment_scope='same_native_joint_raster_pixel_indices',
                alignment_evidence=dict(row_evidence=str(EVIDENCE), sample_id=old['id'],
                    original_joint_tiff=str(rawpath), joint_channel=int(ev[f'{pol}_joint_channel']),
                    raw_to_uint8_exact=True, uint8_to_png_exact=True),
                radiometric_valid=None,
                radiometric_validity_status='unknown_native_valid_data_support',
                valid_mask_status='unknown_not_created', background_validated=False,
                training_qualified=False,
            )
            if branch.get('valid_mask') or branch.get('background_mask'):
                raise ValueError('Unexpected validity/background mask in original record')
        r.update(
            polarizations_coregistered=True,
            coregistration_scope='same_product_joint_raster_grid_no_measured_residual_error',
            actual_registration_error_known=False,
            radiometric_valid=None,
            radiometric_validity_status='unknown_native_valid_data_support',
            valid_mask_status='unknown_not_created',
            background_validated=False, leakage_controlled=False, training_qualified=False,
            qualification_status='alignment_verified_only_not_training_ready',
            qualification_missing=['native_valid_data_support','trusted_background_Q','leakage_control'],
            alignment_restoration=dict(
                source_manifest=str(MANIFEST), source_manifest_sha256=hashes_before[str(MANIFEST)],
                evidence_csv=str(EVIDENCE), evidence_scope='719_source_only',
                restored_polarizations=[p for p in ('vv','vh') if old[p] is None],
                original_roles_and_identity_preserved=True,
            ),
        )
        outputs.append(r)
    if restored != Counter(vv=32,vh=16):
        raise ValueError('Restoration count differs from verified 48 missing branches')
    if sum(r[p] is not None for r in outputs for p in ('vv','vh')) != 1438:
        raise ValueError('Incomplete dual-polarization output')
    # Check all original data fields are preserved; only branch metadata is supplemented.
    for old,new in zip(originals,outputs):
        for key,value in old.items():
            if key not in ('vv','vh') and new[key] != value:
                raise ValueError('Original identity, role, split, or other metadata changed')
        for pol in ('vv','vh'):
            if old[pol] is not None:
                if any(new[pol][k]!=v for k,v in old[pol].items()):
                    raise ValueError('Original branch values changed')
            else:
                ref=new[pol]['geometry_reference_polarization']
                if new[pol]['ship_axis']!=old[ref]['ship_axis'] or new[pol]['detections'] or 'confidence' in new[pol]:
                    raise ValueError('Borrowed geometry is incorrect or fake detections were introduced')
    hashes_after = {str(p): sha256(p) for p in (MANIFEST, SOURCE_IDS, EVIDENCE)}
    if hashes_before != hashes_after:
        raise ValueError('A source/evidence input changed during restoration')
    OUTPUT.write_text(json.dumps(outputs,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    roundtrip=json.loads(OUTPUT.read_text(encoding='utf8'))
    assert roundtrip==outputs
    summary=dict(
        status='alignment_verified_only_not_training_ready',
        source_observations=719, branches_before=1390, branches_after=1438,
        dual_polarization_before=671, dual_polarization_after=719,
        restored_branches=48, restored_by_polarization=dict(restored),
        roles=dict(Counter(r['role'] for r in outputs)),
        geometry_source_counts=dict(Counter(r[p]['geometry_source'] for r in outputs for p in ('vv','vh'))),
        original_input_hashes_unchanged=hashes_before==hashes_after,
        input_sha256_before=hashes_before,input_sha256_after=hashes_after,
        output_manifest=str(OUTPUT),output_sha256=sha256(OUTPUT),
        image_paths_and_shapes_verified=1438,full_raw_uint8_png_replays_verified=1438,
        restored_geometry_checks_passed=48,original_fields_preserved=True,
        validity_masks_created=0,background_masks_created=0,
        radiometric_validity='unknown',background_validated=False,
        leakage_controlled=False,training_qualified=False,
        loader_compatibility='Image/ship_axis schema supported by two_stage.data. The current loader does not enforce training_qualified and treats omitted valid_mask as footprint. Do not train from this draft.',
        native_shapes=dict(shapes),
    )
    SUMMARY.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('input_sha256_before','input_sha256_after','native_shapes')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf8')
    main()
