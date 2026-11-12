"""Export case 7797 from recorded results and exact historical CPU pixel functions."""
import ast
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, gaussian_filter

OUT = Path(__file__).resolve().parent
GALLERY = OUT.parents[1] / 'same_vessel_gallery_20261019'
ROOT = GALLERY / 'causal_frequency_side_20261024'
SOURCE = ROOT / 'round22_classification_coverage'
SNAPSHOT = GALLERY / 'local_condition_module/view_snapshot.json'


def cpu_functions(path, names, namespace):
    # Execute only the existing pure pixel functions; exclude model imports and main().
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def main():
    with (SOURCE / 'interventions.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = [r for r in csv.DictReader(handle) if r['obs_id'] == '7797' and float(r['dose']) == 1]
    row = next(r for r in rows if (r['pol'], r['region'], r['band']) == ('vv', 'body', 'middle'))
    view = next(v for v in json.loads(SNAPSHOT.read_text(encoding='utf-8')) if v['obs_id'] == 7797)['images']['vv']
    original = np.asarray(Image.open(row['image_path']).convert('L'), dtype=float) / 255
    prepare = GALLERY / 'local_condition_module/prepare.py'
    joint = ROOT / 'joint_probe.py'
    probe = ROOT / 'classification_joint_probe.py'
    body = cpu_functions(prepare, ['masks_for'], dict(np=np, Image=Image, ImageDraw=ImageDraw))
    axial = cpu_functions(joint, ['masks_for'], dict(np=np))
    functions = cpu_functions(probe, ['regions', 'bands', 'perturb'], dict(
        np=np, gaussian_filter=gaussian_filter, distance_transform_edt=distance_transform_edt,
        d=SimpleNamespace(ip=SimpleNamespace(masks_for=body['masks_for'])),
        j=SimpleNamespace(masks_for=axial['masks_for'])))
    altered, level, meta = next(v for v in functions['perturb'](original, view)
                                if v[2]['region'] == 'body' and v[2]['band'] == 'middle' and v[2]['dose'] == 1)
    errors = {k: abs(meta[k] - float(row[k])) for k in
              ('l2', 'mean_delta', 'level_l2', 'pixels', 'band_energy_before', 'band_energy_after')}
    assert max(errors.values()) < 1e-12, errors
    difference = altered - original
    difference_limit = 0.02
    assert abs(difference).max() <= difference_limit
    assets = OUT / 'assets'
    assets.mkdir(exist_ok=True)
    paths = {}
    for name, pixels in [('original', original), ('intervened', altered), ('mean_control', level)]:
        path = assets / f'l2_{name}.png'
        gray = Image.fromarray(np.rint(pixels * 255).astype(np.uint8)).convert('RGB')
        gray.resize((512, 512), Image.Resampling.NEAREST).save(path)
        paths[name] = str(path)
    # Fixed signed range: blue negative, white zero, red positive; no image stretching.
    signed = difference / difference_limit
    rgb = np.stack((1 + np.minimum(signed, 0), 1 - abs(signed), 1 - np.maximum(signed, 0)), axis=-1)
    path = assets / 'l2_difference.png'
    Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).resize((512, 512), Image.Resampling.NEAREST).save(path)
    paths['difference'] = str(path)
    data = dict(
        case_id=7797, dataset='24-port source dataset; round22 analyzed a six-port subset',
        mmsi=row['mmsi'], port=row['port'], product=row['product'], overpass=row['overpass'],
        class_id=3, class_name='product/chemical tanker', rival=7, rival_name='offshore supply ship',
        fold=int(row['fold']), pol='vv', region='body', band='middle', dose=1.0,
        base_pred=int(row['base_pred']), pred=int(row['pred']),
        baseline_margin=float(row['baseline_margin']), delta_fixed=float(row['delta_fixed']),
        level_delta=float(row['level_delta']), residual=float(row['residual']),
        margin_after=float(row['baseline_margin']) + float(row['delta_fixed']),
        mean_control_margin_after=float(row['baseline_margin']) + float(row['level_delta']),
        true_score_change=float(row['true_score_change']), rival_score_change=float(row['rival_score_change']),
        margin_definition='m = score[class 3] - score[fixed original strongest wrong class 7]; delta = m(treated) - m(original). Scores are diagnostic decision scores, not probabilities.',
        model_description='Historical frozen full_global diagnostic: frozen ResNet50 conv1/BN/ReLU responses, four-rotation pooled mean/max descriptors and existing fold-specific Ridge classifier; not current complete M1-M3 model. No inference or training in this export.',
        intervention_description='middle = Gaussian(sigma=1, reflect) - Gaussian(sigma=2, reflect), native pixels. Existing perturb() uses tapered target mask and matched L2 across six region/scale operations within each polarization. Only VV is changed; VH stays original.',
        pixel_metadata=meta, reconstruction_errors_against_recorded_cache=errors,
        display=dict(image_range=[0, 1], output_mode='RGB uint8', native_shape=list(original.shape),
                     exported_shape=[512, 512], resampling='nearest, 8x display only',
                     difference_range=[-difference_limit, difference_limit],
                     difference_actual_range=[float(difference.min()), float(difference.max())],
                     difference_colors='negative blue, zero white, positive red',
                     note='PNGs are quantized display assets; recorded scores refer to float64 intervention arrays before PNG quantization. No per-image contrast stretching.'),
        source_files=dict(original_vv=row['image_path'],
                          original_vh=next(r['image_path'] for r in rows if r['pol'] == 'vh'),
                          interventions=str(SOURCE / 'interventions.csv'), baseline=str(SOURCE / 'baseline.csv'),
                          completion_manifest=str(SOURCE / 'manifest.json'), view_snapshot=str(SNAPSHOT),
                          perturb_functions=str(probe), body_mask_function=str(prepare), axial_masks_function=str(joint),
                          montage=str(SOURCE / 'case_rescued_7797.png'),
                          montage_generator=str(ROOT / 'classification_joint_coverage.py')),
        assets=paths,
        condition_table=[dict(pol=r['pol'], region=r['region'], band=r['band'],
                              delta_fixed=float(r['delta_fixed']), level_delta=float(r['level_delta']),
                              l2=float(r['l2']), pred=int(r['pred'])) for r in rows],
        limitations=[
            'Post-hoc illustrative rescue, with final margin only about 9.64e-5; not a robust population gain.',
            'Historical diagnostic intervention, not current complete-model or module efficacy evidence.',
            'VV/VH use their own masks and L2 values; cross-polarization contrast does not isolate physical polarization.',
            'Mean control matches total intensity change, not L2 or the complete pixel distribution.',
            'Original source is an 8-bit display chip, not calibrated SAR backscatter.',
            'Original montage component column is the Gaussian scale component, not the actual pixel difference.'
        ],
    )
    (OUT / 'l2.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(case_id=7797, outputs=paths, metadata=str(OUT / 'l2.json'),
                          maximum_reconstruction_error=max(errors.values()),
                          difference_actual_range=data['display']['difference_actual_range']), indent=2))


if __name__ == '__main__':
    main()
