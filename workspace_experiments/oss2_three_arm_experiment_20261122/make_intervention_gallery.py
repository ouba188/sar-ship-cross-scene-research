"""CPU-only inspection of the locked seed-17 first supplemental batch."""
from pathlib import Path
import hashlib
import json
import os
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parent
CODE = Path('E:/临时会话/port_prototype_model_20261121')
sys.path.insert(0, str(CODE))
from sar_port.two_stage.cache import prepare_record_cached
from sar_port.two_stage.cli import SupplementalInterventions
from sar_port.two_stage.data import InputConfig, collate, _axis
from sar_port.two_stage.geometry import GeometryConfig, rectangle, coverage


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def panel(pixels, q=None, target=None, protection=None):
    rgb = np.repeat(np.round(np.clip(pixels, 0, 1)*255).astype(np.uint8)[..., None], 3, -1)
    if q is not None:
        rgb[q] = np.round(.5*rgb[q] + .5*np.array([0, 220, 255])).astype(np.uint8)
    image = Image.fromarray(rgb).resize((256, 256), Image.Resampling.NEAREST)
    if q is not None:
        draw = ImageDraw.Draw(image)
        for shape, color in ((protection, '#ff8833'), (target, '#ffff00')):
            xy = [(4*(x+.5), 4*(y+.5)) for x, y in shape.exterior.coords]
            draw.line(xy, fill=color, width=2)
    return image


def main():
    torch.set_num_threads(2)
    os.environ['SAR_PORT_PREPARE_CACHE'] = str(ROOT/'prepared_cache_local')
    rows = read(ROOT/'experiment_manifest_local.json')
    cfg = read(ROOT/'configs/disentangle_seed17.json')
    inp = InputConfig(**cfg['input'])
    assert tuple(inp.mean) == (0., 0.) and tuple(inp.std) == (1., 1.)
    geom = GeometryConfig(**cfg['geometry'])
    source_ids = [i for i, row in enumerate(rows) if row['development_split'] == 'train']
    lookup = {row['id']: i for i, row in enumerate(rows)}
    wanted = set(cfg['donor_ids']) | set(cfg['supplemental_batches'][0])
    samples = {lookup[name]: prepare_record_cached(rows[lookup[name]], ROOT, inp, geom)
               for name in sorted(wanted)}
    supplemental = SupplementalInterventions(samples, rows, source_ids, cfg)
    ids = supplemental.batches[0]
    assert len(ids) == 8 and len(supplemental.donors) == 8
    batch = collate([samples[i] for i in ids], 'cpu')
    assert batch['images'].shape == (8, 2, 1, 64, 64)
    original_randint = torch.randint
    traces = []

    def traced_randint(*args, **kwargs):
        result = original_randint(*args, **kwargs)
        traces.append(result.clone())
        return result

    with patch('sar_port.two_stage.cli.torch.randint', side_effect=traced_randint):
        donors = supplemental.apply(batch, ids, samples)
    assert len(traces) == len(ids)
    output = ROOT/'gallery_local'
    output.mkdir(exist_ok=True)
    original = batch['images'][:, :, 0].numpy()
    altered = batch['intervention_images'][:, :, 0].numpy()
    qmasks = batch['background'][:, :, 0].numpy().astype(bool)
    details, shapes = [], []
    for position, (i, j, field) in enumerate(zip(ids, donors, traces)):
        sample, donor = samples[i], samples[j]
        axis = _axis(rows[i]['vv']).copy()
        axis[:2] += (sample.placement['shift_x'], sample.placement['shift_y'])
        target = rectangle(axis)
        protection = target.buffer(inp.protect_margin)
        protected_pixels = coverage(protection, 64, 1) > 0
        shapes.append((target, protection))
        joint_donor_q = donor.background.bool().all(0).flatten()
        donor_pixels = torch.where(joint_donor_q)[0]
        selected_indices = donor_pixels[field]
        assert bool(joint_donor_q[selected_indices].all())
        assert np.array_equal(qmasks[position, 0], qmasks[position, 1])
        assert int(qmasks[position, 0].sum()) == 128
        expected = original[position].copy()
        for p in (0, 1):
            values = donor.images[p].flatten()[selected_indices].reshape(64, 64).numpy()
            expected[p][qmasks[position, p]] = values[qmasks[position, p]]
        assert np.array_equal(expected, altered[position])
        outside_equal, protect_equal, changes, overlap = [], [], [], []
        for p in (0, 1):
            q = qmasks[position, p]
            outside_equal.append(bool(np.array_equal(original[position, p][~q], altered[position, p][~q])))
            protect_equal.append(bool(np.array_equal(original[position, p][protected_pixels], altered[position, p][protected_pixels])))
            changes.append(int(np.count_nonzero(original[position, p] != altered[position, p])))
            overlap.append(int(np.count_nonzero(q & protected_pixels)))
        assert all(outside_equal) and all(protect_equal) and overlap == [0, 0]
        destinations = np.flatnonzero(qmasks[position, 0])
        selected_at_q = selected_indices.numpy()[destinations]
        details.append(dict(query_id=rows[i]['id'], donor_id=rows[j]['id'],
                            class_label=str(rows[i]['label']), query_split=rows[i]['development_split'],
                            donor_split=rows[j]['development_split'], native_hw=sample.placement['native_hw'],
                            canvas=64, q_pixels_per_pol=[128, 128], q_outside_exact_equal=outside_equal,
                            protected_exact_equal=protect_equal, q_protection_overlap_pixels=overlap,
                            changed_pixels_per_pol=changes, vv_vh_same_donor_field_verified=True,
                            actual_matches_reconstructed_field=True, donor_samples_all_inside_joint_q=True,
                            donor_joint_q_pixels=int(joint_donor_q.sum()),
                            source_field_sha256=hashlib.sha256(selected_indices.numpy().tobytes()).hexdigest(),
                            destination_flat_indices=destinations.tolist(),
                            donor_source_flat_indices_shared_by_vv_vh=selected_at_q.tolist(),
                            leakage_controlled=bool(rows[i].get('leakage_controlled', False))))
    font_path = 'C:/Windows/Fonts/arial.ttf'
    font = ImageFont.truetype(font_path, 17)
    small = ImageFont.truetype(font_path, 15)
    title = ImageFont.truetype(font_path, 24)
    columns = ['VV original', 'VV Q + target/protection', 'VV intervened',
               'VH original', 'VH Q + target/protection', 'VH intervened']
    for page in range(2):
        canvas = Image.new('RGB', (1624, 1335), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 12), f'OSS2 seed 17 - first supplemental batch - page {page+1}/2', fill='black', font=title)
        draw.text((16, 46), 'Native 64 x 64; fixed grayscale 0..1; nearest-neighbor display; NO intensity normalization.', fill='black', font=font)
        draw.text((16, 72), 'Cyan: trusted Q (128 pixels/pol). Yellow: target OBB. Orange: 8-pixel protected margin.', fill='black', font=font)
        draw.text((16, 98), 'Replacement samples paired VV/VH source pixels ONLY inside donor Q; only query Q changes.', fill='black', font=font)
        for c, label in enumerate(columns):
            draw.text((16+266*c, 128), label, fill='black', font=small)
        for row_position, position in enumerate(range(4*page, 4*page+4)):
            y = 158 + 290*row_position
            detail = details[position]
            label = (f'{position+1}. {detail["query_id"]} | class {detail["class_label"]} | '
                     f'changed VV/VH {detail["changed_pixels_per_pol"]} | outside Q + protection EXACT unchanged')
            draw.text((16, y), label, fill='black', font=small)
            target, protection = shapes[position]
            for p in (0, 1):
                pics = [panel(original[position, p]),
                        panel(original[position, p], qmasks[position, p], target, protection),
                        panel(altered[position, p])]
                for c, picture in enumerate(pics):
                    canvas.paste(picture, (16+266*(p*3+c), y+24))
        canvas.save(output/f'intervention_page_{page+1:02d}_local.png')
    audit = dict(seed=17, supplemental_occurrence=1, no_training=True, device='cpu',
                 normalization='none; exact native uint8 divided by 255', donor_pool_count=8,
                 actual_distinct_donors=len(set(donors)),
                 first_occurrence_explanation='Identical ordered donor sequences start at index zero for each query.',
                 scope='Only trusted Q is replaced; this is not whole-background intervention or causal-independence certification.',
                 records=details)
    write(ROOT/'intervention_audit_local.json', audit)
    aggregate = dict(seed=17, supplemental_occurrence=1, queries=8, polarizations=2,
                     canvas_hw=[64,64], q_pixels_per_polarization=128, donor_pool_count=8,
                     actual_distinct_donors=len(set(donors)), q_outside_exact_equal_all=True,
                     protected_pixels_exact_equal_all=True, q_protection_overlap_pixels=0,
                     shared_donor_field_verified_all=True, donor_samples_inside_joint_q_all=True,
                     reconstructed_intervention_exact_all=True,
                     changed_pixels_per_pol_total=np.sum([v['changed_pixels_per_pol'] for v in details], axis=0).tolist(),
                     changed_pixels_per_pol_min=np.min([v['changed_pixels_per_pol'] for v in details], axis=0).tolist(),
                     changed_pixels_per_pol_max=np.max([v['changed_pixels_per_pol'] for v in details], axis=0).tolist(),
                     controlled_qualified_queries=sum(v['leakage_controlled'] for v in details),
                     no_training=True, all_queries_and_donors_training_split=True,
                     normalization='none; native uint8 / 255; fixed 0..1 display',
                     visual_review='pending',
                     scope='Trusted-Q-only paired-pixel replacement; no whole-background independence claim')
    write(ROOT/'aggregate_intervention_audit.json', aggregate)
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == '__main__':
    main()
