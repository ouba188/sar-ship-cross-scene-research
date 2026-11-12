"""Export existing historical L3 evidence on CPU; no model loading or inference."""
import csv
import json
import os
import shutil
import sys
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
WORK = OUT.parents[2]
SOURCE = WORK / 'm3_frequency_increment_20261028/m3_true_rescue_12457_20261031'
PACKAGE = WORK / 'sar_m123_current_20261027'
sys.path.insert(0, str(PACKAGE))
from sar_m123.input import prepare
from sar_m123.tokens import slots, REGIONS

CLASSES = ['bulk', 'fishing', 'general', 'chemical/product tanker',
           'container', 'crude tanker', 'tug', 'offshore']
REGION_NAMES = ['body', 'side', 'ends']
plt.rcParams.update({'svg.fonttype': 'none', 'font.family': 'DejaVu Sans'})


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def table(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    assets = OUT / 'l3'
    assets.mkdir(parents=True, exist_ok=True)
    receipt = json.loads((SOURCE / 'REPLAY_RECEIPT.json').read_text(encoding='utf-8'))
    records_path = SOURCE.parent / 'records.json'
    records = {str(r['id']): r for r in json.loads(records_path.read_text(encoding='utf-8'))}
    references = receipt['references']
    manifest = {
        'source_directory': str(SOURCE), 'records_source': str(records_path),
        'receipt_source': str(SOURCE / 'REPLAY_RECEIPT.json'),
        'scope': receipt['scope'], 'candidate': receipt['candidate'], 'seed': receipt['seed'],
        'm3_frequency': False, 'classes': CLASSES,
        'reference_order': [r['id'] for r in references],
        'conditions': receipt['conditions'], 'interventions': receipt['interventions'],
        'selected_relation': receipt['selected'], 'historical_replay_checks': receipt['checks'],
        'images': [], 'array_inventory': {}, 'writeback': {},
        'semantics': [
            'Historical N_level_pattern seed17, fixed 16 source references, M3 frequency disabled.',
            'Same vessel across L1-L3 means query 12457; reference 31512 is same CLASS, not same MMSI.',
            'Source PNGs are 8-bit display chips, not calibrated SAR radiometry.',
            'Native 160 canvases preserve pixel scale; black padding is not an observation.',
            'Body slots 0:8, side 8:16, ends 16:20. Ends are not identified bow/stern.',
            'Removal subtracts a fixed writeback component; T, beta and remaining content are held fixed.',
            'Removal does not erase image pixels and does not retrain or reallocate reference weights.',
            'Margin = true bulk logit minus maximum of the other seven logits; logits are not probabilities.',
            'Effects after nonlinear downstream processing are not additive.',
            'Saved update is the total spatial-M3 residual, not a frequency-on/off difference.',
            'Per-region/per-reference update tensors were not saved in this NPZ and are not reconstructed here.',
            'Update L2 norm measures magnitude, not signed class evidence or causal benefit.',
            'Selected VH/body/31512 slot5-to-slot5 link was selected post hoc by transported content magnitude.',
            'T is transported mass, not a row-normalized probability; beta is reference weight, not class probability.'
        ]
    }
    ids = ['12457'] + [r['id'] for r in references]
    for ident in ids:
        record = records[ident]
        views = prepare(record)
        item = {key: record[key] for key in ('id', 'mmsi', 'label', 'port', 'pass_group', 'split')}
        item['class_en'] = CLASSES[record['label']]
        item['role'] = 'query' if ident == '12457' else 'reference'
        item['views'] = {}
        for p, pol in enumerate(('VV', 'VH')):
            spec, view = record.get(pol.lower()), views[p]
            if view is None:
                item['views'][pol] = {'available': False}
                continue
            stem = f'{ident}_{pol}'
            source_image = Path(spec['image'])
            raw_path, native_path = assets / f'{stem}_raw.png', assets / f'{stem}_native160.png'
            shutil.copyfile(source_image, raw_path)
            Image.fromarray(view['pixels']).save(native_path)
            masks = slots(view['geometry'], view['footprint'])
            mask_path = assets / f'{stem}_geometry.npz'
            np.savez_compressed(mask_path, slots=masks.astype(np.float32),
                                footprint=view['footprint'], geometry=view['geometry'])
            entry = dict(available=True, valid_target=view['valid_target'], source=str(source_image),
                         raw=str(raw_path), native160=str(native_path), geometry_npz=str(mask_path),
                         raw_shape=list(view['raw'].shape), native_shape=list(view['pixels'].shape),
                         placement=view['placement'], geometry=view['geometry'].tolist(),
                         source_spec=spec)
            if ident in ('12457', '31512', '13000'):
                contours = {}
                fig, ax = plt.subplots()
                for name, indices in zip(REGION_NAMES, REGIONS):
                    contour = ax.contour(masks[indices].sum(0), levels=[.05])
                    contours[name] = [segment.tolist() for segment in contour.allsegs[0] if len(segment)]
                plt.close(fig)
                entry['region_contours_native_xy_level_005'] = contours
            item['views'][pol] = entry
        manifest['images'].append(item)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.8), layout='constrained')
    image_lookup = {str(item['id']): item for item in manifest['images']}
    for col, ident in enumerate(('12457', '31512', '13000')):
        for row, pol in enumerate(('VV', 'VH')):
            entry = image_lookup[ident]['views'][pol]
            ax = axes[row, col]
            with Image.open(entry['native160']) as im:
                ax.imshow(np.asarray(im), cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            for name, color in zip(REGION_NAMES, ('#44a468', '#399ed0', '#e3a848')):
                for segment in entry['region_contours_native_xy_level_005'][name]:
                    xy = np.asarray(segment)
                    ax.plot(xy[:, 0], xy[:, 1], color=color, lw=.65)
            ax.set_title(f"{ident} | {pol} | {image_lookup[ident]['class_en']}", fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Actual query and references: native 160-pixel canvases', fontsize=13)
    input_svg = assets / 'query_references.svg'
    fig.savefig(input_svg)
    fig.savefig(assets / 'query_references_preview.png', dpi=120)
    plt.close(fig)
    manifest['query_references_svg'] = str(input_svg)

    archive = assets / 'intermediate.npz'
    shutil.copyfile(SOURCE / 'intermediate.npz', archive)
    matrices = {}
    with np.load(archive) as arrays:
        for key in arrays.files:
            value = arrays[key]
            manifest['array_inventory'][key] = dict(npz=str(archive), key=key,
                                                     shape=list(value.shape), dtype=str(value.dtype))
            if key not in ('query_feature', 'update'):
                matrices[key] = value.tolist()
        before, update = arrays['query_feature'][0], arrays['update'][0]
        after = before + update
        maximum = float(np.linalg.norm(update, axis=1).max())
        for p, pol in enumerate(('VV', 'VH')):
            grid = np.linalg.norm(update[p], axis=0)
            path = assets / f'{pol}_writeback_grid.csv'
            rows = [dict(y=y, x=x, before_channel_mean=float(before[p, :, y, x].mean()),
                         after_channel_mean=float(after[p, :, y, x].mean()),
                         update_channel_mean=float(update[p, :, y, x].mean()),
                         update_l2=float(grid[y, x])) for y in range(10) for x in range(10)]
            table(path, rows)
            fig, ax = plt.subplots(figsize=(4.3, 4.1), layout='constrained')
            im = ax.pcolormesh(np.arange(11)-.5, np.arange(11)-.5, grid,
                               cmap='magma', vmin=0, vmax=maximum, rasterized=False)
            ax.set_aspect('equal'); ax.invert_yaxis()
            ax.set(title=f'{pol}: total spatial-M3 update L2', xlabel='Feature x', ylabel='Feature y')
            fig.colorbar(im, ax=ax)
            svg = assets / f'{pol}_writeback_l2.svg'
            fig.savefig(svg); plt.close(fig)
            manifest['writeback'][pol] = dict(csv=str(path), editable_svg=str(svg),
                                             l2_grid=grid.tolist(), shared_vmin=0, shared_vmax=maximum)
    matrix_path = assets / 'matrices.json'
    dump(matrix_path, matrices)
    manifest['editable_matrices_json'] = str(matrix_path)
    manifest['array_axes'] = {
        'query_feature/update': ['batch=1', 'polarization VV,VH', 'channel=1024', 'y=10', 'x=10'],
        '*_similarity_r/*_transport_r': ['batch=1', 'reference=16 in reference_order', 'query slot within region', 'reference slot within region'],
        '*_branch_similarity_r/*_branch_weights_r': ['batch=1', 'reference=16', 'query slot within region', 'reference slot within region', 'branch: deep,fine,4px,8px,16px'],
        '*_reference_weights': ['batch=1', 'reference=16', 'query slot=20'],
        'selected_values/transported/weighted_content': ['body slot=8', 'projected channel=32'],
        'region_suffix': {'0': 'body / slots 0-7', '1': 'side / slots 8-15', '2': 'ends / slots 16-19'}
    }
    removal_rows = []
    for i, record in enumerate(receipt['interventions']):
        pol = 'VV' if i < 6 else 'VH'
        component = ('all', 'body', 'side', 'ends', 'reference31512', 'reference13000')[i % 6]
        removal_rows.append({**{k: v for k, v in record.items() if k != 'logits'},
                             'polarization': pol, 'component': component,
                             **{f'logit_{j}': value for j, value in enumerate(record['logits'])}})
    removal_path = assets / 'all_removals.csv'
    table(removal_path, removal_rows)
    manifest['all_removals_csv'] = str(removal_path)
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    values = [r['margin'] for r in removal_rows]
    ax.barh(range(12), values, color=['#287da0' if v > 0 else '#bf4545' for v in values])
    ax.set_yticks(range(12), [f"Remove {r['polarization']} {r['component']}" for r in removal_rows])
    ax.invert_yaxis(); ax.axvline(0, color='#333333', lw=.8)
    ax.axvline(receipt['conditions'][1]['margin'], color='#32734b', ls='--', label='Full spatial M3')
    for i, value in enumerate(values):
        ax.text(value + (.006 if value >= 0 else -.006), i, f'{value:+.6f}',
                ha='left' if value >= 0 else 'right', va='center', fontsize=9)
    ax.set_xlim(-.65, .22)
    ax.set(xlabel='Bulk logit - strongest other logit', title='12457: all cached writeback removals')
    ax.legend(loc='lower left')
    chart = assets / 'all_removals.svg'
    fig.savefig(chart); fig.savefig(assets / 'all_removals_preview.png', dpi=130); plt.close(fig)
    manifest['all_removals_editable_svg'] = str(chart)
    manifest['page_assets'] = {
        'query_and_references': {'main_ids': ['12457', '31512', '13000'],
                                 'polarizations': ['VV', 'VH'],
                                 'image_lookup': 'images[].views; use native160 for shared native scale, raw for original chip'},
        'all_region_removals': {'csv': str(removal_path), 'svg': str(chart),
                                'baseline_spatial_margin': receipt['conditions'][1]['margin'],
                                'count': 12, 'regional_count': 6, 'whole_pol_count': 2, 'reference_count': 4}
    }
    dump(OUT / 'l3_assets.json', manifest)
    print(json.dumps({'manifest': str(OUT / 'l3_assets.json'), 'image_records': len(ids),
                      'image_views': sum(v['available'] for i in manifest['images'] for v in i['views'].values()),
                      'cached_arrays': len(manifest['array_inventory']), 'removals': len(removal_rows)}))


if __name__ == '__main__':
    main()
