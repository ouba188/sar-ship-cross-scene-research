"""Full-population, label-free display-image proxies for the C4 diagnosis."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image
import psutil
from threadpoolctl import threadpool_limits

ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
SOURCE = Path('E:/\u4e34\u65f6\u4f1a\u8bdd/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
OUT = ROOT/'c4_condition_diagnosis_20261009/descriptors'
HELPER = Path(__file__).with_name('latent_clue_cards.py')
CACHE = ROOT/'latent_clue_cards_20261008/image_proxies.csv'
MANIFESTS = {
    'self24ports': SOURCE/'manifest.csv',
    'oss2': ROOT/'opensarship2_transfer_20261007/target_manifest.csv',
    'f0921': ROOT/'f0921_detector_recrop_20261007/target_manifest.csv',
}
FEATURES = {
    'visible_footprint_proxy_fraction': 'bright_fraction',
    'elongation_proxy': 'elongation',
    'contrast_proxy': 'contrast',
    'center_excess_proxy': 'center_excess',
}


def load_proxy_helper():
    # Import only the existing numpy-only function, avoiding its analysis imports.
    source = HELPER.read_text(encoding='utf-8')
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == 'image_proxies')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(HELPER), 'exec'), namespace)
    return namespace['image_proxies'], ast.get_source_segment(source, node)


IMAGE_PROXIES, HELPER_SOURCE = load_proxy_helper()


def describe(pair=None, cached=None):
    result = {name: np.nan for name in FEATURES}
    if cached is None:
        if pair is None or pair.shape != (2, 128, 128) or not np.isfinite(pair).all():
            result['proxy_status'] = 'invalid_or_missing_pixels'
        else:
            cached = IMAGE_PROXIES(*pair)
    if cached is not None:
        result.update({name: float(cached[key]) for name, key in FEATURES.items()})
        if not np.isfinite(result['elongation_proxy']) or result['visible_footprint_proxy_fraction'] <= 0:
            result['visible_footprint_proxy_fraction'] = np.nan
            result['elongation_proxy'] = np.nan
            result['proxy_status'] = 'no_bright_excess'
        else:
            result['proxy_status'] = 'ok'
    for name in FEATURES:
        result[name+'_missing'] = not np.isfinite(result[name])
    result['proxy_missing'] = any(result[name+'_missing'] for name in FEATURES)
    return result


def read_pair(paths, pixels=True):
    arrays, sizes = [], {}
    for pol, path in zip(('vv', 'vh'), paths):
        with Image.open(path) as image:
            sizes['native_width_px_'+pol], sizes['native_height_px_'+pol] = image.size
            if pixels:
                arrays.append(np.asarray(image.convert('L').resize(
                    (128, 128), Image.Resampling.BILINEAR)))
    return np.stack(arrays) if pixels else None, sizes


def load_manifest(dataset):
    columns = (['instance_id', 'image_relpath_vv', 'image_relpath_vh'] if dataset == 'self24ports'
               else ['sample_id', 'row'] if dataset == 'f0921'
               else ['sample_id', 'target_manifest_row', 'chip_path_vv', 'chip_path_vh'])
    frame = pd.read_csv(MANIFESTS[dataset], usecols=columns, keep_default_na=False)
    frame['row_id'] = np.arange(len(frame))
    if dataset == 'self24ports':
        frame['sample_id'] = frame.instance_id
    else:
        frame['instance_id'] = ''
    return frame


def run_dataset(dataset, out):
    started = time.perf_counter()
    frame = load_manifest(dataset)
    n = len(frame)
    native_names = [f'native_{dim}_px_{pol}' for pol in ('vv', 'vh') for dim in ('width', 'height')]
    pixels, paths, cache = None, None, {}
    cached_rows, cache_status = 0, 'not_available_for_dataset'
    crop = None
    if dataset == 'f0921':
        directory = MANIFESTS[dataset].parent
        pixels = np.load(directory/'inputs128.npy', mmap_mode='r')
        feature_rows = np.load(directory/'feature_rows.npy', mmap_mode='r')
        if pixels.shape != (n, 2, 128, 128) or pixels.dtype != np.uint8:
            raise ValueError('Unexpected F0921 pixel shape or dtype')
        np.testing.assert_array_equal(feature_rows, frame.row.to_numpy())
        crop = pd.read_csv(directory/'crop_metadata.csv', usecols=['row', 'requested_side'])
        np.testing.assert_array_equal(crop.row.to_numpy(), frame.row.to_numpy())
    else:
        keys = ['image_relpath_vv', 'image_relpath_vh'] if dataset == 'self24ports' else ['chip_path_vv', 'chip_path_vh']
        paths = list(frame[keys].itertuples(index=False, name=None))
        if dataset == 'self24ports' and CACHE.exists():
            values = pd.read_csv(CACHE, usecols=['manifest_row', *FEATURES.values()])
            if values.manifest_row.duplicated().any() or not values.manifest_row.between(0, n-1).all():
                raise ValueError('Descriptor cache row mapping is invalid')
            cache = values.set_index('manifest_row').to_dict('index')
            cache_status = 'reused_after_three_spaced_row_pixel_replays'
            for index in np.linspace(0, len(values)-1, min(3, len(values)), dtype=int):
                row = int(values.manifest_row.iloc[index])
                actual = IMAGE_PROXIES(*read_pair(paths[row])[0])
                if not all(np.isclose(actual[key], cache[row][key], rtol=1e-8, atol=1e-10,
                                      equal_nan=True) for key in FEATURES.values()):
                    cache, cache_status = {}, 'cache_replay_mismatch_recomputed_all_rows'
                    break

    def one(i):
        record = dict(dataset=dataset, row_id=i, sample_id=frame.sample_id.iat[i],
                      instance_id=frame.instance_id.iat[i], input_width_px=128, input_height_px=128,
                      native_size_source='unverified' if dataset == 'f0921' else 'image_header',
                      requested_crop_side_px=float(crop.requested_side.iat[i]) if crop is not None else np.nan,
                      source_row=int(frame.row.iat[i]) if dataset == 'f0921' else np.nan,
                      target_manifest_row=int(frame.target_manifest_row.iat[i]) if dataset == 'oss2' else np.nan,
                      descriptor_source='existing_cache' if i in cache else 'pixels128', read_error='')
        record.update({key: np.nan for key in native_names})
        try:
            if pixels is not None:
                pair = pixels[i]
            else:
                pair, sizes = read_pair(paths[i], pixels=i not in cache)
                record.update(sizes)
            record.update(describe(pair, cache.get(i)))
        except (OSError, ValueError) as error:
            record.update(describe())
            record['proxy_status'] = 'image_read_failed'
            record['read_error'] = str(error)
        record['native_size_missing'] = any(not np.isfinite(record[key]) for key in native_names)
        return record

    present = {name: 0 for name in FEATURES}
    statuses = {}
    native_rows = 0
    peak_rss = 0
    output = out/f'{dataset}.csv'
    process = psutil.Process()
    with output.open('w', encoding='utf-8', newline='') as handle, ThreadPoolExecutor(max_workers=2) as pool:
        writer = None
        last_print = started
        for lo in range(0, n, 128):
            hi = min(n, lo+128)
            records = list(pool.map(one, range(lo, hi)))
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
            writer.writerows(records)
            for record in records:
                statuses[record['proxy_status']] = statuses.get(record['proxy_status'], 0)+1
                native_rows += not record['native_size_missing']
                cached_rows += record['descriptor_source'] == 'existing_cache'
                for name in FEATURES:
                    present[name] += not record[name+'_missing']
            memory = process.memory_info()
            peak_rss = max(peak_rss, memory.rss, getattr(memory, 'peak_wset', 0))
            if peak_rss > 16*1024**3:
                raise MemoryError('Descriptor extraction exceeded the 16 GiB budget')
            now = time.perf_counter()
            if lo == 0 or hi == n or now-last_print >= 20:
                print(f'DESCRIPTORS {dataset} {hi}/{n} {now-started:.1f}s rss={memory.rss/1024**2:.0f}MiB', flush=True)
                last_print = now
    return dict(rows=n, manifest=str(MANIFESTS[dataset]), csv=str(output), coverage=present,
                status_counts=statuses, native_size_rows=native_rows, cached_rows=cached_rows,
                cache_status=cache_status, seconds=time.perf_counter()-started,
                peak_working_set_mib=peak_rss/1024**2,
                input_pixels=str(MANIFESTS[dataset].parent/'inputs128.npy') if dataset == 'f0921'
                else 'Manifest image paths; PIL L conversion and bilinear resize to128, matching C4 read_one',
                native_size_note='F0921 requested_side is ancillary only: actual clipped pre-resize dimensions are not verified.'
                if dataset == 'f0921' else 'Actual PNG header dimensions before resize; not detector bounding-box dimensions.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    process = psutil.Process()
    process.cpu_affinity(process.cpu_affinity()[-2:])
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    provenance = dict(
        analysis_features=list(FEATURES),
        row_id='Zero-based CSV manifest position, without filtering or reordering; NOT the external target_manifest_row or F0921 source row.',
        identifiers='self24ports sample_id aliases instance_id; external instance_id absent, sample_id retained exactly.',
        full_population=True, target_labels_read=False, performance_read=False, model_fits=0,
        gpu_used=False, workers=2, numerical_threads_per_worker=1, cpu_affinity=process.cpu_affinity(),
        memory_budget_gib=16, pixel_batch_rows=128, f0921_pixels_mmap=True,
        helper=str(HELPER), helper_function='image_proxies', helper_source=HELPER_SOURCE,
        descriptor_cache=str(CACHE),
        input_definition='a=(VV128+VH128)/510, float64; both display channels, each uint8. Same recipe across all three datasets.',
        definitions={
            'visible_footprint_proxy_fraction': 'Fraction of all128x128 pixels with a > q50 + 0.5*(q99-q50). Global bright-scattering footprint proxy, not central connected-component or vessel segmentation.',
            'elongation_proxy': 'sqrt((lambda_max+1e-8)/(lambda_min+1e-8)) of 2D covariance weighted by max(a-q50,0); coordinates span [-0.5,0.5]. Global bright-scattering elongation proxy.',
            'contrast_proxy': 'q95(a)-q05(a), unitless display-intensity quantile spread.',
            'center_excess_proxy': 'Mean(a) in central half-width square minus mean(a) in outer eighth-width rim, unitless display-intensity difference.',
        },
        missing='Absent/invalid pixels: all proxies NaN. No positive bright excess: footprint and elongation NaN. Constant-image contrast and center excess remain valid zeros. Each feature has a _missing flag; all rows retained.',
        scale='Native physical pixel spacing and sensor resolution remain unknown and are omitted. Native PNG dimensions and F0921 requested crop side are ancillary only; resized128 is never ground resolution.',
        limitations='Display proxies only: no calibrated radiometry, physical backscatter, true body segmentation, vessel dimensions, sensor resolution, or causal interpretation. Clutter can contribute. Percentile-stretched uint8 inputs do not support physical intensity claims.',
        cache_scope='Existing3718-row self24ports subset reused, all remaining manifest rows computed without any label/performance selection.',
        datasets={}, created_utc=datetime.now(timezone.utc).isoformat())
    with threadpool_limits(limits=1):
        for dataset in MANIFESTS:
            provenance['datasets'][dataset] = run_dataset(dataset, args.output)
    provenance['seconds'] = time.perf_counter()-started
    (args.output/'PROVENANCE.json').write_text(json.dumps(provenance, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(seconds=provenance['seconds'], datasets=provenance['datasets']), indent=2), flush=True)


if __name__ == '__main__':
    main()
