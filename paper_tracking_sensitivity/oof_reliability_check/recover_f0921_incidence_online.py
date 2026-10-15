"""Recover exact Sentinel-1 incidence grids from the public acquisition archive."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import gzip
import json
from pathlib import Path
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from external_knowledge_metadata import OUT as OLD, interpolate, product_key, write

OUT = OLD.parent/'f0921_incidence_recovery_20261010'
BUCKET = 'https://sentinel-s1-l1c.s3.amazonaws.com/'
NS = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}
LOCAL = threading.local()


def session():
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
        LOCAL.session.mount('https://', HTTPAdapter(max_retries=Retry(
            total=3, backoff_factor=.5, status_forcelist=[429, 500, 502, 503, 504])))
    return LOCAL.session


def read_requests():
    frame = pd.read_csv(OLD/'recovery_inventory/acquisition_requests.csv', keep_default_na=False)
    frame['date'] = frame.acquisition_start.str[:8]
    return frame


def catalog_day(date, wanted):
    path = OUT/'catalog'/f'{date}.json'
    if path.exists():
        return json.loads(path.read_text())
    dt = datetime.strptime(date, '%Y%m%d')
    # The target chips have VV/VH inputs. Search IW dual-V first, then ASF for leftovers.
    prefix = f'GRD/{dt.year}/{dt.month}/{dt.day}/IW/DV/'
    params = {'list-type': '2', 'prefix': prefix, 'delimiter': '/', 'max-keys': 1000}
    found, scanned = {}, 0
    while True:
        response = session().get(BUCKET, params=params, timeout=45)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        for item in root.findall('s:CommonPrefixes/s:Prefix', NS):
            product = item.text.rstrip('/').split('/')[-1]
            scanned += 1
            key = '|'.join(product_key(product))
            if key in wanted:
                found.setdefault(key, []).append(dict(product=product, prefix=item.text))
        if root.findtext('s:IsTruncated', namespaces=NS) != 'true':
            break
        params['continuation-token'] = root.findtext('s:NextContinuationToken', namespaces=NS)
    result = dict(date=date, source=BUCKET, prefix=prefix, scanned=scanned, matches=found)
    write(path, result)
    return result


def catalog(workers):
    frame = read_requests()
    groups = list(frame.groupby('date'))
    groups.sort(key=lambda x: -x[1].target_rows.sum())
    matches, failures = {}, []
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(catalog_day, date, {
            '|'.join((r.sensor, r.acquisition_start, r.product_suffix))
            for r in rows.itertuples()}): date for date, rows in groups}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                matches.update(result['matches'])
            except Exception as exc:
                failures.append(dict(date=futures[future], error=str(exc)))
            if i % 100 == 0 or i == len(futures):
                print(f'CATALOG {i}/{len(futures)} days, {len(matches)} exact keys, '
                      f'{len(failures)} failures, {time.time()-started:.0f}s', flush=True)
    write(OUT/'catalog_matches.json', matches)
    write(OUT/'catalog_failures.json', failures)
    return matches


def parse_grid(fragment):
    root = ET.fromstring(fragment)
    points = root.findall('.//geolocationGridPoint')
    grid = np.asarray([[float(p.findtext(k)) for k in
                       ('latitude', 'longitude', 'incidenceAngle')] for p in points])
    if grid.ndim != 2 or grid.shape[1] != 3 or not np.isfinite(grid).all():
        raise ValueError('Missing or nonfinite native geolocation grid')
    if not ((grid[:, 2] > 0) & (grid[:, 2] < 90)).all():
        raise ValueError('Invalid native incidence degrees')
    return grid


def download_grid(record):
    product, prefix = record['product'], record['prefix']
    dest = OUT/'grids'/f'{product}.json'
    if dest.exists():
        return json.loads(dest.read_text())
    mode = product.split('_')[1].lower()
    url = BUCKET+prefix+f'annotation/{mode}-vv.xml'
    transferred = 0
    for size in (262144, 524288, 1048576, 4194304):
        response = session().get(url, headers={'Range': f'bytes=-{size}'}, timeout=55)
        response.raise_for_status()
        payload = response.content
        transferred += len(payload)
        begin, end = payload.find(b'<geolocationGrid>'), payload.find(b'</geolocationGrid>')
        if begin >= 0 and end > begin:
            fragment = payload[begin:end+len(b'</geolocationGrid>')]
            grid = parse_grid(fragment)
            raw = OUT/'xml_grid'/f'{product}.xml.gz'
            raw.parent.mkdir(exist_ok=True)
            raw.write_bytes(gzip.compress(fragment))
            result = dict(product=product, url=url, content_range=response.headers.get('Content-Range'),
                bytes_downloaded=transferred, raw_grid_file=str(raw), grid=grid.tolist(),
                downloaded_utc=datetime.utcnow().isoformat(),
                method='Exact product path; native geolocationGridPoint incidenceAngle; no scene-average surrogate')
            write(dest, result)
            return result
        if response.status_code == 200:
            break
    raise ValueError(f'Complete geolocationGrid not found: {product}')


def download(workers):
    matches = json.loads((OUT/'catalog_matches.json').read_text())
    frame = read_requests().sort_values('target_rows', ascending=False)
    records = []
    for r in frame.itertuples():
        key = '|'.join((r.sensor, r.acquisition_start, r.product_suffix))
        candidates = matches.get(key, [])
        if len(candidates) == 1:
            records.append(candidates[0])
    errors, done, started = [], 0, time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_grid, r): r['product'] for r in records}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
                done += 1
            except Exception as exc:
                errors.append(dict(product=futures[future], error=str(exc)))
            if i % 200 == 0 or i == len(futures):
                print(f'GRIDS {i}/{len(futures)}, {done} saved, {len(errors)} failed, '
                      f'{time.time()-started:.0f}s', flush=True)
    write(OUT/'download_failures.json', errors)


def recover():
    side = pd.read_csv(OLD/'f0921_incidence.csv', keep_default_na=False)
    side['incidence_deg'] = pd.to_numeric(side.incidence_deg, errors='coerce')
    rows = pd.read_csv(OLD/'recovery_inventory/missing_rows.csv', keep_default_na=False)
    np.testing.assert_array_equal(side.sample_id[side.status != 'ok'], rows.sample_id)
    lookup = {v: i for i, v in enumerate(side.sample_id)}
    matches = json.loads((OUT/'catalog_matches.json').read_text())
    fallback_path = OUT/'catalog_fallback.json'
    fallback = {r['key']: r for r in json.loads(fallback_path.read_text())
                if r.get('catalog_exact_matches') == 1} if fallback_path.exists() else {}
    # Repeated Arrow string-column assignment copies the full 123k-row table.
    # Fill owned arrays and attach each column once after all scene interpolation.
    updates = {name: side[name].to_numpy(copy=True) if name == 'incidence_deg'
               else side[name].to_numpy(dtype=object, copy=True)
               for name in ('product', 'incidence_deg', 'status', 'metadata_file', 'method')}
    recovered_products, missing_keys = 0, []
    for key, part in rows[rows.candidate_count == 1].groupby(
            ['sensor', 'acquisition_start', 'product_suffix']):
        candidates = matches.get('|'.join(key), [])
        if len(candidates) != 1:
            if '|'.join(key) in fallback:
                source = fallback['|'.join(key)]
                positions = [lookup[v] for v in part.sample_id]
                updates['product'][positions] = source['product']
                updates['status'][positions] = 'missing_remote_grid'
                updates['method'][positions] = 'ASF exact product identified; native grid unavailable without authenticated access'
            missing_keys.append(dict(sensor=key[0], acquisition_start=key[1],
                product_suffix=key[2], rows=len(part), reason=(
                    'exact_product_found_grid_unavailable' if '|'.join(key) in fallback
                    else 'catalog_missing_or_ambiguous')))
            continue
        product = candidates[0]['product']
        path = OUT/'grids'/f'{product}.json'
        if not path.exists():
            missing_keys.append(dict(sensor=key[0], acquisition_start=key[1],
                product_suffix=key[2], rows=len(part), reason='grid_download_missing'))
            continue
        record = json.loads(path.read_text())
        if product_key(product) != key:
            raise ValueError('Catalog acquisition mismatch')
        angle = interpolate(record['grid'], part[['lat', 'lon']].to_numpy())
        positions = [lookup[v] for v in part.sample_id]
        updates['product'][positions] = product
        updates['incidence_deg'][positions] = angle
        updates['status'][positions] = np.where(np.isfinite(angle), 'ok', 'outside_grid')
        updates['metadata_file'][positions] = str(path)
        updates['method'][positions] = 'Exact sensor/start/suffix online product; native annotation grid; convex interpolation'
        recovered_products += 1
        if recovered_products % 2000 == 0:
            print(f'INTERPOLATE {recovered_products} products', flush=True)
    for name, values in updates.items():
        side[name] = values
    side['differs_from_tile_first'] = (side['product'] != '') & (side['product'] != side['old_tile_product'])
    old = pd.read_csv(OLD/'f0921_incidence.csv', keep_default_na=False)
    old['incidence_deg'] = pd.to_numeric(old.incidence_deg, errors='coerce')
    existing = old.status == 'ok'
    np.testing.assert_array_equal(side.loc[existing, 'incidence_deg'], old.loc[existing, 'incidence_deg'])
    side.to_csv(OUT/'f0921_incidence.csv', index=False)
    pd.DataFrame(missing_keys).to_csv(OUT/'unresolved_acquisitions.csv', index=False)
    good = side.status == 'ok'
    summary = dict(rows=len(side), before_usable=int(existing.sum()), usable=int(good.sum()),
        added=int(good.sum()-existing.sum()), coverage_percent=100*float(good.mean()),
        online_products_used=recovered_products, statuses=side.status.value_counts().to_dict(),
        original_unchanged=True, extrapolation=False, source='Public exact Sentinel-1 GRD annotation geolocation grids')
    write(OUT/'COVERAGE.json', summary)
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['all', 'catalog', 'download', 'recover'], default='all')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ('all', 'catalog'):
        catalog(args.workers)
    if args.stage in ('all', 'download'):
        download(args.workers)
    if args.stage in ('all', 'recover'):
        recover()


if __name__ == '__main__':
    main()
