"""Bounded, anonymous public-source checks; never fetch measurement data."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import time
import urllib.error
import urllib.request

from recover_product_annotations import HERE, read_csv


def check(item):
    name, url, limit, ranged = item
    record = dict(name=name, url=url, checked_utc=datetime.now(timezone.utc).isoformat(),
                  credentials_used=False, requested_range='bytes=0-4095' if ranged else '',
                  status='failed', bytes_read=0)
    headers = {'User-Agent': 'Sentinel-1-annotation-recovery/1.0'}
    if ranged:
        headers['Range'] = 'bytes=0-4095'
    started = time.monotonic()
    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=20) as response:
            record.update(http_status=response.status,
                          content_range=response.headers.get('Content-Range', ''),
                          content_length=response.headers.get('Content-Length', ''))
            data = bytearray()
            while len(data) < limit:
                if time.monotonic() - started > 35:
                    raise TimeoutError('Total read deadline exceeded')
                chunk = response.read(min(65536, limit - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
            path = HERE / 'source_checks' / (name + ('.xml.head' if ranged else '.html'))
            path.write_bytes(data)
            record.update(status='received', bytes_read=len(data), local_path=str(path),
                          stopped_at_byte_cap=len(data) == limit)
            if ranged:
                local = HERE / 'annotations' / (name + '.xml')
                with local.open('rb') as stream:
                    record['prefix_matches_local_annotation'] = stream.read(len(data)) == data
    except urllib.error.HTTPError as error:
        record.update(http_status=error.code, error_type=type(error).__name__)
    except (OSError, ValueError) as error:
        record.update(error_type=type(error).__name__, error=str(error))
    record['elapsed_seconds'] = round(time.monotonic() - started, 3)
    return record


def main():
    (HERE / 'source_checks').mkdir(exist_ok=True)
    jobs = [(row['product'], row['source_url'], 4096, True)
            for row in read_csv(HERE / 'manifest.csv')]
    jobs.extend([
        ('cdse_odata', 'https://documentation.dataspace.copernicus.eu/APIs/OData.html', 2000000, False),
        ('aws_registry', 'https://registry.opendata.aws/sentinel-1/', 500000, False),
        ('sentinel_hub_readme', 'https://roda.sentinel-hub.com/sentinel-s1-l1c/GRD/readme.html', 500000, False),
        ('asf_keywords', 'https://docs.asf.alaska.edu/api/keywords/', 1000000, False),
    ])
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(check, job) for job in jobs]
        for future in as_completed(futures):
            record = future.result()
            results.append(record)
            print(json.dumps(record), flush=True)
    (HERE / 'source_checks.json').write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
