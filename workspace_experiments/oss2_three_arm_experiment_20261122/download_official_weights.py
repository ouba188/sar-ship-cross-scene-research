"""Fetch official torchvision weights in checked HTTP byte ranges."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import time
import requests

URL = 'https://download.pytorch.org/models/convnext_tiny-983f1562.pth'
DEST = Path.home()/'.cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth'
SIZE = 114419221
PART = 4*1024*1024


def fetch(i):
    lo, hi = i*PART, min(SIZE, (i+1)*PART)-1
    path = DEST.with_suffix(f'.part{i}')
    if path.exists() and path.stat().st_size == hi-lo+1:
        return i
    for attempt in range(3):
        try:
            with requests.get(URL, headers={'Range':f'bytes={lo}-{hi}'}, stream=True, timeout=(15,60)) as response:
                response.raise_for_status()
                assert response.status_code == 206
                assert response.headers['Content-Range'] == f'bytes {lo}-{hi}/{SIZE}'
                with path.open('wb') as out:
                    for block in response.iter_content(256*1024): out.write(block)
            assert path.stat().st_size == hi-lo+1
            return i
        except Exception:
            if attempt == 2: raise
            time.sleep(1)


if __name__ == '__main__':
    DEST.parent.mkdir(parents=True, exist_ok=True)
    count = (SIZE+PART-1)//PART
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(fetch, i) for i in range(count)]):
            print('verified range',future.result(),flush=True)
    target = DEST.with_suffix('.verified_download')
    digest = hashlib.sha256()
    with target.open('wb') as out:
        for i in range(count):
            with DEST.with_suffix(f'.part{i}').open('rb') as inp:
                for block in iter(lambda:inp.read(1024*1024),b''):
                    digest.update(block);out.write(block)
    assert target.stat().st_size == SIZE
    assert digest.hexdigest().startswith('983f1562'), digest.hexdigest()
    target.replace(DEST)
    print('official weight SHA256',digest.hexdigest(),flush=True)
