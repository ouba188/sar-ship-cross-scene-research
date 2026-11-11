"""Reuse the exact native OSS2 images and split, without importing old models."""
from pathlib import Path
import importlib.util
import json

import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent.parent


def records():
    rows = json.loads((WORK/'m1_coverage_20261101/manifest.json').read_text(encoding='utf-8'))
    groups = json.loads((WORK/'m1_coverage_20261101/groups.json').read_text(encoding='utf-8'))
    return rows, {k: np.array(v, dtype=np.int64) for k, v in groups.items()}


def native_module():
    spec = importlib.util.spec_from_file_location('native_input', WORK/'sar_m123_standalone/sar_m123/input.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare():
    cache = ROOT/'native_cache'
    if (cache/'complete.json').exists():
        return
    cache.mkdir(exist_ok=True)
    rows, groups = records()
    native = native_module()
    pixels = np.lib.format.open_memmap(cache/'pixels.npy', mode='w+', dtype='uint8', shape=(len(rows),2,160,160))
    footprint = np.lib.format.open_memmap(cache/'footprint.npy', mode='w+', dtype='uint8', shape=pixels.shape)
    available = np.zeros((len(rows),2), np.float32)
    for i, row in enumerate(rows):
        for p, view in enumerate(native.prepare(row)):
            if view is not None and view['valid_target']:
                pixels[i,p] = view['pixels']
                footprint[i,p] = view['footprint']
                available[i,p] = 1
            else:
                pixels[i,p] = 0
                footprint[i,p] = 0
        if i % 200 == 0:
            print('PREPARE', i, len(rows), flush=True)
    pixels.flush(); footprint.flush()
    np.save(cache/'available.npy', available)
    prior = np.load(WORK/'m1_coverage_20261101/cache/a/available.npy', mmap_mode='r')
    assert np.array_equal(prior, available), 'Input availability differs from Full'
    for left, right in [('small','validation'), ('small','evaluation'), ('validation','evaluation')]:
        for key in ('id','mmsi','pass_group'):
            assert not {rows[i][key] for i in groups[left]} & {rows[i][key] for i in groups[right]}
    info = {k: dict(n=len(groups[k]), both=int((available[groups[k]].sum(1)==2).sum()),
                   classes=np.bincount([rows[i]['label'] for i in groups[k]], minlength=4).tolist())
            for k in ('small','validation','evaluation')}
    (cache/'complete.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(info, flush=True)


if __name__ == '__main__':
    prepare()
