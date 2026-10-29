"""Fixed OSS2 fold0 four-class protocol. Standard library only."""
from collections import Counter, defaultdict
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
SOURCE = HERE.parent / 'oss2_m123_replication_20261026'
PACKAGE = WORK / 'sar_m123_standalone'
INIT = WORK / 'sar_m123_current_20261027/weights/init.pt'
MODEL = HERE.parent / 'final_module_ablations_20261029/model.py'
CLASS_INDICES = (0, 2, 3, 4)
CLASSES = ('bulk_carrier', 'general_cargo', 'product_chemical_tanker', 'container_ship')
ARMS = ('full', 'plain')
SEEDS = (17, 29, 43)
ROLES = {'fit_train': 'source_fit', 'fit_validation': 'validation',
         'utility_train': 'source_adapt', 'selection': 'selection', 'evaluation': 'evaluation'}
COUNTS = dict(source_fit=478, validation=228, source_adapt=241, selection=194, evaluation=378)
BUDGET = dict(fit=(8, 24, 160), adapt=(6, 16, 96))
COEFFICIENT = .3683066666126251


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def assert_rows(records, split=None):
    if not records or any(type(r['label']) is not int or not 0 <= r['label'] < 4 for r in records):
        raise ValueError('Expected nonempty four-class records.')
    if any(r['original_class_id'] != CLASS_INDICES[r['label']] for r in records):
        raise ValueError('Original/new class mapping differs from [0,2,3,4].')
    if split is not None and any(r['split'] != split for r in records):
        raise ValueError(f'Expected {split} records only.')


def class_weights4(records, stage):
    assert_rows(records, 'source_fit' if stage == 'fit' else 'source_adapt')
    counts = Counter(r['label'] for r in records)
    if set(counts) != set(range(4)):
        raise ValueError('The fixed source split must contain all four classes.')
    if stage == 'adapt':
        return [len(records)/(4*counts[r['label']]) for r in records]
    if stage != 'fit':
        raise ValueError(stage)
    ships = defaultdict(set)
    repeats = Counter(str(r['mmsi']) for r in records)
    for r in records:
        ships[r['label']].add(str(r['mmsi']))
    return [len(records)/(4*len(ships[r['label']])*repeats[str(r['mmsi'])]) for r in records]


def cohorts():
    return {name: read(HERE/'manifests'/f'{name}.json') for name in (*COUNTS, 'references')}


def checkpoint(arm, seed, stage):
    return HERE/'training'/f'{arm}_s{seed}'/stage/'model.pt'
