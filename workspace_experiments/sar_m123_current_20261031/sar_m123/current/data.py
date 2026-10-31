"""Raw display chips -> native geometry, local slots and Gabor fields."""
import json
from pathlib import Path

import numpy as np
import torch

from ..input import prepare
from .. import frequency, tokens
from ..regions import soft_regions
from ..pool_replacement import grid_occupancy

FREQUENCY_KEYS = ('frequency', 'local_frequency', 'field2', 'field3')
SHAPES = dict(images=(2, 1, 160, 160), masks=(2, 3, 160, 160),
    valid=(2, 1, 160, 160), available=(2,), slot_masks=(2, 20, 10, 10),
    local_slot_masks=(2, 20, 20, 20), frequency=(2, 20, 3, 10, 10),
    local_frequency=(2, 20, 3, 20, 20), field2=(2, 3, 20, 20), field3=(2, 3, 10, 10))


def read_manifest(path):
    path = Path(path).resolve()
    records = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(records, list):
        raise ValueError('Manifest must be a JSON list of observations.')
    for r in records:
        for pol in ('vv', 'vh'):
            if r.get(pol) is not None:
                image = Path(r[pol]['image'])
                r[pol]['image'] = str(image if image.is_absolute() else path.parent/image)
    return records


def energy_for(raw, view, masks):
    field = torch.from_numpy(frequency.extract(raw, masks, view['placement'])['energy_field']).float()
    valid = torch.from_numpy(view['footprint']).float()[None, None]
    product = (torch.from_numpy(masks).float()[:, None]*field[None]).flatten(0, 1)[None]
    result = {}
    for stage, size, name in ((2, 20, 'local_frequency'), (3, 10, 'frequency')):
        result[name] = grid_occupancy(product, (size, size))[0].reshape(20, 3, size, size).numpy()
        occupancy = grid_occupancy(valid, (size, size))
        result[f'field{stage}'] = (grid_occupancy(field[None], (size, size))/occupancy.clamp_min(1e-8))[0].numpy()
    return result


def prepare_record(record, return_views=False):
    views = prepare(record)
    item = {k: np.zeros(shape, np.float32) for k, shape in SHAPES.items()}
    for p, view in enumerate(views):
        if view is None or not view['valid_target']:
            continue
        item['available'][p] = 1
        item['images'][p, 0] = view['pixels'].astype(np.float32)/255
        item['valid'][p, 0] = view['footprint']
        item['masks'][p] = soft_regions(view['geometry'])*view['footprint']
        masks = tokens.slots(view['geometry'], view['footprint'])
        for key, size in (('slot_masks', 10), ('local_slot_masks', 20)):
            item[key][p] = grid_occupancy(torch.from_numpy(masks).float()[None], (size, size))[0].numpy()
        for key, value in energy_for(view['raw'], view, masks).items():
            item[key][p] = value
    return (item, views) if return_views else item


def batch(items, device):
    return {k: torch.as_tensor(np.stack([x[k] for x in items]), device=device) for k in SHAPES}


def pair_mask(queries, references, device):
    # Identity is an exclusion only, never a classifier feature or class label.
    return torch.tensor([[str(q['id']) != str(r['id']) and not (
        q.get('mmsi') and r.get('mmsi') and str(q['mmsi']) == str(r['mmsi']))
        for r in references] for q in queries], device=device, dtype=torch.bool)


def assert_source(records):
    if not records or any(r.get('split') not in ('source_fit', 'source_adapt') for r in records):
        raise ValueError('Training/selection requires explicitly source_fit/source_adapt records.')
    if any(not isinstance(r.get('label'), int) or not 0 <= r['label'] < 8 for r in records):
        raise ValueError('Source records require labels in [0,7].')
