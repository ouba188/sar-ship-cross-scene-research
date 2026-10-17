"""Read only two diagnostic queries under historical/fresh-wrong/fresh-fixed boxes."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'geometry_relation_slots_20261016'))
import extract_slots as slots

OUT = ROOT / 'geometry_recovery_effect_20261016/visual_diagnosis/geometry_trace'


def union_box(records, field):
    grid = (np.arange(56)+.5)/56
    mask = np.zeros((56, 56), bool)
    for record in records:
        b = np.asarray(record[field])
        b = np.concatenate((np.floor(b[:2]), np.ceil(b[2:]))).clip(0, 64)/64
        mask |= ((grid[None] >= b[0]) & (grid[None] < b[2]) &
                 (grid[:, None] >= b[1]) & (grid[:, None] < b[3]))
    return mask


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    trace = json.loads((OUT / 'trace.json').read_text())['cases']
    rows = np.array(sorted({r['global_row'] for r in trace}))
    legacy = slots.load_legacy()
    port, allrows, checkpoint = next(x for x in legacy.inventory() if x[0] == 'Singapore')
    geometry = pd.read_csv(ROOT / 'ship_radar_geometry_design_20261016/geometry_metadata.csv').set_index('global_row')
    model, _ = legacy.backbone('r18')
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['backbone'], strict=True)
    stem = torch.nn.Sequential(model.conv1, model.bn1, model.relu).cuda().eval()
    stem.requires_grad_(False)
    pixels = np.load(legacy.PIXELS, mmap_mode='r')
    oldcache = np.load(ROOT / 'geometry_relation_slots_20261016/slots_cache/Singapore.npz')
    fields = {'historical': 'original_box', 'fresh_old_adapter': 'fresh_legacy_adapter_box',
              'fresh_fixed_adapter': 'fresh_correct_box'}
    output = {'rows': rows}
    store = {flavor: {name: [] for name in ('img_tokens', 'ship_tokens', 'img_areas', 'ship_areas')}
             for flavor in fields}
    errors = []
    for grow in rows:
        position = int(np.flatnonzero(allrows == grow)[0])
        start = position//32*32
        batch = allrows[start:start+32]
        at = position-start
        original_masks, *_ = legacy.spatial_masks(batch)
        records = [r for r in trace if r['global_row'] == grow]
        np.testing.assert_array_equal(union_box(records, 'original_box'), original_masks[at])
        angles = [slots.ship_angle(geometry.loc[g, 'ship_axis_rad'], geometry.loc[g, 'ship_valid'])
                  for g in batch]
        pair = torch.from_numpy(np.array(pixels[batch], dtype=np.float32)).cuda()/255.
        image = legacy.normalize(legacy.rgb(F.interpolate(pair, (112,112), mode='bilinear',
                                             align_corners=False, antialias=True)))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            feature = stem(image)
        feature = feature.float().flatten(2).transpose(1, 2)
        for flavor, field in fields.items():
            masks = original_masks.copy()
            masks[at] = union_box(records, field)
            for prefix in ('img', 'ship'):
                values = [slots.regional_slots(b, a if prefix == 'ship' else 0.)
                          for b, a in zip(masks, angles)]
                pool = torch.from_numpy(np.stack([v[0] for v in values])).cuda().reshape(-1,16,56*56)
                tokens = torch.bmm(pool, feature).reshape(-1,4,4,64)[at].cpu().numpy()
                store[flavor][prefix+'_tokens'].append(tokens)
                store[flavor][prefix+'_areas'].append(values[at][1])
                if flavor == 'historical':
                    idx = int(np.flatnonzero(oldcache['rows'] == grow)[0])
                    err = float(np.abs(tokens-oldcache[prefix+'_tokens'][idx]).max())
                    np.testing.assert_allclose(tokens, oldcache[prefix+'_tokens'][idx], atol=1e-5, rtol=0)
                    np.testing.assert_array_equal(values[at][1], oldcache[prefix+'_areas'][idx])
                    errors.append(dict(global_row=int(grow), prefix=prefix, max_token_error=err))
    for flavor, values in store.items():
        for key, value in values.items():
            output[flavor+'_'+key] = np.stack(value)
    np.savez_compressed(OUT / 'query_box_tokens.npz', **output)
    (OUT / 'token_replay.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),
        original_batch_preserved=True, errors=errors, query_only=True,
        angles_unchanged=True), indent=2), encoding='utf-8')
    print(json.dumps(errors))


if __name__ == '__main__':
    main()
