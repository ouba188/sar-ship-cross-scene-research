"""Fixed tasks and cached local content, before any target result is evaluated."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pass_key(product):
    parts = str(product).split('_')
    return '|'.join(parts[i] for i in (0, 6, 7)) if len(parts) > 7 else 'unknown'


def unique_rows(meta, ids):
    return meta.loc[ids].drop_duplicates('mmsi').row.to_numpy(int)


def make_tasks():
    meta = pd.read_csv(BRIDGE/'cohort.csv', dtype={'date': str, 'mmsi': str})
    geo = pd.read_csv(BRIDGE/'axis_radar_readout_20261019/geometry.csv')
    meta['pass_key'] = geo.resolved_product.map(pass_key)
    meta['pass_known'] = geo.pixel_product_unique & geo.native_window_matches_manifest_raster
    held_dates = ['20250315', '20250401']
    held_ships = set(meta[meta.date.isin(held_dates)].mmsi)
    source = unique_rows(meta, meta.index[~meta.date.isin(held_dates) & ~meta.mmsi.isin(held_ships)])
    validation = source[meta.loc[source].date.eq('20250211')]
    fit = source[~meta.loc[source].date.eq('20250211')]
    fit = fit[~meta.loc[fit].mmsi.isin(meta.loc[validation].mmsi)]
    target_passes = ['S1A|058308|073537', 'S1A|058556|073F16']
    tasks = []
    for index, key in enumerate(target_passes):
        pool = unique_rows(meta, meta.index[meta.pass_key.eq(key) & meta.pass_known])
        order = np.random.default_rng(8100+index).permutation(pool)
        # Retain at least 32 distinct same-pass support vessels, without using labels.
        query = order[:min(24, len(order)-32)]
        query_ships = set(meta.loc[query].mmsi)
        same = order[len(query):]
        other = target_passes[1-index]
        cross = unique_rows(meta, meta.index[meta.pass_key.eq(other) & meta.pass_known
                                             & ~meta.mmsi.isin(query_ships)])
        for line, bank in [('same_pass', same), ('cross_pass', cross)]:
            permutations = [np.random.default_rng(9100+100*index+d).permutation(bank).tolist()
                            for d in range(5)]
            for budget in (16, 32):
                assert len(bank) >= budget
                tasks.append(dict(name=f'{index}_{line}_B{budget}', target_pass=key,
                    line=line, budget=budget, source=source.tolist(), query=query.tolist(),
                    bank=bank.tolist(), supports=[s[:budget] for s in permutations]))
    result = dict(source=source.tolist(), fit=fit.tolist(), validation=validation.tolist(), tasks=tasks)
    (HERE/'tasks.json').write_text(json.dumps(result, indent=2))
    meta.to_csv(HERE/'cohort.csv', index=False)
    pd.DataFrame([dict(task=t['name'], line=t['line'], budget=t['budget'], queries=len(t['query']),
                       bank=len(t['bank']), source=len(source)) for t in tasks]).to_csv(HERE/'capacity.csv', index=False)
    print('SOURCE', len(source), 'FIT', len(fit), 'VALIDATION', len(validation), flush=True)
    return meta


def extract(meta):
    import torch
    import torch.nn.functional as F
    from torchvision.models import resnet50
    from PIL import Image
    torch.set_num_threads(2)
    old = np.load(BRIDGE/'features.npz')
    roles = np.load(BRIDGE/'masks.npz')['regions']
    axes = pd.read_csv(BRIDGE/'orbit_geometry_readout_20261019/axes.csv')
    model = resnet50(weights=None).cuda().eval()
    model.load_state_dict(torch.load('C:/Users/25791/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth',
                                    map_location='cpu', weights_only=True))
    content, masses, positions = [], [], []
    max_replay = 0.
    with torch.inference_mode():
        for start in range(0, len(meta), 8):
            rows = meta.iloc[start:start+8]
            x = np.stack([[np.asarray(Image.open(r.png_vv)), np.asarray(Image.open(r.png_vh))]
                          for r in rows.itertuples()])
            x = torch.tensor(x, device='cuda', dtype=torch.float32)/255
            x = F.interpolate(x, (128,128), mode='bilinear', align_corners=False, antialias=True)
            x = x.flatten(0,1)[:,None].expand(-1,3,-1,-1)
            x = (x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
            f = model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(x)))))
            grid = F.adaptive_avg_pool2d(f, (2,2)).flatten(2).transpose(1,2)
            expected = torch.tensor(old['grid2'][start:start+len(rows)].reshape(len(x),4,256), device='cuda')
            torch.testing.assert_close(grid, expected, rtol=1e-5, atol=2e-5)
            max_replay = max(max_replay, float((grid-expected).abs().max()))
            masks = np.stack([roles[rows.row,0], roles[rows.row,1]+roles[rows.row,2]],1)
            masks = F.interpolate(torch.tensor(masks,device='cuda',dtype=torch.float32),(32,32),mode='area')
            mass = F.avg_pool2d(masks,4)
            fm = f.reshape(len(rows),2,256,32,32)
            numerator = F.avg_pool2d((fm[:,:,None]*masks[:,None,:,None]).reshape(-1,256,32,32),4)
            tokens = numerator.reshape(len(rows),2,2,256,8,8)/mass[:,None,:,None].clamp_min(1e-8)
            tokens = tokens.permute(0,1,2,4,5,3).reshape(len(rows),2,128,256)
            content.append(tokens.cpu().numpy().astype(np.float16))
            masses.append(mass.flatten(1).cpu().numpy())
    yy,xx = np.mgrid[:8,:8]
    xx,yy = (xx.flatten()+.5)*224/8-.5, (yy.flatten()+.5)*224/8-.5
    for a in axes.itertuples():
        valid = np.isfinite(a.half_width) and a.half_width > 0 and a.axis_reliability > 1e-8
        if valid:
            dx,dy = xx-a.center_x, yy-a.center_y
            u = (dx*np.cos(a.ship_axis_rad)+dy*np.sin(a.ship_axis_rad))/a.half_length
            v = (-dx*np.sin(a.ship_axis_rad)+dy*np.cos(a.ship_axis_rad))/a.half_width
        else:
            u,v = np.zeros(64),np.zeros(64)
        p = np.stack([xx/112-1, yy/112-1, np.clip(u,-4,4)/4, np.clip(v,-4,4)/4,
                      np.minimum(abs(u),4)/4, np.minimum(abs(v),4)/4],1)
        p = np.tile(p,(2,1))
        positions.append(np.column_stack([p,np.repeat([1.,0.],64),np.repeat([0.,1.],64)]))
    np.savez_compressed(HERE/'local_tokens.npz', content=np.concatenate(content), mass=np.concatenate(masses),
                        position=np.asarray(positions,np.float32), clean_rows=meta.clean_row.to_numpy())
    (HERE/'extraction.json').write_text(json.dumps(dict(observations=len(meta), slots=128,
        spatial_grid=[8,8], regions=['target','outside'], backbone='frozen_R50_layer1',
        grid_replay_max_error=max_replay, geometry_used_in_extraction=False),indent=2))
    print('EXTRACTION COMPLETE', max_replay, flush=True)


if __name__ == '__main__':
    extract(make_tasks())
