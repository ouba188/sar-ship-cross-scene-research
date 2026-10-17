"""Same frozen stem, complete region-dependent inputs for all task roles."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'geometry_relation_slots_20261016'))
import extract_slots as slots


def masks_for(rows,records,arm):
    grid=(np.arange(56)+.5)/56
    masks=np.zeros((len(rows),56,56),bool)
    for i,grow in enumerate(rows):
        for pol in ('vv','vh'):
            r=records[(int(grow),pol)]
            b=np.asarray(r[arm+'_box'],float)/np.array([r['native_w'],r['native_h']]*2)
            masks[i]|=((grid[None]>=b[0])&(grid[None]<b[2])&
                       (grid[:,None]>=b[1])&(grid[:,None]<b[3]))
        assert masks[i].any(), (grow,arm)
    return masks


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    config=json.loads((HERE/'inventory.json').read_text())
    assert (HERE/'DETECTION_DONE.json').exists()
    records={(r['global_row'],r['pol']):r for r in
             map(json.loads,(HERE/'detections.jsonl').read_text().splitlines())}
    rows=np.asarray(config['rows'],int)
    geo=pd.read_csv(ROOT/'ship_radar_geometry_design_20261016/geometry_metadata.csv').set_index('global_row')
    missing=set(rows)-set(geo.index)
    assert missing.issubset(set(config['gallery_rows'])), 'Missing geometry for a task row'
    geo=geo.reindex(rows)
    legacy=slots.load_legacy()
    model,_=legacy.backbone('r18')
    model.load_state_dict(torch.load(config['source_checkpoint'],map_location='cpu',weights_only=False)['backbone'],strict=True)
    stem=torch.nn.Sequential(model.conv1,model.bn1,model.relu).cuda().eval()
    stem.requires_grad_(False)
    pixels=np.load(legacy.PIXELS,mmap_mode='r')
    arms=('historical','wrong','fixed')
    keys=('ship_peak','tokens','positions','areas','edge','img_tokens','ship_tokens','img_areas','ship_areas')
    output={arm:{key:[] for key in keys} for arm in arms}
    stats=[]; started=time.perf_counter()
    for start in range(0,len(rows),32):
        batch=rows[start:start+32]
        pair=torch.from_numpy(np.array(pixels[batch],np.float32)).cuda()/255.
        image=legacy.normalize(legacy.rgb(F.interpolate(pair,(112,112),mode='bilinear',align_corners=False,antialias=True)))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            feature=stem(image)
        feat=feature.float().flatten(2).transpose(1,2)
        angles=[slots.ship_angle(geo.loc[g,'ship_axis_rad'],geo.loc[g,'ship_valid']) for g in batch]
        for arm in arms:
            box=masks_for(batch,records,arm)
            cells=[legacy.regional_cells(b) for b in box]
            pool=torch.from_numpy(np.stack([v[0] for v in cells])).cuda().reshape(-1,64,56*56)
            output[arm]['tokens'].append(torch.bmm(pool,feat).reshape(-1,4,16,64).cpu().numpy())
            output[arm]['positions'].append(np.stack([v[1] for v in cells]))
            output[arm]['areas'].append(np.stack([v[2] for v in cells]))
            output[arm]['edge'].append(np.array([v[3] for v in cells]))
            peak=feat.transpose(1,2).masked_fill(~torch.from_numpy(box).cuda().flatten(1)[:,None],-torch.inf).amax(2)
            assert torch.isfinite(peak).all()
            output[arm]['ship_peak'].append(peak.cpu().numpy())
            for prefix in ('img','ship'):
                values=[slots.regional_slots(b,a if prefix=='ship' else 0.) for b,a in zip(box,angles)]
                pool=torch.from_numpy(np.stack([v[0] for v in values])).cuda().reshape(-1,16,56*56)
                output[arm][prefix+'_tokens'].append(torch.bmm(pool,feat).reshape(-1,4,4,64).cpu().numpy())
                output[arm][prefix+'_areas'].append(np.stack([v[1] for v in values]))
            stats.extend(dict(global_row=int(g),arm=arm,body_pixels=int(b.sum()),
                              body_fraction=float(b.mean()),edge=bool(c[3]),
                              empty_regions=int((c[2].sum(1)==0).sum())) for g,b,c in zip(batch,box,cells))
        if start%320==0 or start+32>=len(rows):
            print('REGIONS',min(start+32,len(rows)),'/',len(rows),'seconds',round(time.perf_counter()-started,1),flush=True)
    for arm,values in output.items():
        arrays={k:np.concatenate(v) for k,v in values.items()}
        assert all(np.isfinite(a).all() for a in arrays.values())
        np.savez_compressed(HERE/f'features_{arm}.npz',rows=rows,**arrays)
    pd.DataFrame(stats).to_csv(HERE/'region_coverage.csv',index=False)
    # An independent cached old representation catches geometry/order/preprocessing drift.
    with np.load(ROOT/'ship_radar_geometry_design_20261016/spatial_cache/Singapore.npz') as old:
        selected=np.searchsorted(rows,old['rows'])
        np.testing.assert_array_equal(rows[selected],old['rows'])
        errors={}
        for k in ('tokens','positions','areas','ship_peak'):
            new=np.concatenate(output['historical'][k])[selected]
            err=float(np.abs(new-old[k]).max()); errors[k]=err
            np.testing.assert_allclose(new,old[k],atol=2e-5,rtol=1e-5)
    summary=dict(rows=len(rows),arms=list(arms),checkpoint=config['source_checkpoint'],
        seconds=time.perf_counter()-started,historical_cache_errors=errors,
        input_pixels_unchanged=True,all_region_dependent_readouts_rebuilt=True,
        angles_unchanged=True,source_gallery_rows=len(config['gallery_rows']))
    (HERE/'FEATURES_DONE.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
