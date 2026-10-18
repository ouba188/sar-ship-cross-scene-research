"""Build a bounded, explicitly provisional cross-acquisition relation cohort."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from osgeo import gdal
from PIL import Image, ImageDraw

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'land_mask'))
from land_mask import WaterMap, WC


def split_regions(target, other, worldcover):
    body=target.astype(bool)
    neighbor=other.astype(bool)&~body
    available=~(body|neighbor)
    water=available&(worldcover==80)
    rest=available&~water
    return np.stack([body,neighbor,water,rest]).astype(np.uint8)


def polygon(corners, x, y, width, height):
    im=Image.new('1',(width,height))
    ImageDraw.Draw(im).polygon([tuple(p) for p in np.asarray(corners)-[x,y]],fill=1)
    return np.asarray(im,dtype=bool)


def main():
    HERE.mkdir(exist_ok=True)
    gdal.UseExceptions();gdal.SetConfigOption('GDAL_PAM_ENABLED','NO')
    gdal.SetCacheMax(128*1024**2)
    proposals=pd.read_csv(ROOT/'recovery_31cc_2ec7/dense_review/source_review.csv')
    policy=pd.read_csv(ROOT/'land_mask/object_policy.csv')
    cohort=proposals[proposals.unique_containment_proposal].merge(policy[['object_id','exclude_100m','action']],left_on='proposed_object_id',right_on='object_id')
    cohort=cohort[~cohort.exclude_100m].copy()
    counts=cohort.groupby('class_raw').agg(n=('mmsi','size'),products=('product','nunique'))
    classes=sorted(counts[(counts.n>=4)&(counts.products>=2)].index)
    cohort=cohort[cohort.class_raw.isin(classes)].sort_values(['product','class_raw','mmsi']).reset_index(drop=True)
    cohort['label']=cohort.class_raw.map({c:i for i,c in enumerate(classes)})
    assert cohort.mmsi.nunique()==len(cohort)
    objects={}; manifests={}; inputs={};paths={}
    for name in ['object_bridge','object_bridge_b529','object_bridge_31cc_2ec7']:
        batch=ROOT/name
        for obj in json.loads((batch/'objects.json').read_text()):objects[obj['object_id']]=obj
        table=pd.read_csv(batch/'crop_manifest.csv')
        for r in table.to_dict('records'):manifests[r['object_id']]=dict(r,bridge=name)
        inputs[name]=np.load(batch/'inputs128.npy',mmap_mode='r')
        data=json.loads((batch/'INPUTS.json').read_text())
        for r in data['sources']:paths[r['product']]=r['raster_vv']
    all_by_product={p:[o for o in objects.values() if o['product']==p] for p in cohort['product'].unique()}
    rasters={p:gdal.Open(paths[p]) for p in cohort['product'].unique()}
    wm=WaterMap(sorted(WC.glob('*_Map.tif')))
    images=[]; masks=[]; extra=[]
    for r in cohort.to_dict('records'):
        obj=objects[r['object_id']];crop=manifests[r['object_id']]
        x,y,w,h=[int(crop[k]) for k in ['left','top','width','height']]
        target=np.zeros((h,w),bool);other=np.zeros((h,w),bool)
        for candidate in all_by_product[r['product']]:
            dest=target if candidate['object_id']==obj['object_id'] else other
            for member in candidate['members']:
                pts=np.asarray(member['source_corners'])
                if pts[:,0].max()<x or pts[:,0].min()>=x+w or pts[:,1].max()<y or pts[:,1].min()>=y+h:continue
                dest |= polygon(pts,x,y,w,h)
        wc,_=wm.window(rasters[r['product']],(x,y,w,h))
        regions=split_regions(target,other,wc)
        resized=np.stack([np.asarray(Image.fromarray(a).resize((128,128),Image.Resampling.NEAREST)) for a in regions])
        assert np.all(resized.sum(0)==1)
        images.append(inputs[crop['bridge']][int(crop['feature_row'])]);masks.append(resized)
        extra.append(dict(native_vv=crop['native_vv'],native_vh=crop['native_vh'],native_width=w,native_height=h,bridge=crop['bridge'],feature_row=crop['feature_row'],
                          image_other_fraction=float(regions[1].mean()),image_water_fraction=float(regions[2].mean()),image_rest_fraction=float(regions[3].mean()),image_map_unknown_fraction=float((wc==0).mean())))
    cohort=pd.concat([cohort,pd.DataFrame(extra)],axis=1)
    cohort.to_csv(HERE/'cohort.csv',index=False)
    pd.crosstab(cohort.class_raw,cohort['product'].str[-4:]).to_csv(HERE/'class_by_acquisition.csv')
    np.savez_compressed(HERE/'inputs.npz',images=np.stack(images),regions=np.stack(masks),labels=cohort.label.to_numpy(),mmsi=cohort.mmsi.to_numpy(),products=cohort['product'].to_numpy(str))
    config=dict(classes=classes,n=len(cohort),unique_mmsi=cohort.mmsi.nunique(),products=cohort['product'].nunique(),
                eligibility='unique point-in-OBB proposal without competing source; not land-interior; raw class>=4records and>=2acquisitions, chosen before model outcomes',
                label_status='Source metadata joined via unique spatial containment proposals, NOT human-confirmed identity. Exploratory relation probe,NOT formal classification main table.',
                support_budget=16,support_draw_seeds=[0,1,2,3,4],support_sampling='Up to2 per source class then fill to16 without replacement; identical across arms; every source class represented',
                region_order=['target_candidate','other_candidates','remaining_map_water','remaining_nonwater_or_unknown'],
                protocol='Each acquisition held out as query; all supports and centering from other acquisitions; all84MMSI distinct; no target labels for fitting/selection.',
                arms=['global','global_grid','global_region','global_area'],
                feature_plan='Frozen ImageNet ResNet50 V2; last avgpool base,layer1 four local mean pools per polarization; equal local dimensions; fixed0.5 base/local cosine mix; per-pol supports-centered L2 representation. No trainable region parameters.',
                classifier='Support-class mean kernel score,softmax fixed temperature0.1; relationship diagnostic readout,NOT existing full dual-backbone support-adapted model.',
                claim_limits='Map access is extra geographic information; feature receptive fields cross region borders; region pooling is not pure physical component separation; no random backbone seeds for fixed inference.')
    (HERE/'PROTOCOL.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(json.dumps(config,indent=2),flush=True)


if __name__=='__main__':main()
