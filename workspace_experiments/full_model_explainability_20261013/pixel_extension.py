"""Query-only pixel interventions in each frozen port/F full-model context."""

from run_paths import ROOT, OUT, read, write, arrays, Geometry, EPISODES
import argparse
import gc
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import softmax
from threadpoolctl import threadpool_limits
import torch
from torchvision.models import resnet18
from projection_residual_expert import discarded_coordinates

sys.path.insert(0, str(ROOT/'region_cue_probe_20261013'))
sys.path.insert(0, str(ROOT/'region_cue_replication_20261013'))
from transforms_regions import apply, box_mask
from port_transforms import read_images, transform_images, boxes

DEST = OUT/'pixel_extension'
FOLDER_F = Path('E:/Hermes/paper_tracking/sensitivity/f0921_detector_recrop_20261007')
CONDITIONS = ('original', 'target_shuffle', 'outer_shuffle', 'target_flat',
              'sham_flat', 'target_display', 'outer_display')


def identity(values):
    return pd.Series(values).astype(str).str.replace(r'\.0$', '', regex=True).to_numpy()


def f_region(row):
    if row.status != 'detector_dynamic':
        return (0,0,0,0), (0,0,0,0), False, False
    side = float(row.requested_side)
    cx = (float(row.center_x)-float(row.left))*128/side
    cy = (float(row.center_y)-float(row.top))*128/side
    length = float(row.det_len_px)*128/side
    valid = (row.status == 'detector_dynamic' and np.isfinite([cx,cy,length]).all()
             and 0 <= cx < 128 and 0 <= cy < 128 and length > 0)
    if not valid:
        return (0,0,0,0), (0,0,0,0), False, False
    radius = length/2+max(3*128/side, .20*length)
    l,t = np.maximum(np.floor([cx-radius,cy-radius]).astype(int),0)
    r,b = np.minimum(np.ceil([cx+radius,cy+radius]).astype(int),128)
    region = (int(l),int(t),int(r),int(b))
    w,h = r-l,b-t
    valid = 0 < w*h < 128*128
    candidates = [(0,0,w,h),(128-w,0,128,h),(0,128-h,w,128),(128-w,128-h,128,128)]
    candidates.sort(key=lambda a:-((a[0]+a[2])/2-cx)**2-((a[1]+a[3])/2-cy)**2)
    for a in candidates:
        if min(r,a[2]) <= max(l,a[0]) or min(b,a[3]) <= max(t,a[1]):
            return region, tuple(map(int,a)), valid, True
    return region,(0,0,0,0),valid,False


def load_frame(task, data):
    if task['task'].startswith('self24ports/'):
        frame = pd.read_csv(ROOT/'region_cue_replication_20261013/ports/frame.csv',
                            dtype={'sample_id':str,'identity':str},keep_default_na=False)
        frame = frame.iloc[data['rows']].reset_index(drop=True)
        np.testing.assert_array_equal(frame.sample_id.to_numpy(),data['ids'])
        regions = [boxes(row,.20) for row in frame.itertuples()]
        frame['region_valid'] = [all(x[2] for x in a) for a in regions]
        frame['sham_valid'] = [all(x[3] for x in a) for a in regions]
        return frame,None
    frame = pd.read_csv(FOLDER_F/'target_manifest.csv', dtype={'mmsi':str},keep_default_na=False)
    crop = pd.read_csv(FOLDER_F/'crop_metadata.csv',keep_default_na=False)
    np.testing.assert_array_equal(frame.row,crop.row)
    np.testing.assert_array_equal(frame.sample_id.to_numpy(),data['ids'])
    for col in crop.columns:
        if col not in frame:
            frame[col] = crop[col]
    frame['identity'] = identity(frame.mmsi)
    regions = [f_region(row) for row in frame.itertuples()]
    frame['region_valid'] = [a[2] for a in regions]
    frame['sham_valid'] = [a[3] for a in regions]
    frame['product'] = frame.scene
    return frame,np.load(FOLDER_F/'inputs128.npy',mmap_mode='r')


def changed_image(frame, pixels, q, condition):
    row = frame.iloc[q]
    if pixels is None:
        native = read_images(row)
        changed, info = transform_images(native,row,condition,int(q))
        image = np.asarray(Image.fromarray(changed[1]).resize((128,128),Image.Resampling.BILINEAR))
    else:
        pair = np.asarray(pixels[q])
        region,sham,valid,sham_valid = f_region(row)
        mask = np.zeros((128,128),bool)
        kind = 'original'
        if condition != 'original':
            zone,kind = condition.split('_')
            if zone == 'target':
                mask = box_mask((128,128),region)
            elif zone == 'outer':
                mask = ~box_mask((128,128),region)
            elif sham_valid:
                mask = box_mask((128,128),sham)
        changed = apply(pair,mask,kind,20261013+int(q))
        np.testing.assert_array_equal(changed[:,~mask],pair[:,~mask])
        info = dict(region_valid=valid,sham_valid=sham_valid,
                    region_area=(region[2]-region[0])*(region[3]-region[1])/16384,
                    edited_area=float(mask.mean()),changed_fraction=float((changed!=pair).mean()),
                    change_rms=float(np.sqrt(np.mean((changed.astype(float)-pair)**2))),
                    x0=region[0],y0=region[1],x1=region[2],y1=region[3])
        image = changed[1]
    return image,info


def extractor(tail):
    net = resnet18(weights=None)
    ck = torch.load(ROOT/'visual_reliable_baseline/e206_out/ssl_resnet18.pt',
                    map_location='cpu',weights_only=True)
    missing,unexpected = net.load_state_dict(ck['backbone'],strict=False)
    assert set(missing)=={'fc.weight','fc.bias'} and not unexpected
    net.layer4.load_state_dict(torch.load(tail,map_location='cpu',weights_only=True))
    prefix = torch.nn.Sequential(*list(net.children())[:-3]).eval()
    return prefix,net.layer4.eval()


def extract(prefix,tail,frame,pixels,chosen,condition):
    features,controls = [],[]
    with torch.inference_mode():
        for lo in range(0,len(chosen),8):
            batch=[]
            for q in chosen[lo:lo+8]:
                im,info=changed_image(frame,pixels,q,condition)
                batch.append(im.copy())
                controls.append(dict(condition=condition,query_index=int(q),**info))
            x=torch.from_numpy(np.stack(batch)).float()[:,None].repeat(1,3,1,1)/255.
            features.append(tail(prefix(x).half().float()).mean((2,3)).numpy())
    return np.concatenate(features),controls


def predictor(geo,coeff,chosen,data,folder):
    refs=geo.encode(np.unique(data['support']))
    original=geo.encode(chosen)
    kr,_=geo.kernels(original,refs)
    union=np.unique(data['support'])
    parts=np.load(folder/'parts.npy',mmap_mode='r')
    m,stats=geo.ctx.model,geo.ctx.stats
    raw=(np.asarray(geo.raw[data['rows'][chosen]]) if geo.ctx.dataset=='self24ports'
         else np.asarray(geo.raw[chosen]))
    def predict(extra):
        if geo.ctx.dataset=='self24ports':
            bank=np.array(geo.bank[chosen],copy=True)
            bank[:,128:192]=m['pca_ssl'].transform(extra).astype(np.float32)
            from run_protected_drift_activation import make_views
            v=make_views(bank,raw,extra,geo.ctx.geometry)['common'][True]
            br,bd=v['retained'][:,128:],v['complement'][:,2048:]
            pt=np.c_[bank,geo.conditions[chosen]]@coeff['dual_coef']+coeff['dual_intercept']
        else:
            from compact_fusion_source import _expert_inputs
            x=_expert_inputs(raw,extra,m)
            br=(x/np.sqrt(192)).astype(float)[:,128:]
            bd=(discarded_coordinates(extra,m['pca_ssl'])-stats['discarded_center'][2048:])
            bd=bd/stats['discarded_scale'][2048:]/np.sqrt(2560)
            pt=(x@coeff['dual_coef']+coeff['dual_intercept']).astype(np.float32).astype(float)
        kb=.5*(br@refs['br'].T+bd@refs['bd'].T)
        answer=[]
        for draw,support in enumerate(data['support']):
            col=np.searchsorted(union,support)
            expert=pt+(kr[:,col]+kb[:,col])@coeff[f'{draw}_dual_refit_expert']
            answer.append(.5*softmax(parts[draw,0,chosen],axis=1)+.5*softmax(expert,axis=1))
        return np.stack(answer)
    return predict


def measures(task,frame,chosen,data,prob,frozen,dest):
    y=frame.class_id.to_numpy(int)[chosen]
    identities=identity(frame.identity)
    known=np.array([len(a)==9 and a.isdigit() and a!='000000000' for a in identities])
    sham=frame.sham_valid.to_numpy(bool)[chosen]
    records,classes=[],[]
    for draw in range(5):
        same_ship=np.isin(identities[chosen],identities[data['support'][draw]][known[data['support'][draw]]])
        subsets={'all':np.ones(len(y),bool),'sham_valid':sham,
                 'known_no_support_ship':known[chosen]&~same_ship}
        for subset,use in subsets.items():
            if not use.any():continue
            yy=y[use];r=frozen[draw,0,chosen][use];t=prob[0,draw,use].argmax(1)
            _,ix,counts=np.unique(yy,return_inverse=True,return_counts=True)
            w=100./(len(counts)*counts[ix]);rescue=(r!=yy)&(t==yy);harm=(r==yy)&(t!=yy)
            for ci,condition in enumerate(CONDITIONS):
                if condition=='sham_flat' and subset!='sham_valid':continue
                pred=prob[ci,draw,use].argmax(1)
                records.append(dict(task=task['task'],dataset=task['task'].split('/')[0],
                    draw=draw,subset=subset,condition=condition,n=len(yy),classes=len(counts),
                    oa=100*np.mean(pred==yy),ba=float(w@(pred==yy)),
                    delta_oa=100*np.mean((pred==yy).astype(int)-(t==yy)),
                    delta_ba=float(w@((pred==yy).astype(int)-(t==yy))),
                    old_rescue_mass=float(w@rescue),old_rescue_lost=float(w@(rescue&(pred!=yy))),
                    old_harm_removed=float(w@(harm&(pred==yy)))))
                for cl in np.unique(yy):
                    mask=yy==cl
                    classes.append(dict(task=task['task'],dataset=task['task'].split('/')[0],draw=draw,
                        subset=subset,condition=condition,class_id=int(cl),n=int(mask.sum()),
                        recall=100*np.mean(pred[mask]==cl),
                        delta_recall=100*np.mean((pred[mask]==cl).astype(int)-(t[mask]==cl))))
    pd.DataFrame(records).to_csv(dest/'metrics.csv',index=False)
    pd.DataFrame(classes).to_csv(dest/'classes.csv',index=False)


def run(task, sham_only=False):
    dest=(DEST/'sham_supplement' if sham_only else DEST)/task['task']
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'COMPLETE.json').exists():
        print('EXISTS',task['task'],flush=True);return
    folder=OUT/'tasks'/task['task'];started=time.perf_counter()
    data={k:np.load(folder/'input'/(k+'.npy'),mmap_mode='r')
          for k in ('rows','ids','support','support_labels','eval_masks')}
    frame,pixels=load_frame(task,data)
    eligible=data['eval_masks'].all(0)&frame.region_valid.to_numpy(bool)
    if sham_only:
        eligible &= frame.sham_valid.to_numpy(bool)
    rng=np.random.default_rng(20261013);chosen=[]
    for cl in range(8):
        pool=np.flatnonzero(eligible&(frame.class_id.to_numpy()==cl))
        chosen.extend(rng.choice(pool,min(16,len(pool)),replace=False).tolist())
    chosen=np.array(sorted(chosen),int)
    assert len(chosen)>0
    selected=frame.iloc[chosen].copy();selected['query_index']=chosen
    selected.to_csv(dest/'selected.csv',index=False)
    write(dest/'SELECTION_FROZEN.json',dict(task=task['task'],seed=20261013,
        cohort='sham_supplement' if sham_only else 'main',
        query_indices=chosen.tolist(),n=len(chosen),eligible=int(eligible.sum()),
        classes=selected.class_id.value_counts().sort_index().to_dict(),
        sham_n=int(selected.sham_valid.sum()),selection_uses_predictions=False,
        definition='native detection bbox' if pixels is None else 'detector center/length square, not complete bbox'))
    if pixels is not None:
        from osgeo import gdal
        checks=[]
        for scene,group in selected.groupby('scene'):
            date=scene[14:22]
            path=Path('D:/F_0921_relabel_det')/date[:4]/date/(scene+'.tif')
            raster=gdal.Open(str(path))
            assert group.requested_side.max()<=min(raster.RasterXSize,raster.RasterYSize)
            checks.append(dict(scene=scene,width=raster.RasterXSize,height=raster.RasterYSize))
            raster=None
        write(dest/'SCENE_HEADER_CHECK.json',checks)
    coeff=arrays(task['coefficients']);geo=Geometry(task,data,coeff,folder)
    tailpath=(EPISODES/geo.ctx.task/'joint/layer4.pt' if geo.ctx.dataset=='self24ports'
              else geo.ctx.joint/'layer4.pt')
    predict=predictor(geo,coeff,chosen,data,folder)
    frozen=np.load(folder/'predictions.npy',mmap_mode='r')
    parts=np.load(folder/'parts.npy',mmap_mode='r')
    oldextra=np.asarray(geo.extra[chosen]);cached=predict(oldextra)
    for draw in range(5):
        np.testing.assert_array_equal(cached[draw].argmax(1),frozen[draw,7,chosen])
    prefix,tail=extractor(tailpath)
    original,controls=extract(prefix,tail,frame,pixels,chosen,'original')
    probability=predict(original)
    errors=np.linalg.norm(original-oldextra,axis=1)/np.maximum(np.linalg.norm(oldextra,axis=1),1e-12)
    flips=int((probability.argmax(2)!=cached.argmax(2)).sum())
    receipt=dict(task=task['task'],n=len(chosen),checkpoint=str(tailpath),
        feature_max_relative_l2=float(errors.max()),probability_max_error=float(abs(probability-cached).max()),
        original_decision_flips=flips,cached_decision_replay=True,sham_n=int(selected.sham_valid.sum()))
    np.save(dest/'original_extra.npy',original)
    if errors.max()>1e-4 or flips:
        write(dest/'REPLAY_FAILED.json',receipt)
        raise AssertionError('Original replay failed: '+str(receipt))
    write(dest/'ORIGINAL_REPLAY.json',receipt)
    result=[probability]
    for condition in CONDITIONS[1:]:
        extra,rows=extract(prefix,tail,frame,pixels,chosen,condition)
        controls.extend(rows);np.save(dest/(condition+'_extra.npy'),extra)
        result.append(predict(extra))
    result=np.stack(result)
    pd.DataFrame(controls).to_csv(dest/'pixel_controls.csv',index=False)
    np.savez_compressed(dest/'predictions.npz',probability=result,query_indices=chosen,
                        conditions=np.array(CONDITIONS),labels=frame.class_id.to_numpy(int)[chosen],
                        sham_valid=selected.sham_valid.to_numpy(bool))
    measures(task,frame,chosen,data,result,frozen,dest)
    write(dest/'COMPLETE.json',dict(**receipt,draws=5,conditions=CONDITIONS,new_training=False,
                                  seconds=time.perf_counter()-started))
    print('COMPLETE',task['task'],len(chosen),'sham',receipt['sham_n'],
          'replay',receipt['feature_max_relative_l2'],round(time.perf_counter()-started,1),flush=True)
    del geo,prefix,tail,frame,pixels,predict,parts,frozen
    gc.collect()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',nargs='+')
    parser.add_argument('--sham-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1)
    tasks=[t for t in read(ROOT/'paper_core_freeze_20261013/FROZEN_REFERENCE.json')['tasks']
           if not t['task'].startswith('oss2/')]
    if args.task:tasks=[t for t in tasks if t['task'] in args.task]
    with threadpool_limits(limits=1):
        for task in tasks:run(task,args.sham_only)
