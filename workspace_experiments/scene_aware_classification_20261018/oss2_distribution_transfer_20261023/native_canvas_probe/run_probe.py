"""Source-validation-only, frozen 224 versus 288 canvas sensitivity."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '2')
import sys
sys.dont_write_bytecode = True
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import extract_features as extract
import train_heads as heads


def main():
    tick = time.perf_counter()
    assert (HERE/'PROTOCOL_ZH.md').exists()
    table = pd.read_csv(ROOT/'cohort.csv', dtype={'mmsi': str})
    with np.load(ROOT/'readers/fold0_inputs.npz') as z:
        ix = z['validation']
        old_a = z['base'][ix]
        np.testing.assert_array_equal(z['sample_ids'], table.sample_id)
    assert len(ix) == 279
    table = table.iloc[ix].reset_index(drop=True)
    with np.load(ROOT/'features.npz') as z:
        old_feature = z['layer3'][ix]
        assert z['valid'][ix].all()
        np.testing.assert_array_equal(z['sample_ids'][ix], table.sample_id)
    with np.load(ROOT/'calibrated/fold0_seed17.npz') as z:
        old_b = z['base'][ix]
    with np.load(ROOT/'readers/models/fold0_outer.npz') as z:
        w, intercept = z['W'], z['intercept']
    model = extract.network()
    features = {side: np.empty((279, 2048), np.float32) for side in [224, 288]}
    maps224, maps288, mapsdiff, footprints = [], [], [], []
    footprint_error = 0.
    with torch.inference_mode():
        for first in range(0, len(table), 16):
            stop = min(first+16, len(table))
            x, mask = extract.load_batch(table, first, stop)
            x288, mask288 = F.pad(x, (32,32,32,32)), F.pad(mask, (32,32,32,32))
            assert torch.equal(x288[:,:,32:-32,32:-32], x)
            pieces = {224: [], 288: []}
            visual = [[], [], [], []]
            for rotation in range(4):
                z = extract.feature_map(model, x, rotation)
                z288 = extract.feature_map(model, x288, rotation)
                wt = extract.spatial_weights(mask, z, rotation)
                wt288 = extract.spatial_weights(mask288, z288, rotation)
                center = z288[:,:,2:-2,2:-2]
                error = float((wt-wt288[:,:,2:-2,2:-2]).abs().max())
                footprint_error = max(footprint_error, error)
                assert error == 0 and torch.equal(wt.sum((-2,-1)), wt288.sum((-2,-1)))
                for side, zz, ww in [(224,z,wt),(288,z288,wt288)]:
                    pieces[side].append((zz*ww).sum((-2,-1))/ww.sum((-2,-1)))
                for bucket, value in zip(visual, [z.mean(1), center.mean(1),
                        (center-z).square().mean(1).sqrt(), wt[:,0]]):
                    bucket.append(value.rot90(-rotation, (-2,-1)).cpu())
            for side in features:
                features[side][first:stop] = torch.stack(pieces[side]).mean(0).cpu().numpy().reshape(stop-first,2048)
            for target, values in zip([maps224,maps288,mapsdiff,footprints],visual):
                target.append(torch.stack(values).mean(0).numpy().reshape(stop-first,2,14,14))
            print('source validation',stop,'/279',flush=True)
    maps224,maps288,mapsdiff,footprints = [np.concatenate(v) for v in [maps224,maps288,mapsdiff,footprints]]
    a = {side: value.astype(float)@w.T+intercept for side,value in features.items()}
    b = {side: heads.predict(ROOT/'heads/fold0_calibration_seed17.pt',value)[0] for side,value in a.items()}
    replay = dict(feature_max_abs=float(abs(features[224]-old_feature).max()),
                  A_max_abs=float(abs(a[224]-old_a).max()), B_max_abs=float(abs(b[224]-old_b).max()),
                  A_top1_changes=int((a[224].argmax(1)!=old_a.argmax(1)).sum()),
                  B_top1_changes=int((b[224].argmax(1)!=old_b.argmax(1)).sum()),
                  footprint_max_abs=footprint_error)
    (HERE/'replay.json').write_text(json.dumps(replay,indent=2),encoding='utf-8')
    np.testing.assert_allclose(features[224],old_feature,rtol=1e-4,atol=5e-5)
    assert replay['A_max_abs'] < 1e-4 and replay['B_max_abs'] < 1e-4
    df = table.copy()
    df['global_row'] = ix
    sizes = [[Image.open(path).size for path in table[f'chip_path_{pol}']] for pol in ['vv','vh']]
    for pol,size in zip(['vv','vh'],sizes):
        df[f'{pol}_width'],df[f'{pol}_height'] = np.array(size).T
    df['long_side'] = df[['vv_width','vv_height','vh_width','vh_height']].max(axis=1)
    df['size_bin'] = pd.cut(df.long_side,[0,64,128,192,224],labels=['<=64','65-128','129-192','193-224']).astype(str)
    diff = features[288]-features[224]
    df['feature_rms_delta'] = np.sqrt(np.mean(diff**2,axis=1))
    df['feature_relative_l2'] = np.linalg.norm(diff,axis=1)/np.maximum(np.linalg.norm(features[224],axis=1),1e-12)
    df['feature_max_abs'] = abs(diff).max(1)
    for pol,p in enumerate(['vv','vh']):
        dd,ff = diff[:,pol*1024:(pol+1)*1024],features[224][:,pol*1024:(pol+1)*1024]
        df[f'{p}_feature_relative_l2'] = np.linalg.norm(dd,axis=1)/np.maximum(np.linalg.norm(ff,axis=1),1e-12)
    y = df.class_id.to_numpy()
    metrics,groups = [],[]
    for name, scores in [('A',a),('B',b)]:
        for side, score in scores.items():
            df[f'{name}{side}_prediction'] = score.argmax(1)
            for c in range(8): df[f'{name}{side}_{c}'] = score[:,c]
            metrics.append(dict(model=name,side=side,**heads.parent.assessment(y,score)))
        before,after = scores[224].argmax(1),scores[288].argmax(1)
        df[f'{name}_flip'] = before != after
        df[f'{name}_net'] = (after==y).astype(int)-(before==y).astype(int)
        df[f'{name}_outcome'] = np.select([df[f'{name}_net'].eq(1),df[f'{name}_net'].eq(-1),after==y],['rescued','harmed','CC'],default='WW')
        for kind in ['size_bin','class_id']:
            for value,g in df.groupby(kind):
                idx = g.index.to_numpy()
                rec = dict(model=name,grouping=kind,group=value,n=len(g),flips=int(g[f'{name}_flip'].sum()),net=int(g[f'{name}_net'].sum()),
                           feature_relative_l2_median=float(g.feature_relative_l2.median()))
                for side in [224,288]:
                    pred = scores[side][idx].argmax(1)
                    rec[f'OA{side}'] = float(np.mean(pred==y[idx])*100)
                    rec[f'BA_present_classes{side}'] = float(np.mean([np.mean(pred[y[idx]==c]==c) for c in np.unique(y[idx])])*100)
                groups.append(rec)
    df.to_csv(HERE/'predictions.csv',index=False)
    pd.DataFrame(metrics).to_csv(HERE/'metrics.csv',index=False)
    pd.DataFrame(groups).to_csv(HERE/'strata.csv',index=False)
    np.savez_compressed(HERE/'source_features.npz',sample_ids=table.sample_id.to_numpy(str),global_rows=ix,feature224=features[224],feature288=features[288])
    np.savez_compressed(HERE/'coarse_maps.npz',sample_ids=table.sample_id.to_numpy(str),mean224=maps224,mean288_center=maps288,rms_delta=mapsdiff,footprint=footprints)
    cases=[]
    for outcome in ['rescued','harmed','CC','WW']:
        g = df[df.B_outcome.eq(outcome)].sort_values('sample_id')
        if not len(g):
            cases.append(dict(outcome=outcome,available=False));continue
        row=g.iloc[0];i=int(row.name)
        fig,axes=plt.subplots(2,4,figsize=(13,7),constrained_layout=True)
        for p,pol in enumerate(['vv','vh']):
            with Image.open(row[f'chip_path_{pol}']) as im: raw=np.array(im)
            axes[p,0].imshow(raw,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            axes[p,0].set_title(f'{pol.upper()} native {raw.shape[1]}x{raw.shape[0]}')
            high=max(float(maps224[i,p].max()),float(maps288[i,p].max()))
            for col,value,title in [(1,maps224[i,p],'224 mean activation'),(2,maps288[i,p],'288 aligned central 14x14'),(3,mapsdiff[i,p],'C4 mean channel-RMS change')]:
                plot=axes[p,col].imshow(value,cmap='magma',interpolation='nearest',vmin=0,vmax=high if col<3 else None)
                axes[p,col].contour(footprints[i,p]>0,levels=[.5],colors='cyan',linewidths=.6)
                axes[p,col].set_title(title,fontsize=9)
                fig.colorbar(plot,ax=axes[p,col],fraction=.045)
            for ax in axes[p]:ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle(f"{row.sample_id} {outcome}: y={row.class_id}, B224={row.B224_prediction}, B288={row.B288_prediction}\nCyan: nonzero pooling footprint; coarse feature cells, not pixel causality",fontsize=12)
        panel=HERE/f'case_{outcome}.png';fig.savefig(panel,dpi=130);plt.close(fig)
        record=row.to_dict();record.update(outcome=outcome,available=True,panel=str(panel));cases.append(record)
    pd.DataFrame(cases).to_csv(HERE/'cases.csv',index=False)
    info=dict(n=279,fold=0,seed=17,canvases=[224,288],test_evaluated=False,trained=False,seconds=time.perf_counter()-tick,
        replay=replay,feature_relative_l2_quantiles=np.quantile(df.feature_relative_l2,[0,.25,.5,.75,1]).tolist(),
        A_flips=int(df.A_flip.sum()),B_flips=int(df.B_flip.sum()),A_net=int(df.A_net.sum()),B_net=int(df.B_net.sum()),
        B_outcomes=df.B_outcome.value_counts().to_dict(),peak_allocated_bytes=torch.cuda.max_memory_allocated())
    (HERE/'summary.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info),flush=True)
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)


if __name__ == '__main__':main()
