"""Cached footprint and class-response diagnostics; no model fitting."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
OSS = HERE.parent
EIGHT = OSS.parent / 'eight_class_evidence_20261023'


def stats(a):
    a = np.asarray(a, float).ravel()
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(n=0)
    q = np.quantile(a, [.01, .1, .5, .9, .99])
    return dict(n=len(a), mean=float(a.mean()), sd=float(a.std()),
                q01=q[0], q10=q[1], q50=q[2], q90=q[3], q99=q[4])


def geometry(dataset, table):
    if dataset == 'OSS2':
        g = pd.read_csv(OSS / 'native_geometry.csv').rename(columns={'sample_id': 'id'})
        g['side'] = 224
        g['crop_width'], g['crop_height'] = g.width, g.height
        g['pad_left'], g['pad_top'] = (224-g.width)//2, (224-g.height)//2
        g['crop_x'], g['crop_y'], g['valid'] = 0, 0, True
        g['native_pixels'] = g.width*g.height
    else:
        g = pd.read_csv(EIGHT / 'coverage.csv').rename(columns={'instance_id': 'id'})
        g['side'] = 160
        # masks_for partitions every native pixel into target/other/rest.
        snap = json.loads((EIGHT.parents[1] / 'same_vessel_gallery_20261019/local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))
        views = {s['obs_id']: s['images'] for s in snap}
        rows = table.set_index('instance_id')
        g['native_pixels'] = [views[rows.loc[r.id, 'obs_id']][r.pol]['w'] *
                              views[rows.loc[r.id, 'obs_id']][r.pol]['h'] for r in g.itertuples()]
        g['path'] = [rows.loc[r.id, 'image_'+r.pol] for r in g.itertuples()]
    results = []
    for r in g.itertuples():
        # layer3 stride 16; 160/224 are divisible by 16, so exact area bins.
        edges = np.arange(0, r.side, 16)
        wx = np.maximum(0, np.minimum(edges+16, r.pad_left+r.crop_width)-np.maximum(edges, r.pad_left))/16
        wy = np.maximum(0, np.minimum(edges+16, r.pad_top+r.crop_height)-np.maximum(edges, r.pad_top))/16
        w = wy[:, None]*wx[None, :]
        mass = w.sum()
        results.append(dict(dataset=dataset, id=r.id, class_id=r.class_id, pol=r.pol,
            valid=r.valid, side=r.side, retained_pixels=r.crop_width*r.crop_height,
            native_pixels=r.native_pixels, retained_fraction=r.crop_width*r.crop_height/r.native_pixels,
            canvas_coverage=r.crop_width*r.crop_height/r.side**2,
            positive_cells=int((w>0).sum()), effective_cells=mass**2/(w*w).sum(),
            grid_cells=(r.side//16)**2, footprint_mass=mass,
            crop_x=r.crop_x, crop_y=r.crop_y, crop_width=r.crop_width,
            crop_height=r.crop_height, path=r.path))
    return pd.DataFrame(results)


def input_pixels(g):
    # Deterministic label-free systematic sample of actual cohort row order.
    ids = g.id.drop_duplicates().to_numpy()
    chosen = set(ids[np.linspace(0, len(ids)-1, min(128, len(ids)), dtype=int)])
    rows = []
    for r in g[g.id.isin(chosen) & g.valid].itertuples():
        with Image.open(r.path) as im:
            a = np.asarray(im.convert('L'))
        a = a[r.crop_y:r.crop_y+r.crop_height, r.crop_x:r.crop_x+r.crop_width]
        rows.append(dict(dataset=r.dataset, id=r.id, pol=r.pol, pixels=a.size,
            mean=float(a.mean()/255), sd=float(a.std()/255),
            p01=float(np.quantile(a,.01)/255), p99=float(np.quantile(a,.99)/255),
            zero_fraction=float((a==0).mean()), max_fraction=float((a==255).mean()),
            unique_levels=len(np.unique(a))))
    return pd.DataFrame(rows)


def descriptor(mean, mom):
    cols=[]
    for pol in range(2):
        for v in [mean[:,pol],mom[:,pol,:,0],mom[:,pol,:,1]]:
            cols.extend([v,v-v.mean(1,keepdims=True)])
    return np.stack(cols,-1)


def main():
    summaries, details, shift, paired, readers, relation, geometry_frames, pixels, tails, role_geometry = [], [], [], [], [], [], [], [], [], []
    for dataset, root, nfold, idkey, cachepath, readerdir in [
        ('OSS2',OSS,6,'sample_id',OSS/'moments.npz',OSS/'readers'),
        ('ports24_development',EIGHT,3,'instance_id',EIGHT/'calibrated_signed_transfer/signed_features.npz',EIGHT/'baseline')]:
        table=pd.read_csv(root/'cohort.csv',dtype={'mmsi':str})
        geo=geometry(dataset,table)
        geometry_frames.append(geo)
        pixels.append(input_pixels(geo))
        with np.load(cachepath,allow_pickle=True) as z:
            cache={k:z[k] for k in z.files}
        np.testing.assert_array_equal(cache['sample_ids' if dataset=='OSS2' else 'instance_ids'],table[idkey])
        valid=cache['valid'].astype(bool)
        names=list(cache['reader_names'])
        geom=[geo[geo.pol.eq(p)].set_index('id').loc[table[idkey]] for p in ['vv','vh']]
        for fold in range(nfold):
            with np.load(readerdir/f'fold{fold}_inputs.npz',allow_pickle=True) as z:
                d={k:z[k] for k in ['train','validation','test','base','oof']}
            wnorm={}
            for suffix in ['outer']+[f'inner{k}' for k in range(4)]:
                with np.load(readerdir/'models'/f'fold{fold}_{suffix}.npz') as z:
                    w=z['W'].reshape(8,2,1024).transpose(1,0,2)
                    wc=w-w.mean(1,keepdims=True)
                    wnorm[suffix]=np.linalg.norm(wc,axis=-1)
                    for p in range(2):
                        for c in range(8):
                            readers.append(dict(dataset=dataset,fold=fold,reader=suffix,pol=p,coordinate=c,
                                contrast_W_norm=wnorm[suffix][p,c],W_norm=np.linalg.norm(w[p,c]),
                                fit_rows=len(z['fit']),alpha=float(z['alpha'])))
            idx=names.index(f'fold{fold}_outer')
            mean,mom=cache['mean_pol'][idx],cache['moments_pol'][idx]
            om,od=np.zeros_like(mean),np.zeros_like(mom)
            norm=np.zeros_like(mean)
            for k in range(4):
                with np.load(readerdir/'models'/f'fold{fold}_inner{k}.npz') as z:
                    held=z['held']
                j=names.index(f'fold{fold}_inner{k}')
                om[held],od[held],norm[held]=cache['mean_pol'][j,held],cache['moments_pol'][j,held],wnorm[f'inner{k}']
            tr=d['train']
            source=descriptor(om[tr],od[tr])
            mu,sd=source.mean((0,1)),np.maximum(source.std((0,1)),1e-8)
            hi=np.quantile(source,.99,axis=(0,1));lo=np.quantile(source,.01,axis=(0,1))
            for role,ix,mm,rr,nn in [('source_oof',tr,om,od,norm),
                ('source_outer_same_rows',tr,mean,mom,np.broadcast_to(wnorm['outer'],mean.shape)),
                ('validation',d['validation'],mean,mom,np.broadcast_to(wnorm['outer'],mean.shape)),
                ('test',d['test'],mean,mom,np.broadcast_to(wnorm['outer'],mean.shape))]:
                raw=descriptor(mm[ix],rr[ix]);z=(raw-mu)/sd
                for slot in range(12):
                    p=slot//6
                    use=valid[ix,p]
                    a=z[use,:,slot]
                    energy=(a*a).sum(1)
                    order=np.argsort(-energy,kind='stable')
                    for rank,pos in enumerate(order[:3]):
                        row=int(ix[use][pos])
                        tails.append(dict(dataset=dataset,fold=fold,role=role,slot=slot,pol=p,
                            quantity=['mu','mu_classcenter','sigma','sigma_classcenter','kappa','kappa_classcenter'][slot%6],
                            rank=rank+1,id=table.iloc[row][idkey],class_id=int(table.iloc[row].class_id),
                            effective_cells=float(geom[p].iloc[row].effective_cells),
                            sample_z_rms=float(np.sqrt(energy[pos]/8)),
                            total_squared_z_share=float(energy[pos]/energy.sum()),
                            raw_min=float(raw[use,:,slot][pos].min()),raw_max=float(raw[use,:,slot][pos].max())))
                    shift.append(dict(dataset=dataset,fold=fold,role=role,slot=slot,
                        quantity=['mu','mu_classcenter','sigma','sigma_classcenter','kappa','kappa_classcenter'][slot%6],pol=p,
                        z_rms=float(np.sqrt(np.mean(a*a))),z_abs_q99=float(np.quantile(abs(a),.99)),
                        fraction_outside_source_01_99=float(np.mean((raw[use,:,slot]<lo[slot])|(raw[use,:,slot]>hi[slot])))))
                for p in range(2):
                    use=valid[ix,p];ii=ix[use]
                    for key in ['retained_pixels','canvas_coverage','positive_cells','effective_cells']:
                        role_geometry.append(dict(dataset=dataset,fold=fold,role=role,pol=p,
                            quantity=key,**stats(geom[p].iloc[ii][key])))
                    sigma=rr[ii,p,:,0];kappa=rr[ii,p,:,1]
                    values={'mu':mm[ii,p], 'sigma':sigma,'kappa':kappa,
                        'abs_kappa_over_sigma':np.divide(abs(kappa),sigma,out=np.zeros_like(sigma),where=sigma>1e-12),
                        'sigma_over_Wnorm':sigma/np.maximum(nn[ii,p],1e-12)}
                    for key,a in values.items():
                        summaries.append(dict(dataset=dataset,fold=fold,role=role,pol=p,quantity=key,**stats(a)))
                        for c in range(8):
                            details.append(dict(dataset=dataset,fold=fold,role=role,pol=p,quantity=key,coordinate=c,**stats(a[:,c])))
                    cells=geom[p].effective_cells.to_numpy()[ii]
                    # Descriptive associations, not regressions or fitted thresholds.
                    for label,mask in [('all',np.ones(len(ii),bool))]+[(str(c),table.iloc[ii].class_id.to_numpy()==c) for c in range(8)]:
                        if mask.sum()<3:continue
                        for key in ['sigma','sigma_over_Wnorm','abs_kappa_over_sigma']:
                            value=values[key].mean(1)[mask]
                            rho=float(spearmanr(cells[mask],value).statistic) if np.std(cells[mask]) and np.std(value) else np.nan
                            relation.append(dict(dataset=dataset,fold=fold,role=role,pol=p,true_class=label,
                                quantity=key,n=int(mask.sum()),effective_cells_spearman=rho))
            for p in range(2):
                ii=tr[valid[tr,p]]
                for c in range(8):
                    a=od[ii,p,c,0];b=mom[ii,p,c,0]
                    paired.append(dict(dataset=dataset,fold=fold,pol=p,coordinate=c,
                        oof_sigma_mean=float(a.mean()),outer_sigma_mean=float(b.mean()),
                        outer_over_oof_mean=float(b.mean()/a.mean()),
                        same_row_sigma_spearman=float(spearmanr(a,b).statistic)))
            print(dataset,fold,'done',flush=True)
    geo=pd.concat(geometry_frames,ignore_index=True)
    geo.to_csv(HERE/'geometry_rows.csv',index=False)
    geometry_summary=[]
    for (ds,pol),g in geo[geo.valid].groupby(['dataset','pol']):
        for quantity in ['native_pixels','retained_pixels','retained_fraction','canvas_coverage','positive_cells','effective_cells']:
            geometry_summary.append(dict(dataset=ds,pol=pol,quantity=quantity,**stats(g[quantity])))
    pd.DataFrame(geometry_summary).to_csv(HERE/'geometry_summary.csv',index=False)
    pd.concat(pixels,ignore_index=True).to_csv(HERE/'pixel_sample.csv',index=False)
    for name,rows in [('moment_summary',summaries),('moment_by_coordinate',details),
                      ('descriptor_shift',shift),('same_rows_reader_change',paired),
                      ('W_norms',readers),('geometry_moment_association',relation),('descriptor_tails',tails),
                      ('geometry_by_role',role_geometry)]:
        pd.DataFrame(rows).to_csv(HERE/f'{name}.csv',index=False)
    (HERE/'receipt.json').write_text(json.dumps(dict(datasets=2,folds=9,trained_models=0,
        png_sampling='128 systematic cohort rows per dataset, both valid polarizations',
        source_reference='fold source OOF only; quantiles .01/.99 fixed before computation',
        geometry='exact retained rectangular footprint; layer3 stride16; Kish effective cells, not independent RF samples',
        roles=['source_oof','source_outer_same_rows','validation','test']),indent=2),encoding='utf-8')


if __name__=='__main__':main()
