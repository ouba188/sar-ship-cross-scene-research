"""Fixed-prediction comparisons and native panels for the two standard heads."""
import json
import sys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_controls import HERE,ref,parent,restore
sys.path.insert(0,str(parent.HERE))
import analyze_results as old


def render(p):
    table=parent.load_table().set_index('instance_id')
    snapshots={s['obs_id']:s for s in json.loads((old.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    (HERE/'panels').mkdir(exist_ok=True)
    rows=[];missing=[]
    for arm,g in p[p.seed.eq(17)&p.role.eq('test')].groupby('arm'):
        selected=[]
        for outcome in ['rescued','harmed','CC','WW']:
            h=g[g.outcome.eq(outcome)].sort_values('instance_id')
            if len(h):selected.append(h.iloc[0])
            else:missing.append(dict(arm=arm,outcome=outcome))
        if len(selected)<4:
            for outcome in ['CC','WW']:
                selected.extend(q for _,q in g[g.outcome.eq(outcome)].sort_values('instance_id').iloc[1:2].iterrows())
        for q in selected:
            row=table.loc[q.instance_id];y=int(q.class_id)
            b=q[[f'b{c}' for c in range(8)]].to_numpy(float)
            s=q[[f's{c}' for c in range(8)]].to_numpy(float)
            before=max((c for c in range(8) if c!=y),key=lambda c:b[c])
            after=max((c for c in range(8) if c!=y),key=lambda c:s[c])
            fig,axes=plt.subplots(2,2,figsize=(10,11),layout='constrained')
            dimensions=[]
            for j,pol in enumerate(['vv','vh']):
                view=snapshots[int(row.obs_id)]['images'][pol]
                with Image.open(row[f'image_{pol}']) as im:raw=np.array(im.convert('L'))
                pixels,masks,place=old.crop_pad(raw,old.masks_for(view),160)
                dimensions.append(f'{pol}:{raw.shape[1]}x{raw.shape[0]}')
                axes[0,j].imshow(raw,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                axes[0,j].set_title(f'{pol.upper()} original {raw.shape[1]}x{raw.shape[0]}')
                axes[1,j].imshow(pixels,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                axes[1,j].set_title(f'Actual 160 crop/pad; valid={q[f"valid_{pol}"]}; footprint={masks.sum(0).mean():.3f}')
                prior=view['priors'];t=prior.get('target_index');det=prior.get('detections',[])
                if t is not None and 0<=t<len(det):
                    points=np.array(det[t]['corners']);points=np.r_[points,points[:1]]
                    axes[0,j].plot(points[:,0],points[:,1],color='#13c74e',lw=1)
                    points=points-[place['crop_x'],place['crop_y']]+[place['pad_left'],place['pad_top']]
                    axes[1,j].plot(points[:,0],points[:,1],color='#13c74e',lw=1)
                for ax in axes[:,j]:ax.set_xlabel('pixel x');ax.set_ylabel('pixel y')
            mechanism='Direct shared residual; no tanh or kappa carrier' if arm=='free_shared' else 'Fixed class/pol tanh coefficients times kappa; no image gate'
            fig.suptitle(f'{arm} / seed17 / fold{int(q.fold)} / epoch{int(q.epoch)} / {q.outcome}\n'
                f'{q.instance_id} / {row.class_name} / {row.port}\n{mechanism}\n'
                f'true={y}; prediction {int(q.base_prediction)} -> {int(q.prediction)}; strongest rival {before} -> {after}\n'
                f'true margin {b[y]-b[before]:+.5f} -> {s[y]-s[after]:+.5f}; '
                f'fixed old rival change {(s-b)[y]-(s-b)[before]:+.5f}',fontsize=10)
            path=HERE/'panels'/f'{arm}_{q.outcome}_{q.instance_id}.png'
            fig.savefig(path,dpi=110);plt.close(fig)
            rows.append(dict(arm=arm,seed=17,fold=int(q.fold),epoch=int(q.epoch),outcome=q.outcome,instance_id=q.instance_id,
                class_id=y,class_name=row.class_name,port=row.port,base_prediction=int(q.base_prediction),prediction=int(q.prediction),
                old_rival=before,new_rival=after,base_margin=b[y]-b[before],margin=s[y]-s[after],
                old_rival_delta=(s-b)[y]-(s-b)[before],image_vv=row.image_vv,image_vh=row.image_vh,
                dimensions=';'.join(dimensions),valid_vh=bool(q.valid_vh),panel=str(path),mechanism=mechanism))
    pd.DataFrame(rows).to_csv(HERE/'review_cases.csv',index=False)
    return rows,missing


def main():
    p=pd.read_csv(HERE/'predictions.csv',dtype={'mmsi':str})
    original=pd.read_csv(HERE.parent/'predictions.csv',dtype={'mmsi':str})
    strata=[];paired=[];ci=[];matched_classes=[];unique_val=[]
    for (role,arm,seed),g in p.groupby(['role','arm','seed']):
        meta=dict(role=role,arm=arm,seed=seed)
        for unit in ['class_id','fold','valid_vh','pass_group','port']:
            for name,h in g.groupby(unit):strata.append(dict(**meta,unit=unit,name=name,**old.metrics(h)))
        if role=='validation':
            h=g.sort_values(['fold','instance_id']).drop_duplicates('instance_id')
            unique_val.append(dict(**meta,**old.metrics(h)))
        else:
            ci += [dict(**meta,**r) for r in old.intervals(g,g.net.to_numpy(),arm+'-B')]
        key=['fold','instance_id']
        a=g.set_index(key).sort_index()
        for reference in ['pol_preserved','pol_preserved_capped','aggregate_repeated']:
            h=original[original.role.eq(role)&original.seed.eq(seed)&original.arm.eq(reference)].set_index(key).loc[a.index]
            assert np.array_equal(a.class_id,h.class_id)
            diff=a.net.to_numpy()-h.net.to_numpy()
            y=a.class_id.to_numpy()
            am=old.metrics(a);hm=old.metrics(h)
            comparison=arm+'-'+reference
            paired.append(dict(**meta,reference=reference,n=len(a),unique_observations=a.index.get_level_values('instance_id').nunique(),
                net=int(diff.sum()),delta_OA=am['OA']-hm['OA'],delta_BA=am['BA']-hm['BA'],
                delta_NLL=am['NLL']-hm['NLL'],delta_Brier=am['Brier']-hm['Brier']))
            for c in range(8):
                mask=y==c
                matched_classes.append(dict(**meta,reference=reference,class_id=c,n=int(mask.sum()),net=int(diff[mask].sum()),
                    delta_recall=100*float(diff[mask].mean())))
            if role=='test':ci += [dict(**meta,**r) for r in old.intervals(a,diff,comparison)]
    for name,rows in [('stratified_metrics',strata),('matched_metrics',paired),('paired_intervals',ci),
                      ('matched_classes',matched_classes),('validation_unique_metrics',unique_val)]:
        pd.DataFrame(rows).to_csv(HERE/f'{name}.csv',index=False)
    coefficients=[]
    for fold in range(3):
        for seed in ref.SEEDS:
            ck,model=restore(HERE/'models'/f'fold{fold}_static_pol_seed{seed}.pt')
            raw=model.coefficient.detach().numpy();coef=np.tanh(raw)
            for c in range(8):
                for pol in range(2):coefficients.append(dict(fold=fold,seed=seed,epoch=ck['epoch'],class_id=c,
                    pol=['VV','VH'][pol],raw=raw[c,pol],coefficient=coef[c,pol]))
    pd.DataFrame(coefficients).to_csv(HERE/'static_coefficients.csv',index=False)
    cases,missing=render(p)
    summary=dict(panels=[r['panel'] for r in cases],missing_outcomes=missing,images_viewed=False,
        seed_pooling=False,bootstrap_scope='paired fixed predictions; no training/selection uncertainty')
    (HERE/'analysis_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(pd.DataFrame(paired).query("role=='test'").to_string(index=False),flush=True)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
