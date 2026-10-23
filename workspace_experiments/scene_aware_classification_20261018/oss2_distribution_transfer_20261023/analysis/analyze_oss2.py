"""Frozen OSS2 prediction analysis and native 224-pad review; never trains."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import train_heads as run
from extract_features import native_image
sys.path.insert(0, str(run.parent.HERE))
import analyze_results as common

ARMS = ['mean_only','local','additive','extra_score']
PAIRS = [('additive','mean_only'),('additive','local'),('additive','extra_score'),
         ('mean_only','local'),('mean_only','extra_score'),('local','extra_score')]


def highest_other(score, y):
    s=score.copy();s[y]=-np.inf
    return int(s.argmax())


def render(p, table):
    t=table.set_index('sample_id')
    (HERE/'panels').mkdir(exist_ok=True)
    rows,missing=[],[]
    for arm in ARMS:
        g=p[p.arm.eq(arm)&p.seed.eq(17)&p.role.eq('test')]
        for outcome in ['rescued','harmed','CC','WW']:
            hits=g[g.outcome.eq(outcome)].sort_values(['instance_id','fold'])
            if not len(hits):
                missing.append(dict(arm=arm,outcome=outcome));continue
            q=hits.iloc[0];native=t.loc[q.instance_id];y=int(q.class_id)
            a=q[[f'a{c}' for c in range(8)]].to_numpy(float)
            b=q[[f'b{c}' for c in range(8)]].to_numpy(float)
            s=q[[f's{c}' for c in range(8)]].to_numpy(float)
            oldr,newr=highest_other(b,y),highest_other(s,y)
            delta=s-b
            fig,axes=plt.subplots(2,3,figsize=(15,10),layout='constrained')
            dims=[];covers=[]
            for j,pol in enumerate(['vv','vh']):
                path=native[f'chip_path_{pol}']
                with Image.open(path) as im:raw=np.array(im)
                padded,mask=native_image(path)
                h,w=raw.shape;top,left=(224-h)//2,(224-w)//2
                np.testing.assert_array_equal(np.rint(padded[0,top:top+h,left:left+w]*255).astype(np.uint8),raw)
                dims.append(f'{pol}:{w}x{h}');covers.append(float(mask.mean()))
                axes[j,0].imshow(raw,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                axes[j,0].set_title(f'{pol.upper()} original {w}x{h}; uint8 range {raw.min()}..{raw.max()}')
                axes[j,1].imshow(padded[0],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
                axes[j,1].set_title(f'224 native pad; no resize/crop; valid={q[f"valid_{pol}"]}\nfootprint={mask.mean():.4f}')
                for ax in axes[j,:2]:ax.set_xlabel('native pixel x');ax.set_ylabel('native pixel y')
            xx=np.arange(8)
            for offset,score,label,color in [(-.25,a,'Frozen A','#aaaaaa'),(0,b,'Strong B=T(A)','#496979'),(.25,s,arm,'#158964')]:
                axes[0,2].bar(xx+offset,score,width=.24,label=label,color=color)
            axes[0,2].set_xticks(xx);axes[0,2].legend(fontsize=8);axes[0,2].set_title('Actual eight-class scores')
            axes[1,2].bar(xx,delta,color=['#158964' if c==y else '#bd5264' for c in xx])
            axes[1,2].set_xticks(xx);axes[1,2].axhline(0,color='black',lw=.6)
            axes[1,2].set_title('Actual increment score - B; not physical attribution')
            mechanisms={'mean_only':'B + repeated auxiliary per-pol means',
                        'local':'B + auxiliary per-pol sigma/kappa',
                        'additive':'B + jointly trained mean/local branches',
                        'extra_score':'B scores only; images shown for review, NOT head inputs'}
            fig.suptitle(f'OSS2 / {arm} / seed17 / fold{int(q.fold)} / epoch{int(q.epoch)} / {outcome}\n'
                f'{q.instance_id} / {native.class_name} / {native.pass_group}\n{mechanisms[arm]}\n'
                f'true={y}; B winner {b.argmax()} -> {s.argmax()}; rival {oldr} -> {newr}; '
                f'margin {b[y]-b[oldr]:+.5f} -> {s[y]-s[newr]:+.5f}',fontsize=10)
            panel=HERE/'panels'/f'{arm}_{outcome}.png'
            fig.savefig(panel,dpi=100);plt.close(fig)
            rows.append(dict(arm=arm,seed=17,role='test',fold=int(q.fold),epoch=int(q.epoch),outcome=outcome,
                instance_id=q.instance_id,class_id=y,class_name=native.class_name,mmsi=native.mmsi,
                pass_group=native.pass_group,product=native.scene_id,dimensions=';'.join(dims),
                valid_vv=bool(q.valid_vv),valid_vh=bool(q.valid_vh),footprint_vv=covers[0],footprint_vh=covers[1],
                old_winner=int(b.argmax()),final_winner=int(s.argmax()),old_rival=oldr,final_rival=newr,
                old_margin=b[y]-b[oldr],final_margin=s[y]-s[newr],
                delta_true=delta[y],delta_oldrival=delta[oldr],delta_finalrival=delta[newr],
                fixed_oldrival_margin_delta=delta[y]-delta[oldr],
                image_vv=native.chip_path_vv,image_vh=native.chip_path_vh,panel=str(panel),
                selection='seed17 test first instance_id then fold per arm/outcome'))
    pd.DataFrame(rows).to_csv(HERE/'review_cases.csv',index=False)
    return rows,missing


def main():
    status=json.loads((ROOT/'training_summary.json').read_text(encoding='utf-8'))
    frozen=json.loads((ROOT/'selection_frozen.json').read_text(encoding='utf-8'))
    assert status['heads']==90 and frozen['heads']==90 and not frozen['test_labels_used_for_selection']
    table=pd.read_csv(ROOT/'cohort.csv',dtype={'mmsi':str})
    p=pd.read_csv(ROOT/'predictions.csv',dtype={'mmsi':str})
    assert set(p.arm)==set(ARMS) and set(p.seed)=={17,29,43}
    assert set(p.fold)==set(range(6)) and set(p.role)=={'validation','test'}
    aligned=table.set_index('sample_id').loc[p.instance_id]
    np.testing.assert_array_equal(aligned.class_id,p.class_id)
    np.testing.assert_array_equal(aligned.mmsi,p.mmsi)
    assert np.isfinite(p[[f'{prefix}{c}' for prefix in ['a','b','s'] for c in range(8)]].to_numpy()).all()
    for _,g in p[p.role.eq('test')].groupby(['arm','seed']):
        assert len(g)==len(table) and g.instance_id.is_unique
    shared=pd.read_csv(ROOT/'shared_mmsi.csv',dtype={'mmsi':str})
    for ref in ['ports24_manifest','active_eight15251']:
        p['shared_'+ref]=p.mmsi.isin(shared.loc[shared.reference.eq(ref),'mmsi'])
    pooled,strata,paired,ci,classes,unique_val,baseline=[],[],[],[],[],[],[]
    for (role,arm,seed),g in p.groupby(['role','arm','seed']):
        meta=dict(role=role,arm=arm,seed=seed)
        am=common.metrics(g)
        pooled.append(dict(**meta,unique_observations=g.instance_id.nunique(),**am))
        paired.append(dict(**meta,reference='B',n=len(g),net=int(g.net.sum()),
            **{k:am[k] for k in ['delta_OA','delta_BA','delta_NLL','delta_Brier']}))
        for unit in ['fold','class_id','pass_group','product','valid_vh','shared_ports24_manifest','shared_active_eight15251']:
            for name,h in g.groupby(unit):
                strata.append(dict(**meta,unit=unit,name=name,**common.metrics(h)))
        for (fold,c),h in g.groupby(['fold','class_id']):
            classes.append(dict(**meta,fold=fold,class_id=c,**common.metrics(h)))
        if role=='test':
            ci.extend(dict(**meta,**r) for r in common.intervals(g,g.net.to_numpy(),arm+'-B'))
        else:
            h=g.sort_values(['fold','instance_id']).drop_duplicates('instance_id')
            unique_val.append(dict(**meta,**common.metrics(h)))
        if arm=='mean_only':
            for label,prefix in [('A','a'),('B','b')]:
                baseline.append(dict(role=role,seed=seed,baseline=label,
                    **run.parent.assessment(g.class_id.to_numpy(),g[[f'{prefix}{c}' for c in range(8)]].to_numpy())))
    paired_classes=[]
    for arm,reference in PAIRS:
        for (role,seed),g in p[p.arm.eq(arm)].groupby(['role','seed']):
            a=g.set_index(['fold','instance_id']).sort_index()
            h=p[p.arm.eq(reference)&p.seed.eq(seed)&p.role.eq(role)].set_index(['fold','instance_id']).loc[a.index]
            np.testing.assert_array_equal(a.class_id,h.class_id)
            np.testing.assert_array_equal(a[[f'b{c}' for c in range(8)]],h[[f'b{c}' for c in range(8)]])
            meta=dict(role=role,arm=arm,seed=seed)
            am,hm=common.metrics(a),common.metrics(h)
            diff=a.net.to_numpy()-h.net.to_numpy()
            paired.append(dict(**meta,reference=reference,n=len(a),net=int(diff.sum()),
                **{f'delta_{k}':am[k]-hm[k] for k in ['OA','BA','NLL','Brier']}))
            for c,sub in a.groupby('class_id'):
                hh=h.loc[sub.index]
                paired_classes.append(dict(**meta,reference=reference,class_id=c,n=len(sub),
                    net=int(sub.net.sum()-hh.net.sum()),delta_recall=100*float(sub.net.mean()-hh.net.mean())))
            if role=='test':
                ci.extend(dict(**meta,**r) for r in common.intervals(a,diff,arm+'-'+reference))
    for name,rows in [('pooled_metrics',pooled),('stratified_metrics',strata),('class_fold_metrics',classes),
                      ('paired_metrics',paired),('paired_classes',paired_classes),('paired_intervals',ci),
                      ('validation_unique_metrics',unique_val),('baseline_metrics',baseline)]:
        pd.DataFrame(rows).to_csv(HERE/f'{name}.csv',index=False)
    sf=pd.DataFrame(strata)
    for unit,name in [('class_id','class_metrics'),('fold','fold_metrics'),('pass_group','pass_metrics'),
                      ('product','acquisition_metrics'),('valid_vh','vh_metrics'),
                      ('shared_ports24_manifest','shared_ports24_metrics'),('shared_active_eight15251','shared_active_metrics')]:
        sf[sf.unit.eq(unit)].to_csv(HERE/f'{name}.csv',index=False)
    selected=pd.read_csv(ROOT/'selections.csv')
    selected.groupby('arm').agg(heads=('epoch','size'),zero_epochs=('epoch',lambda x:int(x.eq(0).sum())),
        median_epoch=('epoch','median'),max_epoch=('epoch','max')).reset_index().to_csv(HERE/'selection_summary.csv',index=False)
    cases,missing=render(p,table)
    (HERE/'analysis_summary.json').write_text(json.dumps(dict(training_runs=0,prediction_rows=len(p),
        test_unique_per_seed=len(table),test_passes=table.pass_group.nunique(),test_mmsi=table.mmsi.nunique(),
        cases=len(cases),unique_case_objects=len({r['instance_id'] for r in cases}),missing_outcomes=missing,
        images_viewed=False,pixels_resized=0,pixels_cropped=0,pad_side=224,
        ci='Fixed predictions; pass/MMSI clusters 2000 draws seed20261023; not fitting/selection uncertainty'),indent=2),encoding='utf-8')
    print(pd.DataFrame(pooled)[['role','arm','seed','net','OA','BA','delta_OA','delta_BA','NLL','Brier']].to_string(index=False),flush=True)
    print(pd.DataFrame(cases)[['arm','outcome','instance_id','fold','epoch','class_id','old_winner','final_winner','old_margin','final_margin']].to_string(index=False),flush=True)


if __name__=='__main__':main()
