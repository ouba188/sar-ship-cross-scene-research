"""Analyze completed paired scores and render unchanged native query/reference chips."""
from pathlib import Path
import argparse
import html
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(ROOT/'mplconfig')
from obb_geometry import conditions, pipeline_conditions, perturb


def main(args):
    import numpy as np
    import pandas as pd
    from PIL import Image, ImageDraw
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    spec=importlib.util.spec_from_file_location('obb_robustness_experiment',ROOT/'experiment.py')
    ex=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=ex
    spec.loader.exec_module(ex)
    ex.initialize()
    e=ex.e
    e.task.DEVICE='cpu'
    source=ROOT.parents[1]/'m1_coverage_20261101'
    allrows=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    groups=json.loads((source/'groups.json').read_text(encoding='utf-8'))
    ids=np.array(groups['evaluation'])
    rows=[allrows[i] for i in ids]
    refs=[allrows[i] for i in groups['references']]
    y=np.array([r['label'] for r in rows])
    scores={}
    modes={'m2':conditions()}
    if args.pipeline:modes['pipeline']=pipeline_conditions()
    for mode,configs in modes.items():
        for cfg in configs:
            values=[]
            for seed in ex.SEEDS:
                folder=ROOT/f'seed_{seed}'
                receipt=json.loads((folder/'zero_replay.json').read_text())
                assert receipt['predictions_identical'] and receipt['score_max_error']<1e-4
                d=np.load(folder/f'{mode}_{cfg["name"]}.npz')
                assert np.array_equal(d['ids'],ids) and np.isfinite(d['score']).all()
                values.append(d['score'])
            scores[mode,cfg['name']]=np.stack(values)
    clean=scores['m2','zero']
    clean_correct=clean.argmax(-1)==y
    metrics=[];classes=[];individual=[];strata=[]
    audit=pd.read_csv(ROOT/'mask_audit.csv')
    cfg_lookup={cfg['name']:cfg for cfg in conditions()}
    native=audit[audit.region=='body'].copy()
    native['length_delta_px']=[r.length_px*cfg_lookup[r.condition]['value']
        if cfg_lookup[r.condition]['family'] in ('length','scale') else 0. for r in native.itertuples()]
    native['width_delta_px']=[r.width_px*cfg_lookup[r.condition]['value']
        if cfg_lookup[r.condition]['family'] in ('width','scale') else 0. for r in native.itertuples()]
    native['changed_length_px']=native.length_px+native.length_delta_px
    native['changed_width_px']=native.width_px+native.width_delta_px
    native[['condition','index','id','polarization','length_px','width_px','changed_length_px','changed_width_px',
            'length_delta_px','width_delta_px','dx_px','dy_px','displacement_px','angle_delta_deg']].to_csv(
                ROOT/'geometry_parameters.csv',index=False)
    coverage=audit[audit.region=='body'].groupby(['condition','index']).agg(
        width_px=('width_px','mean'),length_px=('length_px','mean'),displacement_px=('displacement_px','mean'),
        coverage=('original_target_covered','mean'),grid20_l1=('grid20_l1','sum'),grid10_l1=('grid10_l1','sum'))
    for (mode,name),stack in scores.items():
        for si,seed in enumerate(ex.SEEDS):
            z=stack[si];pred=z.argmax(1);correct=pred==y
            delta=correct.astype(float)-clean_correct[si]
            rec=[100*correct[y==k].mean() for k in range(4)]
            metrics.append(dict(mode=mode,condition=name,seed=seed,n=len(y),oa=100*correct.mean(),ba=np.mean(rec),
                oa_delta=100*delta.mean(),ba_delta=100*np.mean([delta[y==k].mean() for k in range(4)]),
                rescued=int((correct&~clean_correct[si]).sum()),harmed=int((~correct&clean_correct[si]).sum()),
                flipped=int((pred!=clean[si].argmax(1)).sum()),score_max_change=float(abs(z-clean[si]).max())))
            for label in range(4):classes.append(dict(mode=mode,condition=name,seed=seed,label=label,
                n=int((y==label).sum()),recall=rec[label],recall_delta=100*delta[y==label].mean()))
            cov=coverage.loc[name].reindex(range(len(y)))
            for title,mask in [('width_le8',cov.width_px.to_numpy()<=8),('width_gt8',cov.width_px.to_numpy()>8),
                               ('zero_grid_change',(cov.grid20_l1.to_numpy()+cov.grid10_l1.to_numpy())<1e-7),
                               ('nonzero_grid_change',(cov.grid20_l1.to_numpy()+cov.grid10_l1.to_numpy())>=1e-7)]:
                if mask.any():strata.append(dict(mode=mode,condition=name,seed=seed,stratum=title,n=int(mask.sum()),
                    oa_delta=100*delta[mask].mean(),rescued=int((correct&~clean_correct[si]&mask).sum()),
                    harmed=int((~correct&clean_correct[si]&mask).sum())))
            for j,row in enumerate(rows):
                individual.append(dict(mode=mode,condition=name,seed=seed,index=j,original_index=int(ids[j]),
                    id=row['id'],mmsi=row['mmsi'],orbit=e.task.orbit(row),label=int(y[j]),
                    clean_pred=int(clean[si,j].argmax()),prediction=int(pred[j]),
                    **{f'score_{k}':float(z[j,k]) for k in range(4)},
                    **{f'clean_score_{k}':float(clean[si,j,k]) for k in range(4)},
                    true_margin=float(z[j,y[j]]-np.max(np.delete(z[j],y[j])))))
    df=pd.DataFrame(metrics)
    df.to_csv(ROOT/'metrics.csv',index=False)
    pd.DataFrame(classes).to_csv(ROOT/'per_class.csv',index=False)
    pd.DataFrame(individual).to_csv(ROOT/'scores.csv',index=False)
    pd.DataFrame(strata).to_csv(ROOT/'stratified_changes.csv',index=False)
    summary=df.groupby(['mode','condition']).agg(oa=('oa','mean'),oa_std=('oa','std'),ba=('ba','mean'),
        ba_std=('ba','std'),oa_delta=('oa_delta','mean'),ba_delta=('ba_delta','mean'),
        rescued=('rescued','sum'),harmed=('harmed','sum'),flipped=('flipped','sum'),score_max_change=('score_max_change','max'))
    summary.to_csv(ROOT/'summary.csv')
    # Paired vessel bootstrap, conditional on these three fixed fitted models.
    ship=np.array([str(r['mmsi']) for r in rows])
    orbit=np.array([e.task.orbit(r) for r in rows])
    rng=np.random.default_rng(20261111)
    clusters=[]
    for label in range(4):
        units=[np.flatnonzero(ship==s) for s in np.unique(ship[y==label])]
        assert all(np.all(y[u]==label) for u in units),'Mixed-label ship'
        clusters.append((units,rng.integers(0,len(units),(3000,len(units)))))
    intervals=[];scene_rows=[]
    for (mode,name),stack in scores.items():
        if name=='zero':continue
        delta=(stack.argmax(-1)==y).mean(0)-clean_correct.mean(0)
        total=[];counts=[];class_mean=[]
        for units,draw in clusters:
            sums=np.array([delta[u].sum() for u in units]);sizes=np.array([len(u) for u in units])
            t=sums[draw].sum(1);n=sizes[draw].sum(1)
            total.append(t);counts.append(n);class_mean.append(t/n)
        oa=100*np.sum(total,axis=0)/np.sum(counts,axis=0);ba=100*np.mean(class_mean,axis=0)
        intervals.append(dict(mode=mode,condition=name,ships=len(set(ship)),
            oa_delta=100*delta.mean(),oa_low=np.quantile(oa,.025),oa_high=np.quantile(oa,.975),
            ba_delta=100*np.mean([delta[y==k].mean() for k in range(4)]),
            ba_low=np.quantile(ba,.025),ba_high=np.quantile(ba,.975),bootstrap_draws=3000,bootstrap_seed=20261111,
            scope='perturbed minus clean; class-stratified paired ship bootstrap; fixed fits and observed scenes; unadjusted intervals'))
        for scene in np.unique(orbit):
            mask=orbit==scene
            scene_rows.append(dict(mode=mode,condition=name,orbit=scene,n=int(mask.sum()),
                oa_delta=100*delta[mask].mean(),ba_delta_present=100*np.mean([delta[mask&(y==k)].mean() for k in np.unique(y[mask])])))
    pd.DataFrame(intervals).to_csv(ROOT/'paired_intervals.csv',index=False)
    pd.DataFrame(scene_rows).to_csv(ROOT/'per_scene_changes.csv',index=False)
    audit.groupby(['condition','region']).agg(n=('id','size'),median_displacement_px=('displacement_px','median'),
        mean_area_px=('changed_area_px','mean'),mean_l1_px=('native_l1_px','mean'),mean_iou=('soft_iou','mean'),
        mean_coverage=('original_target_covered','mean'),mean_grid20_l1=('grid20_l1','mean'),
        zero_grid20_count=('grid20_l1',lambda x:int((x<1e-7).sum())),
        zero_grid10_count=('grid10_l1',lambda x:int((x<1e-7).sum()))).to_csv(ROOT/'mask_summary.csv')
    if args.pipeline:
        comparison=[]
        for cfg in pipeline_conditions():
            name=cfg['name']
            for si,seed in enumerate(ex.SEEDS):
                a=scores['pipeline',name][si].argmax(1)==y;b=scores['m2',name][si].argmax(1)==y
                d=a.astype(float)-b
                comparison.append(dict(condition=name,seed=seed,oa_delta=100*d.mean(),
                    ba_delta=100*np.mean([d[y==k].mean() for k in range(4)]),
                    rescued=int((a&~b).sum()),harmed=int((~a&b).sum())))
        pd.DataFrame(comparison).to_csv(ROOT/'pipeline_vs_m2.csv',index=False)
    fig,axes=plt.subplots(3,2,figsize=(11,10),layout='constrained')
    for ax,family in zip(axes.flat,('along','across','scale','angle','length','width')):
        configs=[v for v in conditions() if v['family']==family]
        for mode in modes:
            for metric,color in [('oa_delta','#236dbe'),('ba_delta','#b43c48')]:
                chosen=sorted([conditions()[0]]+[v for v in configs if (mode,v['name']) in scores],key=lambda v:v['value'])
                x=[v['value']*(1 if family=='angle' else 100) for v in chosen]
                yy=[summary.loc[(mode,v['name']),metric] for v in chosen]
                ax.plot(x,yy,marker='o',color=color,linestyle='-' if mode=='m2' else '--',label=f'{mode} {metric[:2].upper()}')
        ax.axhline(0,color='gray',linewidth=.8);ax.set_title(family)
        ax.set_xlabel('Angle error (degrees)' if family=='angle' else 'Signed error (%)')
        ax.set_ylabel('Change from clean (percentage points)');ax.legend(fontsize=8)
    fig.savefig(ROOT/'robustness.png',dpi=180);plt.close(fig)

    def figure(row,rival,cfg,path):
        records=[row]+[next(r for r in refs if r['label']==label and str(r['mmsi'])!=str(row['mmsi'])
            and e.task.orbit(r)!=e.task.orbit(row)) for label in (row['label'],rival)]
        views=[v for r in records for v in e.task.data.prepare(r)]
        widths=[max(180,v['raw'].shape[1]+12) if v is not None else 180 for v in views]
        raw_height=max(v['raw'].shape[0] for v in views if v is not None)
        y_input=raw_height+55;y_overlay=y_input+190
        canvas=Image.new('RGB',(sum(widths),y_overlay+190),'white');draw=ImageDraw.Draw(canvas)
        sources=[];xx=0
        for k,v in enumerate(views):
            r=records[k//2];pol=('VV','VH')[k%2]
            draw.text((xx+5,3),f'{r["id"][-6:]} C{r["label"]} {pol}',fill='black')
            if v is not None:
                canvas.paste(Image.fromarray(v['raw']).convert('RGB'),(xx+5,22))
                canvas.paste(Image.fromarray(v['pixels']).convert('RGB'),(xx+5,y_input))
                canvas.paste(Image.fromarray(v['pixels']).convert('RGB'),(xx+5,y_overlay))
                if v['geometry'] is not None:
                    for g,color in [(v['geometry'],'#00d38a')]+([(perturb(v['geometry'],cfg),'#ff5757')] if k<2 else []):
                        cx,cy,l,w,angle=g[:5];co,si=np.cos(angle),np.sin(angle)
                        points=[(xx+5+cx+u*co-vv*si,y_overlay+cy+u*si+vv*co) for u,vv in
                                [(-l/2,-w/2),(l/2,-w/2),(l/2,w/2),(-l/2,w/2),(-l/2,-w/2)]]
                        draw.line(points,fill=color,width=1)
                draw.text((xx+5,y_overlay+163),f'valid={v["valid_target"]}',fill='black')
                sources.append(dict(role='query' if k<2 else 'clean_reference',id=r['id'],label=r['label'],
                    polarization=pol,path=r[pol.lower()]['image'],geometry=None if v['geometry'] is None else v['geometry'].tolist(),
                    placement=v['placement'],valid=v['valid_target']))
            xx+=widths[k]
        canvas.save(path)
        return sources

    cases=[];(ROOT/'cases').mkdir(exist_ok=True)
    for mode in modes:
        for family in ('along','across','scale','angle','length','width'):
            candidates=[v for v in modes[mode] if v['family']==family and abs(v['value'])==(20 if family=='angle' else .2)]
            for kind in ('rescued','harmed','stable_correct','stable_wrong'):
                best=None
                for cfg in candidates:
                    z=scores[mode,cfg['name']];ok=z.argmax(-1)==y
                    mask={'rescued':ok&~clean_correct,'harmed':~ok&clean_correct,
                        'stable_correct':ok&clean_correct,'stable_wrong':~ok&~clean_correct}[kind]
                    count=mask.sum(0)
                    if not count.max():continue
                    j=int(np.flatnonzero(count==count.max())[0]);si=int(np.flatnonzero(mask[:,j])[0])
                    candidate=(int(count[j]),cfg,j,si)
                    if best is None or candidate[0]>best[0]:best=candidate
                if best is None:continue
                repeats,cfg,j,si=best;before=clean[si,j];after=scores[mode,cfg['name']][si,j]
                rival=int(before.argmax()) if kind=='rescued' else int(after.argmax())
                if rival==y[j]:rival=int(np.where(np.arange(4)==y[j],-np.inf,after).argmax())
                name=f'{mode}_{family}_{kind}_{rows[j]["id"][-6:]}_s{ex.SEEDS[si]}.png'
                paths=figure(rows[j],rival,cfg,ROOT/'cases'/name)
                case=dict(mode=mode,condition=cfg['name'],kind=kind,id=rows[j]['id'],index=j,label=int(y[j]),
                    seed=ex.SEEDS[si],repeats=repeats,before=before.tolist(),after=after.tolist(),
                    file=f'cases/{name}',sources=paths,mask_audit=audit[(audit.condition==cfg['name'])&(audit['index']==j)].to_dict('records'))
                cases.append(case)
    (ROOT/'cases_local.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
    lines=['# Frozen OSS2 query OBB robustness','',
        'Full fixed weights, seeds 17/29/43, 378 paired queries, 32 unchanged source references. No training.',
        'M2-only changes query local_slots/slots. M3 descriptors/masses/writes and its delta remain clean.',
        ('Additional pipeline changes M2 and M3 geometry while preserving image placement; no redetection/recropping.'
         if args.pipeline else 'This run is M2-only. Pipeline sensitivity is not evaluated.'),
        '', '| Mode | Condition | OA | BA | OA change | BA change | Rescue/harm events |',
        '|---|---|---:|---:|---:|---:|---:|']
    for (mode,name),r in summary.iterrows():
        lines.append(f'| {mode} | {name} | {r.oa:.3f} +/- {r.oa_std:.3f} | {r.ba:.3f} +/- {r.ba_std:.3f} | {r.oa_delta:+.3f} | {r.ba_delta:+.3f} | {int(r.rescued)}/{int(r.harmed)} |')
    lines+=['','Events count record x seed, not independent vessels; no test-selected preferred perturbation.',
        'Area/coverage are relative to the supplied clean OBB, not segmentation ground truth. Zero mask change is not evidence of robustness.',
        'Cases selected by cross-seed recurrence then manifest order, both signed extreme errors inspected. Examples do not estimate prevalence.',
        'Panels: original display PNG / unchanged native160 model input / clean green and perturbed red OBB. No rescaling.',
        'Source examples are first legal same-class/confusion-class members of the actual bank, not largest-contribution references.',
        'Mechanism interpretation remains pending manual raw-chip review; attention/appearance alone cannot prove physical causality.']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    body=['<!doctype html><html><meta charset="utf-8"><title>OSS2 OBB robustness</title>',
        '<style>body{font:16px/1.6 system-ui;max-width:1200px;margin:auto;padding:24px}img{max-width:100%;height:auto}section{border-top:1px solid #bbb;margin-top:24px}td,th{padding:5px}</style>',
        '<h1>OSS2 fixed-weight OBB robustness</h1><p><a href="FINDINGS_ZH.md">中文结论与原图分析</a> | <a href="RESULTS.md">Results and scope</a> | <a href="mask_summary.csv">Actual mask changes</a> | <a href="scores.csv">Paired scores</a></p>',
        '<img src="robustness.png" alt="Paired performance changes">',summary.reset_index().to_html(index=False,float_format=lambda x:f'{x:.3f}'),
        '<p>Original display chip / native160 / OBB overlay. Green: clean; red: perturbed query. References stay clean.</p>']
    for case in cases:
        body+=[f'<section><h2>{html.escape(case["mode"]+" "+case["condition"]+" "+case["kind"]+" "+case["id"])}</h2>',
            f'<p>Seed {case["seed"]}; repeated in {case["repeats"]}/3 seeds. Scores {case["before"]} to {case["after"]}.</p>',
            f'<img loading="lazy" src="{case["file"]}" alt="Native query and legal references"></section>']
    (ROOT/'index.html').write_text('\n'.join(body)+'</html>',encoding='utf-8')
    status=json.loads((ROOT/'status.json').read_text())
    status.update(state='analysis_generated_pending_visual_review',analysis_pending=False,complete=False)
    ex.write('status.json',status)
    print('ANALYSIS_GENERATED',len(cases),'native panels; visual review required',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--resources-released',action='store_true')
    parser.add_argument('--pipeline',action='store_true')
    args=parser.parse_args()
    if not args.resources_released:parser.error('Wait for RAM release; no tensor imports before explicit release.')
    main(args)
