"""Reuse stored predictions for paired statistics and native-chip evidence."""
import argparse
import html
import json
import numpy as np
import pandas as pd
import extend as x
import summarize_runs as plots
from paired_statistics import report as paired_report

NAMES8=['Bulk carrier','Fishing','General cargo','Chemical/product','Container','Crude tanker','Tug','Offshore']


def panel(rows,index,before,after,stem,title,references):
    row=rows[index];label=row['label']
    competing_scores=after if after.argmax()!=label else before
    competing=int(np.where(np.arange(len(before))==label,-np.inf,competing_scores).argmax())
    chosen=[row]
    for cls in (label,competing):
        chosen.append(next((r for r in references if r['label']==cls and str(r['mmsi'])!=str(row['mmsi'])
                            and x.e.task.orbit(r)!=x.e.task.orbit(row)),None))
    fig,axes=plots.plt.subplots(2,3,figsize=(9,6),layout='constrained');sources=[]
    for j,rec in enumerate(chosen):
        for p,view in enumerate(x.e.task.data.prepare(rec) if rec else [None,None]):
            ax=axes[p,j];ax.axis('off')
            if view is not None:
                ax.imshow(view['pixels'],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                if view['valid_target']:
                    cx,cy,length,width,angle=view['geometry'][:5]
                    corners=np.array([[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]])*np.array([length,width])/2
                    rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
                    box=corners@rot.T+np.array([cx,cy]);ax.plot(box[:,0],box[:,1],color='#31b99a',linewidth=.8)
            else:ax.text(.5,.5,'Unavailable',ha='center',va='center')
            role=('Query','Same-class reference','Competing reference')[j]
            status='' if view is not None and view['valid_target'] else ' [not used]'
            ident=rec['id'] if rec else 'none';ident=ident if len(ident)<=35 else '...'+ident[-30:]
            clsname=plots.NAMES[rec['label']] if rec else 'none'
            ax.set_title(f'{role} / {("VV","VH")[p]}{status}\n{clsname} / {ident}',fontsize=7)
        if rec:sources.append(dict(id=rec['id'],label=rec['label'],vv=rec.get('vv'),vh=rec.get('vh')))
    mb,ma=plots.margin(before,label),plots.margin(after,label)
    fig.suptitle(f'{title}\nTrue: {plots.NAMES[label]} | {plots.NAMES[int(before.argmax())]} -> {plots.NAMES[int(after.argmax())]}\nTrue-class margin {mb:+.3f} -> {ma:+.3f}',fontsize=10)
    fig.savefig(stem.with_suffix('.png'),dpi=160);plots.plt.close(fig)
    return dict(id=row['id'],label=label,comparison=title,before=before.tolist(),after=after.tolist(),
                margin_before=mb,margin_after=ma,sources=sources,competing_class=competing,
                reference_selection='first lawful actual source reference; after-error class if incorrect after, otherwise before strongest wrong class; illustration, not attention maximum')


def preview(name):
    dest,report=x.setup(name);rows,groups=x.records(name);p=x.read(dest/'protocol.json')
    plots.NAMES=NAMES8 if p['classes']==8 else ['Bulk carrier','General cargo','Chemical/product','Container']
    records=[]
    fig,axes=plots.plt.subplots(2,p['classes'],figsize=(3*p['classes'],6),layout='constrained')
    for cls in range(p['classes']):
        i=next(i for i in groups['source_fit'] if rows[i]['label']==cls)
        row=rows[i]
        for pol,view in enumerate(x.e.task.data.prepare(row)):
            ax=axes[pol,cls];ax.axis('off')
            if view is not None:ax.imshow(view['pixels'],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            ax.set_title(plots.NAMES[cls]+' / '+('VV' if pol==0 else 'VH'),fontsize=10)
        records.append(dict(id=row['id'],label=cls,vv=row.get('vv'),vh=row.get('vh'),scope='Input inspection; no predictions'))
    fig.savefig(report/'input_preview.png',dpi=140);plots.plt.close(fig)
    x.save(report/'input_cases.json',records)


def analyze(name):
    dest,root=x.setup(name);rows,groups=x.records(name);p=x.read(dest/'protocol.json')
    plots.NAMES=NAMES8 if p['classes']==8 else ['Bulk carrier','General cargo','Chemical/product','Container']
    ids=groups['evaluation'];y=np.array([rows[i]['label'] for i in ids]);refs=[rows[i] for i in groups['references']]
    a,_=x.arrays(name);av=np.array(a['available'][ids]);scores={};changes=[];classrows=[];polrows=[];cost=[];stages=[];strictdev=[];classgaps=[];confusions=[]
    source_orbits={x.e.task.orbit(rows[i]) for i in groups['small']}
    devkeep=np.array([x.e.task.orbit(rows[i]) not in source_orbits for i in groups['validation']])
    for arm in p['arms']:
        scores[arm]=[]
        for seed in p['seeds']:
            folder=dest/f'{arm}_s{seed}_lr{p["scales"][arm]:g}'
            d=np.load(folder/'evaluation.npz');assert np.array_equal(d['ids'],ids)
            vd=np.load(folder/'readout/validation.npz');assert np.array_equal(vd['ids'],groups['validation'])
            dy=np.array([rows[i]['label'] for i in groups['validation']])
            strictdev.append(dict(arm=arm,seed=seed,n=int(devkeep.sum()),**x.e.task.metrics(dy[devkeep],vd['score'][devkeep])))
            scores[arm].append(d['score'])
            if arm=='full':scores.setdefault('m3_off',[]).append(d['m3_off'])
            pred=d['score'].argmax(1)
            for true in range(p['classes']):
                for predicted in range(p['classes']):
                    confusions.append(dict(arm=arm,seed=seed,true=true,predicted=predicted,n=int(((y==true)&(pred==predicted)).sum())))
            for cls in range(p['classes']):
                mask=y==cls
                classrows.append(dict(arm=arm,seed=seed,label=cls,n=int(mask.sum()),recall=100*(d['score'].argmax(1)[mask]==cls).mean() if mask.any() else None))
            for split in ('small','validation'):
                sd=np.load(folder/'readout'/f'{split}.npz');sy=np.array([rows[i]['label'] for i in sd['ids']])
                for cls in np.unique(sy):
                    m=sy==cls
                    classgaps.append(dict(arm=arm,seed=seed,split=split,label=cls,n=int(m.sum()),recall=100*(sd['score'].argmax(1)[m]==cls).mean()))
            for phase in p['phases']:
                r=x.read(folder/phase['name']/'receipt.json')
                cost.append(dict(arm=arm,seed=seed,phase=phase['name'],seconds=r['seconds'],steps=r['steps'],examples=r['examples'],trainable=r['trainable'],selected_count=r['selected_count']))
                for split,metrics in r['metrics'].items():stages.append(dict(arm=arm,seed=seed,phase=phase['name'],split=split,**metrics))
    pd.DataFrame(cost).to_csv(root/'training_budget.csv',index=False)
    pd.DataFrame(strictdev).to_csv(root/'strict_development_metrics.csv',index=False)
    pd.DataFrame(stages).to_csv(root/'stage_metrics.csv',index=False)
    pd.DataFrame(classrows).to_csv(root/'per_class.csv',index=False)
    pd.DataFrame(classgaps).to_csv(root/'source_development_classes.csv',index=False)
    pd.DataFrame(confusions).to_csv(root/'confusion_counts.csv',index=False)
    paired_report(root,[rows[i] for i in ids],scores,'full')
    cases=[];gallery=root/'cases';gallery.mkdir(exist_ok=True)
    for arm in scores:
        if arm=='full':continue
        for j,seed in enumerate(p['seeds']):
            full,base=scores['full'][j],scores[arm][j];good=full.argmax(1)==y;before=base.argmax(1)==y
            delta=good.astype(float)-before
            changes.append(dict(arm=arm,seed=seed,oa_delta=100*delta.mean(),ba_delta=100*np.mean([delta[y==c].mean() for c in np.unique(y)]),rescued=int((good&~before).sum()),harmed=int((before&~good).sum())))
            for vv,vh in ((1,1),(1,0),(0,1)):
                m=(av[:,0]==vv)&(av[:,1]==vh)
                if m.any():polrows.append(dict(arm=arm,seed=seed,vv=vv,vh=vh,n=int(m.sum()),oa_delta=100*delta[m].mean()))
        good=scores['full'][0].argmax(1)==y;before=scores[arm][0].argmax(1)==y
        for kind,mask in dict(rescue=good&~before,harm=before&~good,stable_correct=good&before,stable_wrong=~good&~before).items():
            found=np.flatnonzero(mask)
            if not len(found):continue
            k=int(found[0]);filename=f'{arm}_{kind}_{ids[k]}'
            rec=panel(rows,int(ids[k]),scores[arm][0][k],scores['full'][0][k],gallery/filename,f'{name} / {arm} -> full / {kind} / seed17',refs)
            rec.update(file=f'cases/{filename}.png',arm=arm,outcome=kind,index=int(ids[k]),seed=17);cases.append(rec)
    pd.DataFrame(changes).to_csv(root/'paired_changes.csv',index=False)
    pd.DataFrame(polrows).to_csv(root/'polarization_paired.csv',index=False)
    # If the first seed has no M3 flips, inspect the first later seed with each outcome.
    for kind in ('rescue','harm'):
        if any(c['arm']=='m3_off' and c['outcome']==kind for c in cases):continue
        for j,seed in enumerate(p['seeds'][1:],1):
            before=scores['m3_off'][j].argmax(1)==y;good=scores['full'][j].argmax(1)==y
            found=np.flatnonzero(good&~before if kind=='rescue' else before&~good)
            if len(found):
                k=int(found[0]);filename=f'm3_off_{kind}_s{seed}_{ids[k]}'
                rec=panel(rows,int(ids[k]),scores['m3_off'][j][k],scores['full'][j][k],gallery/filename,f'{name} / M3 off -> on / {kind} / seed{seed}',refs)
                rec.update(file=f'cases/{filename}.png',arm='m3_off',outcome=kind,index=int(ids[k]),seed=seed);cases.append(rec)
                break
    x.save(root/'case_records.json',cases)
    table=pd.read_csv(root/'metrics.csv')
    lines=[f'# {name} 当前组件数据集内验证','',
        '本轮只补当前算子的源端配套训练，不重复历史版本；使用共同ImageNet初始化与既定五阶段训练。正式模型及论文正文尚未替换。',
        f'源拟合{len(groups["source_fit"])}，源适配{len(groups["source_adapt"])}，开发{len(groups["validation"])}，评价{len(ids)}，{p["classes"]}类，固定{len(refs)}源参考。',
        '学习率倍率沿用OSS2开发选择：Full/Plain=1，SE=0.3。没有用本轮测试结果调参、挑种子或挑轮次。','',
        '| 配置 | OA均值±标准差 | BA均值±标准差 |','|---|---:|---:|']
    for arm in p['arms']:
        d=table[table.arm==arm];lines.append(f'| {arm} | {d.oa.mean():.3f} ± {d.oa.std():.3f} | {d.ba.mean():.3f} ± {d.ba.std():.3f} |')
    lines+=['','M3开关是固定同权重诊断，不是重新训练的删除消融。各臂都使用自身教师的干预选样，本轮不能单独归因干预训练增益。',
        'SE为相同两个位置的标准通道注意力对照，不冒称整套SE-ResNet或其他论文复现。',
        '原图案例按seed17和清单顺序选救回、误伤、持续正确、持续错误；参考为实际合法源库示例，不是最大贡献参考。统计按船聚类，三个种子不算三套独立数据。']
    if name=='oss1':lines+=['','OSS1测试只有63艘船、一个过境，结论限于该既有跨场景划分，不称全OSS1或新盲测。']
    if name=='ports24':lines+=['','24港测试3075条与源端及开发集均无同船/绝对轨道重叠。开发160条有63条与源共享绝对轨道、不同datatake，额外97条统计见strict_development_metrics.csv；本轮开发结果均不用于选模。开发缺少渔船，BA按实际出现类别平均；测试包含八类。']
    (root/'FINDINGS_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    body=''.join(f'<h2>{html.escape(c["arm"]+" / "+c["outcome"])}</h2><img src="{c["file"]}">' for c in cases)
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{max-width:1100px;margin:30px auto;font-family:Arial}img{width:100%}</style><h1>'+name+' native input evidence</h1><p>Fixed-seed paired outcomes; references illustrate the actual lawful bank.</p>'+body,encoding='utf-8')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preview','analyze']);p.add_argument('dataset');args=p.parse_args();globals()[args.action](args.dataset)
