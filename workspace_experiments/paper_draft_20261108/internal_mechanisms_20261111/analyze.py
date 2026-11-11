"""Summarize matched internal ablations and render native query/reference cases."""
from pathlib import Path
import json
import html
import importlib.util
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.metrics import f1_score
from PIL import Image, ImageDraw
spec=importlib.util.spec_from_file_location('internal_mechanism_experiment',Path(__file__).with_name('run.py'))
experiment=importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
from paired_statistics import report as cluster_report

ROOT, SOURCE, e = experiment.ROOT, experiment.SOURCE, experiment.e
NAMES = {
    'full':'完整模型（复用）', 'm1_mean_only':'M1仅均值反馈',
    'm1_relation_only':'M1仅类别关系反馈', 'm1_no_feedback':'M1无反馈（保留reader）',
    'm3_metric_only':'M3仅通道度量', 'm3_bias_only':'M3仅竞争偏置',
    'm3_neither':'M3单位度量且无偏置',
}
ZH = ['散货船','普通货船','化学品/成品油轮','集装箱船']
SEEDS = (17,29,43)


def metrics(y, score):
    pred=score.argmax(1)
    return dict(oa=100*np.mean(pred==y),ba=100*np.mean([np.mean(pred[y==c]==c) for c in range(4)]),
        macro_f1=100*f1_score(y,pred,labels=range(4),average='macro',zero_division=0),
        nll=np.mean(logsumexp(score.astype(float),axis=1)-score[np.arange(len(y)),y]))


def figure(row, refs, rival, path):
    records=[row]
    for label in (row['label'],rival):
        records.append(next(r for r in refs if r['label']==label and str(r['mmsi'])!=str(row['mmsi']) and e.task.orbit(r)!=e.task.orbit(row)))
    canvas=Image.new('RGB',(6*180,382),'white')
    draw=ImageDraw.Draw(canvas)
    sources=[]
    for k,record in enumerate(records):
        for p,view in enumerate(e.task.data.prepare(record)):
            x=(k*2+p)*180+8
            draw.text((x,4),f"{record['id']} C{record['label']} {('VV','VH')[p]}",fill='black')
            if view is None:
                draw.text((x,65),'Unavailable',fill='red')
                continue
            # Native input first; its unaltered PNG is available through the case record.
            canvas.paste(Image.fromarray(view['pixels']).convert('RGB'),(x,25))
            rgb=np.repeat(view['pixels'][...,None],3,axis=-1)
            mask=view['masks'][0].astype(bool)
            edge=mask & ~(np.roll(mask,1,0)&np.roll(mask,-1,0)&np.roll(mask,1,1)&np.roll(mask,-1,1))
            rgb[edge]=(0,200,125)
            draw.text((x,190),f"valid={view['valid_target']}",fill='black')
            canvas.paste(Image.fromarray(rgb),(x,211))
            sources.append(dict(role='query' if k==0 else 'reference',id=record['id'],label=record['label'],
                polarization=('VV','VH')[p],path=record[('vv','vh')[p]]['image'],valid=view['valid_target'],placement=view['placement']))
    canvas.save(path)
    return sources


def main():
    rows,groups=e.prev.records()
    ids=groups['evaluation']
    query=[rows[i] for i in ids]
    y=np.array([r['label'] for r in query])
    refs=[rows[i] for i in groups['references']]
    scores={};entries=[];class_rows=[];stages=[]
    for arm in NAMES:
        scores[arm]=[]
        for seed in SEEDS:
            folder=SOURCE/f'full_s{seed}_lr1' if arm=='full' else ROOT/f'{arm}_s{seed}'
            if arm!='full' and not (folder/'complete.json').exists():
                raise RuntimeError(f'Incomplete predefined arm: {folder}')
            for split in ('validation','evaluation'):
                d=np.load(folder/f'{split}.npz')
                assert np.array_equal(d['ids'],groups[split])
                z=d['score']
                sy=np.array([rows[i]['label'] for i in groups[split]])
                entries.append(dict(arm=arm,seed=seed,split=split,**metrics(sy,z)))
                if split=='evaluation':
                    scores[arm].append(z)
                    for c in range(4):
                        mask=y==c
                        class_rows.append(dict(arm=arm,seed=seed,label=c,name=ZH[c],n=int(mask.sum()),recall=100*(z.argmax(1)[mask]==c).mean()))
            for phase in ('reference','feedback','readout'):
                receipt=folder/phase/'receipt.json'
                if receipt.exists():
                    r=json.loads(receipt.read_text(encoding='utf-8'))
                    stages.append(dict(arm=arm,seed=seed,phase=phase,steps=r['steps'],examples=r['examples'],seconds=r['seconds']))
    df=pd.DataFrame(entries)
    df.to_csv(ROOT/'metrics.csv',index=False)
    summary=df.groupby(['split','arm'])[['oa','ba','macro_f1','nll']].agg(['mean','std'])
    summary.to_csv(ROOT/'summary.csv')
    pd.DataFrame(class_rows).to_csv(ROOT/'per_class.csv',index=False)
    pd.DataFrame(stages).to_csv(ROOT/'training_budget.csv',index=False)
    cluster_report(ROOT,query,scores,'full')
    paired=[];cases=[];polarized=[]
    available=np.asarray(e.prev.arrays()[0]['available'][ids],dtype=bool)
    dest=ROOT/'cases';dest.mkdir(exist_ok=True)
    for arm in list(NAMES)[1:]:
        for s,seed in enumerate(SEEDS):
            full=scores['full'][s].argmax(1)==y
            control=scores[arm][s].argmax(1)==y
            delta=full.astype(float)-control
            paired.append(dict(control=arm,seed=seed,oa_delta=100*delta.mean(),ba_delta=100*np.mean([delta[y==c].mean() for c in range(4)]),
                rescued=int((full&~control).sum()),harmed=int((~full&control).sum())))
            for vv,vh in ((True,True),(True,False),(False,True)):
                mask=(available[:,0]==vv)&(available[:,1]==vh)
                if mask.any():
                    polarized.append(dict(control=arm,seed=seed,vv=vv,vh=vh,n=int(mask.sum()),
                        oa_delta=100*delta[mask].mean(),ba_delta_present=100*np.mean([delta[mask&(y==c)].mean() for c in np.unique(y[mask])])))
        fstack=np.stack([z.argmax(1)==y for z in scores['full']])
        cstack=np.stack([z.argmax(1)==y for z in scores[arm]])
        masks={'rescue':fstack&~cstack,'harm':~fstack&cstack,'stable_correct':fstack&cstack,'stable_wrong':~fstack&~cstack}
        for kind,mask in masks.items():
            count=mask.sum(0)
            if not count.max():continue
            j=int(np.flatnonzero(count==count.max())[0])
            si=int(np.flatnonzero(mask[:,j])[0]);seed=SEEDS[si]
            before,after=scores[arm][si][j],scores['full'][si][j]
            target=int(y[j])
            rival=int(before.argmax()) if kind=='rescue' else int(after.argmax())
            if rival==target:rival=int(np.where(np.arange(4)==target,-np.inf,after).argmax())
            name=f'{arm}_{kind}_{query[j]["id"]}_s{seed}.png'
            sources=figure(query[j],refs,rival,dest/name)
            margin=lambda z:float(z[target]-np.max(np.delete(z,target)))
            cases.append(dict(control=arm,kind=kind,id=query[j]['id'],label=target,seed=seed,repeats=int(count[j]),
                orbit=e.task.orbit(query[j]),before=before.tolist(),after=after.tolist(),margin_before=margin(before),margin_after=margin(after),
                file=f'cases/{name}',sources=sources))
    paired=pd.DataFrame(paired)
    paired.to_csv(ROOT/'paired_by_seed.csv',index=False)
    pd.DataFrame(polarized).to_csv(ROOT/'polarization_paired.csv',index=False)
    pair_summary=paired.groupby('control').agg(oa_delta=('oa_delta','mean'),ba_delta=('ba_delta','mean'),rescued=('rescued','sum'),harmed=('harmed','sum'),
        positive_oa_seeds=('oa_delta',lambda x:int((x>0).sum())),positive_ba_seeds=('ba_delta',lambda x:int((x>0).sum())))
    pair_summary.to_csv(ROOT/'paired_summary.csv')
    # Accuracy interaction is descriptive under the fixed staged protocol.
    interactions=[]
    test=df[df.split=='evaluation'].set_index(['arm','seed'])
    for family,left,right,none in [('m1','m1_mean_only','m1_relation_only','m1_no_feedback'),('m3','m3_metric_only','m3_bias_only','m3_neither')]:
        for seed in SEEDS:
            result={k:float(test.loc['full',seed][k]-test.loc[left,seed][k]-test.loc[right,seed][k]+test.loc[none,seed][k]) for k in ['oa','ba','macro_f1','nll']}
            interactions.append(dict(family=family,seed=seed,**result))
    pd.DataFrame(interactions).to_csv(ROOT/'factorial_interaction.csv',index=False)
    contrasts=[]
    for family,first,second,none in [('m1','m1_mean_only','m1_relation_only','m1_no_feedback'),('m3','m3_metric_only','m3_bias_only','m3_neither')]:
        for after,before in [(first,none),(second,none),('full',first),('full',second)]:
            for si,seed in enumerate(SEEDS):
                ca=scores[after][si].argmax(1)==y
                cb=scores[before][si].argmax(1)==y
                delta=ca.astype(float)-cb
                contrasts.append(dict(family=family,after=after,before=before,seed=seed,oa_delta=100*delta.mean(),
                    ba_delta=100*np.mean([delta[y==k].mean() for k in range(4)]),rescued=int((ca&~cb).sum()),harmed=int((~ca&cb).sum())))
    contrasts=pd.DataFrame(contrasts)
    contrasts.to_csv(ROOT/'conditional_effects_by_seed.csv',index=False)
    conditional=contrasts.groupby(['family','after','before']).agg(oa_delta=('oa_delta','mean'),ba_delta=('ba_delta','mean'),
        rescued=('rescued','sum'),harmed=('harmed','sum'),positive_ba_seeds=('ba_delta',lambda x:int((x>0).sum())))
    conditional.to_csv(ROOT/'conditional_effects_summary.csv')
    # The factorial design also evaluates channel weighting without the bias term.
    astack=np.stack([z.argmax(1)==y for z in scores['m3_metric_only']])
    bstack=np.stack([z.argmax(1)==y for z in scores['m3_neither']])
    for kind,mask in [('rescue',astack&~bstack),('harm',~astack&bstack)]:
        count=mask.sum(0)
        if not count.max():continue
        j=int(np.flatnonzero(count==count.max())[0]);si=int(np.flatnonzero(mask[:,j])[0]);seed=SEEDS[si]
        before,after=scores['m3_neither'][si][j],scores['m3_metric_only'][si][j]
        target=int(y[j]);rival=int(before.argmax()) if kind=='rescue' else int(after.argmax())
        name=f'm3_channel_without_bias_{kind}_{query[j]["id"]}_s{seed}.png'
        sources=figure(query[j],refs,rival,dest/name)
        margin=lambda z:float(z[target]-np.max(np.delete(z,target)))
        cases.append(dict(control='m3_neither',after_name=NAMES['m3_metric_only'],kind=kind,id=query[j]['id'],label=target,seed=seed,repeats=int(count[j]),
            orbit=e.task.orbit(query[j]),before=before.tolist(),after=after.tolist(),margin_before=margin(before),margin_after=margin(after),
            file=f'cases/{name}',sources=sources))
    (ROOT/'cases_local.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
    unique={c['id']:c for c in cases}
    for start in range(0,len(unique),6):
        sheet=Image.new('RGB',(1080,812),'white')
        draw=ImageDraw.Draw(sheet)
        for j,c in enumerate(list(unique.values())[start:start+6]):
            x,yy=j%3*360,j//3*406
            draw.text((x+5,yy+2),f"Query {c['id']} C{c['label']}",fill='black')
            with Image.open(ROOT/c['file']) as im:
                sheet.paste(im.crop((0,0,360,382)),(x,yy+24))
        sheet.save(dest/f'queries_{start//6+1}.png')
    lines=['# 当前内部机制配套消融结果','',
        'OSS2四分类，训练719、验证228、测试378、固定32源参考，三个训练种子。Full复用；新增18个后缀训练分支。正式模型不变。',
        '各变体从相应机制启用前的对应种子检查点开始，重新训练受影响的后续阶段。M1保留reader与M2引导，测两条反馈；M3无度量指单位权重，不等于历史共享学习度量。','',
        '| 配置 | OA均值±标准差 | BA均值±标准差 | Macro-F1 | NLL |','|---|---:|---:|---:|---:|']
    for arm in NAMES:
        block=df[(df.arm==arm)&(df.split=='evaluation')]
        lines.append(f'| {NAMES[arm]} | {block.oa.mean():.3f} ± {block.oa.std():.3f} | {block.ba.mean():.3f} ± {block.ba.std():.3f} | {block.macro_f1.mean():.3f} | {block.nll.mean():.4f} |')
    lines+=['','## 完整模型相对替换的增量','',
        '| 对照 | OA差 | BA差 | 救回/误伤事件 | OA为正种子数 | BA为正种子数 |','|---|---:|---:|---:|---:|---:|']
    for arm,r in pair_summary.iterrows():
        lines.append(f'| {NAMES[arm]} | {r.oa_delta:+.3f} | {r.ba_delta:+.3f} | {int(r.rescued)}/{int(r.harmed)} | {int(r.positive_oa_seeds)}/3 | {int(r.positive_ba_seeds)}/3 |')
    lines+=['','## 四格中的条件效应','','| 变化 | OA差 | BA差 | BA为正种子 |','|---|---:|---:|---:|']
    for (family,after,before),r in conditional.iterrows():
        lines.append(f'| {NAMES[before]} → {NAMES[after]} | {r.oa_delta:+.3f} | {r.ba_delta:+.3f} | {int(r.positive_ba_seeds)}/3 |')
    lines+=['','统计采用固定三种子同查询配对；事件按记录×种子计，不当独立船数。按船且按类别分层bootstrap区间见paired_intervals.csv，条件化于当前拟合和观测场景，未校正多重比较。',
        '不同候选的源难例权重按同规则重算，而非强制数值一致。变体测的是现有训练日程下的实际机制替换，不是固定所有中间状态的推理干预。',
        '原图案例按跨种子重复次数、随后清单顺序选择；参考为实际固定库的合法样例，不等于贡献最高者。所有图保留原生160裁剪补零像素，上排无叠加，下排目标框边界；未重新拉伸。',
        '本轮没有训练24港或OSS1；不能把OSS2内部贡献数值写成所有数据集均有效。历史负结果保留。']
    (ROOT/'RESULTS_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M1与M3内部消融</title>',
        '<style>body{font:16px/1.7 system-ui;max-width:1150px;margin:24px auto;padding:0 16px;color:#222}img{max-width:100%;height:auto}section{border-top:1px solid #ccc;margin-top:26px}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ccc}</style>',
        '<h1>M1与M3内部机制配套消融</h1><p><a href="RESULTS_ZH.md">结果表</a> · <a href="FINDINGS_ZH.md">结论与原图分析</a> · <a href="PROTOCOL_ZH.md">冻结协议</a></p>',
        '<p>原始尺度输入：左为查询VV/VH，中为同类源参考，右为混淆类源参考；上排无框，下排标注目标框。参考不是贡献最大者。本轮不更换正式模型。</p>']
    for c in cases:
        body+=[f'<section><h2>{html.escape(NAMES[c["control"]])} → {html.escape(c.get("after_name","完整模型"))} / {c["kind"]} / {c["id"]}</h2>',
            f'<p>真实类别：{ZH[c["label"]]}；种子{c["seed"]}，3种子中{c["repeats"]}次；预测{ZH[int(np.argmax(c["before"]))]} → {ZH[int(np.argmax(c["after"]))]}；真实类分差{c["margin_before"]:+.3f} → {c["margin_after"]:+.3f}。</p>',
            f'<img loading="lazy" src="{c["file"]}"></section>']
    (ROOT/'index.html').write_text('\n'.join(body)+'</html>',encoding='utf-8')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
