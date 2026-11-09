"""Fixed-choice paired results, class effects and native-image outcome panels."""
import json
import html
import numpy as np
import pandas as pd
import staged as e
from summarize_runs import panel, ZH
from paired_statistics import report
from analyze_current import metric


def main():
    root=e.ROOT;chosen=json.loads((root/'selected.json').read_text())['scales']
    rows,groups=e.prev.records();ids=groups['evaluation'];queries=[rows[i] for i in ids]
    y=np.array([r['label'] for r in queries]);refs=[rows[i] for i in groups['references']]
    scores={};metrics=[];class_rows=[];stage_rows=[];selection=[];costs=[]
    cohort=[]
    for name in ('source_fit','source_adapt','small','validation','evaluation','references'):
        if name not in groups:continue
        selected_rows=[rows[i] for i in groups[name]]
        cohort.append(dict(cohort=name,n=len(selected_rows),
            ships=len({str(r['mmsi']) for r in selected_rows}),
            scenes=len({e.task.orbit(r) for r in selected_rows}),
            **{f'class_{k}':sum(r['label']==k for r in selected_rows) for k in range(4)}))
    pd.DataFrame(cohort).to_csv(root/'cohort_counts.csv',index=False)
    for arm,scale in chosen.items():
        scores[arm]=[]
        for seed in (17,29,43):
            folder=root/f'{arm}_s{seed}_lr{scale:g}'
            d=np.load(folder/'evaluation.npz');assert np.array_equal(d['ids'],ids)
            z=d['score'];scores[arm].append(z)
            metrics.append(dict(arm=arm,seed=seed,scale=scale,**metric(y,z)))
            for cls in range(4):
                mask=y==cls
                class_rows.append(dict(arm=arm,seed=seed,label=cls,class_name=ZH[cls],n=int(mask.sum()),
                    recall=100*(z.argmax(1)[mask]==cls).mean()))
            for phase in ('carrier','intervention','reference','feedback','readout'):
                r=json.loads((folder/phase/'receipt.json').read_text())
                costs.append(dict(arm=arm,seed=seed,phase=phase,trainable=r['trainable'],
                    steps=r['steps'],presentations=r['examples'],loop_and_posteval_seconds=r['seconds']))
                for split,m in r['metrics'].items():stage_rows.append(dict(arm=arm,seed=seed,phase=phase,split=split,**m))
                if phase=='intervention':selection.append(dict(arm=arm,seed=seed,selected=r['selected_count'],n=241))
    df=pd.DataFrame(metrics);df.to_csv(root/'evaluation_metrics.csv',index=False)
    summary=df.groupby('arm')[['oa','ba','macro_f1','nll']].agg(['mean','std']);summary.to_csv(root/'summary.csv')
    pd.DataFrame(class_rows).to_csv(root/'per_class.csv',index=False)
    pd.DataFrame(stage_rows).to_csv(root/'stage_metrics.csv',index=False)
    pd.DataFrame(selection).to_csv(root/'intervention_counts.csv',index=False)
    pd.DataFrame(costs).to_csv(root/'training_budget.csv',index=False)
    report(root,queries,scores,'full')
    cases=[];changes=[];class_changes=[];availability_changes=[]
    a,_=e.prev.arrays();available=np.array(a['available'][ids],dtype=bool)
    gallery=root/'cases';gallery.mkdir(exist_ok=True)
    for arm in chosen:
        if arm=='full':continue
        for j,seed in enumerate((17,29,43)):
            good=scores['full'][j].argmax(1)==y;before=scores[arm][j].argmax(1)==y
            delta=good.astype(float)-before
            changes.append(dict(arm=arm,seed=seed,delta_oa=100*delta.mean(),
                delta_ba=100*np.mean([delta[y==c].mean() for c in range(4)]),
                rescued=int((good&~before).sum()),harmed=int((before&~good).sum())))
            for cls in range(4):
                mask=y==cls
                class_changes.append(dict(arm=arm,seed=seed,label=cls,class_name=ZH[cls],
                    n=int(mask.sum()),delta_recall=100*delta[mask].mean(),
                    rescued=int((mask&good&~before).sum()),harmed=int((mask&before&~good).sum())))
            for vv,vh in ((True,True),(True,False),(False,True),(False,False)):
                mask=(available[:,0]==vv)&(available[:,1]==vh)
                if not mask.any():continue
                availability_changes.append(dict(arm=arm,seed=seed,vv=vv,vh=vh,n=int(mask.sum()),
                    delta_oa=100*delta[mask].mean(),
                    delta_ba_present=100*np.mean([delta[mask&(y==c)].mean() for c in np.unique(y[mask])]),
                    rescued=int((mask&good&~before).sum()),harmed=int((mask&before&~good).sum())))
        good=scores['full'][0].argmax(1)==y;before=scores[arm][0].argmax(1)==y
        masks=dict(rescue=good&~before,harm=before&~good,stable_correct=good&before,stable_wrong=~good&~before)
        for outcome,mask in masks.items():
            hits=np.flatnonzero(mask)
            if not len(hits):continue
            k=int(hits[0]);name=f'{arm}_{outcome}_s17_{ids[k]}'
            item=panel(rows,int(ids[k]),scores[arm][0][k],scores['full'][0][k],gallery/name,
                f'Staged {arm} -> full / {outcome} / seed17',refs)
            item.update(file=f'cases/{name}.png',arm=arm,outcome=outcome);cases.append(item)
    pd.DataFrame(changes).to_csv(root/'paired_changes.csv',index=False)
    pd.DataFrame(class_changes).to_csv(root/'per_class_paired.csv',index=False)
    pd.DataFrame(availability_changes).to_csv(root/'polarization_paired.csv',index=False)
    e.task.write(root/'case_records.json',cases)
    lines=['# 分阶段配套重训结果','',
        '最终存活算子的共同起点训练，不是历史变结构链逐检查点复制；正式模型未替换。',
        '源fit478→自身教师干预适配241→M3参考训练719→M1反馈719→区域引导读出719。',
        '开发228选预定学习率倍率，评价378只在设置冻结后前向；3个独立任务训练种子。','',
        '| 配置 | LR倍率 | OA均值±标准差 | BA均值±标准差 | 宏F1 |','|---|---:|---:|---:|---:|']
    for arm in chosen:
        d=df[df.arm==arm]
        lines.append(f'| {arm} | {chosen[arm]} | {d.oa.mean():.3f} ± {d.oa.std():.3f} | {d.ba.mean():.3f} ± {d.ba.std():.3f} | {d.macro_f1.mean():.3f} |')
    lines+=['','所有删除组继承Full的开发选定倍率；Plain/SE各有相同两种倍率机会。SE为两个匹配block的标准通道注意力，不冒称全网络SE-ResNet复现。',
        '各组均保留自己的受控干预选样和同一适配损失；本表不能独立证明干预训练增量。普通骨干也经历对应阶段的共同参数/分类头训练。',
        '配对区间见paired_intervals.csv，条件化于现有船群与三个固定训练；开发和评价队列历史已使用，不称新盲测。',
        '原图按固定seed17清单首个救回、误伤、持续正确/错误取例；参考是实际合法源库的示例，不是贡献最大者。']
    (root/'FINDINGS_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    body=''.join(f'<h2>{html.escape(c["arm"]+" / "+c["outcome"])}</h2><img src="{c["file"]}" style="max-width:100%">' for c in cases)
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Staged ablation cases</title><h1>Staged matched outcomes</h1>'+body,encoding='utf-8')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
