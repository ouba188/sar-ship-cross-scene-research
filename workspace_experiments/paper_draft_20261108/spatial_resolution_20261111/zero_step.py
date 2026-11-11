"""Separate the predeclared zero-update structure test from continuation."""
import html
import json
import numpy as np
import pandas as pd
from analyze import ROOT, SEEDS, NAMES, h, e


def main():
    rows, groups = e.prev.records(); ids=groups['evaluation']
    query=[rows[i] for i in ids]; refs=[rows[i] for i in groups['references']]
    y=np.array([r['label'] for r in query]); size=pd.read_csv(ROOT/'test_size.csv')
    scores={arm:[np.load(ROOT/f'{arm}_s{s}/initial_evaluation.npz')['score'] for s in SEEDS] for arm in NAMES}
    pairs=[('dense_current','standard_current'),('dense_mean','standard_mean'),('dense_current','dense_mean')]
    stats, strata, cases=[] ,[], []
    dest=ROOT/'zero_step';dest.mkdir(exist_ok=True)
    class_rows=[]
    for arm,runs in scores.items():
        for seed,z in zip(SEEDS,runs):
            for cls in range(4):
                class_rows.append(dict(arm=arm,seed=seed,label=cls,n=int((y==cls).sum()),recall=100*(z.argmax(1)[y==cls]==cls).mean()))
    pd.DataFrame(class_rows).to_csv(dest/'per_class.csv',index=False)
    for after,before in pairs:
        folder=dest/f'{after}_vs_{before}';folder.mkdir(exist_ok=True)
        h.cluster_report(folder,query,{after:scores[after],before:scores[before]},after)
        ac=np.stack([z.argmax(1)==y for z in scores[after]])
        bc=np.stack([z.argmax(1)==y for z in scores[before]])
        for k,seed in enumerate(SEEDS):
            d=ac[k].astype(float)-bc[k]
            stats.append(dict(after=after,before=before,seed=seed,oa_delta=100*d.mean(),
                ba_delta=100*np.mean([d[y==c].mean() for c in range(4)]),rescued=int((ac[k]&~bc[k]).sum()),harmed=int((~ac[k]&bc[k]).sum())))
            for group in size.group.unique():
                mask=size.group.to_numpy()==group
                strata.append(dict(after=after,before=before,seed=seed,size_group=group,n=int(mask.sum()),
                    oa_delta=100*d[mask].mean(),ba_delta_present=100*np.mean([d[mask&(y==c)].mean() for c in np.unique(y[mask])]),
                    rescued=int((ac[k]&~bc[k]&mask).sum()),harmed=int((~ac[k]&bc[k]&mask).sum())))
        for kind,mask in [('rescue',ac&~bc),('harm',~ac&bc),('stable_correct',ac&bc),('stable_wrong',~ac&~bc)]:
            count=mask.sum(0)
            if not count.max():continue
            j=int(np.flatnonzero(count==count.max())[0]);k=int(np.flatnonzero(mask[:,j])[0])
            az,bz=scores[after][k][j],scores[before][k][j];target=int(y[j])
            rival=int(bz.argmax()) if kind=='rescue' else int(az.argmax())
            if rival==target:rival=int(np.where(np.arange(4)==target,-np.inf,az).argmax())
            name=f'{after}_vs_{before}_{kind}_{query[j]["id"]}_s{SEEDS[k]}.png'
            sources=h.figure(query[j],refs,rival,dest/name)
            margin=lambda z:float(z[target]-np.max(np.delete(z,target)))
            cases.append(dict(after_arm=after,before_arm=before,kind=kind,id=query[j]['id'],label=target,seed=SEEDS[k],repeats=int(count[j]),
                before=bz.tolist(),after=az.tolist(),margin_before=margin(bz),margin_after=margin(az),
                width=float(size.iloc[j].width),length=float(size.iloc[j].length),file=name,sources=sources))
    pd.DataFrame(stats).to_csv(dest/'paired_by_seed.csv',index=False)
    pd.DataFrame(strata).to_csv(dest/'size_strata.csv',index=False)
    summary=pd.DataFrame(stats).groupby(['after','before']).agg(oa_delta=('oa_delta','mean'),ba_delta=('ba_delta','mean'),rescued=('rescued','sum'),harmed=('harmed','sum'),
        positive_oa_seeds=('oa_delta',lambda x:int((x>0).sum())),positive_ba_seeds=('ba_delta',lambda x:int((x>0).sum())))
    summary.to_csv(dest/'paired_summary.csv')
    (dest/'cases_local.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 零步结构对照：权重不更新','','本对照在训练前已保存，未按测试结果回选训练检查点。只改变预先指定的网格/参考比较方式。','',
        '| 前者减后者 | OA差 | BA差 | 救回/误伤事件 | OA/BA正向种子数 |','|---|---:|---:|---:|---:|']
    for (after,before),r in summary.iterrows():
        lines.append(f'| {NAMES[after]} - {NAMES[before]} | {r.oa_delta:+.3f} | {r.ba_delta:+.3f} | {int(r.rescued)}/{int(r.harmed)} | {int(r.positive_oa_seeds)}/3、{int(r.positive_ba_seeds)}/3 |')
    lines+=['','两种子涨、一种子跌仍可保留为开发候选；不能称处处有效或独立确认。本对照不是配套重训比较，不能代替上级目录的完整继续训练结果。',
        '简单均值也受益时，优先归因于共有的网格/读出改变，不能宣称M3特有机制已被证明。']
    (dest/'RESULTS_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>零步网格对照</title>',
        '<style>body{font:16px/1.7 system-ui;max-width:1150px;margin:24px auto;padding:0 16px}img{max-width:100%}section{border-top:1px solid #aaa}</style>',
        '<h1>零步结构对照：权重不更新</h1><p><a href="../FINDINGS_ZH.md">完整分析</a> · <a href="RESULTS_ZH.md">本页统计</a> · <a href="../index.html">继续训练案例</a></p>',
        '<p>原图与合法参考均保持不变，图像不额外拉伸。上排原生像素，下排OBB。参考不是最大贡献参考。</p>']
    for c in cases:
        body += [f'<section><h2>{html.escape(NAMES[c["before_arm"]])} → {html.escape(NAMES[c["after_arm"]])} / {c["kind"]} / {c["id"]}</h2>',
            f'<p>{h.ZH[c["label"]]}，seed{c["seed"]}，{c["repeats"]}/3次；约{c["length"]:.1f}×{c["width"]:.1f}像素；真类分差{c["margin_before"]:+.3f} → {c["margin_after"]:+.3f}。</p>',
            f'<img src="{c["file"]}" loading="lazy"></section>']
    (dest/'index.html').write_text('\n'.join(body)+'</html>',encoding='utf-8')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
