"""Paired resolution effects, native size strata, and actual chip cases."""
from pathlib import Path
import html
import importlib.util
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('resolution_previous_analysis', ROOT.parent/'internal_mechanisms_20261111/analyze.py')
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
e = h.e
SEEDS = (17, 29, 43)
NAMES = {'standard_current':'标准网格＋当前M3', 'dense_current':'密集网格＋当前M3',
         'standard_mean':'标准网格＋参考均值', 'dense_mean':'密集网格＋参考均值'}
PAIRS = [('dense_current','standard_current'), ('dense_mean','standard_mean'),
         ('dense_current','dense_mean'), ('standard_current','standard_mean'),
         ('dense_current','original'), ('standard_current','original')]


def main():
    rows, groups = e.prev.records(); ids = groups['evaluation']
    query = [rows[i] for i in ids]; refs = [rows[i] for i in groups['references']]
    y = np.array([r['label'] for r in query])
    scores, entries, budgets, class_rows = {}, [], [], []
    for arm in NAMES:
        scores[arm] = []
        for seed in SEEDS:
            folder = ROOT/f'{arm}_s{seed}'
            receipt = json.loads((folder/'complete.json').read_text(encoding='utf-8'))
            budgets.append(dict(arm=arm,seed=seed,**{k:receipt[k] for k in ('steps','examples','seconds','peak_gib','trainable')}))
            for split in ('validation','evaluation'):
                for phase in ('initial','final'):
                    d = np.load(folder/f'{"initial_" if phase=="initial" else ""}{split}.npz')
                    assert np.array_equal(d['ids'],groups[split])
                    sy = np.array([rows[i]['label'] for i in groups[split]])
                    z = d['score']; entries.append(dict(arm=arm,seed=seed,split=split,phase=phase,**h.metrics(sy,z)))
                    if split=='evaluation' and phase=='final':
                        scores[arm].append(z)
                        for cls in range(4):
                            class_rows.append(dict(arm=arm,seed=seed,label=cls,n=int((y==cls).sum()),recall=100*(z.argmax(1)[y==cls]==cls).mean()))
    df = pd.DataFrame(entries); df.to_csv(ROOT/'metrics.csv',index=False)
    df.groupby(['phase','split','arm'])[['oa','ba','nll']].agg(['mean','std']).to_csv(ROOT/'summary.csv')
    pd.DataFrame(budgets).to_csv(ROOT/'training_budget.csv',index=False)
    pd.DataFrame(class_rows).to_csv(ROOT/'per_class.csv',index=False)
    for seed in SEEDS:
        orders = [json.loads((ROOT/f'{arm}_s{seed}/orders.json').read_text()) for arm in NAMES]
        assert all(x==orders[0] for x in orders[1:])
    scores['original'] = [np.load(ROOT/f'standard_current_s{seed}/initial_evaluation.npz')['score'] for seed in SEEDS]
    NAMES['original'] = '原完整模型（未追加训练）'
    size_rows = []
    for j,row in enumerate(query):
        widths, lengths = [], []
        for p,view in enumerate(e.task.data.prepare(row)):
            if view is not None and view['valid_target']:
                lengths.append(float(view['geometry'][2])); widths.append(float(view['geometry'][3]))
        width, length = float(np.mean(widths)), float(np.mean(lengths))
        size_rows.append(dict(id=row['id'],mmsi=row['mmsi'],label=row['label'],width=width,length=length,
            group='width_le8' if width<=8 else 'width_8to10' if width<=10 else 'width_gt10'))
    size = pd.DataFrame(size_rows); size.to_csv(ROOT/'test_size.csv',index=False)
    pairs, strata, cases = [], [], []
    (ROOT/'cases').mkdir(exist_ok=True)
    for after,before in PAIRS:
        astack = np.stack([z.argmax(1)==y for z in scores[after]])
        bstack = np.stack([z.argmax(1)==y for z in scores[before]])
        for k,seed in enumerate(SEEDS):
            delta = astack[k].astype(float)-bstack[k]
            pairs.append(dict(after=after,before=before,seed=seed,oa_delta=100*delta.mean(),
                ba_delta=100*np.mean([delta[y==cls].mean() for cls in range(4)]),
                rescued=int((astack[k]&~bstack[k]).sum()),harmed=int((~astack[k]&bstack[k]).sum())))
            for group in size.group.unique():
                mask = size.group.to_numpy()==group
                strata.append(dict(after=after,before=before,seed=seed,size_group=group,n=int(mask.sum()),
                    oa_delta=100*delta[mask].mean(),ba_delta_present=100*np.mean([delta[mask&(y==c)].mean() for c in np.unique(y[mask])]),
                    rescued=int((astack[k]&~bstack[k]&mask).sum()),harmed=int((~astack[k]&bstack[k]&mask).sum())))
        folder = ROOT/f'{after}_vs_{before}'; folder.mkdir(exist_ok=True)
        h.cluster_report(folder,query,{after:scores[after],before:scores[before]},after)
        for kind,mask in [('rescue',astack&~bstack),('harm',~astack&bstack),('stable_correct',astack&bstack),('stable_wrong',~astack&~bstack)]:
            count = mask.sum(0)
            if not count.max(): continue
            j = int(np.flatnonzero(count==count.max())[0]); k = int(np.flatnonzero(mask[:,j])[0])
            az,bz = scores[after][k][j],scores[before][k][j]; target=int(y[j])
            rival = int(bz.argmax()) if kind=='rescue' else int(az.argmax())
            if rival==target: rival=int(np.where(np.arange(4)==target,-np.inf,az).argmax())
            name=f'{after}_vs_{before}_{kind}_{query[j]["id"]}_s{SEEDS[k]}.png'
            sources=h.figure(query[j],refs,rival,ROOT/'cases'/name)
            margin=lambda z:float(z[target]-np.max(np.delete(z,target)))
            cases.append(dict(after_arm=after,before_arm=before,kind=kind,id=query[j]['id'],label=target,seed=SEEDS[k],
                repeats=int(count[j]),before=bz.tolist(),after=az.tolist(),margin_before=margin(bz),margin_after=margin(az),
                width=float(size.iloc[j].width),length=float(size.iloc[j].length),file=f'cases/{name}',sources=sources))
    paired=pd.DataFrame(pairs); paired.to_csv(ROOT/'paired_by_seed.csv',index=False)
    ps=paired.groupby(['after','before']).agg(oa_delta=('oa_delta','mean'),ba_delta=('ba_delta','mean'),
        rescued=('rescued','sum'),harmed=('harmed','sum'),positive_oa_seeds=('oa_delta',lambda x:int((x>0).sum())),positive_ba_seeds=('ba_delta',lambda x:int((x>0).sum())))
    ps.to_csv(ROOT/'paired_summary.csv'); pd.DataFrame(strata).to_csv(ROOT/'size_strata.csv',index=False)
    inter=[]
    for seed in SEEDS:
        d=paired[paired.seed==seed].set_index(['after','before'])
        a=d.loc[('dense_current','standard_current')]; b=d.loc[('dense_mean','standard_mean')]
        inter.append(dict(seed=seed,oa_interaction=a.oa_delta-b.oa_delta,ba_interaction=a.ba_delta-b.ba_delta))
    pd.DataFrame(inter).to_csv(ROOT/'interaction.csv',index=False)
    (ROOT/'cases_local.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 原生小目标空间分辨率结果','','三种子配对继续训练；同一160输入、相同参考和更新预算。','','| 配置 | 测试OA | 测试BA | 验证OA | 验证BA |','|---|---:|---:|---:|---:|']
    t0=df[(df.arm=='standard_current')&(df.phase=='initial')&(df.split=='evaluation')]
    v0=df[(df.arm=='standard_current')&(df.phase=='initial')&(df.split=='validation')]
    lines.append(f'| 原完整模型，未追加训练 | {t0.oa.mean():.3f} ± {t0.oa.std():.3f} | {t0.ba.mean():.3f} ± {t0.ba.std():.3f} | {v0.oa.mean():.3f} | {v0.ba.mean():.3f} |')
    for arm,name in NAMES.items():
        if arm=='original': continue
        t=df[(df.arm==arm)&(df.phase=='final')&(df.split=='evaluation')]
        v=df[(df.arm==arm)&(df.phase=='final')&(df.split=='validation')]
        lines.append(f'| {name} | {t.oa.mean():.3f} ± {t.oa.std():.3f} | {t.ba.mean():.3f} ± {t.ba.std():.3f} | {v.oa.mean():.3f} | {v.ba.mean():.3f} |')
    lines+=['','| 前者减后者 | OA差 | BA差 | 救回/误伤事件 | OA/BA正向种子数 |','|---|---:|---:|---:|---:|']
    for (after,before),r in ps.iterrows():
        lines.append(f'| {NAMES[after]} - {NAMES[before]} | {r.oa_delta:+.3f} | {r.ba_delta:+.3f} | {int(r.rescued)}/{int(r.harmed)} | {int(r.positive_oa_seeds)}/3、{int(r.positive_ba_seeds)}/3 |')
    lines+=['','差为百分点；翻转按记录×种子计，不是独立船数。全体从标准网格训练过的完整模型继续，不能代表从头训练最优配置。',
        '简单参考均值替换的是M3比较，不删除M2；当前M3自身已是类别条件化聚合，不能将本轮命名为精细输运优于均值实验。',
        '干预训练继承于起点，不重做；M1及比较描述保持不变，检查的是后段空间密度。时间为共享GPU环境中的训练时长，不作为部署延时。']
    (ROOT/'RESULTS_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>原生小目标空间分辨率</title>',
        '<style>body{font:16px/1.7 system-ui;max-width:1150px;margin:24px auto;padding:0 16px}img{max-width:100%}section{border-top:1px solid #aaa}</style>',
        '<h1>原生小目标空间分辨率</h1><p>同图像、同训练预算；不缩放船体。案例上排原生像素，下排OBB。参考为实际合法同类/混淆类案例，不是最大贡献参考。</p>',
        '<p><a href="RESULTS_ZH.md">统计结果</a> · <a href="FINDINGS_ZH.md">分析</a> · <a href="PROTOCOL_ZH.md">协议</a></p>']
    for c in cases:
        body += [f'<section><h2>{html.escape(NAMES[c["before_arm"]])} → {html.escape(NAMES[c["after_arm"]])} / {c["kind"]} / {c["id"]}</h2>',
            f'<p>{h.ZH[c["label"]]}，seed{c["seed"]}，{c["repeats"]}/3次；船体约{c["length"]:.1f}×{c["width"]:.1f}像素；分差{c["margin_before"]:+.3f} → {c["margin_after"]:+.3f}。</p>',f'<img src="{c["file"]}" loading="lazy"></section>']
    (ROOT/'index.html').write_text('\n'.join(body)+'</html>',encoding='utf-8')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
