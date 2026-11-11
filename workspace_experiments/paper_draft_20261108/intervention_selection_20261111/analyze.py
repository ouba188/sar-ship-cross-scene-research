"""Complete-pipeline selection ablations, activation diagnostics, and native cases."""
from pathlib import Path
import html
import importlib.util
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('selection_previous_analysis',ROOT.parent/'internal_mechanisms_20261111/analyze.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
e,SOURCE = h.e,h.SOURCE
SEEDS = (17,29,43)
NAMES = {'full':'原干预筛选', 'random_selection':'同类别配额随机筛选', 'no_protection':'无分差保护'}
COEFFICIENT = .3683066666126251


def main():
    rows,groups = e.prev.records()
    ids = groups['evaluation']
    query = [rows[i] for i in ids]
    refs = [rows[i] for i in groups['references']]
    y = np.array([r['label'] for r in query])
    scores,entries,classes,stages,budgets,losses,selection = {},[],[],[],[],[],[]
    for arm in NAMES:
        scores[arm] = []
        for seed in SEEDS:
            folder = SOURCE/f'full_s{seed}_lr1' if arm=='full' else ROOT/f'{arm}_s{seed}'
            if arm!='full': assert (folder/'complete.json').exists()
            for split in ('validation','evaluation'):
                data = np.load(folder/f'{split}.npz')
                assert np.array_equal(data['ids'],groups[split])
                z = data['score'];sy = np.array([rows[i]['label'] for i in groups[split]])
                entries.append(dict(arm=arm,seed=seed,split=split,**h.metrics(sy,z)))
                if split=='evaluation':
                    scores[arm].append(z)
                    for cls in range(4):
                        mask = y==cls
                        classes.append(dict(arm=arm,seed=seed,label=cls,name=h.ZH[cls],n=int(mask.sum()),recall=100*(z.argmax(1)[mask]==cls).mean()))
            for phase in ('intervention','reference','feedback','readout'):
                receipt = json.loads((folder/phase/'receipt.json').read_text(encoding='utf-8'))
                budgets.append(dict(arm=arm,seed=seed,phase=phase,steps=receipt['steps'],examples=receipt['examples'],seconds=receipt['seconds']))
                data = np.load(folder/phase/'validation.npz')
                assert np.array_equal(data['ids'],groups['validation'])
                vy = np.array([rows[i]['label'] for i in groups['validation']])
                stages.append(dict(arm=arm,seed=seed,phase=phase,**h.metrics(vy,data['score'])))
            curve = pd.read_csv(folder/'intervention/curve.csv')
            coefficient = 0. if arm=='no_protection' else COEFFICIENT
            active = coefficient*curve['floor']
            losses.append(dict(arm=arm,seed=seed,steps=len(curve),mean_ce=curve.ce.mean(),mean_keep=curve.keep.mean(),
                diagnostic_floor_mean=curve['floor'].mean(),effective_floor_mean=active.mean(),
                effective_nonzero_steps=int((active>1e-10).sum()),floor_to_ce_ratio=active.mean()/curve.ce.mean(),
                floor_first=active.iloc[0],floor_last=active.iloc[-1]))
            records = json.loads((folder/'intervention/teacher.json').read_text(encoding='utf-8'))
            assert [r['index'] for r in records]==groups['source_adapt'].tolist()
            for cls in range(4):
                for correct in (False,True):
                    subset = [r for r in records if r['label']==cls and (int(np.argmax(r['original']))==cls)==correct]
                    selection.append(dict(arm=arm,seed=seed,label=cls,teacher_correct=correct,n=len(subset),
                        marked=sum(r['useful'] for r in subset),protected=0 if arm=='no_protection' else sum(r['useful'] for r in subset)))
    df = pd.DataFrame(entries)
    df.to_csv(ROOT/'metrics.csv',index=False)
    df.groupby(['split','arm'])[['oa','ba','macro_f1','nll']].agg(['mean','std']).to_csv(ROOT/'summary.csv')
    pd.DataFrame(classes).to_csv(ROOT/'per_class.csv',index=False)
    pd.DataFrame(stages).to_csv(ROOT/'validation_stages.csv',index=False)
    pd.DataFrame(budgets).to_csv(ROOT/'training_budget.csv',index=False)
    pd.DataFrame(losses).to_csv(ROOT/'loss_activation.csv',index=False)
    pd.DataFrame(selection).to_csv(ROOT/'selection_counts.csv',index=False)
    h.cluster_report(ROOT,query,scores,'full')
    extra = ROOT/'random_vs_none';extra.mkdir(exist_ok=True)
    h.cluster_report(extra,query,{a:scores[a] for a in ('random_selection','no_protection')},'random_selection')
    pairs,cases = [],[]
    (ROOT/'cases').mkdir(exist_ok=True)
    for after,before in [('full','random_selection'),('full','no_protection'),('random_selection','no_protection')]:
        astack = np.stack([z.argmax(1)==y for z in scores[after]])
        bstack = np.stack([z.argmax(1)==y for z in scores[before]])
        for k,seed in enumerate(SEEDS):
            delta = astack[k].astype(float)-bstack[k]
            pairs.append(dict(after=after,before=before,seed=seed,oa_delta=100*delta.mean(),
                ba_delta=100*np.mean([delta[y==cls].mean() for cls in range(4)]),
                rescued=int((astack[k]&~bstack[k]).sum()),harmed=int((~astack[k]&bstack[k]).sum())))
        for kind,mask in [('rescue',astack&~bstack),('harm',~astack&bstack),('stable_correct',astack&bstack),('stable_wrong',~astack&~bstack)]:
            count = mask.sum(0)
            if not count.max():continue
            j = int(np.flatnonzero(count==count.max())[0]);k = int(np.flatnonzero(mask[:,j])[0])
            az,bz = scores[after][k][j],scores[before][k][j]
            target = int(y[j]);rival = int(bz.argmax()) if kind=='rescue' else int(az.argmax())
            if rival==target:rival=int(np.where(np.arange(4)==target,-np.inf,az).argmax())
            name = f'{after}_vs_{before}_{kind}_{query[j]["id"]}_s{SEEDS[k]}.png'
            sources = h.figure(query[j],refs,rival,ROOT/'cases'/name)
            margin = lambda z:float(z[target]-np.max(np.delete(z,target)))
            cases.append(dict(after_arm=after,before_arm=before,kind=kind,id=query[j]['id'],label=target,
                seed=SEEDS[k],repeats=int(count[j]),before=bz.tolist(),after=az.tolist(),
                margin_before=margin(bz),margin_after=margin(az),file=f'cases/{name}',sources=sources))
    paired = pd.DataFrame(pairs)
    paired.to_csv(ROOT/'paired_by_seed.csv',index=False)
    ps = paired.groupby(['after','before']).agg(oa_delta=('oa_delta','mean'),ba_delta=('ba_delta','mean'),
        rescued=('rescued','sum'),harmed=('harmed','sum'),positive_oa_seeds=('oa_delta',lambda x:int((x>0).sum())),positive_ba_seeds=('ba_delta',lambda x:int((x>0).sum())))
    ps.to_csv(ROOT/'paired_summary.csv')
    (ROOT/'cases_local.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
    lines = ['# 干预筛选完整流程对照结果','','OSS2四分类，三种子；复用Full，新增六次四阶段后缀训练。','',
        '| 配置 | OA均值±标准差 | BA均值±标准差 | Macro-F1 | NLL |','|---|---:|---:|---:|---:|']
    for arm,name in NAMES.items():
        b = df[(df.arm==arm)&(df.split=='evaluation')]
        lines.append(f'| {name} | {b.oa.mean():.3f} ± {b.oa.std():.3f} | {b.ba.mean():.3f} ± {b.ba.std():.3f} | {b.macro_f1.mean():.3f} | {b.nll.mean():.4f} |')
    lines += ['','| 前者减后者 | OA差 | BA差 | 救回/误伤事件 | OA/BA正向种子数 |','|---|---:|---:|---:|---:|']
    for (after,before),r in ps.iterrows():
        lines.append(f'| {NAMES[after]} - {NAMES[before]} | {r.oa_delta:+.3f} | {r.ba_delta:+.3f} | {int(r.rescued)}/{int(r.harmed)} | {int(r.positive_oa_seeds)}/3、{int(r.positive_ba_seeds)}/3 |')
    lines += ['','差为百分点；翻转事件按记录×种子计，不是独立船数。所有adapt样本均参加CE，筛选仅控制分差保护。',
        '只匹配随机组每类数量，不额外匹配教师置信度或正确状态。完整后缀的难例权重随各组模型按同规则重算。',
        '测试集此前参与开发；bootstrap条件于已存模型和观察到的场景。原图固定参考为实际合法源案例，不是最大贡献参考。']
    (ROOT/'RESULTS_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    body = ['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>干预筛选完整流程对照</title>',
        '<style>body{font:16px/1.7 system-ui;max-width:1150px;margin:24px auto;padding:0 16px}img{max-width:100%;height:auto}section{border-top:1px solid #aaa;margin-top:25px}</style>',
        '<h1>干预筛选完整流程对照</h1><p><a href="RESULTS_ZH.md">结果</a> · <a href="FINDINGS_ZH.md">原图分析</a> · <a href="PROTOCOL_ZH.md">协议</a> · <a href="DATA_REVIEW_ZH.md">源端筛选分布</a></p>',
        '<p>左侧查询VV/VH，中间同类源参考，右侧混淆类源参考；上排原生像素输入，下排标框。不额外拉伸。参考不是最大贡献者；源端选样与目标翻转不构成逐样本因果证明。</p>']
    for c in cases:
        body += [f'<section><h2>{html.escape(NAMES[c["before_arm"]])} → {html.escape(NAMES[c["after_arm"]])} / {c["kind"]} / {c["id"]}</h2>',
            f'<p>真实类别{h.ZH[c["label"]]}，seed{c["seed"]}，三种子中{c["repeats"]}次；预测{h.ZH[int(np.argmax(c["before"]))]} → {h.ZH[int(np.argmax(c["after"]))]}；分差{c["margin_before"]:+.3f} → {c["margin_after"]:+.3f}。</p>',
            f'<img src="{c["file"]}" loading="lazy"></section>']
    (ROOT/'index.html').write_text('\n'.join(body)+'</html>',encoding='utf-8')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
