"""Paired cached-score analysis with unchanged native query/reference panels."""
from pathlib import Path
import html
import importlib.util
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
helper_path = ROOT.parent/'internal_mechanisms_20261111/analyze.py'
spec = importlib.util.spec_from_file_location('previous_internal_analysis', helper_path)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
e, SOURCE = h.e, h.SOURCE
NAMES = {'full':'原三区域完整模型', 'no_m2':'去掉整个M2（复用）', 'merged_readout':'仅合并最终区域读出', 'merged_all':'合并读出及内部回写'}
SEEDS = (17,29,43)


def main():
    rows,groups = e.prev.records()
    ids = groups['evaluation']
    query = [rows[i] for i in ids]
    refs = [rows[i] for i in groups['references']]
    y = np.array([r['label'] for r in query])
    scores,entries,classes,budgets = {},[],[],[]
    for arm in NAMES:
        scores[arm] = []
        for seed in SEEDS:
            folder = SOURCE/f'{arm}_s{seed}_lr1' if arm in ('full','no_m2') else ROOT/f'{arm}_s{seed}'
            if arm not in ('full','no_m2'): assert (folder/'complete.json').exists()
            for split in ('validation','evaluation'):
                data = np.load(folder/f'{split}.npz')
                assert np.array_equal(data['ids'],groups[split])
                z = data['score']
                sy = np.array([rows[i]['label'] for i in groups[split]])
                entries.append(dict(arm=arm,seed=seed,split=split,**h.metrics(sy,z)))
                if split=='evaluation':
                    scores[arm].append(z)
                    for cls in range(4):
                        mask = y==cls
                        classes.append(dict(arm=arm,seed=seed,label=cls,name=h.ZH[cls],n=int(mask.sum()),recall=100*(z.argmax(1)[mask]==cls).mean()))
            for phase in ('carrier','intervention','reference','feedback','readout'):
                receipt = json.loads((folder/phase/'receipt.json').read_text(encoding='utf-8'))
                budgets.append(dict(arm=arm,seed=seed,phase=phase,steps=receipt['steps'],examples=receipt['examples'],seconds=receipt['seconds'],trainable=receipt['trainable']))
    df = pd.DataFrame(entries)
    df.to_csv(ROOT/'metrics.csv',index=False)
    df.groupby(['split','arm'])[['oa','ba','macro_f1','nll']].agg(['mean','std']).to_csv(ROOT/'summary.csv')
    pd.DataFrame(classes).to_csv(ROOT/'per_class.csv',index=False)
    pd.DataFrame(budgets).to_csv(ROOT/'training_budget.csv',index=False)
    h.cluster_report(ROOT,query,scores,'full')
    paired,case_rows = [],[]
    (ROOT/'cases').mkdir(exist_ok=True)
    contrasts = [('full','merged_readout'),('full','merged_all'),('merged_readout','merged_all'),
                 ('merged_readout','no_m2'),('merged_all','no_m2')]
    for after,before in contrasts:
        astack = np.stack([z.argmax(1)==y for z in scores[after]])
        bstack = np.stack([z.argmax(1)==y for z in scores[before]])
        for k,seed in enumerate(SEEDS):
            delta = astack[k].astype(float)-bstack[k]
            paired.append(dict(after=after,before=before,seed=seed,oa_delta=100*delta.mean(),ba_delta=100*np.mean([delta[y==c].mean() for c in range(4)]),rescued=int((astack[k]&~bstack[k]).sum()),harmed=int((~astack[k]&bstack[k]).sum())))
        for kind,mask in [('rescue',astack&~bstack),('harm',~astack&bstack),('stable_correct',astack&bstack),('stable_wrong',~astack&~bstack)]:
            count = mask.sum(0)
            if not count.max(): continue
            j = int(np.flatnonzero(count==count.max())[0])
            k = int(np.flatnonzero(mask[:,j])[0])
            bz,az = scores[before][k][j],scores[after][k][j]
            target = int(y[j])
            rival = int(bz.argmax()) if kind=='rescue' else int(az.argmax())
            if rival==target: rival=int(np.where(np.arange(4)==target,-np.inf,az).argmax())
            filename = f'{after}_vs_{before}_{kind}_{query[j]["id"]}_s{SEEDS[k]}.png'
            sources = h.figure(query[j],refs,rival,ROOT/'cases'/filename)
            margin = lambda z:float(z[target]-np.max(np.delete(z,target)))
            case_rows.append(dict(after_arm=after,before_arm=before,kind=kind,id=query[j]['id'],label=target,seed=SEEDS[k],repeats=int(count[j]),before=bz.tolist(),after=az.tolist(),margin_before=margin(bz),margin_after=margin(az),file=f'cases/{filename}',sources=sources))
    pd.DataFrame(paired).to_csv(ROOT/'paired_by_seed.csv',index=False)
    ps = pd.DataFrame(paired).groupby(['after','before']).agg(oa_delta=('oa_delta','mean'),ba_delta=('ba_delta','mean'),rescued=('rescued','sum'),harmed=('harmed','sum'),positive_oa_seeds=('oa_delta',lambda x:int((x>0).sum())),positive_ba_seeds=('ba_delta',lambda x:int((x>0).sum())))
    ps.to_csv(ROOT/'paired_summary.csv')
    (ROOT/'cases_local.json').write_text(json.dumps(case_rows,ensure_ascii=False,indent=2),encoding='utf-8')
    lines = ['# M2区域身份匹配替换结果','','OSS2四分类，三种子，六次完整五阶段训练；Full和No-M2复用，无新增调参。','',
        '| 配置 | OA均值±标准差 | BA均值±标准差 | Macro-F1 | NLL |','|---|---:|---:|---:|---:|']
    for arm,name in NAMES.items():
        block = df[(df.arm==arm)&(df.split=='evaluation')]
        lines.append(f'| {name} | {block.oa.mean():.3f} ± {block.oa.std():.3f} | {block.ba.mean():.3f} ± {block.ba.std():.3f} | {block.macro_f1.mean():.3f} | {block.nll.mean():.4f} |')
    lines += ['','| 配对差（前者减后者） | OA差 | BA差 | 救回/误伤事件 | OA/BA正向种子数 |','|---|---:|---:|---:|---:|']
    for (after,before),r in ps.iterrows():
        lines.append(f'| {NAMES[after]} − {NAMES[before]} | {r.oa_delta:+.3f} | {r.ba_delta:+.3f} | {int(r.rescued)}/{int(r.harmed)} | {int(r.positive_oa_seeds)}/3、{int(r.positive_ba_seeds)}/3 |')
    lines += ['','单位为百分点；救回/误伤按记录×种子计，不是独立船数。教师筛选与难例权重按原规则各自重算；不是固定所有中间状态的纯推理干预。',
        'M3几何未改，本轮仅检验M2区域身份。测试队列此前参与研发，不称独立新测试。原图参考来自真实合法库，但非最大贡献参考。']
    (ROOT/'RESULTS_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    body = ['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M2区域身份消融</title>',
        '<style>body{font:16px/1.7 system-ui;max-width:1150px;margin:24px auto;padding:0 16px}img{max-width:100%;height:auto}section{border-top:1px solid #aaa;margin-top:25px}</style>',
        '<h1>M2区域身份消融</h1><p><a href="RESULTS_ZH.md">结果</a> · <a href="FINDINGS_ZH.md">原图分析与结论</a> · <a href="PROTOCOL_ZH.md">协议</a></p>',
        '<p>左侧查询VV/VH，中间同类源参考，右侧混淆类源参考；上排原生像素输入，下排标框。不额外拉伸，不按外观挑选案例。参考不是最大贡献者。</p>']
    for c in case_rows:
        body += [f'<section><h2>{html.escape(NAMES[c["before_arm"]])} → {html.escape(NAMES[c["after_arm"]])} / {c["kind"]} / {c["id"]}</h2>',
            f'<p>真实类别{h.ZH[c["label"]]}，seed{c["seed"]}，三种子中{c["repeats"]}次；预测{h.ZH[int(np.argmax(c["before"]))]} → {h.ZH[int(np.argmax(c["after"]))]}；分差{c["margin_before"]:+.3f} → {c["margin_after"]:+.3f}。</p>',
            f'<img src="{c["file"]}" loading="lazy"></section>']
    (ROOT/'index.html').write_text('\n'.join(body)+'</html>',encoding='utf-8')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    main()
