"""Read only saved scores; retain gains, harms, strata, and native image cases."""
from pathlib import Path
import html
import importlib.util
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('previous_case_analysis', ROOT.parent/'internal_mechanisms_20261111/analyze.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
e = h.e
SEEDS = (17,29,43)
NAMES = {'specific':'极化专属（配对初始化）','shared':'极化共享（配对初始化）','historical_full':'历史Full（原初始化，参考）'}


def main():
    rows, groups = e.prev.records()
    ids = groups['evaluation']
    queries = [rows[i] for i in ids]
    refs = [rows[i] for i in groups['references']]
    y = np.array([r['label'] for r in queries])
    availability = np.asarray(e.prev.arrays()[0]['available'][ids], dtype=bool)
    entries, perclass, budgets, divergence, scores = [], [], [], [], {}
    for arm in NAMES:
        scores[arm] = []
        for seed in SEEDS:
            folder = ROOT/f'{arm}_s{seed}' if arm != 'historical_full' else h.SOURCE/f'full_s{seed}_lr1'
            if arm != 'historical_full':
                receipt = json.loads((folder/'complete.json').read_text())
                for module, value in receipt['parameter_divergence'].items():
                    divergence.append(dict(arm=arm, seed=seed, module=module, l2_difference=value))
            for split in ('validation','evaluation'):
                data = np.load(folder/f'{split}.npz')
                assert np.array_equal(data['ids'],groups[split])
                z = data['score']; sy = np.array([rows[i]['label'] for i in groups[split]])
                entries.append(dict(arm=arm,seed=seed,split=split,**h.metrics(sy,z)))
                if split == 'evaluation':
                    scores[arm].append(z)
                    for cls in range(4):
                        mask = y == cls
                        perclass.append(dict(arm=arm,seed=seed,label=cls,name=h.ZH[cls],n=int(mask.sum()),recall=100*(z.argmax(1)[mask]==cls).mean()))
            for phase in ('carrier','intervention','reference','feedback','readout'):
                r = json.loads((folder/phase/'receipt.json').read_text())
                budgets.append(dict(arm=arm,seed=seed,phase=phase,steps=r['steps'],examples=r['examples'],seconds=r['seconds'],trainable=r['trainable']))
    df = pd.DataFrame(entries)
    df.to_csv(ROOT/'metrics.csv',index=False)
    df.groupby(['split','arm'])[['oa','ba','macro_f1','nll']].agg(['mean','std']).to_csv(ROOT/'summary.csv')
    pd.DataFrame(perclass).to_csv(ROOT/'per_class.csv',index=False)
    pd.DataFrame(budgets).to_csv(ROOT/'training_budget.csv',index=False)
    pd.DataFrame(divergence).to_csv(ROOT/'parameter_divergence.csv',index=False)
    h.cluster_report(ROOT, queries, {k:scores[k] for k in ('specific','shared')}, 'specific')
    paired, strata = [], []
    astack = np.stack([z.argmax(1)==y for z in scores['specific']])
    bstack = np.stack([z.argmax(1)==y for z in scores['shared']])
    for si, seed in enumerate(SEEDS):
        delta = astack[si].astype(float)-bstack[si]
        paired.append(dict(seed=seed,oa_delta=100*delta.mean(),ba_delta=100*np.mean([delta[y==cls].mean() for cls in range(4)]),
            rescued=int((astack[si]&~bstack[si]).sum()),harmed=int((~astack[si]&bstack[si]).sum())))
        for vv,vh in ((True,True),(True,False),(False,True)):
            mask = (availability[:,0]==vv)&(availability[:,1]==vh)
            if mask.any():
                strata.append(dict(seed=seed,vv=vv,vh=vh,n=int(mask.sum()),oa_delta=100*delta[mask].mean(),
                    ba_delta=100*np.mean([delta[mask&(y==cls)].mean() for cls in np.unique(y[mask])]),
                    specific_oa=100*astack[si][mask].mean(),shared_oa=100*bstack[si][mask].mean()))
    paired = pd.DataFrame(paired)
    paired.to_csv(ROOT/'paired_by_seed.csv',index=False)
    pd.DataFrame(strata).to_csv(ROOT/'polarization_paired.csv',index=False)
    (ROOT/'cases').mkdir(exist_ok=True)
    cases, used = [], set()
    masks = {'rescue':astack&~bstack,'harm':~astack&bstack,'stable_correct':astack&bstack,'stable_wrong':~astack&~bstack}
    for kind, mask in masks.items():
        count = mask.sum(0)
        # Cover class-specific flips, not only a large-class best example.
        classes = range(4) if kind in ('rescue','harm') else [None]
        for cls in classes:
            eligible = np.flatnonzero((count>0)&((y==cls) if cls is not None else np.ones(len(y),bool)))
            if not len(eligible):
                continue
            j = int(sorted(eligible,key=lambda j:(-count[j],j))[0])
            si = int(np.flatnonzero(mask[:,j])[0])
            bz, az = scores['shared'][si][j], scores['specific'][si][j]
            target = int(y[j]); rival = int(bz.argmax()) if kind=='rescue' else int(az.argmax())
            if rival==target:
                rival = int(np.where(np.arange(4)==target,-np.inf,az).argmax())
            filename = f'{kind}_{queries[j]["id"]}_s{SEEDS[si]}.png'
            sources = h.figure(queries[j],refs,rival,ROOT/'cases'/filename)
            margin = lambda z:float(z[target]-np.max(np.delete(z,target)))
            cases.append(dict(kind=kind,id=queries[j]['id'],label=target,seed=SEEDS[si],repeats=int(count[j]),
                before=bz.tolist(),after=az.tolist(),margin_before=margin(bz),margin_after=margin(az),
                file=f'cases/{filename}',sources=sources))
    (ROOT/'cases_local.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
    lines = ['# OSS2极化共享参数对照结果','','两种输入都保留；两组初始函数相同，差别是模块参数是否允许按极化分化。','',
        '| 配置 | 测试OA | 测试BA | 验证OA | 验证BA |','|---|---:|---:|---:|---:|']
    for arm,name in NAMES.items():
        values = []
        for split in ('evaluation','validation'):
            block = df[(df.arm==arm)&(df.split==split)]
            values += [f'{block[k].mean():.3f} ± {block[k].std():.3f}' for k in ('oa','ba')]
        lines.append('| '+name+' | '+' | '.join(values)+' |')
    lines += ['',f"专属减共享：OA {paired.oa_delta.mean():+.3f}、BA {paired.ba_delta.mean():+.3f}个百分点；OA/BA正向种子分别{int((paired.oa_delta>0).sum())}/3、{int((paired.ba_delta>0).sum())}/3。",
        f"救回{paired.rescued.sum()}、误伤{paired.harmed.sum()}记录×种子事件，不是独立船数。",'',
        '历史Full单独列出，不代替同初始化的直接对照。共享减少独立参数数目，不能只凭本对照分离极化语义和容量效应。',
        '同一训练规则下教师选样和难例权重各自重算。三训练种子，区间按船分层重采样，测试场景并非全新研发未见样本。',
        '参数分化、逐类与逐场景、极化缺失分层及训练成本见配套CSV；原图案例不是总体比例或物理原因证明。']
    (ROOT/'RESULTS_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    body = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>极化共享参数对照</title>',
        '<style>body{font:16px/1.7 system-ui;max-width:1150px;margin:24px auto;padding:0 16px}img{max-width:100%}section{border-top:1px solid #bbb;margin-top:24px}</style>',
        '<h1>OSS2极化参数共享与专属</h1><p><a href="RESULTS_ZH.md">结果表</a> · <a href="FINDINGS_ZH.md">结论与原图回看</a> · <a href="PROTOCOL_ZH.md">协议</a></p>',
        '<p>预测由共享组→专属组；左查询VV/VH，中同类参考，右混淆类参考。原生160裁剪补零，未额外拉伸；上排原图，下排检测框。参考为真实合法库成员，不是最大贡献者。</p>']
    for case in cases:
        body += [f'<section><h2>{case["kind"]} / {case["id"]} / {h.ZH[case["label"]]}</h2>',
                 f'<p>seed{case["seed"]}，出现{case["repeats"]}/3；{h.ZH[int(np.argmax(case["before"]))]} → {h.ZH[int(np.argmax(case["after"]))]}；分差{case["margin_before"]:+.3f} → {case["margin_after"]:+.3f}。</p>',
                 f'<img src="{case["file"]}"></section>']
    (ROOT/'index.html').write_text('\n'.join(body)+'</html>',encoding='utf-8')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    main()
