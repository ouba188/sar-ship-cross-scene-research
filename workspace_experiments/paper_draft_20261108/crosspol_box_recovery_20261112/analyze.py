"""Paired statistics and native cases for fixed-weight query recovery."""
import html
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pilot

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('recovery_stats',ROOT.parent/'r50_tuned_20261111/analyze.py')
h=importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
sys.path.insert(0,str(pilot.WORK/'sar_m123_standalone'))
from sar_m123 import input as native


def main():
    rows,groups=pilot.records()
    metric_rows=[];paired=[];ci=[];class_rows=[];case_rows=[]
    for split in ('validation','evaluation'):
        ids=np.array(groups[split]);y=np.array([rows[i]['label'] for i in ids])
        missing=np.array([rows[i].get('vv') is None or rows[i].get('vh') is None for i in ids])
        ships=np.array([str(rows[i]['mmsi']) for i in ids])
        scores={key:np.stack([np.load(ROOT/f'{split}_s{s}.npz')[key] for s in pilot.SEEDS]) for key in ('base','recovered')}
        pred={key:z.argmax(-1) for key,z in scores.items()}
        masks=dict(all=np.ones(len(ids),bool),recovered=missing,both=~missing,
            restore_vv=np.array([rows[i].get('vv') is None for i in ids]),
            restore_vh=np.array([rows[i].get('vh') is None for i in ids]))
        passes=np.array(['_'.join(rows[i]['pass_group'].split('_')[:2]) for i in ids])
        masks.update({f'pass:{p}':passes==p for p in sorted(set(passes))})
        states=h.outcome_masks(pred['recovered'],pred['base'],y)
        for group,mask in masks.items():
            if not mask.any():continue
            for k,seed in enumerate(pilot.SEEDS):
                a=h.metrics(y[mask],pred['base'][k,mask]);b=h.metrics(y[mask],pred['recovered'][k,mask])
                for method,values in (('base',a),('recovered',b)):
                    metric_rows.append(dict(split=split,group=group,seed=seed,method=method,n=int(mask.sum()),**values))
                paired.append(dict(split=split,group=group,seed=seed,n=int(mask.sum()),
                    **{m:b[m]-a[m] for m in h.METRICS},
                    **{kind:int(state[k,mask].sum()) for kind,state in states.items()}))
            if split=='evaluation' and group in ('all','recovered'):
                ci += [dict(group=group,**v) for v in h.cluster_intervals(y[mask],ships[mask],pred['recovered'][:,mask],pred['base'][:,mask],2000,9012)]
        for c in range(4):
            for k,seed in enumerate(pilot.SEEDS):
                mask=y==c
                class_rows.append(dict(split=split,label=c,class_name=h.LABELS[c],seed=seed,n=int(mask.sum()),
                    before=float((pred['base'][k,mask]==c).mean()*100),
                    after=float((pred['recovered'][k,mask]==c).mean()*100)))
        if split!='evaluation':continue
        for kind,state in states.items():
            chosen=h.choose_cases(state&missing[None,:],y,ids,2)
            for j in chosen:
                k=int(np.flatnonzero(state[:,j])[0]);idx=int(ids[j]);q=rows[idx]
                restored,audit=pilot.recover(q)
                wrong=int(pred['base'][k,j] if kind=='rescued' else pred['recovered'][k,j])
                if wrong==y[j]:
                    temp=scores['recovered'][k,j].copy();temp[y[j]]=-np.inf;wrong=int(temp.argmax())
                refs=[]
                for title,c in (('Same-class ref',int(y[j])),('Confusion ref',wrong)):
                    possible=[i for i in groups['references'] if rows[i]['label']==c and str(rows[i]['mmsi'])!=str(q['mmsi'])
                        and '_'.join(rows[i]['pass_group'].split('_')[:2])!='_'.join(q['pass_group'].split('_')[:2])]
                    if possible: refs.append((title,rows[possible[0]]))
                path=ROOT/'panels'/f'{kind}_{idx}_s{pilot.SEEDS[k]}.png'
                caption=f'{kind} | s{pilot.SEEDS[k]} | truth={int(y[j])}, before={pred["base"][k,j]}, after={pred["recovered"][k,j]}'
                provenance=h.draw_panel(native,restored,refs,path,caption)
                evidence=[]
                for kk,s in enumerate(pilot.SEEDS):
                    margin=lambda z:float(z[y[j]]-np.delete(z,y[j]).max())
                    evidence.append(dict(seed=s,before=int(pred['base'][kk,j]),after=int(pred['recovered'][kk,j]),
                        before_margin=margin(scores['base'][kk,j]),after_margin=margin(scores['recovered'][kk,j])))
                case_rows.append(dict(kind=kind,index=idx,id=q['id'],label=int(y[j]),restored=audit['restored'],
                    repeats=int(state[:,j].sum()),file='panels/'+path.name,evidence=evidence,paths=provenance,
                    displayed_refs=[r['id'] for _,r in refs],review='pending'))
    pd.DataFrame(metric_rows).to_csv(ROOT/'metrics.csv',index=False)
    pd.DataFrame(paired).to_csv(ROOT/'paired.csv',index=False)
    pd.DataFrame(ci).to_csv(ROOT/'intervals.csv',index=False)
    pd.DataFrame(class_rows).to_csv(ROOT/'classes.csv',index=False)
    df=pd.DataFrame(metric_rows)
    summary=df.groupby(['split','group','method'])[['oa','ba','macro_f1']].agg(['mean','std'])
    summary.to_csv(ROOT/'summary.csv')
    pilot.save('cases_local.json',case_rows)
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>跨极化借框固定权重验证</title>',
        '<style>body{font:16px/1.6 system-ui;margin:24px}section{border-top:1px solid #999;padding:16px 0}img{max-width:none}pre{white-space:pre-wrap}</style>',
        '<h1>跨极化借框：固定完整模型，不重训</h1>',
        '<p>仅恢复缺框查询极化。参考库仍为原32条，不补框。上轮三种子检查点不变。VV/VH保持原始像素尺度；图中恢复后输入和原参考均为实际输入，无显示增强。</p>',
        '<p>参考为实际合法库中首个同类/混淆类案例，不表示贡献最大。恢复同时改变图像输入、区域、可用标记及相应参考关系，不是单独模块消融。</p>',
        '<pre>'+html.escape(summary.to_string())+'</pre>']
    for c in case_rows:
        body += [f'<section><h2>{c["kind"]} / {c["id"]}</h2>',
            f'<p>恢复{c["restored"].upper()}；同状态{c["repeats"]}/3种子。原流程该极化整路不输入。</p>',
            f'<div style="overflow:auto"><img src="{c["file"]}"></div>',
            '<pre>'+html.escape(json.dumps(c['evidence'],ensure_ascii=False,indent=2))+'</pre></section>']
    (ROOT/'index.html').write_text('\n'.join(body),encoding='utf-8')
    print(summary.loc[(slice(None),['all','recovered','restore_vv','restore_vh'],slice(None)),:].to_string())


if __name__=='__main__':main()
