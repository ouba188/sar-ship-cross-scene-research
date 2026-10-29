"""CPU summaries and native-chip panels for completed M3 repair arms."""
import csv
import html
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SEEDS=(17,29,43)


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


base=module('repair_analysis_base',ROOT/'spatial_m12_refit_20261029/analysis.py')
native=module('repair_native_input',HERE.parents[2]/'sar_m123_standalone/sar_m123/input.py')


def main():
    cohorts=json.loads((ROOT/'spatial_m12_full_20261029/cohorts.json').read_text(encoding='utf-8'))
    rows,refs=cohorts['development'],cohorts['references']
    y=np.array([r['label'] for r in rows])
    scores={}
    for seed in SEEDS:
        for path in (HERE/'predictions').glob(f'*_s{seed}.npz'):
            condition=path.stem.rsplit('_s',1)[0]
            if condition.startswith('before_') and condition!='before_control':
                continue
            scores[condition,seed]=base.load_score(path,rows)
    conditions=sorted({k[0] for k in scores})
    out=HERE/'analysis'
    out.mkdir(exist_ok=True)
    table=[]
    summary={}
    for c in conditions:
        values=[]
        for s in SEEDS:
            m=base.metrics(scores[c,s],y)
            values.append(m)
            table.append(dict(condition=c,seed=s,**m))
        summary[c]={k:dict(mean=float(np.mean([v[k] for v in values])),sd=float(np.std([v[k] for v in values],ddof=1)))
                    for k in ('OA','BA','NLL','Brier')}
    base.write_csv(out/'metrics.csv',table)
    base.dump(out/'summary.json',summary)
    comparisons=[('before_control','after_control'),('after_control','after_fast_compare'),
                 ('after_control','after_metric'),('before_control','after_metric')]
    if 'compare_only_metric' in conditions:
        comparisons += [('before_control','compare_only_metric'),('before_control','value_only_metric')]
    events=[]
    for old,new in comparisons:
        for s in SEEDS:
            a,b=scores[old,s],scores[new,s]
            for i,r in enumerate(rows):
                ap,bp=int(a[i].argmax()),int(b[i].argmax())
                ac,bc=ap==y[i],bp==y[i]
                aa=a[i].copy();aa[y[i]]=-np.inf
                rival=int(aa.argmax())
                events.append(dict(comparison=f'{new}-minus-{old}',seed=s,id=str(r['id']),label=int(y[i]),
                    port=r['port'],pass_group=r['pass_group'],state=('CC' if ac else 'rescued') if bc else ('harmed' if ac else 'WW'),
                    old_prediction=ap,new_prediction=bp,rival=rival,
                    margin_old=float(a[i,y[i]]-a[i,rival]),margin_new=float(b[i,y[i]]-b[i,rival]),
                    delta_margin=float(b[i,y[i]]-b[i,rival]-a[i,y[i]]+a[i,rival])))
    base.write_csv(out/'events.csv',events)
    text=['# M3 repair fixed-endpoint results','',
          'Historical development160 / 3 passes / 7 present classes. Source-fit continuation only. No new blind test.',
          'Same-checkpoint compare_only/value_only swaps are diagnostics, not retrained contenders.','',
          '| Condition | OA | BA | NLL | Brier |','|---|---:|---:|---:|---:|']
    for c,v in summary.items():
        text.append('| '+c+' | '+' | '.join(f"{v[k]['mean']:.4f} +/- {v[k]['sd']:.4f}" for k in ('OA','BA','NLL','Brier'))+' |')
    text += ['','## Outcome events (three orders, not independent ships)','']
    for old,new in comparisons:
        es=[e for e in events if e['comparison']==f'{new}-minus-{old}']
        text.append(f"- {new} minus {old}: "+', '.join(f"{st}={sum(e['state']==st for e in es)}" for st in ('rescued','harmed','CC','WW'))
                    +f"; all prediction changes={sum(e['old_prediction']!=e['new_prediction'] for e in es)}")
    (out/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    # No category is silently filled with a different transition.
    chosen=[]
    for comp in ('after_metric-minus-after_control','after_metric-minus-before_control'):
        for state in ('rescued','harmed','CC','WW'):
            pool=[e for e in events if e['comparison']==comp and e['state']==state and e['id'] not in {x['id'] for x in chosen}]
            if pool:
                chosen.append(sorted(pool,key=lambda e:(-abs(e['delta_margin']),e['id'],e['seed']))[0])
    folder=out/'cases';folder.mkdir(exist_ok=True)
    lookup={str(r['id']):r for r in rows}
    blocks=[];sources=[]
    for number,case in enumerate(chosen,1):
        q=lookup[case['id']]
        allowed=[r for r in refs if base.eligible(q,r)]
        selected=[r for r in allowed if r['label'] in {case['label'],case['rival'],case['new_prediction']}]
        case['reference_ids']=[str(r['id']) for r in selected]
        roles=['QUERY']+['TRUE REF' if r['label']==case['label'] else 'RIVAL REF' for r in selected]
        blocks.append('<h2>'+html.escape(f"{number}: {case['id']} / {case['state']} / {case['comparison']} / seed{case['seed']}")+'</h2><pre>'+html.escape(json.dumps(case,indent=2))+'</pre>')
        for pol in ('vv','vh'):
            filename=f'case_{number:02d}_{pol}.png'
            sources.extend(base.panel([q,*selected],roles,pol,folder/filename,native))
            blocks.append(f'<h3>{pol.upper()}</h3><img src="{filename}">')
    base.dump(folder/'cases.json',chosen)
    base.dump(folder/'native_sources.json',sources)
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>M3 repair cases</title><style>body{font:14px Arial;margin:24px}img{image-rendering:pixelated}pre{white-space:pre-wrap}</style><h1>M3 repair cases</h1><p>Native pixels; actual eligible references, not assumed highest-weight references. Outcome-selected examples. VV/VH shown separately.</p>'+''.join(blocks),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print('CASES',json.dumps(chosen))


if __name__=='__main__':
    main()
