"""Summarize paired development results, retaining all search trials."""
from pathlib import Path
import csv
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
ARMS=('baseline','independent','gate','pcea')


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def main():
    rows=[];selected={}
    for path in sorted(ROOT.glob('*_s*_lr*_e*/receipt.json')):
        r=read(path);b=r['best']['metrics'];last=r['last']
        scope=r.get('optimization_scope','joint')
        row=dict(run=path.parent.name,scope=scope,arm=r['arm'],seed=r['seed'],lr=r['backbone_lr'],
            epochs=r['epochs'],selected_epoch=r['best']['epoch'],oa=b['oa'],ba=b['ba'],
            macro_f1=b['macro_f1'],last_oa=last['oa'],last_ba=last['ba'],seconds=r['seconds'],
            branch_parameters=r['branch_parameters'])
        rows.append(row)
        key=(scope,r['seed'],r['arm'])
        if key not in selected or r['best']['key']>selected[key][0]:
            selected[key]=(r['best']['key'],path.parent)
    if not rows: return
    with (ROOT/'all_runs.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    chosen={str(s):{a:str(v[1].relative_to(ROOT)) for (sc,s1,a),v in selected.items() if s1==s and sc=='joint'}
            for s in sorted({k[1] for k in selected if k[0]=='joint'})}
    branch={str(s):{a:str(v[1].relative_to(ROOT)) for (sc,s1,a),v in selected.items() if s1==s and sc=='branch'}
            for s in sorted({k[1] for k in selected if k[0]=='branch'})}
    (ROOT/'selected_branch.json').write_text(json.dumps(branch,indent=2),encoding='utf-8')
    (ROOT/'selected_validation.json').write_text(json.dumps(chosen,indent=2),encoding='utf-8')
    pairs=[]
    for seed,arms in chosen.items():
        if len(arms)!=4: continue
        results={a:np.load(ROOT/d/'validation.npz') for a,d in arms.items()}
        for a,b in (('pcea','baseline'),('pcea','independent'),('pcea','gate'),('gate','baseline')):
            x,y=results[a],results[b]
            assert np.array_equal(x['ids'],y['ids']) and np.array_equal(x['label'],y['label'])
            ca,cb=x['score'].argmax(1)==x['label'],y['score'].argmax(1)==y['label']
            pairs.append(dict(seed=int(seed),candidate=a,control=b,n=len(ca),
                rescue=int((ca&~cb).sum()),harm=int((~ca&cb).sum()),net=int(ca.sum()-cb.sum()),
                delta_oa=float((ca.mean()-cb.mean())*100)))
    last_pairs=[]
    for rate in sorted({r['lr'] for r in rows if r['scope']=='joint'}):
        trials={r['arm']:r for r in rows if r['seed']==17 and r['scope']=='joint' and r['lr']==rate}
        if len(trials)!=4:continue
        arrays={a:np.load(ROOT/r['run']/'last_validation.npz') for a,r in trials.items()}
        for a,b in (('pcea','baseline'),('pcea','independent'),('pcea','gate')):
            ca=arrays[a]['score'].argmax(1)==arrays[a]['label']
            cb=arrays[b]['score'].argmax(1)==arrays[b]['label']
            last_pairs.append(dict(lr=rate,candidate=a,control=b,rescue=int((ca&~cb).sum()),
                harm=int((~ca&cb).sum()),net=int(ca.sum()-cb.sum())))
    (ROOT/'last_epoch_pairs.json').write_text(json.dumps(last_pairs,indent=2),encoding='utf-8')
    work=ROOT.parent.parent
    manifest=read(work/'m1_coverage_20261101/manifest.json')
    groups=read(work/'m1_coverage_20261101/groups.json')
    av=np.load(ROOT.parent/'recent_baselines_20261111/native_cache/available.npy')
    coverage={}
    for split in ('small','validation'):
        ix=groups[split]
        coverage[split]=dict(observations=len(ix),ships=len({manifest[i]['mmsi'] for i in ix}),
            passes=len({manifest[i]['pass_group'] for i in ix}),dual_available=int(av[ix].all(1).sum()),
            classes=[dict(label=c,n=sum(manifest[i]['label']==c for i in ix),
                dual=sum(manifest[i]['label']==c and bool(av[i].all()) for i in ix)) for c in range(4)])
    (ROOT/'data_coverage.json').write_text(json.dumps(coverage,indent=2),encoding='utf-8')
    (ROOT/'paired_counts.json').write_text(json.dumps(pairs,indent=2),encoding='utf-8')
    lines=['# PCEA paired development training','','All values are validation-selected; not held-out confirmation.','',
           '| Scope | Arm | Seed | LR setting | Epoch | OA | BA | Last OA | Last BA |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['scope']} | {r['arm']} | {r['seed']} | {r['lr']:g} | {r['selected_epoch']} | {r['oa']:.3f} | {r['ba']:.3f} | {r['last_oa']:.3f} | {r['last_ba']:.3f} |")
    lines+=['','## Paired Predictions','', '| Seed | Candidate | Control | Rescue | Harm | Net | OA delta |',
            '|---:|---|---|---:|---:|---:|---:|']
    for r in pairs:
        lines.append(f"| {r['seed']} | {r['candidate']} | {r['control']} | {r['rescue']} | {r['harm']} | {r['net']} | {r['delta_oa']:+.3f} |")
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':main()
