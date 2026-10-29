"""Source-only score alignment after M3 continuation; no target centering."""
import csv
import json
from pathlib import Path
from collections import Counter
import numpy as np

HERE=Path(__file__).resolve().parent
WORK=HERE.parents[2]
PUB=HERE.parent/'final_oss2_replication_20261029'


def metrics(z,y):
    p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
    pred=z.argmax(1)
    return dict(oa=100*float((pred==y).mean()),
        ba=100*float(np.mean([(pred[y==c]==c).mean() for c in np.unique(y)])),
        nll=float(-np.log(p[np.arange(len(y)),y].clip(1e-30)).mean()),
        brier=float(np.square(p-np.eye(z.shape[1])[y]).sum(1).mean()))


def source_weights(rows):
    classes=sorted({r['label'] for r in rows})
    counts=Counter((r['label'],str(r['mmsi'])) for r in rows)
    vessels={c:len({str(r['mmsi']) for r in rows if r['label']==c}) for c in classes}
    w=np.array([1/(len(classes)*vessels[r['label']]*counts[r['label'],str(r['mmsi'])]) for r in rows])
    assert np.isclose(w.sum(),1)
    return w


def main():
    all_metrics=[]
    for ds in ('private','oss2'):
        manifest=(WORK/'sar_m123_current_20261027/manifests/source_adapt.json' if ds=='private'
                  else PUB/'manifests/source_adapt.json')
        records=json.loads(manifest.read_text(encoding='utf-8'))
        lookup={str(r['id']):r for r in records}
        for seed in (17,29,43):
            for arm in ('ce','rank','hybrid'):
                root=HERE/ds/'predictions'
                sp=root/f'{arm}_source_s{seed}.npz'
                qp=root/f'{arm}_s{seed}.npz'
                if not sp.exists() or not qp.exists() or not (HERE/ds/'training'/f'{arm}_s{seed}.json').exists():
                    continue
                source=np.load(sp);q=np.load(qp)
                rows=[lookup[str(i)] for i in source['ids']]
                assert np.array_equal(source['y'],[r['label'] for r in rows])
                w=source_weights(rows)
                shift=(w[:,None]*(source['score'].astype(float)-source['baseline'])).sum(0)
                shift-=shift.mean()
                z=q['score'].astype(float)-shift
                candidate=arm+'_guard'
                np.savez(root/f'{candidate}_s{seed}.npz',score=z,baseline=q['baseline'],y=q['y'],ids=q['ids'])
                d=q['score'].astype(float)-q['baseline'];d-=d.mean(1,keepdims=True)
                target_shift=d.mean(0)
                # The query aggregate is diagnostic only; never used to form z or source_shift.
                report=dict(dataset=ds,seed=seed,arm=candidate,parent_arm=arm,
                    source_manifest=str(manifest),source_ids=source['ids'].tolist(),source_shift=shift.tolist(),
                    source_count=len(rows),query_count=len(z),evaluation_used=False,target_used_for_shift=False,
                    source_shift_norm=float(np.linalg.norm(shift)),
                    diagnostic_query_shift_distance=float(np.linalg.norm(shift-target_shift)),
                    before=metrics(q['score'],q['y']),baseline=metrics(q['baseline'],q['y']),after=metrics(z,q['y']))
                out=HERE/ds/'alignment';out.mkdir(exist_ok=True)
                (out/f'{candidate}_s{seed}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
                for variant,scores in [('saved_baseline',q['baseline']),(arm,q['score']),(candidate,z)]:
                    if variant=='saved_baseline' and arm!='ce':continue
                    all_metrics.append(dict(dataset=ds,seed=seed,arm=variant,**metrics(scores,q['y'])))
    out=HERE/'alignment_summary';out.mkdir(exist_ok=True)
    with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(all_metrics[0]));w.writeheader();w.writerows(all_metrics)
    lines=['# Source-only alignment diagnostic','','|Data|Arm|Orders|OA|BA|NLL|Brier|','|---|---|---|---|---|---|---|']
    for ds in ('private','oss2'):
        for arm in ('saved_baseline','ce','rank','hybrid','ce_guard','rank_guard','hybrid_guard'):
            rs=[r for r in all_metrics if r['dataset']==ds and r['arm']==arm]
            if rs:lines.append('|'+ds+'|'+arm+'|'+str(len(rs))+'|'+'|'.join(f'{np.mean([r[k] for r in rs]):.6f}' for k in ('oa','ba','nll','brier'))+'|')
    lines += ['','Shift uses source_adapt labels for class balance and source predictions only. No query centering and no target prior estimation.',
              'This is output alignment, not evidence that matching learned new physical semantics. Final evaluation remains unused.']
    (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':main()
