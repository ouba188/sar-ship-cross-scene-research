"""Review source-reader repair without selecting a new epoch or position."""
import html
from experiment import *
sys.path.insert(0, str(task.ROOT))
import review_task as gallery


def main():
    dest = ROOT/'source_reader_calibrated'
    rows, train, ids, rid = task.records()
    a = task.arrays()
    y = np.array([rows[i]['label'] for i in ids])
    arms = ('pre_class_gate','post_class_gate','pre_mean_gate','post_mean_gate')
    scores = {'anchor':np.load(ROOT/'anchor.npz')['score'], 'none':np.load(ROOT/'none_17/final.npz')['score']}
    summary, pairs, classes = [], [], []
    for arm in arms:
        for mode in ('final','off','vv_only','vh_only','uniform_context','spatial_mean'):
            path = dest/(arm+'_17')/(mode+'.npz')
            if path.exists():
                with np.load(path) as z:
                    assert np.array_equal(z['ids'],ids)
                    scores[arm if mode=='final' else arm+'_'+mode] = z['score']
        scores[arm+'_uncalibrated'] = np.load(ROOT/(arm+'_17')/'final.npz')['score']
    for name,z in scores.items():
        summary.append(dict(arm=name,**task.metrics(y,z)))
        for cls in range(4):
            mask = y==cls
            classes.append(dict(arm=name,label=cls,n=int(mask.sum()),correct=int(((z.argmax(1)==y)&mask).sum())))
    comparisons = [(arm,control) for arm in arms for control in ('none','anchor',arm+'_uncalibrated',arm+'_off')]
    for pos in ('pre','post'):
        comparisons += [(pos+'_class_gate',pos+'_mean_gate')]
        comparisons += [(pos+'_class_gate',pos+'_class_gate_'+mode) for mode in ('uniform_context','spatial_mean','vv_only','vh_only')]
    comparisons.append(('post_class_gate','pre_class_gate'))
    for candidate,control in comparisons:
        cc,bb = scores[candidate].argmax(1),scores[control].argmax(1)
        mc,mb = [task.metrics(y,scores[k]) for k in (candidate,control)]
        pairs.append(dict(candidate=candidate,control=control,rescues=int(((cc==y)&(bb!=y)).sum()),
                          harms=int(((cc!=y)&(bb==y)).sum()),changes=int((cc!=bb).sum()),
                          delta_oa=mc['oa']-mb['oa'],delta_ba=mc['ba']-mb['ba'],delta_nll=mc['nll']-mb['nll']))
    pd.DataFrame(summary).to_csv(dest/'all_metrics.csv',index=False)
    pd.DataFrame(pairs).to_csv(dest/'pairs.csv',index=False)
    pd.DataFrame(classes).to_csv(dest/'class_counts.csv',index=False)
    focus='post_class_gate'
    good=scores[focus].argmax(1)==y
    no=scores['none'].argmax(1)==y
    mean=scores['post_mean_gate'].argmax(1)==y
    groups=[('rescue',good&~no),('harm',~good&no),('unique_mean_rescue',good&~mean),
            ('stable_correct',good&no&mean),('stable_wrong',~good&~no&~mean)]
    rank=abs(gallery.margins(scores[focus],y)-gallery.margins(scores['none'],y))
    gallery.ROOT=dest
    selected,cases,status=set(),[],[]
    shown={k:scores[k] for k in ('anchor','none')+arms}
    shown['baseline']=scores['anchor']
    for name,mask in groups:
        pool=sorted(np.flatnonzero(mask),key=lambda i:-rank[i])
        pos=next((int(i) for i in pool if int(i) not in selected),None)
        status.append(dict(group=name,n=len(pool),selected=pos))
        if pos is None:continue
        selected.add(pos)
        case=gallery.draw_case(rows,a,rid,int(ids[pos]),name,shown,y,pos,focus=focus)
        case['pathways']={k:dict(prediction=int(z[pos].argmax()),margin=float(gallery.margins(z[pos:pos+1],y[pos:pos+1])[0])) for k,z in scores.items() if k.startswith(focus)}
        cases.append(case)
    task.write(dest/'cases.json',dict(groups=status,cases=cases))
    body=['<html><meta charset="utf-8"><title>M1 source-reader repair</title>',
          '<style>body{font-family:Arial;margin:24px}img{max-width:100%}table{border-collapse:collapse}td,th{padding:5px;border:1px solid #bbb}</style>',
          '<h1>Source-only reader calibration: matched M1 positions</h1>',
          '<p>Source719 calibration only; historical development228; seed17; fixed epoch4. No final-test selection. Native VV/VH; reference examples are pooled-cosine selections, not maximum attention.</p>',
          pd.DataFrame([r for r in summary if r['arm'] in shown]).to_html(index=False),pd.DataFrame(pairs).to_html(index=False)]
    for case in cases:
        body+=['<h2>'+html.escape(case['group']+' '+case['id'])+'</h2>','<img src="'+case['image']+'">','<pre>'+html.escape(json.dumps(dict(evidence=case['evidence'],pathways=case['pathways']),indent=2))+'</pre>']
    (dest/'index.html').write_text('\n'.join(body+['</html>']),encoding='utf-8')
    print(pd.DataFrame([r for r in summary if r['arm'] in shown]).to_string(index=False))
    print(pd.DataFrame(pairs).to_string(index=False))


if __name__=='__main__':main()
