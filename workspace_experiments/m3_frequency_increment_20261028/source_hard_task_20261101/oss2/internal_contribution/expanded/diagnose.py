"""Post-hoc pathway attribution on cases, never a new predictive selection rule."""
import html
import pandas as pd
from run import *
import review_task as gallery


def main():
    current.setup(DEVICE)
    rows,train,rid,groups=expanded_records();allids=np.concatenate(list(groups.values()))
    lookup={int(i):j for j,i in enumerate(allids)}
    y=np.array([rows[i]['label'] for i in allids]);seeds=(17,29,43)
    scores={s:{n:np.load(HERE/f'{n}_{s}.npz')['score'] for n in ('evidence','reference','none','evidence_off','reference_off')} for s in seeds}
    for s in seeds:scores[s]['baseline']=np.load(HERE/'baseline.npz')['score'][allids]
    a={p.stem:np.load(p,mmap_mode='r') for p in (HERE/'cache').glob('*.npy')}
    cases=json.loads((HERE/'cases.json').read_text());gallery.ROOT=HERE
    eligible=np.array([lookup[int(i)] for i in groups['evaluation'] if rows[i]['label']==2])
    correct={n:np.stack([scores[s][n].argmax(1)==y for s in seeds]) for n in ('evidence','evidence_off','reference')}
    for name,event in [('chemical_direct_harm',~correct['evidence']&correct['evidence_off']),
                       ('chemical_direct_rescue',correct['evidence']&~correct['evidence_off'])]:
        ranked=sorted([int(j) for j in eligible if event[:,j].any()],key=lambda j:(-event[:,j].sum(),int(allids[j])))
        if not ranked:continue
        j=ranked[0];s=seeds[int(np.flatnonzero(event[:,j])[0])]
        case=gallery.draw_case(rows,a,rid,int(allids[j]),name,scores[s],y,j,focus='evidence')
        case.update(seed=s,eligible=len(ranked));cases.append(case)
    chosen=np.array(sorted({c['index'] for c in cases}));out=[]
    for seed in seeds:
        model=Model('aggregate',seed)
        model.restore(torch.load(ROOT/f'aggregate_{seed}/last.pt',map_location='cpu',weights_only=True)['trainable_state'])
        forward=model.matcher.forward
        z={n:scores[seed][n][[lookup[int(i)] for i in chosen]] for n in ('evidence','evidence_off','reference')}
        for p,name in enumerate(('vv_only','vh_only')):
            def masked(*args,pol=p,**kwargs):
                result=forward(*args,**kwargs);delta=result[0].clone();delta[:,1-pol]=0
                return (delta,*result[1:])
            model.matcher.forward=masked
            z[name]=predict(model,a,rows,chosen,rid)['score']
        model.matcher.forward=forward
        err=float(abs(z['vv_only']+z['vh_only']-z['evidence_off']-z['evidence']).max())
        assert err<1e-4,err
        for j,i in enumerate(chosen):
            yy=np.array([rows[i]['label']]);item=dict(seed=seed,id=rows[i]['id'],label=int(yy[0]),additive_error=err)
            for name,score in z.items():
                item[name+'_prediction']=int(score[j].argmax())
                item[name+'_margin']=float(gallery.margins(score[j:j+1],yy)[0])
            # Fixed competitor permits additive evidence accounting; max-wrong margin need not add.
            wrong=int(np.where(np.arange(4)==yy[0],-np.inf,z['evidence'][j]).argmax())
            item['competitor']=wrong
            for name in ('vv_only','vh_only'):
                d=z[name][j]-z['evidence_off'][j]
                item[name+'_pair_delta']=float(d[yy[0]]-d[wrong])
            out.append(item)
        del model
    frame=pd.DataFrame(out);frame.to_csv(HERE/'case_pathways.csv',index=False)
    write(HERE/'diagnostic_cases.json',cases)
    body=['<html><meta charset="utf-8"><title>M3 pathway diagnostic</title><style>body{font-family:Arial;margin:24px}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:4px}</style>',
          '<h1>Frozen M3 VV / VH writeback attribution</h1><p>Diagnostic ablation only. Original images and base paths unchanged. No per-query deployable selection, no retraining. Pair delta uses the fixed strongest wrong class under full E; ordinary max-wrong margins are not additive.</p>']
    for c in cases:body += ['<h2>'+html.escape(c['group']+' '+c['id'])+'</h2>',frame[frame.id==c['id']].to_html(index=False),'<img src="'+c['image']+'">']
    (HERE/'pathways.html').write_text('\n'.join(body)+'</html>',encoding='utf-8')
    print(frame.to_string(index=False))


if __name__=='__main__':main()
