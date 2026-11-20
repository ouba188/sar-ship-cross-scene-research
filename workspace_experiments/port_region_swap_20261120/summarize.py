from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw

P=Path(__file__).resolve().parent
read=lambda f:json.loads((P/f).read_text(encoding='utf-8'))
write=lambda f,d:(P/f).write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')

def margin(z,y):
    return z[y]-max(v for k,v in enumerate(z) if k!=y)

def main():
    rows=read('observations_local.json'); banks=read('banks_local.json'); slot=banks['slot']
    result={}; examples=[]
    for name in ('0:1','2:6'):
        rs=[r for r in rows if name in r['pairs']]
        if not rs:
            result[name]={'n':0}; continue
        z=np.array([r['pairs'][name] for r in rs])
        y=np.array([r['label'] for r in rs])
        m=np.array([[[margin(a,c),margin(b,c)] for a,b in row] for c,row in zip(y,z)])
        d=m[:,:,0]-m[:,:,1]; ix=d[:,1:]-d[:,:1]
        pred=z.argmax(-1); correct=pred==y[:,None,None]
        soft=np.array([r['soft'] for r in rs]); sc=soft.argmax(-1)==y[:,None]
        result[name]={'n':len(rs),'ports':len({r['port'] for r in rs}),'passages':len({r['passage'] for r in rs}),
            'by_class':{str(c):int((y==c).sum()) for c in (3,4)},
            'soft_correct':sc.sum(0).tolist(),'soft_rescue_vs_original':((~sc[:,:1])&sc[:,1:]).sum(0).tolist(),
            'soft_harm_vs_original':(sc[:,:1]&(~sc[:,1:])).sum(0).tolist(),
            'median_abs_A_minus_B':np.median(abs(d),axis=0).tolist(),
            'max_abs_A_minus_B':abs(d).max(0).tolist(),
            'A_vs_B_pred_changed':(pred[:,:,0]!=pred[:,:,1]).sum(0).tolist(),
            'A_rescue_over_B':(correct[:,:,0]&(~correct[:,:,1])).sum(0).tolist(),
            'A_harm_over_B':((~correct[:,:,0])&correct[:,:,1]).sum(0).tolist(),
            'median_abs_interaction':np.median(abs(ix),axis=0).tolist(),
            'max_abs_interaction':abs(ix).max(0).tolist(),
            'both_replacements_reverse':{str(t):int(((d[:,:1]*d[:,1:]<0).all(1)&(abs(d).min(1)>t)).sum()) for t in (0,.01,.05)},
            'soft_bank_abs_margin_shift_median':np.median(abs(np.array([[margin(v,r['label']) for v in r['soft']] for r in rs])[:,1:]-
               np.array([[margin(r['soft'][0],r['label'])] for r in rs])),axis=0).tolist()}
        # Post-hoc: fix the original soft rival to separate competitor changes.
        rival=np.where(np.arange(8)[None,:]!=y[:,None],soft[:,0],-np.inf).argmax(-1)
        fm=np.stack([zz[...,c]-zz[...,j] for zz,c,j in zip(z,y,rival)])
        fd=fm[:,:,0]-fm[:,:,1]
        result[name].update(A_correct=correct[:,:,0].sum(0).tolist(),B_correct=correct[:,:,1].sum(0).tolist(),
            either_correct=correct.any(-1).sum(0).tolist(),
            fixed_rival_both_reverse_gt_01=int(((fd[:,:1]*fd[:,1:]<0).all(1)&(abs(fd).min(1)>.01)).sum()))
        if name=='0:1':
            result[name]['chemical_size']={}
            for label,sel in [('near',[]),('far',[])]:
                for j,r in enumerate(rs):
                    if r['label']!=3 or not r['length']: continue
                    L=float(r['length'])
                    if not np.isfinite(L) or L<=0: continue
                    far=min(max(L/k,k/L) for k in (102,106))>1.5
                    if far==(label=='far'): sel.append(j)
                result[name]['chemical_size'][label]=dict(n=len(sel),soft_correct=sc[sel].sum(0).tolist())
        # First cases in seeded order for each state; position switch and bank switch separately.
        for axis,pre,post in [('region',pred[:,0,1],pred[:,0,0]),('bank',soft[:,0].argmax(-1),soft[:,1].argmax(-1))]:
            for cat,mask in [('rescue',(pre!=y)&(post==y)),('harm',(pre==y)&(post!=y)),
                            ('correct',(pre==y)&(post==y)),('wrong',(pre!=y)&(post!=y))]:
                ids=np.flatnonzero(mask)
                if len(ids):
                    j=ids[0]
                    examples.append(dict(id=rs[j]['id'],pair=name,axis=axis,cat=cat,delta=d[j].tolist(),
                        pre=int(pre[j]),post=int(post[j]),margins=m[j].tolist()))
    write('summary.json',result); write('cases_local.json',examples)
    cases={}
    for e in examples: cases.setdefault(e['id'],[]).append(e)
    lookup={r['id']:r for r in rows}
    for n,(key,es) in enumerate(cases.items()):
        r=lookup[key]; y=r['label']; rival=max((k for k in range(8) if k!=y),key=lambda k:r['soft'][0][k])
        cols=[('query',r)]+[(f'old {j}',v) for j,v in enumerate(banks['banks'][0]) if v['label'] in {y,rival,3,4}]+[
                (f'new {b}',banks['banks'][b][slot]) for b in (1,2)]
        out=Image.new('RGB',(len(cols)*164,374),(110,110,110)); draw=ImageDraw.Draw(out)
        for x,(lab,row) in enumerate(cols):
            draw.text((x*164+2,0),f"{lab} c{row['label']}",fill='white')
            for k,pol in enumerate(('vv','vh')):
                if row[pol]:
                    with Image.open(row[pol]) as im:
                        im=im.convert('L')
                        assert max(im.size)<=164, im.size
                        out.paste(im,(x*164+(164-im.width)//2,20+164*k+(164-im.height)//2))
        draw.text((2,350),'VV top, VH bottom | '+key+' '+es[0]['axis']+' '+es[0]['cat'],fill='white')
        out.save(P/f'panel_{n:02d}_local.png')
    print(json.dumps(result,indent=1))
if __name__=='__main__': main()
