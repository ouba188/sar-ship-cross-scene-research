"""Native query/reference evidence panels; no model execution or image rescaling."""
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
PAPER,WORK=ROOT.parent,ROOT.parent.parent
NAMES=('Bulk','General cargo','Chemical/product tanker','Container')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def change(x,m,valid,idx,name):
    out=x.copy()
    if name=='clean':return out
    p=0 if name.startswith('VV') else 1
    if 'mean50_keep' in name:
        keep=int(name[-1])
        for r in range(3):
            if not keep&(1<<r) and m[p,r].any():
                out[p,m[p,r]]=.5*x[p,m[p,r]]+.5*x[p,m[p,r]].mean()
    elif 'gain' in name:
        area=valid[p].astype(bool)&~m[p,0]
        out[p,area]*=float(name.split('gain')[1])
    elif 'permute' in name:
        r=('body','ends','side').index(name.split('_')[1])
        ix=np.flatnonzero(m[p,r])
        order=np.random.default_rng(16000+int(idx)*6+p*3+r).permutation(len(ix))
        out[p].flat[ix]=x[p].flat[ix[order]]
    return out


def margin(z,y):
    rival=z.copy();rival[y]=-np.inf
    return float(z[y]-rival.max()),int(rival.argmax())


def main():
    groups=read(WORK/'m1_coverage_20261101/groups.json')
    rows=read(WORK/'m1_coverage_20261101/manifest.json')
    ids=np.asarray(groups['validation']);refs=groups['references']
    y=np.asarray([rows[i]['label'] for i in ids])
    cache=PAPER/'recent_baselines_20261111/native_cache'
    pixels=np.load(cache/'pixels.npy',mmap_mode='r')
    valid=np.load(cache/'footprint.npy',mmap_mode='r')
    av=np.load(cache/'available.npy')
    masks=np.load(ROOT/'native_regions.npy',mmap_mode='r')
    folder=ROOT/'predictions/full_s17'
    original=np.load(folder/'clean.npz')['score']
    clean=original.argmax(1)
    candidates=[]
    # Predeclared structural controls, with median-margin examples, not best-looking cases.
    for condition in ('VV_mean50_keep3','VH_mean50_keep3','VV_body_permute','VH_body_permute'):
        z=np.load(folder/(condition+'.npz'))['score']
        pred=z.argmax(1)
        p=0 if condition.startswith('VV') else 1
        r=2 if 'keep3' in condition else 0
        live=(av[ids,p]>0)&masks[:,p,r].any((-2,-1))
        states={'rescued':(clean!=y)&(pred==y),'harmed':(clean==y)&(pred!=y),
                'stable_correct':(clean==y)&(pred==y),'stable_wrong':(clean!=y)&(pred!=y)}
        for state in ('rescued','harmed'):
            members=np.flatnonzero(states[state]&live)
            if len(members):
                delta=np.array([margin(z[j],y[j])[0]-margin(original[j],y[j])[0] for j in members])
                j=int(members[np.argsort(abs(delta),kind='stable')[len(members)//2]])
                candidates.append((condition,state,j,z))
        if condition=='VH_mean50_keep3':
            for state in ('stable_correct','stable_wrong'):
                members=np.flatnonzero(states[state]&live)
                if len(members):candidates.append((condition,state,int(members[len(members)//2]),z))
    out=ROOT/'gallery';out.mkdir(exist_ok=True)
    manifest=[]
    for condition,state,j,z in candidates:
        idx=int(ids[j]);row=rows[idx];label=int(y[j])
        oldmargin,rival=margin(original[j],label)
        newmargin,newrival=margin(z[j],label)
        orbit=lambda item:'_'.join(item['pass_group'].split('_')[:2])
        legal=[i for i in refs if rows[i]['mmsi']!=row['mmsi'] and orbit(rows[i])!=orbit(row)]
        bank=[]
        for c in (label,rival):
            members=[i for i in legal if rows[i]['label']==c]
            bank.append(next((i for i in members if av[i].all()),members[0] if members else None))
        x=np.asarray(pixels[idx],dtype=np.float32)/255
        v=np.asarray(valid[idx]);m=np.asarray(masks[j]);changed=change(x,m,v,idx,condition)
        fig,axes=plt.subplots(2,6,figsize=(15,7.6))
        for p in range(2):
            for ax in axes[p]:ax.set_axis_off()
            boxes=np.argwhere(v[p]>0)
            sl=(slice(None),slice(None)) if not len(boxes) else (slice(boxes[:,0].min(),boxes[:,0].max()+1),slice(boxes[:,1].min(),boxes[:,1].max()+1))
            for col,image in ((0,x[p]),(1,x[p]),(2,changed[p])):
                axes[p,col].imshow(image[sl],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
            colors=np.zeros((160,160,4))
            for r,color in enumerate(((.1,.7,.5,.25),(.9,.5,.1,.3),(.3,.5,.9,.3))):colors[m[p,r]]=color
            axes[p,1].imshow(colors[sl],interpolation='nearest')
            diff=changed[p]-x[p];scale=max(float(abs(diff).max()),1/255)
            axes[p,3].imshow(diff[sl],cmap='coolwarm',vmin=-scale,vmax=scale,interpolation='nearest')
            axes[p,3].set_title(f'delta, +/-{scale:.3f}',fontsize=9)
            for col,ref in enumerate(bank,4):
                if ref is not None:
                    vv=np.asarray(valid[ref,p]);points=np.argwhere(vv>0)
                    if len(points):
                        rr=slice(points[:,0].min(),points[:,0].max()+1);cc=slice(points[:,1].min(),points[:,1].max()+1)
                        axes[p,col].imshow(pixels[ref,p][rr,cc],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                    axes[p,col].set_title(('same-class' if col==4 else 'rival-class')+' ref\n'+rows[ref]['id'].split('_')[-1],fontsize=9)
            axes[p,0].set_title(('VV' if p==0 else 'VH')+f' original ({int(av[idx,p])} available)',fontsize=10)
            axes[p,1].set_title('body / ends / side',fontsize=10)
            axes[p,2].set_title('operated input',fontsize=10)
        fig.suptitle(f"{row['id']} | {state} | seed17 | true={NAMES[label]}\n{condition}: {NAMES[clean[j]]} -> {NAMES[z[j].argmax()]} | strongest-rival margin {oldmargin:+.3f} -> {newmargin:+.3f}",fontsize=11)
        fig.text(.5,.02,'Padding trimmed for display only; native pixels retained, no per-panel stretch. Reference examples are actual legal bank members, not ranked influence.',ha='center',fontsize=8)
        fig.subplots_adjust(left=.015,right=.99,bottom=.10,top=.84,wspace=.20,hspace=.34)
        stem=f'{condition}_{state}_{idx}'
        fig.savefig(out/(stem+'.png'),dpi=160);fig.savefig(out/(stem+'.svg'));plt.close(fig)
        path_effects=[]
        if 'mean50' in condition:
            for arm in ('full','m1_feedback_off','m2_write_off','m2_uniform','m3_off'):
                af=ROOT/'predictions'/f'{arm}_s17'
                before=np.load(af/'clean.npz')['score'][j]
                after=np.load(af/(condition+'.npz'))['score'][j]
                path_effects.append(dict(arm=arm,clean_margin=margin(before,label)[0],
                    operated_margin=margin(after,label)[0],clean_prediction=NAMES[before.argmax()],
                    operated_prediction=NAMES[after.argmax()]))
        manifest.append(dict(id=row['id'],index=idx,label=label,seed=17,state=state,condition=condition,
            mmsi=row['mmsi'],pass_group=row['pass_group'],old_score=original[j].tolist(),new_score=z[j].tolist(),
            old_margin=oldmargin,new_margin=newmargin,old_rival=rival,new_rival=newrival,
            reference_indices=bank,reference_ids=[rows[i]['id'] if i is not None else None for i in bank],
            input_sources=row.get('source_paths'),path_effects=path_effects,image=stem+'.png',svg=stem+'.svg'))
    (out/'cases.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    body=['<!doctype html><meta charset="utf-8"><title>Regional dependency evidence</title>',
          '<style>body{font:16px system-ui;max-width:1500px;margin:24px auto}img{width:100%}section{border-top:1px solid #bbb;padding:20px 0}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #bbb}</style>',
          '<h1>区域证据依赖：固定模型输入操作</h1><p>OSS2验证集；seed17；原图与操作图使用同一模型。均值衰减与像素置换不是相同扰动剂量。图中参考仅显示实际库成员，不声称是最高贡献案例。</p>']
    body.extend(['<p><a href="../FINDINGS_ZH.md">研究结论</a> | <a href="../CASES_ZH.md">原图案例分析</a></p>',
        '<h2>三个种子的总体诊断</h2><p>极化效应矩阵限定到同一批双极化可用样本；路径结果为固定权重关闭，不是重新训练消融。</p>',
        '<img src="../paired_polarization_effects.png"><p><a href="../paired_polarization_effects.svg">可编辑极化效应矩阵</a></p>',
        '<img src="../path_net_effects.png"><p><a href="../path_net_effects.svg">可编辑路径效应图</a></p>'])
    for case in manifest:
        body.append(f'<section><h2>{html.escape(case["id"])} / {case["state"]}</h2><p>{case["condition"]}</p><img src="{case["image"]}"><a href="{case["svg"]}">Editable SVG</a></section>')
        if case['path_effects']:
            body.append('<table><tr><th>固定权重路径</th><th>原图分差</th><th>操作后分差</th><th>原图预测</th><th>操作后预测</th></tr>')
            for e in case['path_effects']:
                body.append(f'<tr><td>{e["arm"]}</td><td>{e["clean_margin"]:+.3f}</td><td>{e["operated_margin"]:+.3f}</td><td>{e["clean_prediction"]}</td><td>{e["operated_prediction"]}</td></tr>')
            body.append('</table><p>路径撤除不是重新训练消融；这里观察同一已训练网络是否依赖该路径。</p>')
    (out/'index.html').write_text('\n'.join(body),encoding='utf-8')
    print(json.dumps(dict(cases=len(manifest),path=str(out/'index.html'))))


if __name__=='__main__':main()
