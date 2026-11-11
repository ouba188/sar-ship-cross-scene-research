"""Paired same-query analysis and raw-chip cases from completed runs only."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent
STORE=Path('C:/Users/25791/AppData/Local/sar_ports24_fullscale_20261110')
sys.path.insert(0,str(ROOT.parents[1]/'sar_m123_standalone'))
from sar_m123.input import prepare


def main():
    rows=json.loads((STORE/'manifest.json').read_text(encoding='utf-8'))
    groups=json.loads((STORE/'groups.json').read_text(encoding='utf-8'))
    protocol=json.loads((STORE/'protocol.json').read_text(encoding='utf-8'))
    ids=np.array(groups['evaluation']);query=[rows[i] for i in ids];y=np.array([r['label'] for r in query])
    predictions={};per_class=[];per_port=[]
    for arm in protocol['arms']:
        for seed in protocol['seeds']:
            d=np.load(STORE/f"{arm}_s{seed}_lr{protocol['scales'][arm]:g}"/'evaluation.npz')
            assert np.array_equal(d['ids'],ids)
            predictions[arm,seed]=d['score']
            right=d['score'].argmax(1)==y
            for cls in range(8):
                mask=y==cls;per_class.append(dict(arm=arm,seed=seed,label=cls,n=int(mask.sum()),recall=100*right[mask].mean()))
            for port in sorted({r['port'] for r in query}):
                mask=np.array([r['port']==port for r in query])
                ba=np.mean([right[mask&(y==cls)].mean() for cls in np.unique(y[mask])])
                per_port.append(dict(arm=arm,seed=seed,port=port,n=int(mask.sum()),oa=100*right[mask].mean(),ba=100*ba))
    pd.DataFrame(per_class).to_csv(ROOT/'per_class.csv',index=False)
    pd.DataFrame(per_port).to_csv(ROOT/'per_port.csv',index=False)
    changes=[];cases=[];used=set();seeds=protocol['seeds']
    ships=np.unique([r['mmsi'] for r in query]);ship_index={v:i for i,v in enumerate(ships)}
    si=np.array([ship_index[r['mmsi']] for r in query]);rng=np.random.default_rng(20261110)
    for baseline in ('plain','se'):
        full=np.stack([predictions['full',s].argmax(1)==y for s in seeds])
        other=np.stack([predictions[baseline,s].argmax(1)==y for s in seeds])
        diff=(full.astype(float)-other).mean(0)
        boot=[]
        for _ in range(1000):
            counts=np.bincount(rng.integers(0,len(ships),len(ships)),minlength=len(ships))[si]
            oa=np.average(diff,weights=counts)
            ba=np.mean([np.average(diff[y==c],weights=counts[y==c]) for c in range(8) if counts[y==c].sum()])
            boot.append([100*oa,100*ba])
        interval=np.quantile(boot,[.025,.975],axis=0)
        changes.append(dict(baseline=baseline,oa_delta=100*diff.mean(),ba_delta=100*np.mean([diff[y==c].mean() for c in range(8)]),
            oa_low=interval[0,0],oa_high=interval[1,0],ba_low=interval[0,1],ba_high=interval[1,1],
            rescue_events=int((full&~other).sum()),harm_events=int((~full&other).sum())))
        categories=[('rescue',full&~other),('harm',~full&other)]
        if baseline=='plain':
            categories.extend([('stable_correct',full&other),('stable_wrong',~full&~other)])
        for kind,mask in categories:
            rank=mask.sum(0);seen_classes=set();picked=0
            for j in sorted(range(len(ids)),key=lambda k:(-int(rank[k]),int(ids[k]))):
                if rank[j]==0 or j in used or y[j] in seen_classes:continue
                seed=next(s for k,s in enumerate(seeds) if mask[k,j]);used.add(j);seen_classes.add(y[j]);picked+=1
                bscore=predictions[baseline,seed][j];fscore=predictions['full',seed][j]
                if kind=='stable_correct':
                    alternatives=fscore.copy();alternatives[y[j]]=-np.inf
                    rival=int(alternatives.argmax())
                else:
                    rival=int(bscore.argmax() if kind=='rescue' else fscore.argmax())
                reference_ids=[next(i for i in groups['references'] if rows[i]['label']==c) for c in (int(y[j]),rival)]
                item=dict(id=query[j]['id'],label=int(y[j]),port=query[j]['port'],mmsi=query[j]['mmsi'],
                    baseline=baseline,kind=kind,seed=seed,repeated=int(rank[j]),baseline_score=bscore.tolist(),full_score=fscore.tolist(),
                    references=[rows[i]['id'] for i in reference_ids])
                canvas=Image.new('RGB',(6*178,208),'white');draw=ImageDraw.Draw(canvas)
                for group,row in enumerate([query[j]]+[rows[i] for i in reference_ids]):
                    for pol,view in enumerate(prepare(row)):
                        x=(group*2+pol)*178
                        draw.text((x+2,2),f"{row['id']} C{row['label']} {('VV','VH')[pol]}",fill='black')
                        if view is not None and view['valid_target']:
                            im=np.repeat(view['pixels'][...,None],3,axis=2)
                            m=view['masks'][0].astype(bool)
                            edge=m&~(np.roll(m,1,0)&np.roll(m,-1,0)&np.roll(m,1,1)&np.roll(m,-1,1))
                            im[edge]=(0,255,0);canvas.paste(Image.fromarray(im),(x+2,25))
                        else:draw.text((x+8,70),'Unavailable',fill='red')
                dest=ROOT/'cases';dest.mkdir(exist_ok=True)
                item['panel']=f"cases/{baseline}_{kind}_{item['id']}.png";canvas.save(ROOT/item['panel']);cases.append(item)
                if picked==2:break
    pd.DataFrame(changes).to_csv(ROOT/'paired_changes.csv',index=False)
    (ROOT/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(changes).to_string(index=False))
    html=['<!doctype html><meta charset="utf-8"><title>24港全量源训练案例</title><style>body{font-family:Arial;max-width:1150px;margin:24px auto}img{max-width:100%}section{margin-bottom:35px}</style>',
          '<h1>全量合法源训练：完整模型与对照</h1><p>每图从左到右：查询VV/VH、同类源参考VV/VH、混淆类源参考VV/VH。参考按类取固定库首例，仅说明实际参考内容，不宣称最高贡献。</p>']
    for case in cases:
        html.append(f"<section><h2>{case['id']} / {case['port']} / {case['baseline']} / {case['kind']}</h2><p>种子{case['seed']}；三种子中{case['repeated']}次；真实类{case['label']}；对照预测{np.argmax(case['baseline_score'])}，完整模型{np.argmax(case['full_score'])}</p><img src='{case['panel']}'></section>")
    (ROOT/'index.html').write_text('\n'.join(html),encoding='utf-8')


if __name__=='__main__':main()
