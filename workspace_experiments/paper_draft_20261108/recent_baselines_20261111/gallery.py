"""Native-pixel success/failure comparison, using frozen predictions only."""
import html
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from data import ROOT,records,native_module

LABELS=['Bulk carrier','General cargo','Chemical/product tanker','Container ship']


def draw_case(row,folder,name,caption,reference_rows):
    native=native_module()
    selected=[row]+reference_rows
    canvas=Image.new('RGB',(max(900,len(selected)*200),455),'white')
    draw=ImageDraw.Draw(canvas)
    draw.text((10,5),caption,fill='black')
    paths=[]
    for j,r in enumerate(selected):
        x=10+j*200
        draw.text((x,35),('Query ' if j==0 else 'Source ')+r['id'].split('_')[-1],fill='black')
        draw.text((x,50),LABELS[r['label']],fill='black')
        for p,pol in enumerate(('vv','vh')):
            y=75+185*p
            view=native.prepare_view(r.get(pol))
            draw.text((x,y),pol.upper(),fill='black')
            if view is None or not view['valid_target']:
                draw.text((x+20,y+65),'Unavailable',fill='gray');continue
            im=Image.fromarray(view['pixels']).convert('RGB')
            # Keep every displayed source pixel at 1:1, without new contrast stretch.
            canvas.paste(im,(x,y+16))
            paths.append(dict(id=r['id'],pol=pol,path=r[pol]['image']))
    canvas.save(folder/name)
    return paths


def main():
    rows,groups=records();ids=groups['evaluation'];y=np.array([rows[i]['label'] for i in ids])
    selected=json.loads((ROOT/'selected.json').read_text(encoding='utf-8'));lr=selected['lr']
    full=[];base=[]
    for seed in (17,29,43):
        a=np.load(ROOT.parent/'staged_ablation_20261109'/f'full_s{seed}_lr1/evaluation.npz')
        b=np.load(ROOT/f'cpinet_s{seed}_lr{lr:g}/evaluation.npz')
        assert np.array_equal(a['ids'],b['ids'])
        full.append(a['score']);base.append(b['score'])
    fc=np.stack([x.argmax(1)==y for x in full]);bc=np.stack([x.argmax(1)==y for x in base])
    case_masks={'full_rescue':fc&~bc,'cpinet_rescue':bc&~fc,'both_correct':fc&bc,'both_wrong':~fc&~bc}
    folder=ROOT/'cases';folder.mkdir(exist_ok=True)
    cases=[];used=set()
    for kind,mask in case_masks.items():
        for c in range(4):
            count=mask.sum(0);valid=np.flatnonzero((y==c)&(count>0))
            valid=sorted(valid,key=lambda j:(-count[j],rows[ids[j]]['id']))
            if not valid:continue
            j=next((j for j in valid if j not in used),valid[0]);used.add(j)
            k=int(np.flatnonzero(mask[:,j])[0]);seed=(17,29,43)[k]
            row=rows[ids[j]];fs,bs=full[k][j],base[k][j]
            competitor=int(bs.argmax()) if kind=='full_rescue' else int(fs.argmax())
            if competitor==c:competitor=int(np.where(np.arange(4)==c,-np.inf,bs).argmax())
            refs=[]
            for label in (c,competitor):
                legal=[rows[i] for i in groups['references'] if rows[i]['label']==label
                       and rows[i]['mmsi']!=row['mmsi'] and rows[i]['pass_group']!=row['pass_group']]
                if legal:refs.append(legal[0])
            margin=lambda z:float(z[c]-np.delete(z,c).max())
            name=f'{kind}_{row["id"]}_s{seed}.png'
            caption=f'{kind} | seed {seed} | CPINet: {LABELS[bs.argmax()]} | Full: {LABELS[fs.argmax()]}'
            paths=draw_case(row,folder,name,caption,refs)
            cases.append(dict(kind=kind,id=row['id'],seed=seed,label=c,repeats=int(count[j]),
                cpinet_prediction=int(bs.argmax()),full_prediction=int(fs.argmax()),
                cpinet_margin=margin(bs),full_margin=margin(fs),file='cases/'+name,paths=paths))
    (ROOT/'cases_local.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
    out=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>CPINet与Full原图对照</title>',
         '<style>body{font:16px/1.7 system-ui;max-width:1050px;margin:24px auto;padding:0 16px}img{max-width:100%}section{border-top:1px solid #aaa}</style>',
         '<h1>CPINet独立复现与Full：原始chip对照</h1>',
         '<p>图像为相同native160输入，原像素1:1显示，无额外拉伸。左右为查询、Full固定库中的合法同类及混淆类源参考；参考仅是实际案例示例，不是最大贡献解释。CPINet没有参考库。</p>',
         '<p>分差来自各模型自己的分数尺度，不能直接把两者相减解释为同量纲因果效应。重复次数是三个训练种子中的判对/错状态次数。</p>']
    for c in cases:
        out.extend([f'<section><h2>{c["kind"]} / {c["id"]} / {c["repeats"]}/3</h2>',
                    f'<p>真类：{LABELS[c["label"]]}；seed{c["seed"]}。CPINet分差{c["cpinet_margin"]:+.3f}，Full分差{c["full_margin"]:+.3f}。</p>',
                    f'<img loading="lazy" src="{html.escape(c["file"])}"></section>'])
    (ROOT/'index.html').write_text('\n'.join(out)+'</html>',encoding='utf-8')
    print('GALLERY',len(cases),flush=True)


if __name__=='__main__':main()
