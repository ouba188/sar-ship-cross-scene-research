"""CPU statistics and native-input panels for the fixed computational replay."""
import csv
import html
import importlib.util
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent.parent
SEEDS=(17,29,43)
ARMS=('factual','no_m3','no_vv','no_vh','no_body','no_side','no_ends','unit_metric','no_competition','simple_mean')
LABELS=('Bulk carrier','General cargo','Chemical/product tanker','Container ship')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def table(name,rows):
    with (ROOT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def margin(z,y):
    wrong=z.copy();wrong[np.arange(len(y)),y]=-np.inf
    return z[np.arange(len(y)),y]-wrong.max(1)


def metrics(z,y):
    pred=z.argmax(1)
    cm=np.bincount(y*4+pred,minlength=16).reshape(4,4)
    prob=np.exp(z-z.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
    return dict(oa=100*(pred==y).mean(),ba=100*np.mean(np.diag(cm)/cm.sum(1)),
        macro_f1=100*np.mean(2*np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1),1)),
        nll=float(-np.log(prob[np.arange(len(y)),y].clip(1e-12)).mean()),
        margin=float(margin(z,y).mean()))


def legal(query,ref):
    return query['mmsi']!=ref['mmsi'] and '_'.join(query['pass_group'].split('_')[:2])!='_'.join(ref['pass_group'].split('_')[:2])


def native_api():
    spec=importlib.util.spec_from_file_location('cf_native',WORK/'sar_m123_standalone/sar_m123/input.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def font(size): return ImageFont.truetype('C:/Windows/Fonts/arial.ttf',size)


def draw_panel(case,rows,native):
    # Native160 pixels, without radiometric enhancement or resizing.
    canvas=Image.new('RGB',(900,790),'white');d=ImageDraw.Draw(canvas)
    d.text((15,10),f"{case['kind']} | seed {case['seed']} | query {case['id']}",font=font(18),fill='black')
    d.text((15,36),f"Truth: {LABELS[case['label']]} | unchanged native160 inputs; same model/32 references",font=font(15),fill='black')
    specimens=[('Query',case['index']),('Same-class reference',case['same_reference']),('Competing-class reference',case['foil_reference'])]
    for col,(role,idx) in enumerate(specimens):
        left=20+col*295
        if idx is None:
            d.text((left,65),role+' unavailable',font=font(14),fill='black');continue
        row=rows[idx];views=native.prepare(row)
        d.text((left,63),role,font=font(15),fill='black')
        d.text((left,84),row['id'].replace('pub2_opensarship2_',''),font=font(14),fill='black')
        d.text((left,104),LABELS[row['label']],font=font(14),fill='black')
        for p,view in enumerate(views):
            top=145+p*188
            d.text((left,top-20),('VV','VH')[p],font=font(14),fill='black')
            if view is None or not view['valid_target']:
                d.rectangle((left,top,left+159,top+159),fill='#eeeeee')
                d.text((left+12,top+65),'Not used',font=font(14),fill='black')
            else:
                tile=Image.fromarray(np.asarray(view['pixels'],np.uint8)).convert('RGB')
                canvas.paste(tile,(left,top))
    d.text((20,510),'True-class margin: correct class minus strongest wrong class (recomputed per intervention)',font=font(15),fill='black')
    values=case['margins'];bound=max(.05,max(abs(v) for v in values.values()))
    center=555;scale=235/bound
    d.line((center,540,center,760),fill='#aaaaaa',width=1)
    for j,arm in enumerate(ARMS):
        y=540+j*22;v=values[arm];end=center+v*scale
        d.text((20,y),arm,font=font(14),fill='black')
        d.rectangle((min(center,end),y+2,max(center,end),y+15),fill='#137f73' if v>0 else '#bd4d4d')
        d.text((805,y),f'{v:+.4f}',font=font(14),fill='black')
    d.text((20,772),'References: highest mean beta within each legal class; displayed weights are not causal attribution.',font=font(12),fill='#555555')
    path=ROOT/'gallery'/f"{case['kind']}_s{case['seed']}_{case['index']}.png"
    canvas.save(path);return path.name


def main():
    (ROOT/'gallery').mkdir(exist_ok=True)
    rows=read(WORK/'m1_coverage_20261101/manifest.json')
    groups=read(WORK/'m1_coverage_20261101/groups.json')
    records=[];perclass=[];events=[];pool=[];datasets={}
    for seed in SEEDS:
        receipt=read(ROOT/f'private/seed{seed}_receipt.json')
        assert receipt['n']==378 and receipt['actual_factual_arm']=='aggregate'
        p=dict(np.load(ROOT/f'private/seed{seed}.npz'));datasets[seed]=p
        ids,y=p['ids'],p['label']
        assert np.array_equal(ids,groups['evaluation'])
        assert np.array_equal(y,[rows[int(i)]['label'] for i in ids])
        assert all(np.isfinite(p[a]).all() for a in ARMS)
        good=p['factual'].argmax(1)==y; off=p['no_m3'].argmax(1)==y
        rescue=good&~off; harm=~good&off
        for a in ARMS:
            correct=p[a].argmax(1)==y
            records.append(dict(seed=seed,arm=a,**metrics(p[a],y),
                factual_wins=int((good&~correct).sum()),alternative_wins=int((~good&correct).sum()),
                m3_rescues=int(rescue.sum()),m3_harms=int(harm.sum()),
                rescues_lost=int((rescue&~correct).sum())))
            for c in range(4):
                mask=y==c
                perclass.append(dict(seed=seed,arm=a,label=c,class_name=LABELS[c],n=int(mask.sum()),
                    recall=100*correct[mask].mean(),margin=float(margin(p[a],y)[mask].mean())))
        for j,idx in enumerate(ids):
            kind='rescued' if rescue[j] else 'harmed' if harm[j] else 'stable_correct' if good[j] else 'stable_wrong'
            margins={a:float(margin(p[a][j:j+1],y[j:j+1])[0]) for a in ARMS}
            event=dict(seed=seed,index=int(idx),id=rows[int(idx)]['id'],label=int(y[j]),kind=kind,
                factual_prediction=int(p['factual'][j].argmax()),off_prediction=int(p['no_m3'][j].argmax()),
                simple_prediction=int(p['simple_mean'][j].argmax()),**margins)
            events.append(event)
            pool.append(dict(event=event,j=j,margins=margins,
                dependency=bool(rescue[j] and p['simple_mean'][j].argmax()!=y[j]),
                effect=abs(margins['factual']-margins['no_m3'])))
    table('metrics_by_seed.csv',records);table('per_class.csv',perclass);table('private/events.csv',events)
    summary=[]
    for arm in ARMS:
        part=[x for x in records if x['arm']==arm]
        summary.append(dict(arm=arm,**{m:float(np.mean([x[m] for x in part])) for m in ('oa','ba','macro_f1','nll','margin')},
            oa_sd=float(np.std([x['oa'] for x in part],ddof=1)),ba_sd=float(np.std([x['ba'] for x in part],ddof=1)),
            factual_wins=sum(x['factual_wins'] for x in part),alternative_wins=sum(x['alternative_wins'] for x in part),
            rescues_lost=sum(x['rescues_lost'] for x in part)))
    table('summary.csv',summary)
    selected=[];used=set()
    def take(candidates,count,kind=None):
        for item in sorted(candidates,key=lambda x:(-x['effect'],x['event']['seed'],x['event']['index'])):
            if item['event']['index'] in used:continue
            used.add(item['event']['index']);selected.append((item,kind or item['event']['kind']));count-=1
            if count==0:break
    take([x for x in pool if x['dependency']],2,'conditional_rescue')
    for kind,count in (('rescued',3),('harmed',3),('stable_correct',2),('stable_wrong',2)):
        take([x for x in pool if x['event']['kind']==kind],count)
    native=native_api();cases=[];htmlparts=[]
    for item,kind in selected:
        event=item['event'];seed=event['seed'];idx=event['index'];j=item['j'];p=datasets[seed]
        y=event['label'];z=p['no_m3'][j].copy();z[y]=-np.inf;foil=int(z.argmax())
        meanbeta=p['beta'][j].mean((0,1))
        refs=[]
        for cls in (y,foil):
            legal_candidates=[k for k,i in enumerate(p['reference_ids'])
                if rows[int(i)]['label']==cls and legal(rows[idx],rows[int(i)]) and meanbeta[k]>0]
            refs.append(int(p['reference_ids'][max(legal_candidates,key=lambda k:meanbeta[k])]) if legal_candidates else None)
        case=dict(seed=seed,index=idx,id=event['id'],label=y,kind=kind,margins=item['margins'],
                  same_reference=refs[0],foil_reference=refs[1],foil=foil,
                  predictions={a:int(p[a][j].argmax()) for a in ARMS},
                  review='pending_native_image_review')
        case['panel']=draw_panel(case,rows,native);cases.append(case)
        cells=''.join(f'<tr><td>{a}</td><td>{LABELS[case["predictions"][a]]}</td><td>{case["margins"][a]:+.5f}</td></tr>' for a in ARMS)
        htmlparts.append(f'<section><h2>{html.escape(kind)}: {html.escape(event["id"])} / seed {seed}</h2>'
            f'<img src="gallery/{case["panel"]}"><table><tr><th>操作</th><th>预测</th><th>真实类分差</th></tr>{cells}</table></section>')
    write(ROOT/'private/cases.json',cases)
    header=''.join(f'<tr><td>{x["arm"]}</td><td>{x["oa"]:.3f}</td><td>{x["ba"]:.3f}</td><td>{x["rescues_lost"]}</td></tr>' for x in summary)
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>M3个体反事实证据</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:0 18px;color:#222}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px 12px;text-align:left}section{border-top:1px solid #aaa;margin-top:30px;padding-top:12px}img{max-width:100%;height:auto}p{line-height:1.7}</style>'
    page+='<h1>M3固定模型个体反事实证据</h1><p>OSS2四分类，378张相同测试图，三个冻结微调模型，32个源参考。模型内部证据替换，不是物理成像反事实。当前事实路径为聚合版回写，不是精细运输回写。</p>'
    page+='<p>23次救回、19次误伤；三种子是同一批测试船的重复模型。simple_mean仍保留编码器、类别证据比较和输出映射，替代的是条件化参考组织。案例规则：优先两例事实正确而no_m3及simple_mean均错误；再按绝对分差改变量选择各组，查询去重，不代表组内频率。原图人工审阅状态见FINDINGS_ZH.md。</p>'
    page+='<table><tr><th>操作</th><th>OA</th><th>BA</th><th>23次救回中撤销次数</th></tr>'+header+'</table>'+''.join(htmlparts)+'</html>'
    (ROOT/'index.html').write_text(page,encoding='utf-8')
    rescue_ids={e['index'] for e in events if e['kind']=='rescued'}
    write(ROOT/'analysis_receipt.json',dict(state='statistics_and_gallery_complete_review_pending',n=378,seeds=list(SEEDS),
        m3_rescue_events=sum(e['kind']=='rescued' for e in events),m3_harm_events=sum(e['kind']=='harmed' for e in events),
        rescued_unique_records=len(rescue_ids),rescued_unique_mmsi=len({rows[i]['mmsi'] for i in rescue_ids}),
        panels=len(cases),training=False,formal_model_changed=False))
    print('COMPLETE',len(cases),'panels',flush=True)


if __name__=='__main__':main()
