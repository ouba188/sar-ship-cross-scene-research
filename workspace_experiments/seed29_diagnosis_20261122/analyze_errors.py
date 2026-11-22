"""Paired validation errors and native SAR panels; no model/test access."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent/'oss2_three_arm_experiment_20261122'
CLASSES = ('bulk_carrier','general_cargo','product_chemical_tanker','container_ship')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def main():
    records = {r['id']:r for r in read(EXP/'experiment_manifest_local.json')
               if r['development_split'] in ('val','validation')}
    atoms = {r['id']:r for r in read(EXP/'atom_metadata_local.json') if r['development_split'] in ('val','validation')}
    assert len(records)==len(atoms)==227
    inputs={}; predictions={}
    for seed in (17,29):
        for arm in ('baseline','disentangle'):
            path=EXP/'runs'/f'seed{seed}'/f'{arm}_base_validation.json'
            rows=read(path)['predictions']
            predictions[seed,arm]={r['id']:r for r in rows}
            assert len(rows)==227 and set(predictions[seed,arm])==set(records)
            assert all(str(r['label'])==str(records[r['id']]['label']) for r in rows)
            inputs[f'seed{seed}_{arm}']=sha(path)
    passes=sorted({r['pass_group'] for r in records.values()})
    groups={'all':sorted(records)}
    for k,name in enumerate(CLASSES): groups['class/'+name]=[i for i,r in records.items() if int(r['label'])==k]
    for name in ('1','2','more_than_2'):
        groups['atoms/'+name]=[i for i in records if str(atoms[i]['atom_count'])==name or (name=='more_than_2' and atoms[i]['atom_count']>2)]
    for k,p in enumerate(passes): groups[f'acquisition/{k+1}']=[i for i,r in records.items() if r['pass_group']==p]
    for k,p in enumerate(passes):
        for c,name in enumerate(CLASSES):
            groups[f'acquisition_class/{k+1}/{name}']=[i for i,r in records.items() if r['pass_group']==p and int(r['label'])==c]
    correct={(seed,arm):{i:str(r['prediction'])==str(r['label']) for i,r in pred.items()} for (seed,arm),pred in predictions.items()}
    def summarize(seed,ids):
        b,d=correct[seed,'baseline'],correct[seed,'disentangle']
        nb,nd=sum(b[i] for i in ids),sum(d[i] for i in ids)
        return dict(n=len(ids),baseline_correct=nb,disentangle_correct=nd,
            harmed=sum(b[i] and not d[i] for i in ids),rescued=sum(not b[i] and d[i] for i in ids),
            delta_accuracy_pp=100*(nd-nb)/len(ids) if ids else None)
    summaries={name:{str(seed):summarize(seed,ids) for seed in (17,29)} for name,ids in groups.items()}
    harmed={seed:{i for i in records if correct[seed,'baseline'][i] and not correct[seed,'disentangle'][i]} for seed in (17,29)}
    rescued={seed:{i for i in records if not correct[seed,'baseline'][i] and correct[seed,'disentangle'][i]} for seed in (17,29)}
    def transitions(ids):
        counts=Counter((CLASSES[int(records[i]['label'])],CLASSES[int(predictions[29,'disentangle'][i]['prediction'])]) for i in ids)
        return [dict(true_class=a,disentangle_prediction=b,count=n) for (a,b),n in sorted(counts.items())]
    result=dict(scope='Paired validation error diagnosis; descriptive strata, no causal attribution',test_access=False,
        source_prediction_sha256=inputs,groups=summaries,
        class_order=list(CLASSES),seed29_harmed_transitions=transitions(harmed[29]),
        seed29_rescued_transitions=transitions(rescued[29]),
        overlap=dict(harmed_both=len(harmed[17]&harmed[29]),harmed_only29=len(harmed[29]-harmed[17]),
                     rescued_both=len(rescued[17]&rescued[29])),
        ports='All validation ports unknown; acquisition strata are not port labels')
    write(ROOT/'aggregate_error_diagnosis.json',result)
    private=ROOT/'cases_local'; private.mkdir(exist_ok=True)
    chosen=[]
    for kind,pool in (('harmed',harmed[29]),('rescued',rescued[29])):
        for cls in range(4):
            # Fixed class-stratified ID order, not selection by visual appearance.
            for sid in sorted(i for i in pool if int(records[i]['label'])==cls)[:2 if kind=='harmed' else 1]:
                chosen.append(dict(id=sid,kind=kind,true_class=CLASSES[cls],
                    baseline_prediction=CLASSES[int(predictions[29,'baseline'][sid]['prediction'])],
                    disentangle_prediction=CLASSES[int(predictions[29,'disentangle'][sid]['prediction'])],
                    atom_count=atoms[sid]['atom_count'],harmed_seed17=sid in harmed[17]))
    write(private/'case_index_local.json',chosen)
    from PIL import Image,ImageDraw
    for start in range(0,len(chosen),4):
        subset=chosen[start:start+4]
        sheet=Image.new('RGB',(1050,260*len(subset)),(245,245,245)); draw=ImageDraw.Draw(sheet)
        for pos,case in enumerate(subset):
            rec=records[case['id']]; y=pos*260
            text=f"{start+pos+1}. {case['kind']}  {case['true_class']} -> {case['disentangle_prediction']} | atoms={case['atom_count']}"
            draw.text((10,y+5),text,fill='black')
            draw.text((10,y+23),f"Native pixels, nearest display enlargement; orange = target OBB. Baseline={case['baseline_prediction']}",fill='black')
            for p,pol in enumerate(('vv','vh')):
                spec=rec[pol]
                with Image.open(spec['image']) as im: native=im.convert('RGB')
                width,height=native.size
                overlay=ImageDraw.Draw(native)
                axis=spec['ship_axis']; cx,cy=axis['center']; c,s=math.cos(axis['angle']),math.sin(axis['angle'])
                corners=[(cx+c*u-s*v,cy+s*u+c*v) for u,v in
                         [(-axis['length_px']/2,-axis['width_px']/2),(axis['length_px']/2,-axis['width_px']/2),
                          (axis['length_px']/2,axis['width_px']/2),(-axis['length_px']/2,axis['width_px']/2)]]
                overlay.line(corners+[corners[0]],fill=(255,150,0),width=1)
                factor=max(1,min(4,185//max(width,height)))
                shown=native.resize((width*factor,height*factor),Image.Resampling.NEAREST)
                x=20+p*510; sheet.paste(shown,(x,y+65)); draw.text((x,y+46),f'{pol.upper()} native {width}x{height}',fill='black')
        sheet.save(private/f'page_{start//4+1:02d}_local.png')
    print(json.dumps(dict(seed29=summaries['all']['29'],class_groups={k:v['29'] for k,v in summaries.items() if k.startswith('class/')},
        atoms={k:v['29'] for k,v in summaries.items() if k.startswith('atoms/')},overlap=result['overlap'],panels=(len(chosen)+3)//4),indent=2))


if __name__=='__main__': main()
