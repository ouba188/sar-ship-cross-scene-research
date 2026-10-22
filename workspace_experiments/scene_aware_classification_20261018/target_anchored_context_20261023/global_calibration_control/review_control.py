"""Replay global calibration and inspect representative native outcomes."""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
EXP=PARENT.parent/'context_expansion_20261022'
EIGHT=PARENT.parent/'eight_class_evidence_20261023'
LOCAL=EIGHT.parents[1]/'same_vessel_gallery_20261019/local_condition_module'
sys.path.insert(0,str(LOCAL))
from native_input import crop_pad
from prepare import masks_for


def main():
    p=pd.read_csv(HERE/'predictions.csv')
    small=pd.read_csv(EXP/'readout/targets.csv').set_index('instance_id')
    big=pd.read_csv(EIGHT/'cohort.csv').reset_index().set_index('instance_id')
    valid=np.load(EIGHT/'features.npz',allow_pickle=True)['valid']
    snapshot={v['obs_id']:v for v in json.loads((LOCAL/'view_snapshot.json').read_text(encoding='utf-8'))}
    checks=[]
    for (cohort,fold,arm),part in p.groupby(['cohort','fold','arm']):
        saved=np.load(HERE/f'{cohort}_fold{fold}_{arm}.npz')
        c=8 if cohort=='eight' else 3
        s=part[[f'b{k}' for k in range(c)]].to_numpy()*saved['scale']+saved['bias']
        err=float(np.abs(s-part[[f's{k}' for k in range(c)]].to_numpy()).max())
        assert err<1e-6 and np.array_equal(s.argmax(1),part.prediction),(cohort,fold,arm,err)
        root=EIGHT/'baseline' if cohort=='eight' else (PARENT if cohort=='old' else PARENT/'partition_replication')
        dtype=np.load(root/f'fold{fold}_inputs.npz')['base'].dtype
        exact=part[[f'b{k}' for k in range(c)]].to_numpy().astype(dtype)*float(saved['scale'])+saved['bias']
        exact_error=float(np.abs(exact-part[[f's{k}' for k in range(c)]].to_numpy()).max())
        assert exact_error<1e-10
        checks.append(dict(cohort=cohort,fold=fold,arm=arm,float64_error=err,original_dtype_error=exact_error))
    pd.DataFrame(checks).to_csv(HERE/'replay.csv',index=False)
    p['net']=p.prediction.eq(p.class_id).astype(int)-p.base_prediction.eq(p.class_id).astype(int)
    p['outcome']=np.select([p.net.eq(1),p.net.eq(-1),p.prediction.eq(p.class_id)],['rescue','harm','correct'],default='wrong')
    rows=[]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13)
    for cohort,part in p[p.arm.eq('scale_bias')].groupby('cohort'):
        for outcome in ['rescue','harm','correct','wrong']:
            r=part[part.outcome.eq(outcome)].sort_values('instance_id').iloc[0]
            meta=(big if cohort=='eight' else small).loc[r.instance_id]
            panel=Image.new('RGB',(770,330),'white')
            draw=ImageDraw.Draw(panel)
            draw.text((8,5),f'{cohort} {r.instance_id} {meta.class_name}: {outcome}',font=font,fill='black')
            draw.text((8,25),f'Base {int(r.base_prediction)} -> global scale+bias {int(r.prediction)}; truth={int(r.class_id)}',font=font,fill='black')
            for j,pol in enumerate(['vv','vh']):
                if cohort=='eight':
                    path=meta['image_'+pol]
                    image=np.asarray(Image.open(path).convert('L'))
                    pixels,_,_=crop_pad(image,masks_for(snapshot[int(meta.obs_id)]['images'][pol]),160)
                    image=Image.fromarray(pixels).convert('RGB')
                    used=bool(valid[int(meta['index']),j])
                else:
                    path=str(EXP/'readout/native_targets'/f'{r.instance_id}_{pol}.png')
                    image=Image.open(path).convert('RGB')
                    used=True
                draw.text((8+j*200,53),f'{pol.upper()} {image.width}x{image.height}'+('' if used else ' NOT USED'),font=font,fill='black')
                panel.paste(image,(8+j*200,77))
                rows.append(dict(cohort=cohort,fold=int(r.fold),instance_id=r.instance_id,outcome=outcome,pol=pol,path=path,used=used,
                    class_id=int(r.class_id),base_prediction=int(r.base_prediction),prediction=int(r.prediction)))
            draw.text((8,260),'No spatial or sample-specific conditioning: the same class offsets apply to every sample in a fold.',font=font,fill='black')
            draw.text((8,283),'Pixels unchanged. A flip alone does not demonstrate new local evidence or a physical cause.',font=font,fill='black')
            panel.save(HERE/f'case_{cohort}_{outcome}_{r.instance_id}.png')
    pd.DataFrame(rows).to_csv(HERE/'review_cases.csv',index=False)
    print(pd.DataFrame(rows).drop_duplicates(['cohort','instance_id'])[['cohort','instance_id','outcome','class_id','base_prediction','prediction']].to_string(index=False))


if __name__=='__main__':main()
