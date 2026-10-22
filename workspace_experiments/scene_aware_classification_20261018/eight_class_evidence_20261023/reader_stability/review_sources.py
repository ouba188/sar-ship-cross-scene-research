"""Review source examples without portraying in-sample gains as transfer gains."""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
LOCAL=PARENT.parents[1]/'same_vessel_gallery_20261019/local_condition_module'
sys.path.insert(0,str(LOCAL))
from native_input import crop_pad
from prepare import masks_for


def main():
    source=pd.read_csv(HERE/'source_events.csv')
    cohort=pd.read_csv(PARENT/'cohort.csv')
    lookup=cohort.reset_index().set_index('instance_id')
    valid=np.load(PARENT/'features.npz',allow_pickle=True)['valid']
    snapshot={v['obs_id']:v for v in json.loads((LOCAL/'view_snapshot.json').read_text(encoding='utf-8'))}
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13)
    rows=[]
    for a,b in [(True,True),(False,False),(False,True),(True,False)]:
        r=source[source.inner_correct.eq(a)&source.outer_in_sample_correct.eq(b)].sort_values(['instance_id','fold']).iloc[0]
        meta=lookup.loc[r.instance_id]
        panel=Image.new('RGB',(740,320),'white')
        draw=ImageDraw.Draw(panel)
        draw.text((8,5),f'{r.instance_id} {meta.class_name} source fold{r.fold} inner{r.inner}',font=font,fill='black')
        draw.text((8,24),f'Correct: held={a}, outer in-sample={b}; weighted sign mismatch={r.mass_weighted_sign_disagreement:.3f}',font=font,fill='black')
        draw.text((8,43),'Same pixels; different source readers. This is NOT a generalization rescue/harm experiment.',font=font,fill='black')
        d=np.load(PARENT/'baseline'/f'fold{r.fold}_inputs.npz')
        i=int(meta['index'])
        for j,pol in enumerate(['vv','vh']):
            original=np.asarray(Image.open(meta['image_'+pol]).convert('L'))
            pixels,masks,placement=crop_pad(original,masks_for(snapshot[int(meta.obs_id)]['images'][pol]),160)
            draw.text((8+j*200,66),pol.upper()+(' used' if valid[i,j] else ' NOT USED'),font=font,fill='black')
            panel.paste(Image.fromarray(pixels).convert('RGB'),(8+j*200,88))
        draw.text((8,262),f'Predictions: held={d["oof"][i].argmax()}, outer in-sample={d["base"][i].argmax()}; truth={r.class_id}',font=font,fill='black')
        draw.text((8,284),'Actual 160 crop/pad, no resizing. Original geometry and channel-validity flags unchanged.',font=font,fill='black')
        panel.save(HERE/f'case_{r.instance_id}.png')
        rows.append(dict(r,obs_id=int(meta.obs_id),image_vv=meta.image_vv,image_vh=meta.image_vh,
                         vh_valid=bool(valid[i,1]),inner_prediction=int(d['oof'][i].argmax()),outer_prediction=int(d['base'][i].argmax())))
    pd.DataFrame(rows).to_csv(HERE/'review_cases.csv',index=False)
    print(pd.DataFrame(rows)[['instance_id','vh_valid','inner_prediction','outer_prediction']].to_string(index=False))


if __name__=='__main__':
    main()
