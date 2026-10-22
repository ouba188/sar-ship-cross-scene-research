"""Show original pixels beside the exact pooled regions, including disabled VH."""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[1] / 'same_vessel_gallery_20261019'
sys.path.insert(0, str(OLD / 'local_condition_module'))
from native_input import crop_pad
from prepare import masks_for


def main():
    frame = pd.read_csv(HERE/'cohort.csv')
    snapshots = {s['obs_id']: s for s in json.loads((OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    coverage = pd.read_csv(HERE/'coverage.csv').set_index(['instance_id', 'pol'])
    picked = frame.sort_values('obs_id').groupby('class_id', sort=True).head(1)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 13)
    canvas = Image.new('RGB', (740, 40+len(picked)*218), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), 'Native pixels / pooled regions. Green=target; cyan=outside 3px guard of all detected boats.', font=font, fill='black')
    draw.text((8, 21), 'No resizing. Black padding excluded from pooling, not from convolution receptive fields.', font=font, fill='black')
    selected = []
    for j, row in enumerate(picked.itertuples()):
        y = 40+j*218
        draw.text((8,y), f'{row.instance_id} | {row.class_name} | {row.port}', font=font, fill='black')
        for p, pol in enumerate(('vv', 'vh')):
            view = snapshots[row.obs_id]['images'][pol]
            raw = np.asarray(Image.open(getattr(row, 'image_'+pol)).convert('L'))
            pixels, masks, placement = crop_pad(raw, masks_for(view), 160)
            fg = masks[0] > 0
            occupied = torch.from_numpy(masks[:2].sum(0))[None,None]
            dilated = F.max_pool2d(occupied,7,stride=1,padding=3)[0,0].numpy()>0
            bg = (masks.sum(0)>0)&~dilated
            rgb = np.repeat(pixels[...,None],3,-1).astype(float)
            color=np.zeros_like(rgb);color[fg]=[40,255,70];color[bg]=[30,220,255]
            alpha=.25*(fg|bg)[...,None]
            overlay=Image.fromarray(np.uint8(rgb*(1-alpha)+color*alpha))
            valid=bool(coverage.loc[(row.instance_id,pol),'valid'])
            x=8+p*364
            label=pol.upper()+(' native' if valid else ' native (NOT USED)')
            draw.text((x,y+19),label,font=font,fill='black' if valid else '#b91c1c')
            draw.text((x+178,y+19),'readout mask' if valid else 'mask not used',font=font,fill='black')
            canvas.paste(Image.fromarray(pixels).convert('RGB'),(x,y+40))
            canvas.paste(overlay,(x+178,y+40))
            selected.append(dict(instance_id=row.instance_id,obs_id=row.obs_id,class_name=row.class_name,pol=pol,
                                 image_path=getattr(row,'image_'+pol),valid=valid,**placement))
    canvas.save(HERE/'region_coverage.png')
    pd.DataFrame(selected).to_csv(HERE/'review_cases.csv',index=False)


if __name__=='__main__':
    main()
