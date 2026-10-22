"""Render actual native-scale VV/VH inputs and selected scene detections."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from osgeo import gdal

HERE = Path(__file__).resolve().parent / 'ports24'
FONT = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)


def render(ids, name):
    t = pd.read_csv(HERE/'targets.csv').set_index('instance_id')
    edges = pd.read_csv(HERE/'neighbors.csv')
    canvas = Image.new('RGB', (1160, 290*len(ids)), 'white')
    draw = ImageDraw.Draw(canvas)
    for r, iid in enumerate(ids):
        y = r*290
        row = t.loc[iid]
        draw.text((8,y+3),f'{iid} | {row.port} | {row.class_name} | N={row.neighbors}', fill='black',font=FONT)
        original = Image.open(row.original_chip).convert('RGB')
        canvas.paste(original,(8,y+46))
        draw.text((8,y+25),'original crop VV',fill='black',font=FONT)
        for col,pol in enumerate(('vv','vh')):
            x = 150+col*170
            im = Image.open(HERE/'native_targets'/f'{iid}_{pol}.png').convert('RGB')
            canvas.paste(im,(x,y+46))
            draw.text((x,y+25),'input '+pol.upper(),fill='black',font=FONT)
        for col,pol in enumerate(('vv','vh')):
            x = 510+col*310
            ds = gdal.Open(row['raster_'+pol])
            side = int(np.ceil(2400/row.pixel_m))
            left,top = int(row.x-side//2), int(row.y-side//2)
            a = ds.ReadAsArray(left,top,side,side)
            im = Image.fromarray(a).convert('RGB')
            d = ImageDraw.Draw(im)
            wx,wy,w,h=json.loads(row.target_window)
            d.rectangle((wx-left,wy-top,wx-left+w,wy-top+h),outline='#00ff99',width=1)
            for edge in edges[edges.instance_id.eq(iid)].itertuples():
                pts=np.asarray(json.loads(edge.source_points))-[left,top]
                d.line([tuple(p) for p in np.vstack([pts,pts[0]])], fill='#ffb020',width=1)
                d.text(tuple(pts.mean(0)),str(edge.detection_id),font=FONT,fill='#ffff66')
            canvas.paste(im,(x,y+44))
            draw.text((x,y+25),'scene '+pol.upper()+' | native pixels',fill='black',font=FONT)
            ds = None
    path=HERE/name
    canvas.save(path)
    return str(path)


if __name__=='__main__':
    t=pd.read_csv(HERE/'targets.csv')
    ids=t.sort_values('neighbors',ascending=False).groupby('port',sort=False).head(1).head(4).instance_id.tolist()
    print(render(ids,'input_review.png'))
