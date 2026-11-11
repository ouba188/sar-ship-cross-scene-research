"""Small preparation gallery, no model imports and no performance claims."""
from pathlib import Path
import csv
import html
import json
import math
import sys

sys.dont_write_bytecode=True
from PIL import Image, ImageDraw
from obb_geometry import conditions, perturb

ROOT=Path(__file__).resolve().parent


def main():
    checks=json.loads((ROOT/'light_checks.json').read_text())
    source=json.loads((ROOT/'native_sources.json').read_text(encoding='utf-8'))
    lookup={(r['id'],r['polarization']):r for r in source}
    cfgs={r['name']:r for r in conditions()}
    rows=list(csv.DictReader((ROOT/'light_native_audit.csv').open(encoding='utf-8')))
    panels=[]
    for item in checks['selected']:
        r=lookup[item['id'],item['polarization']];p=r['placement']
        with Image.open(r['source']) as im:raw=im.convert('L')
        native=Image.new('L',(160,160),0)
        native.paste(raw.crop((p['crop_x'],p['crop_y'],p['crop_x']+p['crop_width'],p['crop_y']+p['crop_height'])),
                     (p['pad_left'],p['pad_top']))
        canvas=Image.new('RGB',(760,max(240,raw.height+65)),'white');draw=ImageDraw.Draw(canvas)
        draw.text((8,5),f'{r["id"]} {r["polarization"]}  L={r["geometry"][2]:.2f}px W={r["geometry"][3]:.2f}px',fill='black')
        canvas.paste(raw.convert('RGB'),(8,38));draw.text((8,22),'Original display chip',fill='black')
        for col,name in enumerate(('zero','across_p10','angle_p05')):
            x=200+col*182
            canvas.paste(native.convert('RGB'),(x,38))
            for g,color in [(r['geometry'],'#00d38a')]+([(perturb(r['geometry'],cfgs[name]),'#ff5757')] if name!='zero' else []):
                cx,cy,l,w,a=g[:5];co,si=math.cos(a),math.sin(a)
                pts=[(x+cx+u*co-v*si,38+cy+u*si+v*co) for u,v in
                     [(-l/2,-w/2),(l/2,-w/2),(l/2,w/2),(-l/2,w/2),(-l/2,-w/2)]]
                draw.line(pts,fill=color,width=1)
            draw.text((x,22),name,fill='black')
            measure=next(v for v in rows if v['id']==r['id'] and v['polarization']==r['polarization']
                         and v['condition']==name and v['region']=='body')
            draw.text((x,204),f'd={float(measure["displacement_px"]):.3f}px',fill='black')
            draw.text((x,220),f'body L1={float(measure["native_l1_px"]):.2f}px2',fill='black')
        filename=f'preview_{r["id"][-6:]}_{r["polarization"]}.png'
        canvas.save(ROOT/filename)
        panels.append((r,filename))
    page=['<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>OBB preparation only</title><style>body{font:16px/1.7 system-ui;max-width:1100px;margin:24px auto;padding:0 18px}img{max-width:100%;height:auto}section{border-top:1px solid #ccc;margin-top:24px}</style>',
        '<h1>OSS2 OBB 鲁棒性：准备状态</h1><p><strong>尚未执行三种子推理，不能据此判断模型鲁棒性。</strong>等待主代理完成训练并明确释放资源。</p>',
        '<p><a href="READINESS_ZH.md">实现、运行命令和未完成项</a> | <a href="light_native_audit.csv">小型原生几何检查</a> | <a href="light_grid_audit.csv">已有网格变化检查</a></p>',
        f'<p>726 个有效极化视图，目标中位尺寸 {checks["median_length_px"]:.2f} × {checks["median_width_px"]:.2f} px。原图未缩放；绿框为原 OBB，红框为扰动 OBB。面积差采用软 mask，图上画的是几何轮廓。</p>']
    for record,filename in panels:
        page.extend([f'<section><h2>{record["id"]} / {record["polarization"]}</h2>',
                     f'<img src="{filename}" alt="Native query geometry preview"><p>{html.escape(record["source"])}</p></section>'])
    (ROOT/'preparation.html').write_text('\n'.join(page)+'</html>',encoding='utf-8')
    print('PREVIEW_READY',len(panels),'native panels; torch imported:', 'torch' in sys.modules)


if __name__=='__main__':main()
