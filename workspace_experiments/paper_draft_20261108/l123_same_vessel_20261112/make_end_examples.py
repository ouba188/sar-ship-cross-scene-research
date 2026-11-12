"""New axis-end pixel examples for obs12457, without model evaluation."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, distance_transform_edt
from pptx import Presentation
from pptx.util import Inches
from pptx.oxml.xmlchemy import OxmlElement

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[1]
OUT=ROOT/'axis_end_examples';OUT.mkdir(exist_ok=True)
AS=OUT/'images';AS.mkdir(exist_ok=True)
G=WORK/'same_vessel_gallery_20261019'
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
loader=module('cpu_loader',ROOT.parent/'l123_editable_20261112/prepare_l2.py')
cpu=loader.cpu_functions
body=cpu(G/'local_condition_module/prepare.py',['masks_for'],dict(np=np,Image=Image,ImageDraw=ImageDraw))
axis=cpu(G/'causal_frequency_side_20261024/joint_probe.py',['masks_for'],dict(np=np))
probe=cpu(G/'causal_frequency_side_20261024/classification_joint_probe.py',['regions','bands'],dict(
    np=np,gaussian_filter=gaussian_filter,d=SimpleNamespace(ip=SimpleNamespace(masks_for=body['masks_for'])),j=SimpleNamespace(masks_for=axis['masks_for'])))
snap=next(x for x in json.loads((G/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8')) if x['obs_id']==12457)
obs=pd.read_csv(ROOT/'ports24/l1_observations.csv').set_index('obs_id').loc[12457]
proposals=[]
for pol in ('vv','vh'):
    source=snap['images'][pol]
    im=np.asarray(Image.open(obs['image_relpath_'+pol]).convert('L'),dtype=float)/255
    assert max(im.shape)<160  # Whole native chip is inside historical crop/pad input.
    masks=probe['regions'](im,source);mask=masks['end']
    assert mask.sum()>=8 and not (mask&masks['body']).any()
    taper=np.minimum(distance_transform_edt(mask)/2,1)*mask
    for band,component in probe['bands'](im)[0].items():
        delta=np.clip(im-.5*taper*component,0,1)-im
        proposals.append(dict(pol=pol,band=band,im=im,mask=mask,taper=taper,delta=delta,
                              norm=float(np.linalg.norm(delta)),source=obs['image_relpath_'+pol]))
old=json.loads((ROOT/'expanded/l2_assets.json').read_text(encoding='utf-8'))
oldtarget=old['records'][0]['l2']
# Extend the existing fixed dose; never amplify beyond the bounded proposal.
target=min(oldtarget,min(p['norm'] for p in proposals))
limit=.03
actualmax=max(float(abs(target*p['delta']/p['norm']).max()) for p in proposals)
if actualmax>limit:limit=float(np.ceil(actualmax*100)/100)
records=[];arrays={}
def gray(name,pixels):
    Image.fromarray(np.rint(pixels*255).astype(np.uint8)).save(AS/f'{name}.png')
    Image.fromarray(np.rint(pixels*65535).astype(np.uint16)).save(AS/f'{name}_16bit.png')
    return str(AS/f'{name}.png')
for p in proposals:
    im,mask,taper=p['im'],p['mask'],p['taper']
    after=im+target*p['delta']/p['norm'];change=after-im
    basis=taper*((1-im) if change.sum()>=0 else im)
    control=im+change.sum()/basis.sum()*basis if basis.sum() else im.copy()
    assert abs(change[~mask]).max(initial=0)==0
    assert abs(np.linalg.norm(change)-target)<1e-12
    assert abs((control-im).sum()-change.sum())<1e-12
    assert min(after.min(),control.min())>=0 and max(after.max(),control.max())<=1
    key=p['pol']+'_end_'+p['band']
    paths={k:gray(key+'_'+k,v) for k,v in [('original',im),('after',after),('control',control)]}
    rgb=np.repeat(im[...,None],3,axis=-1);rgb[mask]=.62*rgb[mask]+.38*np.array([.05,.85,.65])
    paths['region']=str(AS/f'{key}_region.png');Image.fromarray(np.rint(rgb*255).astype(np.uint8)).save(paths['region'])
    v=change/limit;rgb=np.stack((1+np.minimum(v,0),1-abs(v),1-np.maximum(v,0)),axis=-1)
    paths['difference']=str(AS/f'{key}_difference.png');Image.fromarray(np.rint(rgb*255).astype(np.uint8)).save(paths['difference'])
    arrays.update({key+'_'+k:a for k,a in [('original',im),('after',after),('control',control),('difference',change),('mask',mask)]})
    records.append(dict(pol=p['pol'],region='end',band=p['band'],paths=paths,source=p['source'],
        shape=list(im.shape),pixels=int(mask.sum()),changed_pixels=int(np.count_nonzero(change)),l2=target,
        max_abs_change=float(abs(change).max()),mean_delta=float(change.mean()),prediction=None,delta_margin=None))
np.savez_compressed(OUT/'exact_pixels.npz',**arrays)
info=dict(obs_id=12457,mmsi='371234000',status='pixel examples only; no model inference, no measured classification effect',
    old_body_side_l2=oldtarget,end_l2=target,same_dose_as_old=abs(target-oldtarget)<1e-12,
    difference_limit=limit,records=records,
    definition='Detected ship-axis coordinates u,v: L/2 < abs(u) <= L/2+8; abs(v)<=W/2+2; remove target body. Units: native pixels. Both ends together, not identified bow/stern.',
    control='Same total grayscale change, not same L2; nonselected polarization stays original.',
    source_functions=str(G/'causal_frequency_side_20261024/joint_probe.py'))
(OUT/'assets.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')

d=module('drawing',ROOT.parent/'l123_editable_20261112/build_deck.py')
d.prs=Presentation();d.prs.slide_width=Inches(18);d.prs.slide_height=Inches(10)
for pol in ('vv','vh'):
    s=d.new(f'L2 轴端干预示例：12457 / {pol.upper()}',
      '新增像素级示例，尚未进行模型预测；沿检测船轴的两端邻域，不把两端指认为真实船首、船尾。',len(d.prs.slides)+1)
    cols=[('original','原图'),('region','轴端区域示意'),('after','实际干预后'),('difference','改后 − 原图'),('control','同均值变化对照')]
    xs=[1.35,3.95,6.55,9.15,11.75]
    for x,(_,label) in zip(xs,cols):d.txt(s,label,x,1.95,2.45,.4,17,bold=True)
    d.txt(s,'本项改动',14.45,1.95,2.65,.4,17,bold=True)
    for j,r in enumerate([r for r in records if r['pol']==pol]):
        y=2.65+j*3.1
        d.txt(s,'细尺度' if r['band']=='fine' else '中尺度',.5,y+.63,.85,.8,16,d.C['teal'],True)
        for x,(key,_) in zip(xs,cols):
            h,w=r['shape'];pw,ph=2*w/72,2*h/72
            d.pic(s,r['paths'][key],x+(2-pw)/2,y+(2-ph)/2,pw,ph)
        d.txt(s,f"轴端区域 {r['pixels']} 像素\n改动L2 {target:.6f}\n最大改动 {r['max_abs_change']:.4f}\n分类效果：未评估",14.45,y+.1,2.8,1.8,16)
        d.txt(s,'另一极化保持原样；框内与其他区域不变。',1.35,y+2.16,12,.38,14,d.C['muted'])
        d.line(s,.7,y+2.7,17.2,y+2.7)
    d.txt(s,f'差异色标：蓝 −{limit:.2f} / 白 0 / 红 +{limit:.2f}；绿色仅作轴端区域标注，不送入模型。',.7,8.8,16.5,.4,15)
    dose_note='改动L2与前八项相同。' if info['same_dose_as_old'] else '为保持弱干预，四项统一采用较低L2；不能与前八项直接比较效果。'
    d.txt(s,'原图、改后及对照固定灰度[0,1]，无单独拉伸。'+dose_note,.7,9.38,16,.36,12,d.C['muted'])
    d.note(s,json.dumps(dict(scope=info['status'],definition=info['definition'],records=[r for r in records if r['pol']==pol]),ensure_ascii=False))
for slide in d.prs.slides:
    for sh in slide.shapes:
        sp=sh._element.find('{http://schemas.openxmlformats.org/presentationml/2006/main}spPr')
        if sp is not None:sp.append(OxmlElement('a:effectLst'))
        if sh.has_text_frame:
            for para in sh.text_frame.paragraphs:
                for run in para.runs:
                    ea=OxmlElement('a:ea');ea.set('typeface','Microsoft YaHei');run._r.get_or_add_rPr().append(ea)
d.prs.save(OUT/'12457_轴端干预示例_可编辑.pptx')
parts=['<!doctype html><meta charset="utf-8"><title>12457轴端干预示例</title><style>body{font:16px Microsoft YaHei;margin:30px}table{border-collapse:collapse}td,th{padding:12px;border-bottom:1px solid #ccc}img{width:180px;image-rendering:pixelated}</style><h1>12457 轴端干预示例</h1><p>像素级示例，尚未计算分类效果。两端由检测船轴定义，不区分真实首尾。</p><table><tr><th>条件</th><th>原图</th><th>轴端区域</th><th>干预后</th><th>差异</th><th>同均值对照</th></tr>']
for r in records:
    parts.append(f"<tr><td>{r['pol'].upper()} / {'细尺度' if r['band']=='fine' else '中尺度'}</td>")
    for k in ('original','region','after','difference','control'):
        path=Path(r['paths'][k]).relative_to(OUT).as_posix();parts.append(f'<td><a href="{path}"><img src="{path}"></a></td>')
    parts.append('</tr>')
parts.append(f'</table><p>四项像素L2={target:.9f}；原始灰度[0,1]，差异统一±{limit:.2f}。图片与浮点数组另存。</p>')
(OUT/'index.html').write_text(''.join(parts),encoding='utf-8')
print(json.dumps({k:info[k] for k in ('old_body_side_l2','end_l2','same_dose_as_old','difference_limit')},indent=2))
