"""Expanded editable same-vessel evidence, with exact intervention images."""
from pathlib import Path
import importlib.util
import json
import html
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.util import Inches
from pptx.oxml.xmlchemy import OxmlElement

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'expanded'
spec=importlib.util.spec_from_file_location('draw',ROOT.parent/'l123_editable_20261112/build_deck.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
d.prs=Presentation(ROOT/'L1-L3_同船12457_可编辑.pptx')
data=json.loads((OUT/'l2_assets.json').read_text(encoding='utf-8'))
records=data['records']
H=ROOT.parents[1]/'m3_frequency_increment_20261028/m3_true_rescue_12457_20261031'
receipt=json.loads((H/'REPLAY_RECEIPT.json').read_text(encoding='utf-8'))
rem=pd.read_csv(H/'local_removal.csv')

def new(title,sub):return d.new(title,sub,len(d.prs.slides)+1)
def foot(s,text):d.txt(s,text,.65,9.35,16.0,.47,12,d.C['muted'])
def rawpic(s,path,x,y,w=2.0):
    with Image.open(path) as im: iw,ih=im.size
    pw,ph=w*iw/72,w*ih/72
    d.pic(s,path,x+(w-pw)/2,y+(w-ph)/2,pw,ph)

s=new('L2 如何改图：只改变指定极化、区域和尺度',
      '12457 / 散货船；使用历史实验函数恢复浮点干预图，数值沿用2026-10-27完整模型seed17。')
for x,title,body in [(0.7,'① 指定输入','每次只改变VV或VH中的一幅图。\n另一极化、检测框、模型权重和\n16个源参考保持不变。'),
 (6.5,'② 指定局部成分','区域：框内 / 侧向。\n细尺度：原图 − σ=1高斯平滑。\n中尺度：σ=1平滑 − σ=2平滑。'),
 (12.3,'③ 限幅并匹配强度','在区域边界平滑衰减改动。\n八种操作使用相同像素L2改变量。\n同时构造总灰度变化相同的对照。')]:
    d.tag(s,title,x,2.0,4.95,d.C['teal'])
    d.txt(s,body,x,2.65,4.95,1.45,19)
d.txt(s,'xᴵ = x + a · δ，δ = clip(x − 0.5 wᵣ Bₛ(x), 0, 1) − x',.75,4.55,16.2,.6,25,bold=True)
d.txt(s,'a将各操作的改变量L2统一到0.104540；wᵣ是区域边缘衰减权重，Bₛ是选定尺度成分。',.75,5.3,16.1,.6,18)
d.txt(s,'记录分差：m = 正确类别分数 − 当前最高错误类别分数；Δm = 干预后m − 原图m。',.75,6.2,16.1,.65,20)
d.conclusion(s,'下四页逐项展示全部8种干预，不只挑改善最大的操作。',.75,7.35,16.1,d.C['teal'])
d.txt(s,'原图 / 改后 / 同均值对照均固定灰度[0,1]；差异图采用统一±0.03色标，蓝为降低、红为增加。',.75,8.2,16.1,.6,17)
foot(s,'同均值对照只匹配总灰度变化，不匹配L2或全部像素分布。历史full路径包含重算的显式频率描述，不能全部归因于纯图像路径。')
d.note(s,'Exact historical functions and reconstruction diagnostics: '+str(OUT/'l2_assets.json'))

for pol in ('vv','vh'):
  for region,regionzh in [('body','框内'),('side','侧向')]:
    pair=[r for r in records if r['pol']==pol and r['region']==region]
    s=new(f'L2 实际干预图：{pol.upper()} · {regionzh}',
          '同一观测12457；上行细尺度，下行中尺度。每一行是一项独立干预，另一极化始终不变。')
    cols=[('original','原图'),('region','操作区域示意'),('after','实际干预后'),('difference','改后 − 原图'),('control','同均值变化对照')]
    xs=[1.35,3.95,6.55,9.15,11.75]
    for x,(_,label) in zip(xs,cols):d.txt(s,label,x,1.96,2.3,.4,17,bold=True)
    d.txt(s,'分类分差变化',14.45,1.96,2.65,.4,17,bold=True)
    for j,r in enumerate(pair):
      y=2.65+j*3.1
      d.txt(s,'细尺度' if r['band']=='fine' else '中尺度',.5,y+.63,.85,.95,16,d.C['teal'],True)
      for x,(key,_) in zip(xs,cols):rawpic(s,r['paths'][key],x,y)
      d.txt(s,f"结构 Δm {r['delta_margin']:+.4f}\n对照 Δm {r['control_delta']:+.4f}\n改后 m {r['after_margin']:+.4f}\n预测：仍为散货船",14.45,y+.12,2.7,1.7,16)
      d.txt(s,f"改动 {r['changed_pixels']} 像素；最大绝对改动 {max(abs(r['difference_min']),abs(r['difference_max'])):.4f}",1.35,y+2.14,12,.4,14,d.C['muted'])
      d.line(s,.7,y+2.69,17.25,y+2.69)
    d.txt(s,'差异色标',.75,8.79,1.3,.35,14)
    for k in range(30):
      v=-1+2*k/29;rgb=(1+min(v,0),1-abs(v),1-max(v,0))
      d.rect(s,2.1+k*.07,8.83,.072,.15,fill=''.join(f'{round(q*255):02X}' for q in rgb))
    d.txt(s,'−0.03       0       +0.03',2.0,9.02,2.6,.27,11)
    d.txt(s,'绿色仅为区域标注，不送入模型；灰度图未单独拉伸。浮点改动较小，差异图便于查看。',5.0,8.85,12.2,.5,15)
    foot(s,'图像保持原生像素比例；8位PNG用于显示，另提供16位PNG及精确浮点数组。所有分差取自历史缓存，非PNG重新预测。')
    d.note(s,json.dumps(pair,ensure_ascii=False))

s=new('L2 条件效应与强度对照：保留完整八项结果',
      '同一原图、同一模型；八项结构操作的像素L2相同。横向比较的是当前模型的判别响应，不是物理散射因果结论。')
headers=['干预条件','原图 m','结构后 m','结构 Δm','对照 Δm','预测变化']
xx=[.8,5.0,7.4,9.8,12.2,14.5]
for x,t in zip(xx,headers):d.txt(s,t,x,2.05,2.5,.43,18,bold=True)
for j,r in enumerate(records):
    y=2.85+j*.62
    label=f"{r['pol'].upper()} / {'框内' if r['region']=='body' else '侧向'} / {'细' if r['band']=='fine' else '中'}"
    vals=[label,f"{r['base_margin']:+.4f}",f"{r['after_margin']:+.4f}",f"{r['delta_margin']:+.4f}",f"{r['control_delta']:+.4f}",'正确 → 正确']
    for k,(x,t) in enumerate(zip(xx,vals)):
        d.txt(s,t,x,y,4 if k==0 else 2.5,.38,17,d.C['teal'] if k==3 and r['delta_margin']>=0 else d.C['red'] if k==3 else d.C['ink'])
    d.line(s,.8,y+.45,17.1,y+.45)
d.conclusion(s,'同一种尺度处理在VV与VH、框内与侧向可产生不同方向的效果；本例L2没有预测翻转。',.8,8.26,16.3,d.C['teal'])
foot(s,'分差逐次重新选择最高错误类。对照数值不能与其他阶段模型的分差直接比较。')

assets_path=OUT/'l3_assets.json'
if assets_path.exists():
    assets=json.loads(assets_path.read_text(encoding='utf-8'))
    images={r['id']:r for r in assets['images']}
    s=new('L3 实际查询与参考：原图没有被修改',
          '12457是待分类散货船；31512是同类但不同船的参考，13000是海工参考。后两者来自该次实际使用的16个源参考库。')
    for j,(oid,label) in enumerate([('12457','待分类查询 · 散货船'),('31512','同类参考 · 散货船'),('13000','异类参考 · 海工船')]):
        x=.75+j*5.75;r=images[oid]
        d.txt(s,f'{oid} / {label}',x,2.05,5.2,.43,20,bold=True)
        d.txt(s,r['port']+' / '+r['pass_group'],x,2.65,5.2,.5,13,d.C['muted'])
        for k,pol in enumerate(('VV','VH')):
            d.txt(s,pol,x+k*2.55,3.26,2.1,.35,16,d.C['blue'],True)
            rawpic(s,r['views'][pol]['raw'],x+k*2.55,3.75,2.2)
    d.txt(s,'完整M3：+0.1376 / 判对；逐个撤除参考回写后：',.75,6.4,16.4,.5,20,bold=True)
    for j,(pol,oid) in enumerate([('VV','31512'),('VH','31512'),('VV','13000'),('VH','13000')]):
        row=rem[rem.condition.eq(f'移除{pol}参考{oid}回写')].iloc[0]
        d.txt(s,f"撤除 {pol} / {oid}\nm = {row.margin:+.4f}\n仍判为散货船",.75+j*4.25,7.15,3.95,1.2,18)
    foot(s,'图像本身保持不变，撤除的是该参考的内部回写。单个参考移除未翻转预测，不等于全部参考无用；回写效应经过非线性下游，不可直接相加。')
    d.note(s,'Actual reference identities and source paths: '+str(assets_path))

    s=new('L3 回写发生在哪里：真实特征图上的更新幅度',
          '读取既有缓存 update[1,2,1024,10,10]；每格显示1024个通道更新的L2范数，不重新推理。')
    vmax=assets['writeback']['VV']['shared_vmax']
    for pol,x in [('VV',1.0),('VH',9.3)]:
        d.txt(s,pol+' / 10 × 10 特征网格',x,2.03,6.9,.45,22,bold=True)
        vals=assets['writeback'][pol]['l2_grid']
        for iy,row in enumerate(vals):
          for ix,value in enumerate(row):
            t=value/vmax
            rgb=[round(255*(1-t)+v*t) for v in (20,125,121)]
            d.rect(s,x+ix*.52,2.75+iy*.52,.52,.52,fill=''.join(f'{v:02X}' for v in rgb),line='D8E2E5')
            if value>.01:d.txt(s,f'{value:.1f}',x+ix*.52+.015,2.75+iy*.52+.16,.49,.23,10,'FFFFFF' if t>.55 else d.C['ink'])
        d.txt(s,f'共同色标：0 至 {vmax:.2f}',x,8.13,6.9,.35,16,d.C['muted'])
    foot(s,'幅度图说明回写位置与大小，不是带正负号的类别贡献，也不单独证明该位置有益；其分类作用需结合下一页的撤除对照。')

s=new('L3 同一输入的路径反事实：移除哪里会撤销救回？',
      '历史 N_level_pattern / seed17；原始VV/VH图像不改变；固定匹配T、分支权重及其余计算，只移除指定回写再计算下游。')
for p,x in [('VV',.8),('VH',9.1)]:
    d.tag(s,p+' 回写路径',x,2.02,7.7,d.C['gold'])
    rows=[('M3完整开启',receipt['conditions'][1]['margin']),('M3全部关闭',receipt['conditions'][0]['margin'])]
    for region in ('全部','框内','侧向','轴端'):
        val=float(rem.loc[rem.condition.eq(f'移除{p}{region}回写'),'margin'].iloc[0])
        rows.append((f'撤除{region}',val))
    d.bars(s,rows,x,3.05,7.7,4.3,.9,19)
d.conclusion(s,'VV框内、VH框内与VH侧向回写撤除后，本例由正确回到错误；轴端撤除后仍正确。',.8,8.08,16.3,d.C['gold'])
foot(s,'这是固定计算图上的个体反事实。移除并不改变原始chip，因此L3没有“改后的SAR图”；应展示改变的内部路径和输出。')

s=new('三层证据如何相互补充，而不是混作同一次实验',
      '同船身份联接L1；同一观测12457联接L2与L3。保留原始证据，明确各层回答的问题。')
rows=[('L1 自然观测','不同场景下，同船会不会判得不一样？','4次观测 / 4个港口；旧诊断器出现对错变化。','揭示需要解释的现象。'),
 ('L2 输入干预','改变指定局部成分，会怎样改变判别？','8项同强度操作；极化、区域、尺度的作用方向不同。','支持条件化利用，不能把局部变化一律当噪声。'),
 ('L3 内部反事实','已发生的M3救回，依赖哪些回写？','固定输入；移除特定区域回写可撤销救回。','定位参与决策的补充路径，不是物理世界反事实。')]
for i,(a,b,c,e) in enumerate(rows):
    y=2.0+i*1.85
    d.tag(s,a,.8,y,2.4,[d.C['blue'],d.C['teal'],d.C['gold']][i])
    d.txt(s,b,3.5,y,13.5,.45,23,bold=True)
    d.txt(s,c,3.5,y+.6,13.5,.45,19)
    d.txt(s,e,3.5,y+1.14,13.5,.45,18,d.C['muted'])
d.txt(s,'设计启示：按极化与区域保留证据身份，再检验参考补充是否真正改变分类；不是要求所有响应对所有船型都有效。',.8,8.0,16.3,.8,20)
foot(s,'L3简单聚合也可救回12457，因此本例不证明精细匹配不可替代。不同阶段模型的分差不组成连续性能增益。')

target=OUT/'L1-L3_同船12457_完整干预图_可编辑.pptx'
for slide in d.prs.slides:
    for shape in slide.shapes:
        sp=shape._element.find('{http://schemas.openxmlformats.org/presentationml/2006/main}spPr')
        if sp is not None and sp.find('{http://schemas.openxmlformats.org/drawingml/2006/main}effectLst') is None:
            sp.append(OxmlElement('a:effectLst'))
        if shape.has_text_frame:
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    rp=run._r.get_or_add_rPr()
                    ea=rp.find('{http://schemas.openxmlformats.org/drawingml/2006/main}ea')
                    if ea is None:ea=OxmlElement('a:ea');rp.append(ea)
                    ea.set('typeface','Microsoft YaHei')
        assert shape.left>=0 and shape.top>=0 and shape.left+shape.width<=d.prs.slide_width+2000 and shape.top+shape.height<=d.prs.slide_height+2000
d.prs.save(target)
parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>12457 干预原图与结果</title><style>body{font:16px Microsoft YaHei,sans-serif;margin:32px;color:#203039}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ccd4d7;text-align:left}img{width:180px;image-rendering:pixelated}small{color:#56666e}</style><h1>12457：全部8种干预前后图</h1><p>固定灰度[0,1]；差异统一±0.03。区域图绿色仅作标注。历史完整模型seed17，16个参考固定，另一极化不变。</p><table><tr><th>条件</th><th>原图</th><th>操作区域</th><th>实际干预后</th><th>改后−原图</th><th>同均值对照</th><th>分差变化</th></tr>']
for r in records:
    parts.append('<tr><td>'+html.escape(' / '.join(r[k] for k in ('pol','region','band')))+'</td>')
    for k in ('original','region','after','difference','control'):
        path=Path(r['paths'][k]).relative_to(OUT).as_posix()
        parts.append(f'<td><a href="{path}"><img src="{path}"></a></td>')
    parts.append(f"<td>结构 {r['delta_margin']:+.4f}<br>对照 {r['control_delta']:+.4f}<br>预测仍正确</td></tr>")
parts.append('</table><p><small>精确浮点图：l2_exact_pixels.npz；16位PNG与显示PNG在l2_images/。本页只恢复旧实验图像，不重训、不重新预测。</small></p></html>')
(OUT/'index.html').write_text(''.join(parts),encoding='utf-8')
print(target)
