"""Editable native PowerPoint evidence panels; no model evaluation or training."""
from pathlib import Path
import csv
import importlib.util
import json
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement

OUT = Path(__file__).resolve().parent
WORK = OUT.parents[1]
ASSETS = OUT / 'assets'
R22 = WORK / 'same_vessel_gallery_20261019/causal_frequency_side_20261024/round22_classification_coverage'
CF = OUT.parent / 'm3_counterfactual_20261112'
C = dict(ink='203039', muted='56666E', line='C7D3D8', blue='346A91', teal='147D79',
         gold='9A651B', red='B0444B', pale='F4F7F8', white='FFFFFF')
FONT = 'Microsoft YaHei'
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(18), Inches(10)
sources = []


def txt(s, text, x, y, w, h, size=18, color=None, bold=False, align=None):
    sh = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = sh.text_frame
    tf.clear(); tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(.02)
    tf.margin_top = tf.margin_bottom = Inches(.015)
    for i, line in enumerate(str(text).split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line; p.font.name = FONT; p.font.size = Pt(size)
        p.font.bold = bold; p.font.color.rgb = RGBColor.from_string(color or C['ink'])
        p.space_before = p.space_after = Pt(0)
        if align is not None: p.alignment = align
    return sh


def rect(s,x,y,w,h,fill=None,line=None,radius=False):
    sh=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
                          Inches(x),Inches(y),Inches(w),Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb=RGBColor.from_string(fill or C['white'])
    sh.line.color.rgb=RGBColor.from_string(line or fill or C['white'])
    sh.line.width=Pt(.8)
    if radius: sh.adjustments[0]=.06
    return sh


def line(s,x1,y1,x2,y2,color=None,width=1.2,arrow=False):
    sh=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x1),Inches(y1),Inches(x2),Inches(y2))
    sh.line.color.rgb=RGBColor.from_string(color or C['line']);sh.line.width=Pt(width)
    if arrow:
        tail=OxmlElement('a:tailEnd');tail.set('type','triangle');sh._element.spPr.find('{http://schemas.openxmlformats.org/drawingml/2006/main}ln').append(tail)
    return sh


def pic(s,path,x,y,w,h=None):
    with Image.open(path) as im: iw,ih=im.size
    if h is None:h=w*ih/iw
    else:
        k=min(w/iw,h/ih); x+=(w-iw*k)/2;y+=(h-ih*k)/2;w=iw*k;h=ih*k
    return s.shapes.add_picture(str(path),Inches(x),Inches(y),Inches(w),Inches(h))


def tag(s,text,x,y,w,color,size=15):
    rect(s,x,y,w,.38,fill=color)
    txt(s,text,x+.07,y+.035,w-.14,.3,size,C['white'],True)


def new(title,sub,num):
    s=prs.slides.add_slide(prs.slide_layouts[6])
    txt(s,title,.55,.35,16.9,.65,30,bold=True)
    txt(s,sub,.57,1.04,16.85,.43,15,C['muted'])
    line(s,.55,1.62,17.45,1.62)
    txt(s,f'{num:02d}',16.8,9.56,.6,.28,12,C['muted'],align=PP_ALIGN.RIGHT)
    return s


def conclusion(s,text,x,y,w,color):
    rect(s,x,y,w,.62,fill='F2F6F7')
    rect(s,x,y,.045,.62,fill=color)
    txt(s,text,x+.14,y+.12,w-.27,.43,16,color,True)


def bars(s,rows,x,y,w,h,bound=None,fontsize=15):
    # Native vector bars, with independently editable labels and values.
    bound=bound or max(abs(v) for _,v in rows)*1.16
    labelw=w*.36; chartw=w*.43; zero=x+labelw+chartw/2
    step=h/len(rows)
    line(s,zero,y-.1,zero,y+h-.1,C['muted'],.9)
    txt(s,'0',zero-.1,y-.31,.25,.25,11,C['muted'])
    for j,(label,v) in enumerate(rows):
        yy=y+j*step
        txt(s,label,x,yy,w*.35,.36,fontsize)
        extent=v/bound*chartw/2
        rect(s,min(zero,zero+extent),yy+.07,max(abs(extent),.006),.22,
             fill=C['teal'] if v>=0 else C['red'])
        txt(s,f'{v:+.4f}',x+w*.81,yy,w*.19,.36,fontsize,
            C['teal'] if v>=0 else C['red'])


def note(s,text):
    s.notes_slide.notes_text_frame.text=text


def stage_header(s,level,name,en,x,color):
    tag(s,level,x,1.91,.6,color,16)
    txt(s,name,x+.76,1.86,4.25,.48,23,color,True)
    txt(s,en,x+.76,2.38,4.2,.34,13,C['muted'])


def copy_l1():
    t=pd.read_csv(R22/'interventions.csv')
    for oid in (7032,7035):
        for pol in ('vv','vh'):
            row=t[(t.obs_id==oid)&(t.pol==pol)].iloc[0]
            dest=ASSETS/f'l1_{oid}_{pol}.png'
            shutil.copyfile(row.image_path,dest)
            sources.append(dict(asset=dest.name,source=row.image_path,role=f'L1 {oid} {pol}'))
    return pd.read_csv(R22/'baseline.csv').set_index('obs_id')


def copy_l3():
    cases=json.loads((CF/'private/cases.json').read_text(encoding='utf-8'))
    case=next(c for c in cases if c['index']==1294 and c['seed']==29)
    records=json.loads((WORK/'m1_coverage_20261101/manifest.json').read_text(encoding='utf-8'))
    spec=importlib.util.spec_from_file_location('native_input',WORK/'sar_m123_standalone/sar_m123/input.py')
    api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    for role,idx in [('query',case['index']),('same',case['same_reference']),('foil',case['foil_reference'])]:
        for pol,view in zip(('vv','vh'),api.prepare(records[idx])):
            assert view and view['valid_target']
            dest=ASSETS/f'l3_{role}_{pol}.png'
            Image.fromarray(view['pixels']).save(dest)
            sources.append(dict(asset=dest.name,source=records[idx][pol]['image'],role=f'L3 {role} {pol} native160'))
    return case


def overview(base,l2,case):
    s=new('跨场景 SAR 船舶分类的分层判别证据分析',
          '受因果层级启发：自然观测发现变化，输入干预检验作用，个体反事实定位决策依赖。',1)
    xs=(.65,6.45,12.25);cols=(C['blue'],C['teal'],C['gold'])
    for x,lv,name,en,col in zip(xs,('L1','L2','L3'),('自然观测','受控输入干预','个体决策反事实'),
                               ('Observation','Controlled input intervention','Individual decision counterfactual'),cols):
        stage_header(s,lv,name,en,x,col)
    for x in (6.0,11.8):line(s,x,2.94,x,8.78)
    for x in (5.75,11.55):line(s,x,2.12,x+.45,2.12,C['muted'],1.5,True)
    x=xs[0]
    txt(s,'同一艘散货船 · Santos',x,2.94,5,.4,18,bold=True)
    txt(s,'7032 / 场景 A',x+.57,3.44,2,.32,14)
    txt(s,'7035 / 场景 B',x+2.78,3.44,2,.32,14)
    for r,pol in enumerate(('vv','vh')):
        yy=3.88+r*1.55
        txt(s,pol.upper(),x,yy+.55,.48,.32,14,C['muted'])
        for c,oid in enumerate((7032,7035)):pic(s,ASSETS/f'l1_{oid}_{pol}.png',x+.6+c*2.2,yy,1.38)
    txt(s,'散货船：判对',x+.49,7.04,2.1,.36,17,C['teal'],True)
    txt(s,'集装箱船：判错',x+2.65,7.04,2.3,.36,17,C['red'],True)
    txt(s,'m = +0.0504',x+.57,7.48,2.1,.33,15)
    txt(s,'m = −0.1880',x+2.77,7.48,2.1,.33,15)
    conclusion(s,'同船身份不变，响应与判断仍可变化。',x,8.16,5.05,cols[0])
    x=xs[1]
    txt(s,'固定图像其余部分与诊断模型',x,2.94,5.05,.4,18,bold=True)
    txt(s,'VV / VH × 框内 / 轴端 / 侧向 × 细 / 中尺度',x,3.43,5.05,.48,14,C['muted'])
    for c,(key,label) in enumerate((('original','原图'),('intervened','指定干预后'))):
        pic(s,ASSETS/f'l2_{key}.png',x+.35+c*2.55,4.13,1.75)
        txt(s,label,x+.35+c*2.55,3.87,2,.3,14)
    line(s,x+2.22,4.98,x+2.7,4.98,cols[1],1.5,True)
    txt(s,'例：7797，VV框内中尺度；VH保持原样',x,6.09,5.05,.42,14)
    bars(s,[('指定干预 Δm',l2['delta_fixed']),('均值对照 Δm',l2['level_delta'])],x,6.88,5.05,1.04,.022,14)
    conclusion(s,'作用取决于极化、区域、尺度和类别。',x,8.16,5.05,cols[1])
    x=xs[2]
    txt(s,'固定查询、权重与32个源参考',x,2.94,5.05,.4,18,bold=True)
    for c,pol in enumerate(('vv','vh')):
        txt(s,pol.upper(),x+.45+c*2.5,3.51,2,.3,14)
        pic(s,ASSETS/f'l3_query_{pol}.png',x+.45+c*2.5,3.85,1.52)
    txt(s,'同一查询002422：仅替换M3内部消息',x,5.62,5.05,.38,14)
    m=case['margins']
    bars(s,[('完整M3',m['factual']),('无M3',m['no_m3']),('简单均值',m['simple_mean']),('无框内消息',m['no_body'])],x,6.3,5.05,1.72,.11,14)
    conclusion(s,'定位这次救回实际依赖的补充路径。',x,8.16,5.05,cols[2])
    txt(s,'L1–L2：24港来源的历史冻结诊断器；L3：OSS2 Full-Tuned / seed29。各层分差不跨模型比较。',.65,9.05,16.5,.33,13,C['muted'])
    txt(s,'L3为模型内部计算反事实，不表示已恢复真实成像物理因果。箭头表示分析递进。',.65,9.46,15.5,.3,12,C['muted'])
    note(s,'Three panels are separate evidence settings, not the same sample flowing through a network. '+str(R22)+' ; '+str(CF))


def l1_slide(base):
    s=new('L1  自然观测：同一艘船，为什么会得到不同判断？',
          '24港来源的同船配对案例；MMSI 314590000；真实类别：散货船；港口：Santos。',2)
    for col,(oid,scene,date,platform) in enumerate(((7032,'场景 A','2025-11-22','S1C / 005124'),(7035,'场景 B','2025-12-22','S1A / 062425'))):
        xx=.8+col*8.5
        tag(s,scene,xx,1.98,1.1,C['blue'])
        txt(s,f'观测 {oid}  |  {date}  |  {platform}',xx+1.3,2,6.3,.45,19)
        for j,pol in enumerate(('vv','vh')):
            px=xx+j*3.65
            txt(s,pol.upper(),px,2.68,2,.4,18,C['muted'],True)
            pic(s,ASSETS/f'l1_{oid}_{pol}.png',px,3.2,3.15)
        val=float(base.loc[oid,'baseline_margin'])
        label='散货船 / 判对' if oid==7032 else '集装箱船 / 判错'
        txt(s,label,xx,6.72,5,.45,25,C['teal'] if oid==7032 else C['red'],True)
        txt(s,f'散货船 − 集装箱船分差：{val:+.4f}',xx,7.35,7.5,.5,22)
    line(s,8.8,2.02,8.8,7.93)
    conclusion(s,'发现：船只身份不变时，局部图像响应及其类别判别结果仍可能随场景变化。',.8,8.25,16.35,C['blue'])
    txt(s,'原始64×64像素chip，同倍率显示，不重新拉伸灰度。两次观测由同一折的冻结诊断器预测。',.8,9.04,16.3,.35,14,C['muted'])
    txt(s,'自然配对说明变化同时发生；不单凭外观将差异归因为朝向、旁瓣、邻船或海况。',.8,9.47,15.5,.3,12,C['muted'])
    note(s,'Sources: '+str(R22/'baseline.csv')+' and '+str(R22/'interventions.csv')+'. Observation IDs 7032 and7035, fold0. Both original rival is class4 container. Exact margin .0503727531 / -.1879588328.')


def heatmap(s,x,y,w,h):
    table=pd.read_csv(R22/'by_class.csv')
    cond=[('vv','body','fine'),('vv','side','fine'),('vv','end','middle'),('vh','body','fine'),('vh','side','fine'),('vh','end','middle')]
    cs=[0,3,4,5]; labels=['散货船','成品/化学品船','集装箱船','普通货船']
    sub=table[(table.scope=='all')&(table.dose==1)].set_index(['pol','region','band','class_id'])
    vals=np.array([[sub.loc[(*c,k),'ship_mean']*1000 for k in cs] for c in cond])
    lim=float(abs(vals).max());labw=2.0;cw=(w-labw)/4;rh=(h-.65)/6
    counts=[int(sub.loc[(*cond[0],k),'ships']) for k in cs]
    for j,(lab,n) in enumerate(zip(labels,counts)):
        txt(s,f'{lab}\n{n}船',x+labw+j*cw,y,cw,.65,12,align=PP_ALIGN.CENTER)
    region={'body':'框内','side':'侧向','end':'轴端'}
    for i,c in enumerate(cond):
        yy=y+.68+i*rh
        txt(s,f'{c[0].upper()} · {region[c[1]]} · '+('细' if c[2]=='fine' else '中'),x,yy+.07,labw-.12,.35,13)
        for j,v in enumerate(vals[i]):
            t=abs(v)/lim
            target=np.array([185,64,72]) if v>0 else np.array([48,111,165])
            rgb=np.rint(255*(1-t*.82)+target*t*.82).astype(int)
            fill=''.join(f'{a:02X}' for a in rgb)
            rect(s,x+labw+j*cw,yy,cw-.04,rh-.04,fill=fill)
            txt(s,f'{v:+.2f}',x+labw+j*cw,yy+.07,cw-.04,.35,13,align=PP_ALIGN.CENTER)
    return vals.tolist()


def l2_slide(l2):
    s=new('L2  受控输入干预：局部响应的作用不是固定的',
          '固定历史诊断器；24港来源样本7797，Singapore，成品/化学品船；本例仅改变VV框内中尺度，VH不变。',3)
    items=[('original','原始 VV','原图分差 −0.016102'),('intervened','指定干预','Δm = +0.016199'),('mean_control','匹配均值对照','Δm = +0.006679'),('difference','实际像素改变量','蓝：降低；红：提高；±0.02')]
    for j,(key,title,caption) in enumerate(items):
        xx=.7+j*4.35
        txt(s,title,xx,1.94,3.75,.42,21,C['teal'],True)
        pic(s,ASSETS/f'l2_{key}.png',xx,2.5,2.55)
        txt(s,caption,xx,5.23,3.95,.4,15)
    txt(s,'条件效应矩阵：按船平均 Δm × 10³',.7,5.94,8.2,.42,19,bold=True)
    heatmap(s,.7,6.49,8.6,2.58)
    txt(s,'冷色：干预后更不利；暖色：干预后更有利。展示选定条件与四类。',.7,9.2,8.9,.4,11,C['muted'])
    txt(s,'本例为何发生救回？',10,5.94,6.9,.42,21,C['teal'],True)
    txt(s,'正确类分数变化      −0.004668\n竞争类分数变化      −0.020867\n类别分差净变化      +0.016199',10,6.58,6.8,1.48,21)
    txt(s,'不是增强正确类，而是更强地削弱错误竞争类。\n干预后分差仅 +0.000096：属于近边界翻转。',10,8.17,7.05,.84,17)
    txt(s,'Δm固定原图最强错误类；不是概率或准确率。均值对照匹配总灰度变化，不匹配全部结构。',.7,9.63,15.5,.28,11,C['muted'])
    note(s,'L2 scope: historical frozen full_global diagnostic, not current Full. Native64 images shown nearest-neighbor8x without per-image stretch. Values from '+str(OUT/'l2.json')+'. Matrix source '+str(R22/'by_class.csv')+', scope=all,dose=1,mean per vessel. Model scores fixed rival class7. Mean control is not L2-matched. Case float pixels reconstructed exactly; PNG is display only.')


def l3_slide(case):
    s=new('L3  个体决策反事实：这次判对，依赖哪条证据路径？',
          'OSS2样本002422 / 散货船 / Full-Tuned seed29；固定原图、模型权重与32个合法源参考，重算全部后续网络。',4)
    roles=[('query','查询 002422'),('same','同类参考 006482'),('foil','混淆类参考 006719')]
    for j,(role,label) in enumerate(roles):
        xx=.65+j*2.5
        txt(s,label,xx,2,2.38,.42,16,C['blue'] if j==0 else C['muted'],True)
        for i,pol in enumerate(('vv','vh')):
            yy=2.91+i*2.37
            txt(s,pol.upper(),xx,yy-.35,2,.3,14,C['muted'])
            pic(s,ASSETS/f'l3_{role}_{pol}.png',xx,yy,1.98)
    line(s,8.15,2,8.15,8.45)
    txt(s,'同一输入，改变内部消息，再计算分类结果',8.6,2,8.6,.55,22,C['gold'],True)
    tag(s,'M3消息干预',8.6,2.9,2.2,C['gold'],17)
    line(s,10.95,3.1,11.48,3.1,C['gold'],1.5,True)
    txt(s,'layer4＋深层区域读出 → 分类头',11.6,2.85,5.6,.55,18)
    m=case['margins']
    bars(s,[('完整M3',m['factual']),('撤除M3',m['no_m3']),('简单类别均值',m['simple_mean']),('撤除框内消息',m['no_body'])],8.6,4.13,8.55,2.65,.115,19)
    txt(s,'完整M3判为散货船；其余三组判为成品/化学品船。',8.6,7.03,8.4,.55,18)
    conclusion(s,'该次救回需要当前组织的框内参考补充。',8.6,7.87,8.55,C['gold'])
    txt(s,'分差 = 正确类分数 − 本次前向的最高错误类分数；每个替代路径重新求竞争类。',.65,8.85,16.5,.4,14,C['muted'])
    txt(s,'黑边为补零；移除消息不等于遮挡像素。简单均值保留编码器与输出映射；显示参考按类内权重选取，不等于单参考因果归因。',.65,9.34,16.4,.42,12,C['muted'])
    note(s,'Current aggregate M3, NOT historical fine transport. Fixed-seed computational counterfactual, not separate retraining. Case '+json.dumps(case,ensure_ascii=False)+'. Full report: '+str(CF/'FINDINGS_ZH.md'))


def main():
    ASSETS.mkdir(exist_ok=True)
    base=copy_l1();case=copy_l3()
    l2=json.loads((OUT/'l2.json').read_text(encoding='utf-8'))
    overview(base,l2,case);l1_slide(base);l2_slide(l2);l3_slide(case)
    target=OUT/'L1-L3_分层判别证据_可编辑.pptx'
    for s in prs.slides:
        for sh in s.shapes:
            sp=sh._element.find('{http://schemas.openxmlformats.org/presentationml/2006/main}spPr')
            if sp is not None:
                sp.append(OxmlElement('a:effectLst'))
    prs.save(target)
    all_shapes=[]
    for i,s in enumerate(prs.slides,1):
        for sh in s.shapes:
            assert sh.left>=0 and sh.top>=0 and sh.left+sh.width<=prs.slide_width+2000 and sh.top+sh.height<=prs.slide_height+2000,(i,sh.name)
        all_shapes.append(dict(slide=i,objects=len(s.shapes),text=sum(sh.has_text_frame and bool(sh.text.strip()) for sh in s.shapes),pictures=sum(sh.shape_type==13 for sh in s.shapes)))
    plot_rows=[dict(level='L1',condition=str(oid),value=float(base.loc[oid,'baseline_margin'])) for oid in (7032,7035)]
    plot_rows += [dict(level='L2_case',condition=k,value=l2[k]) for k in ('baseline_margin','delta_fixed','level_delta','margin_after','true_score_change','rival_score_change')]
    plot_rows += [dict(level='L3',condition=k,value=v) for k,v in case['margins'].items()]
    with (OUT/'figure_values.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['level','condition','value']);writer.writeheader();writer.writerows(plot_rows)
    ht=pd.read_csv(R22/'by_class.csv')
    ht[(ht.scope=='all')&(ht.dose==1)&ht.class_id.isin([0,3,4,5])].to_csv(OUT/'heatmap_values.csv',index=False,encoding='utf-8-sig')
    (OUT/'source_assets.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'build_receipt.json').write_text(json.dumps(dict(file=str(target),slides=all_shapes,editable='native text, shapes, connectors and data bars; only SAR pixels are raster',render_review='pending'),ensure_ascii=False,indent=2),encoding='utf-8')
    print(target)
    print(json.dumps(all_shapes))


if __name__=='__main__':main()
