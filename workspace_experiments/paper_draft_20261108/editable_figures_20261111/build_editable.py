"""Native editable PowerPoint and SVG charts from verified experiment tables."""
from pathlib import Path
import csv
import json
import math
import xml.etree.ElementTree as ET

from pptx import Presentation
from pptx.util import Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment

ROOT = Path(__file__).resolve().parent
W, H = 960, 540
FONT = 'Microsoft YaHei'
INK, GRAY, LINE = '#20252B', '#616973', '#D8DDE2'
NEG, POS, TEAL, GOLD = '#397FA6', '#C64D56', '#287C83', '#B57926'
SVGN = 'http://www.w3.org/2000/svg'
ET.register_namespace('', SVGN)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def rgb(hexcode):
    return RGBColor.from_string(hexcode.lstrip('#'))


def blend(a, b, fraction):
    a, b = a.lstrip('#'), b.lstrip('#')
    return '#' + ''.join(f'{round(int(a[i:i+2],16)*(1-fraction)+int(b[i:i+2],16)*fraction):02X}'
                         for i in (0,2,4))


def diverging(value, limit):
    return blend('#FAFAFA', POS if value >= 0 else NEG, min(1,abs(value)/limit))


def sequential(value):
    return blend('#F5F8FA', '#195A78', max(0,min(1,value/100)))


class Page:
    """One geometry definition produces both editable native shapes and SVG."""
    def __init__(self, name, title, subtitle, notes):
        self.name, self.notes, self.items = name, notes, []
        self.text(32,18,896,36,title,24,bold=True)
        self.text(32,58,896,38,subtitle,13.5,color=GRAY)

    def text(self,x,y,w,h,text,size=16,color=INK,align='left',bold=False):
        self.items.append(dict(kind='text',x=x,y=y,w=w,h=h,text=str(text),size=size,color=color,align=align,bold=bold))

    def rect(self,x,y,w,h,fill,stroke=None,width=.6):
        self.items.append(dict(kind='rect',x=x,y=y,w=w,h=h,fill=fill,stroke=stroke,width=width))

    def line(self,x1,y1,x2,y2,color=INK,width=1):
        self.items.append(dict(kind='line',x1=x1,y1=y1,x2=x2,y2=y2,color=color,width=width))

    def marker(self,x,y,size,color,shape='circle',filled=True):
        self.items.append(dict(kind='marker',x=x,y=y,size=size,color=color,shape=shape,filled=filled))

    def foot(self,text):
        self.text(32,486,896,43,text,12,color=GRAY)

    def save_svg(self):
        root=ET.Element(f'{{{SVGN}}}svg',dict(width=f'{W}pt',height=f'{H}pt',viewBox=f'0 0 {W} {H}'))
        ET.SubElement(root,f'{{{SVGN}}}rect',dict(x='0',y='0',width=str(W),height=str(H),fill='white'))
        for p in self.items:
            if p['kind']=='text':
                anchor={'left':'start','center':'middle','right':'end'}[p['align']]
                tx=p['x']+({'left':0,'center':p['w']/2,'right':p['w']}[p['align']])
                lines=p['text'].split('\n')
                for j,line in enumerate(lines):
                    el=ET.SubElement(root,f'{{{SVGN}}}text',{
                        'x':str(tx),'y':str(p['y']+p['size']*.92+j*p['size']*1.16),
                        'font-family':FONT,'font-size':str(p['size']),'fill':p['color'],
                        'text-anchor':anchor,'font-weight':'bold' if p['bold'] else 'normal'})
                    el.text=line
            elif p['kind']=='rect':
                ET.SubElement(root,f'{{{SVGN}}}rect',dict(x=str(p['x']),y=str(p['y']),width=str(p['w']),height=str(p['h']),
                    fill=p['fill'] or 'none',stroke=p['stroke'] or 'none',**{'stroke-width':str(p['width'])}))
            elif p['kind']=='line':
                ET.SubElement(root,f'{{{SVGN}}}line',dict(x1=str(p['x1']),y1=str(p['y1']),x2=str(p['x2']),y2=str(p['y2']),
                    stroke=p['color'],**{'stroke-width':str(p['width'])}))
            else:
                size=p['size'];x=p['x'];y=p['y']
                style=dict(fill=p['color'] if p['filled'] else 'white',stroke=p['color'],**{'stroke-width':'1.5'})
                if p['shape']=='circle':
                    ET.SubElement(root,f'{{{SVGN}}}circle',dict(cx=str(x),cy=str(y),r=str(size/2),**style))
                elif p['shape']=='square':
                    ET.SubElement(root,f'{{{SVGN}}}rect',dict(x=str(x-size/2),y=str(y-size/2),width=str(size),height=str(size),**style))
                else:
                    points=f'{x},{y-size/2} {x+size/2},{y} {x},{y+size/2} {x-size/2},{y}'
                    ET.SubElement(root,f'{{{SVGN}}}polygon',dict(points=points,**style))
        ET.ElementTree(root).write(ROOT/'svg'/f'{self.name}.svg',encoding='utf-8',xml_declaration=True)

    def add_ppt(self, prs):
        slide=prs.slides.add_slide(prs.slide_layouts[6])
        slide.background.fill.solid();slide.background.fill.fore_color.rgb=rgb('#FFFFFF')
        for p in self.items:
            if p['kind']=='text':
                box=slide.shapes.add_textbox(Pt(p['x']),Pt(p['y']),Pt(p['w']),Pt(p['h']))
                tf=box.text_frame;tf.clear();tf.word_wrap=False
                tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0
                tf.vertical_anchor=MSO_ANCHOR.TOP
                for j,line in enumerate(p['text'].split('\n')):
                    par=tf.paragraphs[0] if j==0 else tf.add_paragraph()
                    par.alignment={'left':PP_ALIGN.LEFT,'center':PP_ALIGN.CENTER,'right':PP_ALIGN.RIGHT}[p['align']]
                    par.space_before=par.space_after=Pt(0);par.line_spacing=1.0
                    run=par.add_run();run.text=line
                    run.font.name=FONT;run.font.size=Pt(p['size']);run.font.bold=p['bold'];run.font.color.rgb=rgb(p['color'])
            elif p['kind']=='line':
                shape=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Pt(p['x1']),Pt(p['y1']),Pt(p['x2']),Pt(p['y2']))
                shape.line.color.rgb=rgb(p['color']);shape.line.width=Pt(p['width'])
            else:
                if p['kind']=='rect':
                    x,y,w,h=p['x'],p['y'],p['w'],p['h'];kind=MSO_AUTO_SHAPE_TYPE.RECTANGLE
                    fill,stroke,width=p['fill'],p['stroke'],p['width']
                else:
                    x,y=p['x']-p['size']/2,p['y']-p['size']/2;w=h=p['size']
                    kind={'circle':MSO_AUTO_SHAPE_TYPE.OVAL,'square':MSO_AUTO_SHAPE_TYPE.RECTANGLE,'diamond':MSO_AUTO_SHAPE_TYPE.DIAMOND}[p['shape']]
                    fill=p['color'] if p['filled'] else '#FFFFFF';stroke=p['color'];width=1.5
                shape=slide.shapes.add_shape(kind,Pt(x),Pt(y),Pt(max(w,.05)),Pt(max(h,.05)))
                shape._element.spPr.append(OxmlElement('a:effectLst'))
                if fill:shape.fill.solid();shape.fill.fore_color.rgb=rgb(fill)
                else:shape.fill.background()
                if stroke:shape.line.color.rgb=rgb(stroke);shape.line.width=Pt(width)
                else:shape.line.fill.background()
        slide.notes_slide.notes_text_frame.text=self.notes


def colorbar(page,x,y,w,limit,div=True):
    for i in range(80):
        val=-limit+2*limit*(i+.5)/80 if div else 100*(i+.5)/80
        page.rect(x+i*w/80,y,w/80+.15,10,diverging(val,limit) if div else sequential(val))
    ticks=[-limit,-limit/2,0,limit/2,limit] if div else [0,25,50,75,100]
    for v in ticks:
        px=x+w*((v+limit)/(2*limit) if div else v/100)
        page.text(px-30,y+13,60,20,f'{v:g}',11,align='center',color=GRAY)


CLASS_SHORT={'bulk_carrier':'散货船','fishing_vessel':'渔船','general_cargo':'普通货船',
 'product_chemical_tanker':'化学品/\n成品油轮','container_ship':'集装箱船',
 'crude_oil_tanker':'原油轮','tug_towing':'拖轮/拖带','offshore_supply':'海工补给船'}


def conditions():
    pages=[]
    main=read_csv(ROOT/'condition_data'/'condition_long.csv')
    extra=[r for r in read_csv(ROOT/'condition_data'/'round22_condition_long.csv')
           if r['scope']=='expansion_only' and float(r['dose'])==1]
    settings=[('01_condition_joint',main,'joint','OSS1联合极化诊断读出',10),
              ('02_condition_single',main,'single','OSS1对应单极化诊断读出',10),
              ('03_condition_24port',extra,'joint','24港历史诊断读出',None)]
    for name,data,readout,title,limit in settings:
        rows=[r for r in data if r['readout']==readout]
        if not rows:raise ValueError((name,'no records'))
        classes=sorted({(int(r['class_id']),r['class_name']) for r in rows})
        keys=[(p,r,s) for p in ('VV','VH') for r in ('body','side','axis_end') for s in ('fine','mid')]
        lookup={(r['polarization'],r['region'],r['scale'],int(r['class_id'])):r for r in rows}
        vals=[float(r['delta_margin'])*1000 for r in rows]
        if limit is None:limit=math.ceil(max(map(abs,vals))/5)*5
        detail=('1165次观测 · 969艘船 · 22过境组 · 船等权 · 共同区域掩膜' if name!='03_condition_24port'
                else '209次观测 · 104艘船 · 31过境组 · 船等权 · 各极化自身区域掩膜')
        notes=(ROOT/'condition_data'/'README.md').read_text(encoding='utf-8')
        p=Page(name,'条件效应矩阵｜'+title,detail+'\n数值：Δm × 10³；高剂量预设1.0，区域×尺度的削弱操作；非当前Full分类结果',notes)
        x,y,cw,ch=220,151,680/len(classes),24
        for j,(c,cl) in enumerate(classes):
            ns={int(r['n_ships']) for r in rows if int(r['class_id'])==c}
            n=str(min(ns)) if len(ns)==1 else f'{min(ns)}–{max(ns)}'
            label=CLASS_SHORT.get(cl,cl)
            mark='†' if min(ns)<20 else ''
            p.text(x+j*cw,99,cw,49,label+f'\nn={n}{mark}',13.5,align='center')
        for i,(pol,region,scale) in enumerate(keys):
            region_cn={'body':'框内','side':'侧向','axis_end':'轴端'}[region]
            scale_cn={'fine':'细尺度','mid':'中尺度'}[scale]
            p.text(35,y+i*ch+3,174,ch,f'{pol} · {region_cn} · {scale_cn}',14,align='right')
            for j,(c,cl) in enumerate(classes):
                row=lookup[pol,region,scale,c];v=float(row['delta_margin'])*1000
                p.rect(x+j*cw,y+i*ch,cw,ch,diverging(v,limit),'#FFFFFF',.6)
                p.text(x+j*cw,y+i*ch+3,cw,ch,f'{v:+.2f}',13,color='#FFFFFF' if abs(v)>limit*.68 else INK,align='center')
            if i==5:p.line(x,y+6*ch,x+680,y+6*ch,INK,1.1)
        colorbar(p,x,452,680,limit)
        p.foot('正值：削弱该分量后，真实类对固定竞争类的分差增大；负值相反。不是准确率增量。\n† 少于20艘船，仅标注样本量、不表示显著性。不同历史诊断器不合并估计。')
        pages.append(p)
    return pages


def confusion_pages(results):
    pages=[]
    for ds,name in [('OSS2','04_confusion_OSS2'),('24ports','05_confusion_24ports')]:
        d=results[ds];k=len(d['labels']);display='OSS2四分类' if ds=='OSS2' else '自建24港八分类'
        p=Page(name,display+'｜Plain 与完整模型的混淆矩阵',
            f"{d['observations']}条测试观测 · {d['ships']}艘船 · 三训练种子平均 · 行归一化（%）",
            'Counts from archived predictions; each row averages the same cohort over seeds17/29/43. No ensemble. Source: result_data/figures.json and confusion_by_seed.csv.')
        size=288;cell=size/k;y=138
        short=['散货','普通货','化学品\n油轮','集装箱'] if k==4 else ['散货','渔船','普通货','化学品\n油轮','集装箱','原油轮','拖轮','海工']
        for arm,x in [('plain',114),('full',578)]:
            p.text(x,101,size,28,'Plain' if arm=='plain' else 'Full model',20,align='center',bold=True)
            for i,label in enumerate(short):
                p.text(x-80,y+i*cell+cell/2-16,72,34,label,13,align='right')
                xlabel=label if k==4 else ['散货','渔船','普通\n货船','化/成\n油轮','集装\n箱船','原油\n轮','拖轮','海工'][i]
                p.text(x+i*cell,y+size+6,cell,40,xlabel,11.5 if k==8 else 14,align='center')
                for j,value in enumerate(d['confusion'][arm][i]):
                    p.rect(x+j*cell,y+i*cell,cell,cell,sequential(value),'#FFFFFF',.8)
                    p.text(x+j*cell,y+i*cell+cell/2-9,cell,22,f'{value:.1f}',12 if k==8 else 17,
                           color='#FFFFFF' if value>57 else INK,align='center')
            p.text(x,y+size+43,size,22,'预测类别',13,align='center',color=GRAY)
        p.text(32,103,75,24,'真实类别',12,color=GRAY)
        p.foot('两图共用0–100%的色阶；对角线为该类召回率。n按独立观测计，未将三种子扩算为三倍样本。\n逐格原始计数、百分比及每类样本量见配套数据表；所有类别与混淆均保留。')
        pages.append(p)
    return pages


def horizontal_scene(name,title,groups,subtitle):
    rows=sorted(groups,key=lambda r:r['oa_delta'],reverse=True)
    p=Page(name,title,subtitle,'Paired Full-minus-Plain differences; same query set and seed. BA over classes present in each group. Horizontal error bars: SD across 3 training seeds, not a confidence interval.')
    n=len(rows);top=132;rh=min(55 if n<=8 else 17.8,310/n);bottom=top+n*rh
    extreme=[r[key]+s*r[key.replace('delta','std')] for r in rows for key in ('oa_delta','ba_delta') for s in (-1,1)]
    low=math.floor(min(min(extreme),0)/10)*10;high=math.ceil(max(max(extreme),0)/10)*10
    if high==low:high=low+10
    for key,x in [('oa_delta',266),('ba_delta',620)]:
        width=260;mapx=lambda value:x+(value-low)/(high-low)*width
        p.text(x,101,width,24,'ΔOA（百分点）' if key=='oa_delta' else 'ΔBA（百分点）',17,align='center',bold=True)
        for tick in range(int(low),int(high)+1,20 if high-low>=60 else 10):
            px=mapx(tick);p.line(px,top-4,px,bottom+2,LINE,.5)
            p.text(px-25,bottom+9,50,22,f'{tick}',12,align='center',color=GRAY)
        p.line(mapx(0),top-4,mapx(0),bottom+2,GRAY,1)
        for i,row in enumerate(rows):
            yy=top+i*rh+rh/2;val=row[key];sd=row[key.replace('delta','std')]
            color=TEAL if val>0 else POS if val<0 else GRAY
            p.rect(min(mapx(0),mapx(val)),yy-rh*.25,abs(mapx(val)-mapx(0)),rh*.5,color)
            p.line(mapx(val-sd),yy,mapx(val+sd),yy,INK,.75)
            for end in (val-sd,val+sd):p.line(mapx(end),yy-2.4,mapx(end),yy+2.4,INK,.75)
            p.text(x+width+7,yy-8,55,20,f'{val:+.1f}',12,color=color)
    for i,row in enumerate(rows):
        label=row['group']
        if label.startswith('S1A_'):label=label.replace('S1A_','')
        p.text(35,top+i*rh+rh/2-8,215,21,f"{label}  (n={row['n']})",12.5,align='right')
    p.foot('条形：三种子配对增量均值；误差线：训练种子标准差（不是置信区间）。\n场景BA只对该组实际出现类别等权；保留小样本与负增益组。n为测试观测数。')
    return p


def overpass_page(groups):
    code={r['group']:f'S{i+1:02d}' for i,r in enumerate(sorted(groups,key=lambda r:r['group']))}
    rows=sorted(groups,key=lambda r:r['oa_delta'])
    p=Page('07_scenes_overpasses','自建24港｜各过境组的性能增量分布',
           f'{len(rows)}个原协议过境组 · Full − Plain · 三种子配对平均；按ΔOA升序，两个面板顺序相同',
           'Use original pass_group keys, not silently merged orbit numbers. Scene codes map in result_data/scene_codes.csv. BA is present-class BA. Every group retained, small-n groups outlined.')
    x,w=82,820;step=w/len(rows)
    for key,y in [('oa_delta',126),('ba_delta',314)]:
        height=111;limit=max(20,math.ceil(max(abs(r[key]) for r in rows)/20)*20)
        mapy=lambda v:y+height/2-v/limit*height/2
        for tick in (-limit,0,limit):
            p.line(x,mapy(tick),x+w,mapy(tick),GRAY if tick==0 else LINE,.9 if tick==0 else .4)
            p.text(38,mapy(tick)-9,36,22,f'{tick:g}',12,align='right',color=GRAY)
        p.text(x,y-31,w,24,'ΔOA（百分点）' if key=='oa_delta' else 'ΔBA（百分点）',16,bold=True)
        for i,r in enumerate(rows):
            val=r[key];xx=x+i*step+step*.1;color=TEAL if val>=0 else POS
            p.rect(xx,min(mapy(0),mapy(val)),step*.8,abs(mapy(val)-mapy(0)),
                   color if r['n']>=10 else '#FFFFFF',color,.7)
            if i%10==0 or i==len(rows)-1:p.text(xx-15,y+height+8,40,20,code[r['group']],11,align='center',color=GRAY)
        pos=sum(r[key]>1e-9 for r in rows);neg=sum(r[key]<-1e-9 for r in rows)
        p.text(610,y-31,292,25,f'正 / 平 / 负：{pos} / {len(rows)-pos-neg} / {neg}',13,align='right')
    p.foot('实心：n≥10；空心：n<10。每个条形是一组，不代表独立港口，也未按样本量改变条宽。\n场景编号、原始过境键、样本数及类别覆盖见scene_codes.csv；不以小样本高增益代替总体结果。')
    with (ROOT/'result_data'/'scene_codes.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['scene_code','pass_group','n','present_classes','oa_delta','ba_delta'])
        writer.writeheader()
        for r in rows:writer.writerow(dict(scene_code=code[r['group']],pass_group=r['group'],**{k:r[k] for k in ['n','present_classes','oa_delta','ba_delta']}))
    return p


def reference_page():
    data=read_csv(ROOT/'reference_data'/'reference_plot_ready.csv')
    p=Page('09_reference_budget_historical','源参考数量｜历史候选的已测规模点',
           '固定权重与查询 · 单模型种子17 · 非当前RCEW完整验证 · 未测预算不补点',
           (ROOT/'reference_data'/'README.md').read_text(encoding='utf-8'))
    styles={'random':(TEAL,'circle','随机库'),'coverage':(GOLD,'square','覆盖库'),'original':(INK,'diamond','原始库')}
    for j,(kind,(color,shape,label)) in enumerate(styles.items()):
        xx=290+j*150;p.marker(xx,105,9,color,shape,kind=='random');p.text(xx+14,95,115,24,label,14)
    for dataset,x in [('private',100),('oss2',550)]:
        subset=[r for r in data if r['dataset']==dataset]
        for metric,y in [('oa',160),('ba',335)]:
            width,height=310,87;vals=[float(r[metric+'_mean']) for r in subset]
            lo=math.floor(min(vals))-1;hi=math.ceil(max(vals))+1
            if hi-lo<2:hi=lo+2
            mx=lambda k:x+math.log2(k)/4*width
            my=lambda v:y+height-(v-lo)/(hi-lo)*height
            p.text(x,y-29,width,24,('自建开发集（160条）' if dataset=='private' else 'OSS2验证集（228条）')+' · '+metric.upper(),15,bold=True)
            for i in range(3):
                v=lo+(hi-lo)*i/2;yy=my(v)
                p.line(x,yy,x+width,yy,LINE,.6);p.text(x-48,yy-8,40,22,f'{v:g}',12,align='right',color=GRAY)
            for k in (1,2,4,8,16):
                px=mx(k);p.line(px,y+height,px,y+height+4,GRAY,.6)
                p.text(px-17,y+height+8,34,20,str(k),12,align='center')
            for kind in ('coverage','random','original'):
                series=sorted([r for r in subset if r['strategy']==kind],key=lambda r:int(r['references_per_class']))
                color,shape,label=styles[kind]
                points=[(mx(int(r['references_per_class'])),my(float(r[metric+'_mean']))) for r in series]
                for a,b in zip(points,points[1:]):p.line(*a,*b,color,1.7)
                for xx,yy in points:p.marker(xx,yy,12 if kind=='coverage' else 8 if kind=='random' else 17,color,shape,kind=='random')
            p.text(x,y+height+29,width,20,'每类源参考船数（log₂刻度）',12,align='center',color=GRAY)
    p.foot('只测得：自建每类2/4，OSS2每类8/16；每点仅一次模型/参考配置，无误差条。纵轴局部放大。\nOSS2各策略实测点完全重合。缺少同协议延时数据，未添加延时曲线。')
    return p


def workbook():
    wb=Workbook();index=wb.active;index.title='文件索引'
    index.append(['数据表','CSV文件','说明'])
    index.append(['范围','当前分类结果与历史机制/参考诊断分开','缺失结果不填零'])
    paths=[]
    for folder in ('result_data','condition_data','reference_data'):
        paths += [p for p in (ROOT/folder).glob('*.csv') if 'selected_observations' not in p.name and 'membership' not in p.name]
    for i,path in enumerate(paths,1):
        title=f'{i:02d}_{path.stem}'[:31];ws=wb.create_sheet(title)
        with path.open(encoding='utf-8-sig',newline='') as f:
            for row in csv.reader(f):
                values=[]
                for cell in row:
                    try:values.append(float(cell) if any(c in cell.lower() for c in ('.','e')) else int(cell))
                    except ValueError:values.append(cell)
                ws.append(values)
        ws.freeze_panes='B2';ws.auto_filter.ref=ws.dimensions
        for cell in ws[1]:cell.font=Font(bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='286A78')
        for col in ws.columns:
            letter=col[0].column_letter
            ws.column_dimensions[letter].width=min(42,max(12,max(len(str(c.value or '')) for c in list(col)[:80])+2))
        index.append([title,str(path.relative_to(ROOT)),'实测数据；定义见对应README'])
    index.column_dimensions['A'].width=35;index.column_dimensions['B'].width=80;index.column_dimensions['C'].width=50
    wb.save(ROOT/'图表数据.xlsx')


def main():
    (ROOT/'svg').mkdir(exist_ok=True)
    results=json.loads((ROOT/'result_data'/'figures.json').read_text(encoding='utf-8'))
    pages=conditions()+confusion_pages(results)
    pages += [horizontal_scene('06_scenes_ports','自建24港｜港口水域中的性能增量',
        results['24ports']['scenes']['port'],'3075条测试观测 · 18个测试港口 · Full − Plain · 按ΔOA均值排序')]
    pages += [overpass_page(results['24ports']['scenes']['pass_group'])]
    pages += [horizontal_scene('08_scenes_OSS2','OSS2｜各测试场景的性能增量',
        results['OSS2']['scenes']['pass_group'],'378条测试观测 · 5个过境组 · Full − Plain · 与主结果相同查询')]
    pages += [reference_page()]
    prs=Presentation();prs.slide_width=Pt(W);prs.slide_height=Pt(H)
    for page in pages:page.save_svg();page.add_ppt(prs)
    prs.save(ROOT/'实验图表_全部可编辑.pptx')
    workbook()
    (ROOT/'figure_manifest.json').write_text(json.dumps([
        dict(slide=i+1,name=p.name,objects=len(p.items),notes=p.notes) for i,p in enumerate(pages)],ensure_ascii=False,indent=2),encoding='utf-8')
    print('Native slides:',len(pages),'objects:',sum(len(p.items) for p in pages))


if __name__=='__main__':main()
