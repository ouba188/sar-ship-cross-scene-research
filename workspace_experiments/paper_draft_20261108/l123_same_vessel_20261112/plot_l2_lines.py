"""Plot recorded eight-condition effects; editable native chart and SVG."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_MARKER_STYLE, XL_TICK_MARK, XL_TICK_LABEL_POSITION
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.oxml.xmlchemy import OxmlElement

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'expanded/l2_line_chart';OUT.mkdir(exist_ok=True)
records=json.loads((ROOT/'expanded/l2_assets.json').read_text(encoding='utf-8'))['records']
labels=[f"{r['pol'].upper()}\n{'框内' if r['region']=='body' else '侧向'}·{'细尺度' if r['band']=='fine' else '中尺度'}" for r in records]
ys=[r['delta_margin'] for r in records]
yc=[r['control_delta'] for r in records]
with (OUT/'data.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['条件','结构干预Δm','同均值对照Δm'])
    w.writerows(zip([s.replace('\n','-') for s in labels],ys,yc))
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':12,'axes.unicode_minus':False,'svg.fonttype':'none','pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(12,6.1))
fig.subplots_adjust(left=.085,right=.98,bottom=.19,top=.81)
x=np.arange(8)
ax.axvspan(-.4,3.5,color='#346A91',alpha=.035)
ax.axvspan(3.5,7.4,color='#147D79',alpha=.035)
ax.axvline(3.5,color='#C7D3D8',lw=1,ls=':')
ax.axhline(0,color='#687780',lw=1.1,ls='--')
ax.plot(x,ys,'-o',color='#147D79',lw=2,ms=7,label='结构干预')
ax.plot(x,yc,'--s',color='#B76A31',lw=1.8,ms=6,label='同均值变化对照')
for i,(a,b) in enumerate(zip(ys,yc)):
    ax.annotate(f'{a:+.4f}',(i,a),xytext=(0,10 if a>b else -17),textcoords='offset points',ha='center',color='#147D79',fontsize=10)
    ax.annotate(f'{b:+.4f}',(i,b),xytext=(0,-17 if a>b else 10),textcoords='offset points',ha='center',color='#A15824',fontsize=10)
ax.set_xticks(x,labels);ax.set_xlim(-.4,7.4);ax.set_ylim(-.14,.08)
ax.set_yticks(np.arange(-.12,.081,.04))
ax.set_ylabel('分类分差变化 Δm')
ax.grid(axis='y',alpha=.18)
ax.spines[['top','right']].set_visible(False)
ax.legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=2,frameon=False)
fig.suptitle('L2 条件效应与强度对照',x=.085,y=.97,ha='left',fontsize=19,fontweight='bold')
fig.text(.085,.903,'同一观测12457 · 历史完整模型seed17 · 8项结构干预的像素L2均为0.104540',fontsize=11,color='#56666E')
fig.text(.085,.045,'Δm > 0：正确类别相对最高错误类的优势增加。横轴为离散条件，连线不表示连续变化。',fontsize=10,color='#56666E')
for ext in ('png','svg','pdf'):fig.savefig(OUT/f'L2_八项条件效应折线图.{ext}',dpi=220)
plt.close(fig)

prs=Presentation();prs.slide_width=Inches(14);prs.slide_height=Inches(7.7)
s=prs.slides.add_slide(prs.slide_layouts[6])
def text(value,x,y,w,h,size=18,color='203039',bold=False):
    sh=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    tf=sh.text_frame;tf.word_wrap=True
    tf.text=value
    for p in tf.paragraphs:
        for r in p.runs:
            r.font.name='Microsoft YaHei';r.font.size=Pt(size);r.font.bold=bold;r.font.color.rgb=RGBColor.from_string(color)
            ea=OxmlElement('a:ea');ea.set('typeface','Microsoft YaHei');r._r.get_or_add_rPr().append(ea)
text('L2 条件效应与强度对照',.5,.25,13,.5,26,bold=True)
text('同一观测12457 · 历史完整模型seed17 · 8项结构干预的像素L2均为0.104540',.5,.86,13,.42,14,'56666E')
data=CategoryChartData();data.categories=labels
data.add_series('结构干预',ys);data.add_series('同均值变化对照',yc);data.add_series('零效应',[0.]*8)
chart=s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(.65), Inches(1.48), Inches(12.65), Inches(5.35),data).chart
chart.has_legend=True;chart.legend.position=XL_LEGEND_POSITION.TOP;chart.legend.include_in_layout=False
chart.legend.font.name='Microsoft YaHei';chart.legend.font.size=Pt(13)
chart.value_axis.minimum_scale=-.14;chart.value_axis.maximum_scale=.08;chart.value_axis.major_unit=.04
chart.value_axis.has_major_gridlines=True
chart.value_axis.tick_labels.number_format='0.00'
chart.value_axis.tick_labels.font.size=Pt(12)
chart.category_axis.tick_labels.font.name='Microsoft YaHei';chart.category_axis.tick_labels.font.size=Pt(12)
chart.category_axis.tick_label_position=XL_TICK_LABEL_POSITION.LOW
chart.category_axis.major_tick_mark=XL_TICK_MARK.NONE
chart.value_axis.has_title=True;chart.value_axis.axis_title.text_frame.text='分类分差变化 Δm'
for i,(color,marker) in enumerate([('147D79',XL_MARKER_STYLE.CIRCLE),('B76A31',XL_MARKER_STYLE.SQUARE),('687780',XL_MARKER_STYLE.NONE)]):
    series=chart.series[i];series.format.line.color.rgb=RGBColor.from_string(color);series.format.line.width=Pt(2 if i<2 else 1)
    series.marker.style=marker
    if i<2:
        series.marker.size=7;series.marker.format.fill.solid();series.marker.format.fill.fore_color.rgb=RGBColor.from_string(color)
        series.marker.format.line.color.rgb=RGBColor.from_string(color)
    else:
        dash=OxmlElement('a:prstDash');dash.set('val','dash');series.format.line._get_or_add_ln().append(dash)
text('Δm > 0：正确类别相对最高错误类的优势增加。横轴为离散条件，连线不表示连续变化。',.5,7.0,13,.4,13,'56666E')
s.notes_slide.notes_text_frame.text='Source: '+str(ROOT/'expanded/l2_assets.json')+'\nNative editable chart with embedded workbook. No new training/inference. Same-mean control matches total grayscale change, not pixel L2. All 8 treatments remain correct.'
prs.save(OUT/'L2_八项条件效应折线图_可编辑.pptx')
print(OUT)
