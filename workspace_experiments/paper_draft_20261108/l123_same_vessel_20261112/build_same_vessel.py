"""One vessel across three historical evidence settings, without new inference."""
from pathlib import Path
import importlib.util
import json
import shutil
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.util import Inches
from pptx.oxml.xmlchemy import OxmlElement

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[1]
old=ROOT.parent/'l123_editable_20261112/build_deck.py'
spec=importlib.util.spec_from_file_location('draw_helpers',old)
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
d.prs=Presentation();d.prs.slide_width=Inches(18);d.prs.slide_height=Inches(10)
P=ROOT/'ports24'
A=ROOT/'same_vessel_assets';A.mkdir(exist_ok=True)
l1=pd.read_csv(P/'l1_observations.csv').set_index('obs_id')
l2=pd.read_csv(P/'l2_same_obs.csv')
l2=l2[(l2.obs_id==12457)&(l2.seed==17)].set_index(['pol','region','band'])
hist=WORK/'m3_frequency_increment_20261028/m3_true_rescue_12457_20261031'
replay=json.loads((hist/'REPLAY_RECEIPT.json').read_text(encoding='utf-8'))
rem=pd.read_csv(hist/'local_removal.csv').set_index('condition')
assert replay['case']=='12457' and int(l1.loc[12457,'mmsi'])==371234000
for oid in (12457,12458,12459,12460):
    assert int(l1.loc[oid,'mmsi'])==371234000 and int(l1.loc[oid,'class_id'])==0
    for pol in ('vv','vh'):
        shutil.copyfile(l1.loc[oid,'image_relpath_'+pol],A/f'{oid}_{pol}.png')


def native(s,oid,pol,x,y,w):
    path=A/f'{oid}_{pol}.png'
    with Image.open(path) as im:iw,ih=im.size
    # Same display pixels-per-inch for all original 64/72-pixel chips.
    pw,ph=w*iw/72,w*ih/72
    d.pic(s,path,x+(w-pw)/2,y+(w-ph)/2,pw,ph)


def effects(s,x,y,w):
    d.txt(s,'局部弱抑制后的分差变化 Δm',x,y,w,.38,17,bold=True)
    d.txt(s,'VV',x+2.6,y+.6,1,.3,16,d.C['teal'],True)
    d.txt(s,'VH',x+3.8,y+.6,1,.3,16,d.C['teal'],True)
    for j,(region,band,label) in enumerate([('body','fine','框内 · 细尺度'),('body','middle','框内 · 中尺度'),('side','fine','侧向 · 细尺度'),('side','middle','侧向 · 中尺度')]):
        yy=y+1.14+j*.58
        d.txt(s,label,x,yy,2.35,.4,16)
        for k,pol in enumerate(('vv','vh')):
            v=float(l2.loc[(pol,region,band),'delta_dynamic'])
            d.txt(s,f'{v:+.4f}',x+2.6+k*1.2,yy,1.15,.4,16,d.C['teal'] if v>=0 else d.C['red'])
        d.line(s,x,yy+.45,x+w,yy+.45)


def main():
    s=d.new('同一艘散货船的 L1–L3 判别证据链',
            'MMSI 371234000；以M3真实救回的观测12457为主例，反查同船自然观测和既有输入干预。',1)
    xs=(.65,6.45,12.25)
    for x,lv,name,en,col in zip(xs,('L1','L2','L3'),('同船自然观测','同图输入干预','内部路径反事实'),
         ('同船12457与12459','同一观测12457','同一观测12457'),(d.C['blue'],d.C['teal'],d.C['gold'])):
        d.stage_header(s,lv,name,en,x,col)
    for x in (6,11.8):d.line(s,x,2.94,x,8.85)
    for x in (5.75,11.55):d.line(s,x,2.12,x+.45,2.12,d.C['muted'],1.5,True)
    x=xs[0]
    d.txt(s,'12457 / Jebel Ali',x+.38,3.02,2.3,.35,15,bold=True)
    d.txt(s,'12459 / Qingdao',x+2.74,3.02,2.3,.35,15,bold=True)
    d.txt(s,'2025-04-10',x+.55,3.45,2.0,.3,13,d.C['muted'])
    d.txt(s,'2025-06-21',x+2.85,3.45,2.0,.3,13,d.C['muted'])
    for j,pol in enumerate(('vv','vh')):
        yy=3.99+j*1.64
        d.txt(s,pol.upper(),x,yy+.58,.48,.3,14,d.C['muted'])
        native(s,12457,pol,x+.58,yy,1.45)
        native(s,12459,pol,x+2.88,yy,1.45)
    d.txt(s,'普通货船 / 判错',x+.36,7.43,2.35,.4,17,d.C['red'],True)
    d.txt(s,'散货船 / 判对',x+2.77,7.43,2.25,.4,17,d.C['teal'],True)
    d.conclusion(s,'同一身份，在不同场景出现不同判断。',x,8.2,5.05,d.C['blue'])
    x=xs[1]
    d.txt(s,'原图12457；权重与16个源参考固定',x,3.03,5.05,.7,17,bold=True)
    effects(s,x,3.8,5.05)
    d.txt(s,'同一原图：不同区域、尺度及极化\n的局部改变，作用方向并不一致。',x,7.35,5.05,.73,16)
    d.conclusion(s,'局部响应有用与否，不能一刀切判断。',x,8.2,5.05,d.C['teal'])
    x=xs[2]
    d.txt(s,'保持原图不变，只撤除内部回写',x,3.03,5.05,.7,17,bold=True)
    vals=[('M3开启',replay['conditions'][1]['margin']),('M3关闭',replay['conditions'][0]['margin']),
          ('无VV框内',float(rem.loc['移除VV框内回写','margin'])),('无VH框内',float(rem.loc['移除VH框内回写','margin'])),('无VH侧向',float(rem.loc['移除VH侧向回写','margin']))]
    d.bars(s,vals,x,4.36,5.05,2.65,.95,15)
    d.txt(s,'M3开启判为散货船；其余四组判错。\n撤除局部回写可撤销这次救回。',x,7.35,5.05,.73,16)
    d.conclusion(s,'定位真实参与救回的区域补充路径。',x,8.2,5.05,d.C['gold'])
    d.txt(s,'模型阶段：L1＝10月10日双路诊断器；L2＝10月27日完整模型seed17；L3＝N_level_pattern / seed17空间M3。',.65,9.03,16.5,.43,13,d.C['muted'])
    d.txt(s,'同船、同观测联接，不是同一模型逐步升级曲线。L2/L3均以当前最高错误类计分；不同阶段数值不直接比较。',.65,9.5,15.7,.35,12,d.C['muted'])
    d.note(s,'L1 all 4 observations have same MMSI371234000 and fold1. L2 delta_dynamic recalculates strongest competitor; all eight interventions keep prediction correct in seed17. L3 freezes T,beta then removes local writeback. Simple aggregate also rescues: no claim fine matching is uniquely necessary. Sources '+str(P)+' ; '+str(hist))

    s=d.new('同船追溯：四次观测、四个港口、同一诊断器',
            '同一MMSI 371234000 / 同一类别散货船 / 同一fold1；原始chip保持相同显示像素尺度，无灰度重拉伸。',2)
    labels={0:'散货船',4:'集装箱船',5:'普通货船'}
    for j,oid in enumerate((12457,12458,12459,12460)):
        x=.7+j*4.35;r=l1.loc[oid]
        d.txt(s,f'{oid} / {r.port}',x,1.98,4.0,.42,21,d.C['blue'],True)
        d.txt(s,str(r.acq_time)[:10],x,2.48,3.8,.32,16,d.C['muted'])
        for k,pol in enumerate(('vv','vh')):
            yy=3.1+k*2.14
            d.txt(s,pol.upper(),x,yy+.65,.5,.35,16,d.C['muted'])
            native(s,oid,pol,x+.7,yy,1.86)
        correct=int(r.dual_prediction)==0
        d.txt(s,labels[int(r.dual_prediction)]+' / '+('判对' if correct else '判错'),x,7.4,3.9,.45,21,d.C['teal'] if correct else d.C['red'],True)
        d.txt(s,f'分差 {r.dual_margin:+.4f}',x,8.04,3.9,.4,18)
    d.txt(s,'这些自然观测是L1证据；同一张12457另有L2输入干预与L3内部路径对照，三层不再依赖不同船舶拼接。',.7,8.83,16.5,.48,16)
    d.txt(s,'跨港口同时伴随采集时相、观测条件等变化；本图不将差异归结为纯港口效应。',.7,9.49,16,.35,13,d.C['muted'])
    d.note(s,'Exact natural observation and prediction sources/line numbers in ports24/l1_observations.csv. All same-fold historical dual diagnostic. No new model run.')
    for slide in d.prs.slides:
        for sh in slide.shapes:
            sp=sh._element.find('{http://schemas.openxmlformats.org/presentationml/2006/main}spPr')
            if sp is not None:sp.append(OxmlElement('a:effectLst'))
            assert sh.left>=0 and sh.top>=0 and sh.left+sh.width<=d.prs.slide_width+2000 and sh.top+sh.height<=d.prs.slide_height+2000
    target=ROOT/'L1-L3_同船12457_可编辑.pptx';d.prs.save(target)
    print(target)


if __name__=='__main__':main()
