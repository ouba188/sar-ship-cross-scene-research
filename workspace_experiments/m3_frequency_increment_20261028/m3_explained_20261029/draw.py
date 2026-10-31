"""Explanatory M3 schematic and actual cached-case evidence, no training."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyArrowPatch

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans'],
    'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
    'ytick.labelsize': 7, 'legend.fontsize': 8, 'figure.titlesize': 9,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6,
    'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.width': .6,
    'ytick.major.width': .6, 'legend.frameon': False,
})
CATEGORICAL = ['#2166AC', '#B2182B', '#1B7837', '#F1A340', '#762A83', '#666666']
DIVERGING = ['#2166AC', '#F7F7F7', '#B2182B']
mpl.rcParams.update({'pdf.fonttype':42, 'svg.fonttype':'none', 'savefig.bbox':'tight', 'savefig.dpi':300})

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent/'joint_m3_readout'
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
mpl.rcParams.update({'font.sans-serif':['Microsoft YaHei','Arial'], 'axes.unicode_minus':False})
BLUE, RED, GREEN, ORANGE, _, GRAY = CATEGORICAL
INK = '#222222'


def save(fig, name):
    for extension in ('png','pdf','svg'):
        fig.savefig(ROOT/f'{name}.{extension}',dpi=300,bbox_inches='tight',facecolor='white')
    fig.savefig(ROOT/f'{name}_preview.png',dpi=140,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def schematic():
    fig, ax = plt.subplots(figsize=(15,9))
    ax.set(xlim=(0,100),ylim=(0,100)); ax.axis('off')
    fig.suptitle('M3：空间—频域条件化证据匹配与融合',fontsize=21,x=.09,ha='left',y=.98)
    ax.text(2,94,'核心：先决定参考哪里、参考多少，再把参考信息写回查询特征。VV / VH分别执行，后续分类器汇合。',fontsize=12,color=GRAY)

    def box(x,y,w,h,title,body,color=BLUE,fill='#F3F7FA',dashed=False):
        ax.add_patch(Rectangle((x,y),w,h,facecolor=fill,edgecolor=color,lw=1.35,ls='--' if dashed else '-'))
        ax.text(x+1.5,y+h-3,title,va='top',fontsize=13,fontweight='bold',color=color)
        ax.text(x+1.5,y+h-8,body,va='top',fontsize=10.5,linespacing=1.7,color=INK)
    def arrow(points,color=BLUE):
        for a,b in zip(points[:-2],points[1:-1]):
            ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=1.45)
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=14,color=color,lw=1.45))
    box(2,66,18,20,'① 区域证据读出','查询 + 源参考库\n框内 / 侧向 / 轴端\n深、浅层与频带描述')
    box(27,66,22,20,'② 条件化比较','逐局部对计算相似度\n动态融合空间与频域比较\n得到综合相似度 S')
    box(56,66,19,20,'③ 区域内可微匹配','部分最优传输 T\n允许部分响应不匹配\n不是强制逐点一一对齐')
    box(81,66,18,20,'⑥ 后续分类','更新特征进入后续层\n输出各船型类别分数\n核查正确类与错类分差',GREEN,'#F2F7F3')
    box(27,37,22,19,'参考内容分支 V','参考特征归一化\n1×1投影 → 区域聚合\n独立于比较用描述',GREEN,'#F2F7F3')
    box(56,37,19,19,'④ 聚合参考信息','T 对齐参考局部内容\n按参考权重 W 混合\n得到查询各区域更新',GREEN,'#F2F7F3')
    box(81,37,18,19,'⑤ 区域回写','按区域掩膜铺回特征图\n输出投影 × 增益 γ\nF′q = Fq + ΔF',GREEN,'#F2F7F3')
    arrow([(20,76),(27,76)])
    arrow([(49,76),(56,76)])
    arrow([(65.5,66),(65.5,56)])
    ax.text(66.7,60.5,'对应关系',fontsize=9,color=BLUE)
    arrow([(10,66),(10,46.5),(27,46.5)],GREEN)
    ax.text(11,49,'参考特征',fontsize=9,color=GREEN)
    arrow([(49,46.5),(56,46.5)],GREEN)
    arrow([(75,46.5),(81,46.5)],GREEN)
    arrow([(90,56),(90,66)],GREEN)
    arrow([(19,66),(19,28),(90,28),(90,37)],GRAY)
    ax.text(46,29.5,'原查询特征 Fq 直连：保留原有证据',fontsize=11,color=GRAY)
    box(2,3,48,19,'新读出只替换②中的频域比较（实验候选）',
        '正式：频带能量加权CNN摘要 → 比较相似性\n候选：频带均值强度 + 中心化局部排列 → 比较相似性\n虚线部分不代表已接入正式默认模型。',ORANGE,'#FFFDF7',True)
    box(56,3,43,19,'图中每一步都能对应可视化证据',
        '① 原图与区域　② 分项相似度　③ 匹配矩阵\n④ 实际参考权重　⑤ 回写差异　⑥ 类别分差\n权重和热图只是中间证据，需配合路径关闭对照。',GRAY,'#FAFAFA')
    save(fig,'M3_flow_ZH')


def evidence():
    sys.path.insert(0,str(BASE))
    import review
    import torch
    j, exp = review.joint, review.exp
    exp.DEVICE='cpu'
    exp.current.setup('cpu')
    rows,nf,refs,rid,arrays=exp.load()
    cache=j.pair_cache()
    index=next(i for i,r in enumerate(rows) if str(r['id'])=='24470')
    model=review.restore('N_level_pattern',43)
    with torch.no_grad():
        on, ds=j.forward(model,'N_level_pattern',cache,arrays,np.array([index]),rid,rows,details=True)
        for op in model.m3[0]: op.frequency=False
        off, do=j.forward(model,'N_level_pattern',cache,arrays,np.array([index]),rid,rows,details=True)
    refids=[str(rows[i]['id']) for i in rid]
    si,ci=refids.index('24527'),refids.index('31512')
    live=np.asarray(arrays['mass'][index,1])>0
    won=ds[1]['reference_weights'][0].detach().numpy()[:,live].mean(1)
    woff=do[1]['reference_weights'][0].detach().numpy()[:,live].mean(1)
    transport=ds[1]['transport'][0][0,si].detach().numpy()
    raw_transport=transport.copy()
    transport=transport/np.maximum(transport.sum(1,keepdims=True),1e-20)
    update=(ds[1]['update']-do[1]['update'])[0].detach().norm(dim=0).numpy()
    labels=np.array([3])
    mon=float(review.margin(on.detach().numpy(),labels)[0])
    moff=float(review.margin(off.detach().numpy(),labels)[0])
    table=pd.read_csv(BASE/'case_24470_band_dependency.csv').set_index('intervention')
    assert abs(mon-table.loc['all_on','true_margin'])<5e-4
    assert abs(moff-table.loc['all_frequency_off','true_margin'])<5e-4
    np.savez_compressed(ROOT/'case24470_traces.npz',transport=raw_transport,transport_row_normalized=transport,
        reference_weights_on=won,reference_weights_off=woff,frequency_update_norm=update,
        logits_on=on.detach().numpy(),logits_off=off.detach().numpy())
    (ROOT/'evidence.json').write_text(json.dumps(dict(query='24470',seed=43,model='N_level_pattern candidate',
        reference_ids=refids,same_reference='24527',other_reference='31512',
        margin_on=mon,margin_off=moff,mean_weight_on=won.tolist(),mean_weight_off=woff.tolist(),
        source_files=[str(BASE/'case_24470_band_dependency.csv'),str(BASE/'training/N_level_pattern_43/m3.pt')]),
        ensure_ascii=False,indent=2),encoding='utf-8')

    fig=plt.figure(figsize=(15,12))
    fig.suptitle('M3实例证据：24470 的频域信息如何进入判别',fontsize=20,x=.06,ha='left',y=.99)
    fig.text(.06,.952,'候选读出 N，种子43；真实标签：化学品/成品油轮。此图展示路径贡献，不代表优于仅空间对照。',fontsize=11,color=GRAY)
    grid=fig.add_gridspec(3,3,left=.065,right=.96,bottom=.085,top=.89,wspace=.32,hspace=.58,height_ratios=(1.1,1,.9))
    for col,(idx,title) in enumerate(((index,'A  查询船 24470'),(int(rid[si]),'B  同类参考 24527'),(int(rid[ci]),'C  混淆类参考 31512（散货）'))):
        sub=grid[0,col].subgridspec(1,2,wspace=.07)
        views=exp.data.prepare(rows[idx])
        for p,v in enumerate(views):
            ax=fig.add_subplot(sub[0,p])
            ax.imshow(v['pixels'],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            masks=exp.tokens.slots(v['geometry'],v['footprint'])
            ax.contour(masks[:8].sum(0),levels=[.05],colors=[GREEN],linewidths=.6)
            ax.set_title(('VV','VH')[p],fontsize=11)
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
        bounds=grid[0,col].get_position(fig)
        fig.text(bounds.x0,bounds.y1+.018,title,fontsize=12,fontweight='bold')
        fig.text(bounds.x0,bounds.y0-.015,'原生尺度裁剪/补零；绿色为框内区域',fontsize=9,color=GRAY)
    ax=fig.add_subplot(grid[1,0])
    im=ax.imshow(transport,cmap='Blues',vmin=0,vmax=1,interpolation='nearest',aspect='auto')
    ax.set_title('D  对应③：实际局部匹配',fontsize=12,loc='left',pad=12)
    ax.set_xlabel('同类参考的框内槽位',fontsize=10);ax.set_ylabel('查询框内槽位',fontsize=10)
    ax.set_xticks(range(8));ax.set_yticks(range(8))
    fig.colorbar(im,ax=ax,fraction=.04,pad=.03)
    ax.text(0,-.32,'VH → 参考24527；逐行归一化仅供显示。\n原始匹配量另存；不是正确类别概率。',transform=ax.transAxes,fontsize=9,color=GRAY,va='top')
    ax=fig.add_subplot(grid[1,1])
    im=ax.imshow(update,cmap='magma',interpolation='nearest')
    ax.set_title('E  对应⑤：回写在哪里改变',fontsize=12,loc='left',pad=12)
    ax.set_xticks([]);ax.set_yticks([])
    fig.colorbar(im,ax=ax,fraction=.04,pad=.03)
    ax.text(0,-.14,'VH频域开启/关闭的回写差异范数。\n10×10特征单元；不等于物理散射归因。',transform=ax.transAxes,fontsize=9,color=GRAY,va='top')
    ax=fig.add_subplot(grid[1,2])
    names=['全开','全关','关VV','关VH']
    values=[float(table.loc[k,'true_margin']) for k in ('all_on','all_frequency_off','VV_all_off','VH_all_off')]
    ax.bar(range(4),values,color=[BLUE,GRAY,BLUE,GRAY],width=.6)
    ax.axhline(0,color=INK,lw=.8)
    for k,v in enumerate(values): ax.text(k,v+(.001 if v>=0 else -.001),f'{v:+.4f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=10)
    ax.set(xticks=range(4),xticklabels=names,ylim=(-.016,.013))
    ax.set_title('F  对应⑥：是否改变判断',fontsize=12,loc='left',pad=12)
    ax.set_ylabel('正确类 − 最强错误类分数',fontsize=10)
    ax.text(0,-.24,'分差 > 0 判对；关闭VH频域后判错。\n仅空间配套训练仍判对：不能宣称新增优势。',transform=ax.transAxes,fontsize=9,color=GRAY,va='top')
    ax=fig.add_subplot(grid[2,:])
    x=np.arange(len(rid))
    ax.bar(x-.18,woff,width=.34,color=GRAY,label='频域关闭')
    ax.bar(x+.18,won,width=.34,color=BLUE,label='频域开启')
    for k in (si,refids.index('7790')): ax.axvspan(k-.48,k+.48,color=GREEN,alpha=.09,zorder=0)
    ax.set_xticks(x,refids,fontsize=9)
    ax.set_title('G  对应④：真正写入的参考混合权重，而非相似度排名',fontsize=12,loc='left',pad=12)
    ax.set_ylabel('平均参考权重',fontsize=10)
    ax.set_xlabel('全部16条参考；VH有效查询槽位平均；浅绿区是同类参考',fontsize=10)
    ax.legend(fontsize=10,ncol=2,loc='upper right')
    fig.text(.065,.025,'读图结论：频域会改变匹配与参考混合；本例的微小判别改善依赖VH频域组。热图本身不能证明原因，需与固定权重的关闭实验合看。',fontsize=11,color=INK)
    save(fig,'M3_case_evidence_ZH')
    print('MARGINS',mon,moff,'WEIGHTS_SUM',won.sum(),woff.sum(),flush=True)


if __name__=='__main__':
    ROOT.mkdir(exist_ok=True)
    schematic()
    evidence()
    print('DONE',ROOT,flush=True)
