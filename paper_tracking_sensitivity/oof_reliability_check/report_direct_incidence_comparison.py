"""Summarize fixed same-information comparisons without selecting parameters."""

import numpy as np
import pandas as pd

from run_direct_incidence_comparison import OUT, SOURCE, MODES


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names={'reference':'原联合版本','same_vessel_correction':'同船响应校正',
           'direct_condition':'直接输入入射角','correction_plus_condition':'校正+直接条件'}
    df=pd.read_csv(OUT/'metrics.csv'); pm=pd.read_csv(OUT/'port_means.csv'); summary=pd.read_csv(OUT/'summary.csv')
    effects=[]
    for a,b,name in [('reference','same_vessel_correction','correction_without_direct'),
                     ('reference','direct_condition','direct_without_correction'),
                     ('direct_condition','correction_plus_condition','correction_given_direct'),
                     ('same_vessel_correction','correction_plus_condition','direct_given_correction'),
                     ('same_vessel_correction','direct_condition','direct_minus_correction')]:
        left=pm[pm['mode']==a].set_index('port'); right=pm[pm['mode']==b].set_index('port')
        for port in left.index:
            row=dict(port=port,comparison=name)
            for metric in ['full_oa','full_ba','base_oa','base_ba','expert_oa','expert_ba']:
                row[metric+'_delta_pp']=100*(right.loc[port,metric]-left.loc[port,metric])
            effects.append(row)
    effects=pd.DataFrame(effects); effects.to_csv(OUT/'paired_effects.csv',index=False)
    effects.groupby('comparison').mean(numeric_only=True).to_csv(OUT/'paired_effect_means.csv')
    full8=pd.read_csv(SOURCE/'manifest.csv',usecols=['port','class_id']).groupby('port').class_id.nunique()
    held=effects[effects.port.map(full8).eq(8)]
    held.groupby('comparison').mean(numeric_only=True).to_csv(OUT/'all8_paired_effect_means.csv')
    y=pd.read_csv(SOURCE/'manifest.csv',usecols=['class_id']).class_id.to_numpy(); recalls=[]
    for folder in sorted((OUT/'ports').iterdir()):
        with np.load(folder/'policies.npz') as f:
            truth=y[f['query_rows']]; pred=f['predictions']; masks=f['eval_mask']
        for draw,ev in enumerate(masks):
            for c in range(8):
                use=ev & (truth==c)
                if not use.any(): continue
                for m,mode in enumerate(MODES):
                    recalls.append(dict(port=folder.name,draw=draw,class_id=c,mode=mode,n=int(use.sum()),
                                        recall=float(np.mean(pred[draw,m,use]==c))))
    pd.DataFrame(recalls).to_csv(OUT/'class_recalls.csv',index=False)
    lines=['# 同船响应校正与直接条件输入：同信息对照', '',
        '2026-10-09。固定四方案、23目标港、原5组64标签、5%交接。所有目标预测先冻结再统一评价，原主模型保持不变。', '',
        '## 要回答什么', '',
        '有入射角带来收益，不自动说明同船配对学到的响应有额外价值。本轮比较原联合版本、同船校正、直接条件输入，以及二者结合。原方案和校正方案直接复用已冻结结果，另外两组重新拟合源分类头与原同组64标签适配。',
        '四组条件可用性相同，t=(angle-38)/5及t平方，缺失时两维置0；无额外缺失指示或角度插补。同船校正仍用既有源同船回归的角度分量，强度1。',
        '直接条件方案在两个源分类头增加独立的两维条件系数，源岭回归强度1；基础适配追加u/sqrt(2)，专家核追加0.25*u*u_support.T/2，适配岭参数仍为8。权重固定，未搜索。',
        '这里的同信息指推理侧入射角字段与标注预算一致；同船校正训练另使用源船身份与位置配对。不是等参数或等假设空间对照。直接条件增加自由系数；同船校正还改变视觉度量的交叉项，不能仅凭胜负唯一识别物理去噪机制。门控参数、几何、骨干、R18和原标注/交接预算均保持一致。', '',
        '## 完整识别', '',
        'OA/BA先按每港5组划分平均，再按港等权。23港共享八个输出类别，但不是每港八类齐全；另外报告八类齐全12港。BA平均查询实际出现类别召回。', '',
        '| 目标范围 | 方案 | OA (%) | BA (%) | 相对原OA变化pp | 相对原BA变化pp |',
        '|---|---|---:|---:|---:|---:|']
    for group in ['all23','all8_target12']:
        for mode in MODES:
            r=summary[(summary['group']==group)&(summary['mode']==mode)].iloc[0]
            lines.append(f'| {group} | {names[mode]} | {100*r.full_oa:.4f} | {100*r.full_ba:.4f} | {r.full_oa_delta_pp:+.4f} | {r.full_ba_delta_pp:+.4f} |')
    lines+=['', '## 最重要的机制比较', '',
        '| 比较 | 23港OA差pp | 23港BA差pp | OA提高港数 | BA提高港数 |',
        '|---|---:|---:|---:|---:|']
    for name,g in effects.groupby('comparison'):
        lines.append(f'| {name} | {g.full_oa_delta_pp.mean():+.4f} | {g.full_ba_delta_pp.mean():+.4f} | {(g.full_oa_delta_pp>1e-10).sum()}/23 | {(g.full_ba_delta_pp>1e-10).sum()}/23 |')
    lines+=['', 'correction_given_direct检验已直接输入入射角后，同船校正还增加多少收益；direct_minus_correction直接比较两种利用同一信息的方式。', '',
        '## 标注划分差异', '', '| 划分 | 方案 | 港均OA (%) | 港均BA (%) |','|---|---|---:|---:|']
    for (draw,mode),r in df.groupby(['draw','mode'])[['full_oa','full_ba']].mean().iterrows():
        lines.append(f'| {draw} | {names[mode]} | {100*r.full_oa:.4f} | {100*r.full_ba:.4f} |')
    lines+=['', '## 保留的边界', '',
        '这是重复使用的开发数据，不是独立确认；候选选择来自历史研发。固定门控没有随新接收模型重训，因此另存基础/专家指标区分接收模型与路由变化。不能把平均提点写成所有港都提高，也不能把同信息对照写成已经证明因果漂移。',
        '5项新增测试覆盖相同条件编码、原变换保留、零条件源拟合复现、独立原始空间加权岭公式及零条件预测复现。所有23港原方案/校正方案首组分数重放误差见verification.csv。', '',
        '逐港数值见port_means.csv，配对差异见paired_effects.csv，逐类召回见class_recalls.csv。结论与处置见DECISION_ZH.md。', '',
        '![同信息对照](same_information_comparison.png)']
    (OUT/'FINDINGS_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),constrained_layout=True)
    for ax,group,title in zip(axes,['all23','all8_target12'],['All 23 target ports','12 targets with all 8 classes']):
        s=summary[summary['group']==group].set_index('mode'); x=np.arange(3)
        for offset,field,label,color in [(-.18,'full_oa_delta_pp','OA','#197d80'),(.18,'full_ba_delta_pp','BA','#a63b63')]:
            bars=ax.bar(x+offset,[s.loc[m,field] for m in MODES[1:]],.34,label=label,color=color)
            ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=8)
        ax.set_xticks(x,['Vessel correction','Direct condition','Both'])
        ax.axhline(0,color='#777777',lw=.8); ax.set_title(title); ax.set_ylabel('Change from reference (pp)')
        ax.margins(y=.22); ax.legend(fontsize=8)
    fig.savefig(OUT/'same_information_comparison.png',dpi=170); plt.close(fig)
    print(effects.groupby('comparison').mean(numeric_only=True).to_string())


if __name__=='__main__': main()
