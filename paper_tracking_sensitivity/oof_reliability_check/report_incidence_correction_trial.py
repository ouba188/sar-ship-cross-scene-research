"""Compact combined evidence report; read saved policies without new fits."""

import numpy as np
import pandas as pd

from run_incidence_correction_trial import OUT, PROBE, SOURCE


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    summary=pd.read_csv(OUT/'summary.csv')
    pm=pd.read_csv(OUT/'port_means.csv')
    changed=pm[pm['mode']=='incidence_corrected']
    response=pd.read_csv(PROBE/'response_means.csv')
    frame=pd.read_csv(SOURCE/'manifest.csv',usecols=['class_id'])
    recalls=[]
    for folder in sorted((OUT/'ports').iterdir()):
        with np.load(folder/'policies.npz') as f:
            truth=frame.loc[f['query_rows'],'class_id'].to_numpy()
            for draw,ev in enumerate(f['eval_mask']):
                for c in range(8):
                    use=ev & (truth==c)
                    if not use.any():
                        continue
                    before=np.mean(f['predictions'][draw,0,use]==c)
                    after=np.mean(f['predictions'][draw,1,use]==c)
                    recalls.append(dict(port=folder.name,draw=draw,class_id=c,n=int(use.sum()),
                        reference_recall=before,corrected_recall=after,delta_pp=100*(after-before)))
    pd.DataFrame(recalls).to_csv(OUT/'class_recalls.csv',index=False)
    lines=['# 入射角相关漂移：识别与校正实验', '',
        '2026-10-09。结论：发现可在留出船与留出月份上复现的入射角相关隐空间响应；固定条件校正使完整联合模型平均BA/OA同时小幅提高。它是有效候选，不是已识别的纯噪声或物理因果分量。', '',
        '## 从哪里来', '',
        '代表性SAFE的复用由原导出脚本明确记录，不适合逐产品读取时间和轨道。此次改用已有独立缓存：fetch_s1_grid.py通过完整产品编号访问公开AWS annotation，提取经纬度与入射角网格；现有sidecar按裁片位置插值。生产路径已核查，本轮没有重新逐景下载XML。',
        '全38091行中28628行有获准位置来源和有效入射角，约75.16%。历史坐标、缺失坐标不填补。returned_georef_center是VV/VH均值或单可用视图的参考位置，不宣称两视图完全共注册。', '',
        '## 识别流程', '',
        '源数据排除目标港、目标身份和产品；按同港同MMSI相邻时相配对。用入射角及其平方的变化预测隐特征差，同时加入位置位移对照。固定保留约1/3船身份、3/6/9/12月份进行外推诊断。',
        '在同时换船和换月份的诊断中，候选稳定/漂移联合空间的平均预测skill：入射角2.7805%，位置-0.5249%，二者联合2.4844%。联合比仅位置多3.0093个百分点；这不是分类BA/OA，也不是已确认的因果解释比例。入射角及联合模型在23个重叠源划分均为正。',
        '仅换船的结果更弱：角度平均0.9187%，联合平均0.2886%，分别18/23和13/23为正。原128维保留特征的同时换船换月skill为负，未复制同样的泛化。23个源划分高度重叠，不是23项独立证据。', '',
        '## 校正怎么接入', '',
        '令t=(入射角-38)/5。源同船差分拟合得到角度/位置联合响应W，只取角度两行计算shift=[t,t^2]W_angle，校正z_new=z-shift。参考角度38度、强度1固定，不依据目标查询标签搜索；不删除整片子空间，不校正位置项。',
        '通过既有线性映射将同一校正同步传到源/目标R50路径，重新拟合两个源分类头，并用原同一组64标签重新适配基础模型与专家。R18、骨干、几何及归一化不变。没有有效入射角的行不直接移动特征，但共享分类头变化仍可能改变其预测。',
        '重算置信度与5%交接；原门控参数未重训。因此这不是全系统所有组件都重训的最终版本。基础和专家的独立指标也提高，收益不只是交接排名变化。全部23港预测冻结后才统一读查询标签评价。', '',
        '## 完整八输出识别', '',
        '先平均每港5组划分，再按港等权。23港并非每港都有八类；另外单列查询八类齐全的12港。BA按各查询集实际出现类别平均召回，不给模型目标类别真值白名单。', '',
        '| 目标范围 | 方案 | OA (%) | BA (%) | OA变化pp | BA变化pp |',
        '|---|---|---:|---:|---:|---:|']
    for group in ['all23','all8_target12']:
        for mode in ['reference','incidence_corrected']:
            r=summary[(summary['group']==group)&(summary['mode']==mode)].iloc[0]
            lines.append(f'| {group} | {mode} | {100*r.full_oa:.4f} | {100*r.full_ba:.4f} | {r.full_oa_delta_pp:+.4f} | {r.full_ba_delta_pp:+.4f} |')
    lines+=['', '23港基础分支OA/BA分别提高0.6439/0.3437个百分点，专家分支分别提高0.3370/0.2578个百分点。平均交接数量仍为79，没有放宽预算。',
        'OA逐港19升4降，BA逐港14升9降；5组标签划分的港均OA全部提高，BA为4升1降。Port Said的BA下降1.0036个百分点、Sydney Botany下降0.5571个百分点，不能称为所有港都有效。', '',
        '| 港口 | OA变化pp | BA变化pp |', '|---|---:|---:|']
    for _,r in changed.sort_values('full_ba_delta_pp').iterrows():
        lines.append(f'| {r.port} | {r.full_oa_delta_pp:+.4f} | {r.full_ba_delta_pp:+.4f} |')
    lines+=['', '## 什么已经成立，什么还没有', '',
        '- 已有证据：同船差分中的入射角相关响应可在部分严格留出组合中复现；源端拟合的固定响应校正在既有跨港识别流程中有平均正收益。',
        '- 尚无证据：响应是纯随机噪声、入射角的无混杂物理因果效应、所有船型共享相同响应、或完整跨数据集泛化。船舶姿态、状态、背景等未测因素仍可能混杂。',
        '- 入射角是外部条件信息，不是只凭原隐空间无监督发现因果变量。线性源分类头对校正特征的读出，也可等价写成原特征加受约束的角度条件项；因此收益还不能唯一归因为去除了物理噪声，而非利用了条件信息。',
        '- 给定表示的响应留出验证，不是整个表示学习流程独立验证。既有几何使用了外层源留出船/月份；全部数据也属于历史开发数据。', '',
        '## 实现与后续', '',
        '19项针对性测试通过，覆盖身份/月划分、非线性差分、日期变更线位移、已知响应、缺失/参考角度恒等及原联合预测器重放。23港第一组支持划分的原分数重放误差均为0。',
        '保留本候选和原主模型，不自动合并。下一步应优先区分条件信息收益与校正机制收益：同一入射角信息、同一标注预算下的直接条件读出对照，可回答同船响应估计是否提供额外价值；若要更强因果结论，还需要观测几何/姿态的更充分测量或受控成像干预。', '',
        '详表：summary.csv、port_means.csv、metrics.csv、class_recalls.csv；识别阶段见../conditioned_vessel_drift_20261009/FINDINGS_ZH.md。', '',
        '![条件响应与完整分类收益](condition_response_and_accuracy.png)']
    (OUT/'FINDINGS_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

    fig,axes=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
    x=np.arange(3)
    for offset,space,label,color in [(-.18,'retained128','Original 128D','#197d80'),
                                   (.18,'stable_drift','Candidate stable + drift','#a63b63')]:
        view=response[(response.space==space)&(response['split']=='new_ship_month')].set_index('model')
        axes[0].bar(x+offset,[100*view.loc[k,'skill'] for k in ['incidence','position','incidence_position']],
                    .34,label=label,color=color)
    axes[0].set_xticks(x,['Incidence','Position','Both'])
    axes[0].set_ylabel('Pair prediction skill (%)')
    axes[0].set_title('Held vessels and acquisition months')
    axes[0].axhline(0,color='#777777',lw=.8)
    axes[0].legend(fontsize=8)
    view=summary[summary['mode']=='incidence_corrected'].set_index('group')
    x=np.arange(2)
    for offset,field,label,color in [(-.18,'full_oa_delta_pp','OA','#197d80'),(.18,'full_ba_delta_pp','BA','#a63b63')]:
        heights=[view.loc[k,field] for k in ['all23','all8_target12']]
        bars=axes[1].bar(x+offset,heights,.34,label=label,color=color)
        axes[1].bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
    axes[1].set_xticks(x,['All 23 ports','12 ports with all 8 classes'])
    axes[1].set_ylabel('Full-model improvement (pp)')
    axes[1].set_title('Fixed correction + head/support refit')
    axes[1].set_ylim(0,.8)
    axes[1].legend(fontsize=8)
    fig.savefig(OUT/'condition_response_and_accuracy.png',dpi=170)
    plt.close(fig)
    print('Wrote FINDINGS_ZH.md, class_recalls.csv and condition_response_and_accuracy.png')


if __name__=='__main__':
    main()
