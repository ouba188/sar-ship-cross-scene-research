"""Read out the bounded cross-port direction experiment, including losses."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_stable_port_pilot import OUT, ARMS


NAMES = ['散货船', '渔船', '杂货船', '成品油/化学品船', '集装箱船', '原油轮', '拖轮/拖带船', '海工补给船']
SHORT = ['Bulk', 'Fishing', 'General cargo', 'Product/chemical', 'Container', 'Crude oil', 'Tug/towing', 'Offshore']
ARM_ZH = {'joint_reference': '现有完整联合模型', 'kernel_shrink_control': '仅缩放原核',
          'pooled_direction': '加普通类间方向', 'stable_direction': '加跨港稳定方向'}


def main():
    complete = json.loads((OUT/'COMPLETE.json').read_text(encoding='utf-8'))
    assert complete['ports'] == 23 and complete['evaluation_rows'] == 460
    metrics = pd.read_csv(OUT/'metrics.csv')
    assert not metrics.duplicated(['port', 'draw', 'arm']).any()
    values = ['acc', 'ba', 'expert_acc', 'expert_ba', 'selected', 'evaluated', 'rescued', 'harmed']
    tasks = metrics.groupby(['port', 'arm'])[values].mean().reset_index()
    average = tasks.groupby('arm')[values].mean().reindex(ARMS)
    pairs = pd.read_csv(OUT/'heldout_pair_auc.csv')
    pm = pairs.groupby(['port', 'class_c', 'class_d'])[['pooled_auc', 'stable_auc']].mean().reset_index()
    pm['draws'] = pairs.groupby(['port', 'class_c', 'class_d']).size().to_numpy()
    pm['delta_auc'] = pm.stable_auc-pm.pooled_auc
    port_auc = pm.groupby('port')[['pooled_auc', 'stable_auc', 'delta_auc']].mean()
    grouped = pm.groupby(['class_c', 'class_d']).agg(
        ports=('port', 'nunique'), pooled_auc=('pooled_auc', 'mean'), stable_auc=('stable_auc', 'mean'),
        min_stable_auc=('stable_auc', 'min'), max_stable_auc=('stable_auc', 'max'),
        ports_above_half=('stable_auc', lambda v: int((v > .5).sum())),
        delta_auc=('delta_auc', 'mean')).reset_index().sort_values('stable_auc', ascending=False)
    port_table = tasks.pivot(index='port', columns='arm', values=values)
    comparisons = {}
    for reference in ('joint_reference', 'kernel_shrink_control', 'pooled_direction'):
        differences = pd.DataFrame({v: 100*(port_table[v]['stable_direction']-port_table[v][reference])
                                    for v in values[:4]})
        comparisons[reference] = dict(mean_delta_pp=differences.mean().to_dict(),
            positive_ports=differences.gt(1e-9).sum().to_dict(),
            negative_ports=differences.lt(-1e-9).sum().to_dict(),
            zero_ports=differences.abs().le(1e-9).sum().to_dict())
    summary = dict(status='completed_not_promoted', model_classes=8, target_ports=23, draws=5,
        arms=list(ARMS), aggregate_rule='mean draws within port, then equal-port mean',
        mean_metrics=average.reset_index().to_dict(orient='records'), stable_comparisons=comparisons,
        pair_auc_scope='fixed oriented class-pair scores on held-port queries; >=5 query rows per class; no adaptation or target fit',
        pair_auc_aggregation='mean available draws per port/pair, then equal pairs within port, then equal ports',
        eligible_pair_draw_rows=len(pairs), eligible_port_pairs=len(pm),
        pooled_pair_auc=float(port_auc.pooled_auc.mean()), stable_pair_auc=float(port_auc.stable_auc.mean()),
        stable_above_half_port_pairs=int((pm.stable_auc > .5).sum()),
        pooled_above_half_port_pairs=int((pm.pooled_auc > .5).sum()),
        stable_beats_pooled_port_pairs=int((pm.delta_auc > 0).sum()),
        stable_beats_pooled_ports=int((port_auc.delta_auc > 0).sum()),
        source_only=True, encoder_training=False, causal_identification=False,
        main_model_changed=False, external_datasets_tested=False,
        decision='Keep cross-port discriminative signal as diagnostic evidence; do not add this fixed25percent direction kernel to the main model.')
    (OUT/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    tasks.to_csv(OUT/'port_means.csv', index=False, encoding='utf-8-sig')
    average.to_csv(OUT/'mean_metrics.csv', encoding='utf-8-sig')
    pm.to_csv(OUT/'port_pair_auc.csv', index=False, encoding='utf-8-sig')
    grouped.to_csv(OUT/'class_pair_summary.csv', index=False, encoding='utf-8-sig')
    port_auc.to_csv(OUT/'port_auc.csv', encoding='utf-8-sig')

    figures = OUT/'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), layout='constrained')
    for ax, ref, title in zip(axes, ['joint_reference', 'pooled_direction'], ['Stable vs current joint', 'Stable vs pooled directions']):
        delta = 100*(port_table['ba']['stable_direction']-port_table['ba'][ref])
        ax.barh(delta.index, delta, color=np.where(delta >= 0, '#39866A', '#C35862'))
        ax.axvline(0, color='#333333', linewidth=.8)
        ax.invert_yaxis()
        ax.set_title(title+f'\nMean {delta.mean():+.4f} pp', loc='left', fontweight='bold')
        ax.set_xlabel('Full-system BA change (percentage points)')
        ax.grid(axis='x', alpha=.18)
        ax.set_axisbelow(True)
    fig.savefig(figures/'01_full_model_port_deltas.png', dpi=200, bbox_inches='tight')
    fig.savefig(figures/'01_full_model_port_deltas.pdf', bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout='constrained')
    matrix = np.full((8, 8), np.nan)
    for row in grouped.itertuples():
        matrix[row.class_c, row.class_d] = row.stable_auc
    im = axes[0].imshow(np.ma.masked_invalid(matrix), cmap='YlGnBu', vmin=.5, vmax=1.)
    axes[0].set_xticks(range(8), SHORT, rotation=45, ha='right', fontsize=8)
    axes[0].set_yticks(range(8), SHORT, fontsize=9)
    for i in range(8):
        for j in range(8):
            if np.isfinite(matrix[i,j]):
                axes[0].text(j, i, f'{matrix[i,j]:.3f}', ha='center', va='center', fontsize=8,
                             color='white' if matrix[i,j]>.85 else '#222222')
    axes[0].set_title('Held-port pair discrimination\nMean available ports per pair', loc='left')
    fig.colorbar(im, ax=axes[0], shrink=.65, label='Stable-direction AUC')
    axes[1].scatter(pm.pooled_auc, pm.stable_auc, s=15, alpha=.55, color='#247BA0')
    axes[1].plot([0,1], [0,1], linestyle='--', color='#666666')
    axes[1].set_xlim(0,1.02)
    axes[1].set_ylim(0,1.02)
    axes[1].set_xlabel('Pooled-direction AUC')
    axes[1].set_ylabel('Stable-direction AUC')
    axes[1].set_aspect('equal')
    axes[1].set_title(f'{len(pm)} held-port/class-pair means\nAbove diagonal: {summary["stable_beats_pooled_port_pairs"]}', loc='left')
    fig.savefig(figures/'02_held_port_pair_auc.png', dpi=200, bbox_inches='tight')
    fig.savefig(figures/'02_held_port_pair_auc.pdf', bbox_inches='tight')
    plt.close(fig)

    lines = ['# 跨港稳定判别方向：首轮挖掘结果', '', '日期：2026-10-08。状态：实验完成，候选不并入主模型。', '',
        '## 结论先行', '',
        '挖到了可跨港复用的统计判别信息，但本轮稳定性加权接入没有带来完整模型增益。'
        '不将它称为因果变量、物理不变量或已生效的新模块。', '',
        f'23个目标港、5次原支持集抽样、4个匹配臂，共460条结果。原联合模型的预测、交接集合和专家预测全部精确复现。'
        f'固定成对方向在未参与拟合的目标港上，等港平均AUC为普通方向{summary["pooled_pair_auc"]:.6f}、'
        f'稳定方向{summary["stable_pair_auc"]:.6f}。'
        f'{summary["stable_above_half_port_pairs"]}/{len(pm)}个可评估港口—类别对的稳定方向AUC大于0.5；'
        f'{summary["stable_beats_pooled_ports"]}/23港的平均成对AUC略有提高。'
        '这些是描述性结果，没有据此宣称独立显著性或完全不变。', '',
        '## 挖掘位置与算法', '',
        '只使用冻结R50的2048维视觉表征，不使用港口设施、AIS轨迹、目标查询标签或新增标注。'
        '选择R50是因为它不受现有逐港M3训练改变，源/目标处于同一编码器坐标系。'
        '当前联合模型的R18、M1、M2、M3、源分类先验、Base64及门控参数均保持原样。'
        '本轮没有对R18、其他骨干或图像空间注意力做不变性发现。', '',
        '每个外层目标港使用既有身份/产品排除后的源行；预处理也使用该源集合的R50均值和标准差。'
        '对类别c<d，在每个至少各有5条样本的源港，计算同港均值差delta_e=mu_e,c-mu_e,d。'
        '至少3个源港支持该类别对才保留；本轮每个外层任务均得到28个类别对方向。', '',
        '普通方向为m=mean_e(delta_e)，源港等权。稳定加权为a_j=m_j^2/mean_e(delta_e,j^2)，'
        '再令v_j=m_j*a_j并做L2归一化。跨港方向变化较大的坐标权重较低；'
        '这不是强制整体域分布对齐，也不是可辨识因果图。'
        '28个投影分别按源行拟合均值/标准差，最后除以sqrt(28)。', '',
        '源方向的留一港余弦仅是条件于外层源预处理的内部描述，不冒称完整内层留港检验。'
        '真正的外推读出在未参与方向拟合的外层目标港进行。', '',
        '## 接入完整模型的公平比较', '',
        '设Kjoint=0.5Kretained+0.5Kcomplement（不含偏置）。所有臂保持同一源先验、'
        '同64标签、残差ridge=8、5%上限及原接收者一致门控；每个核只有一个偏置常数。', '',
        '| 实验臂 | 残差核 |', '| --- | --- |',
        '| 原联合模型 | Kjoint+1 |', '| 仅缩放原核 | 0.75Kjoint+1 |',
        '| 普通类间方向 | 0.75Kjoint+0.25Kpooled+1 |',
        '| 跨港稳定方向 | 0.75Kjoint+0.25Kstable+1 |', '',
        '25%为本轮预先固定的新增方向权重，没有根据查询结果搜索或改权重。'
        '缩放对照用于区分信息收益与原核权重/正则化改变；普通方向对照用于区分稳定性加权与一般源监督信息。'
        '全部23港策略保存后，才统一读取查询指标。', '',
        '| 实验臂 | 完整OA % | 完整BA % | 全量专家OA % | 全量专家BA % |',
        '| --- | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        row = average.loc[arm]
        lines.append(f'| {ARM_ZH[arm]} | {100*row.acc:.6f} | {100*row.ba:.6f} | {100*row.expert_acc:.6f} | {100*row.expert_ba:.6f} |')
    lines += ['', '所有准确率先平均各港5次抽样，再对23港等权平均；不按样本数池化。', '',
        '| 稳定方向相对 | Δ完整OA pp | Δ完整BA pp | Δ专家OA pp | Δ专家BA pp | 完整BA正/零/负港数 |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for ref, result in comparisons.items():
        v = result['mean_delta_pp']
        signs = '/'.join(str(result[s]['ba']) for s in ('positive_ports','zero_ports','negative_ports'))
        lines.append(f'| {ARM_ZH[ref]} | {v["acc"]:+.6f} | {v["ba"]:+.6f} | '
                     f'{v["expert_acc"]:+.6f} | {v["expert_ba"]:+.6f} | {signs} |')
    lines += ['', f'![逐港完整模型BA变化]({(figures/"01_full_model_port_deltas.png").as_posix()})', '',
        '缩放对照已经带来专家BA变化；稳定方向与普通方向相比的专家BA差值只有'
        f'{comparisons["pooled_direction"]["mean_delta_pp"]["expert_ba"]:+.6f}个百分点。'
        '因此不能把新增分支称为有实际效用的稳定信息升级，亦不能把所有退化归因于方向本身。', '',
        '## 哪些类别关系跨港可读', '',
        f'共有{len(pairs)}个合格的港口—类别对—抽样观测，平均可用抽样后得到{len(pm)}个港口—类别对。'
        '仅在该轮查询中两类各至少5条时计算AUC；不足样本的对不纳入该项诊断，但其样本始终保留在完整8类OA/BA评价中。'
        '分数方向预先固定：c类为正，不能见目标结果后反转分数。AUC不是8分类准确率，0.797不能写成79.7%的船型识别准确率。', '',
        f'![未参与拟合港口上的类别对AUC]({(figures/"02_held_port_pair_auc.png").as_posix()})', '',
        '| 类别对 | 可评估港数 | 普通AUC | 稳定AUC | 稳定AUC最小/最大 | AUC>0.5港数 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in grouped.itertuples():
        lines.append(f'| {NAMES[row.class_c]} / {NAMES[row.class_d]} | {row.ports} | {row.pooled_auc:.4f} | '
                     f'{row.stable_auc:.4f} | {row.min_stable_auc:.4f}/{row.max_stable_auc:.4f} | {row.ports_above_half} |')
    lines += ['', '上表先对每个港口/类别对的可用抽样平均，再按该类别对可用港口等权平均。'
        '它与开头“各港先平均其类别对、再等港平均”的总AUC不是同一汇总分母。'
        '部分稀有类只能在少数港/抽样中诊断，未把它们写成24港均成立。', '',
        '## 本轮判断与下一步边界', '',
        '1. 找到了跨港可复用的类别判别关系，尤其部分包含渔船的类别对；但普通等港方向已经有接近的AUC，'
        '稳定性加权只带来很小的成对读出增量。', '',
        '2. 本轮接入没有改善现有专家或完整系统。可读信息不等于互补信息，也不等于在少量标签和固定预算下可用的纠错信息。'
        '当前实验不能进一步区分信息冗余、核几何、目标适配或门控变化各自的贡献。', '',
        '3. 不并入主模型，不修改既有论文结果，不追加六个数据集扫描，也不在同一查询结果上反复调权重找正数。'
        '保留方向银行、逐港预测、对照和负结果。', '',
        '4. 若继续，不是再证明这些方向能区分类别，而是先定位哪些跨港可读信息与当前专家已有判别不同，'
        '再判断是否存在值得接入的增量。当前还没有这一步的结果。', '',
        '5. 跨港可读仍可能包含各港共享的非因果相关性；本轮没有图像物理因素干预、因果结构识别或独立新数据确认。'
        '准确命名为“跨港稳定判别方向诊断”，不称为因果发现成功。', '',
        '## 工件与复现', '',
        '- PROTOCOL.json：固定方法与读出定义。',
        '- metrics.csv / port_means.csv / mean_metrics.csv：全部460行和等港汇总。',
        '- heldout_pair_auc.csv / port_pair_auc.csv / class_pair_summary.csv：成对方向证据。',
        '- class_recall.csv：所有样本保留的逐类查询计数及召回。',
        '- ports/<port>/source_direction_bank.npz：源拟合方向与标准化参数。',
        '- ports/<port>/*_policies.npz：四臂完整冻结预测及交接集合。',
        '- summary.json / COMPLETE.json：机器可读结论与完成状态。', '',
        '代码位于E:/Hermes/paper_tracking/sensitivity/oof_reliability_check：'
        'stable_port_directions.py、run_stable_port_pilot.py、report_stable_port_pilot.py；'
        'test_stable_port_directions.py的6项测试通过，覆盖合成环境中的干扰翻转、固定源统计、类别对可用性及分块边界。', '']
    (OUT/'STABLE_DIRECTION_FINDINGS_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    print(average.to_string())
    print(json.dumps({k: summary[k] for k in ['pooled_pair_auc','stable_pair_auc','stable_beats_pooled_ports','stable_comparisons']}, indent=2))


if __name__ == '__main__':
    main()
