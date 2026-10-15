"""Report source evidence, matched spectral controls and retained negative results."""

import json
import numpy as np
import pandas as pd
from channel_operator_report import table
from spectral_residual import BASE, OUT, FEATURES, masks, transform, reconstruct


def source_figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    weights = pd.read_csv(OUT/'source_band_weights.csv')
    names = ['低频', '中频', '高频']
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout='constrained')
    for ax, col, title in zip(axes, ['class_centroid_signal', 'same_vessel_far_drift', 'weight'],
                              ['源类别均值差异', '同船条件明显变化的差异', '固定组合权重']):
        ax.bar(names, weights[col], color=['#347d9b', '#a84a64', '#769847'])
        ax.set_title(title)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('源域统计：不是纯语义 / 纯噪声鉴定')
    fig.savefig(OUT/'source_band_evidence.png', dpi=150)
    plt.close(fig)
    pairs = pd.read_csv(OUT/'source_pair_drift.csv')
    candidate = pairs[pairs.group.eq('SP_far')].copy()
    candidate['total_drift'] = candidate[['low_drift', 'middle_drift', 'high_drift']].sum(axis=1)
    chosen = candidate.sort_values(['total_drift', 'pair_id']).iloc[len(candidate)//2]
    obs = pd.read_csv(BASE/'observations.csv')
    pair = obs.iloc[[int(chosen.a), int(chosen.b)]]
    prefix = np.load(FEATURES/'source_layer3.npy', mmap_mode='r')
    h = np.array(prefix[pair.manifest_row], dtype=np.float32)
    components = [reconstruct(transform(h)*m) for m in masks()]
    maps = [np.sqrt(np.mean(v*v, axis=1)) for v in components]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5), layout='constrained')
    for i, (_, row) in enumerate(pair.iterrows()):
        with Image.open(row.image_relpath_vh) as image:
            axes[i, 0].imshow(np.asarray(image.convert('L')), cmap='gray', vmin=0, vmax=255)
        angle = f'{row.incidence_deg:.1f}' if pd.notna(row.incidence_deg) else '缺失'
        axes[i, 0].set_title(f'{row.port} | {str(row.acq_time)[:10]}\n入射角 {angle} | {row.platform}', fontsize=10)
        for j, values in enumerate(maps):
            axes[i, j+1].imshow(values[i], cmap='viridis', vmin=0, vmax=float(values.max()), interpolation='nearest')
            axes[i, j+1].set_title(names[j]+'分量的通道RMS', fontsize=10)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f'同一MMSI {int(chosen.mmsi)}，同港不同条件；源候选中漂移居中的一对\n'
                 '右侧为8×8内部响应，不是注意力或杂波定位；每列两行共用色标，不跨频带比较颜色强度', fontsize=11)
    fig.savefig(OUT/'source_same_vessel_bands.png', dpi=150)
    plt.close(fig)
    pair.to_csv(OUT/'source_visualized_pair.csv', index=False)


def run():
    done = json.loads((OUT/'COMPLETE.json').read_text())
    rows = pd.read_csv(OUT/'results.csv')
    for ref, suffix in [('head_only', 'head'), ('full', 'full')]:
        values = rows[rows.arm.eq(ref)][['port', 'draw', 'OA', 'BA']].rename(columns={m: m+'_'+suffix for m in ('OA', 'BA')})
        rows = rows.merge(values, on=['port', 'draw'])
        for m in ('OA', 'BA'):
            rows['delta_'+m+'_vs_'+suffix] = rows[m]-rows[m+'_'+suffix]
    cols = ['OA', 'BA', 'delta_OA_vs_head', 'delta_BA_vs_head', 'delta_OA_vs_full', 'delta_BA_vs_full']
    means = rows.groupby('arm')[cols].mean().reset_index()
    ports = rows.groupby(['port', 'arm'])[cols].mean().reset_index()
    rows.to_csv(OUT/'paired_results.csv', index=False)
    means.to_csv(OUT/'summary.csv', index=False)
    ports.to_csv(OUT/'port_summary.csv', index=False)
    classes = pd.read_csv(OUT/'by_class.csv')
    task_groups = classes.groupby(['port', 'draw', 'arm', 'coverage']).recall.mean().reset_index()
    coverage = task_groups.groupby(['arm', 'coverage']).agg(recall=('recall', 'mean'), task_groups=('draw', 'size')).reset_index()
    coverage.to_csv(OUT/'coverage_summary.csv', index=False)
    stability = rows.groupby('arm').agg(positive_vs_full=('delta_BA_vs_full', lambda v: int((v > 0).sum())),
        negative_vs_full=('delta_BA_vs_full', lambda v: int((v < 0).sum()))).reset_index()
    stability.to_csv(OUT/'stability.csv', index=False)
    update = pd.read_csv(OUT/'update_diagnosis.csv').groupby('arm').mean(numeric_only=True).reset_index()
    update.to_csv(OUT/'update_summary.csv', index=False)
    weights = pd.read_csv(OUT/'source_band_weights.csv')
    drift = pd.read_csv(OUT/'source_drift_summary.csv')
    names = dict(head_only='只适配分类头', full='全频残差', middle='固定中频残差', source_guided='源数据组合残差')
    if not (OUT/'source_same_vessel_bands.png').exists() or not (OUT/'source_band_evidence.png').exists():
        source_figures()
    lines = ['# 空间频率分解与保留主路的残差补偿', '',
             '## 总体结果', '',
             table(means.assign(arm=means.arm.map(names)), ['arm']+cols), '',
             'OA/BA为百分数，差值为百分点。3港各3次64标签支持抽样等权，抽样共享查询人群，不是9个独立数据集。', '',
             '结论：中频版比匹配的全频版高0.347个百分点BA，但比只适配分类头仅高0.040个百分点；9次任务中相对全频只有4次为正。源数据组合版比全频低0.093个百分点。本轮没有得到可确认的性能突破，不能把中频版相对较少的退化说成已经挖出了有效因果因素。', '',
             '## 这次与旧通道路线有何不同', '',
             '- 在冻结R18的layer3之后、整个layer4之前插入残差，输入256通道8×8空间图；不在旧4×4通道图上继续修改。',
             '- 这里低/中/高频指内部特征图的空间变化快慢，不是SAR传感器的电磁波段，也不直接等于某种散射机制。',
             '- 保留原特征主路，只添加经过选择的空间频率分量：h_new=h+gate*branch(h)。每版仅256个可训练系数，从0初始化，界限为-0.25到0.25。',
             '- 全频、中频和源数据组合三版具有相同参数量、插入位置、24步Adam和训练损失；不包含token注意力，不是Causal-Tune完整复现。',
             '- 源域逐通道匹配分支RMS到原特征RMS，减少不同频带能量大小带来的修正强度混淆；实际训练后的更新幅度仍单独报告。',
             '- 沿用前轮修正后的目标缺类损失处理、源分数保持和源八类监督。原939个源标签、9个64目标支持划分、查询身份排除均不变。',
             '- 与历史1024参数、不同插入位置的仿射适配比较不能单独归因于频率；本轮频率作用看三个严格匹配版本之间的对照。', '',
             '## 源数据给出的频带证据', '',
             table(weights, list(weights.columns)), '',
             '权重由源类别均值差异除以同船条件变化差异，再归一化得到。固定高斯低通半径0.2和0.7，其差为中频；半径按8×8频率平面角点归一化。没有目标港截止频率搜索。', '',
             '低频47.6%、中频32.0%、高频20.4%是固定源统计组合，不是逐样本自适应注意力。类别均值差异可能包含港口相关性；同船差异也可能包含有效状态变化，因此不能称为纯语义或纯噪声分解。', '',
             '![源频带统计](source_band_evidence.png)', '',
             table(drift, list(drift.columns)), '',
             '先船内平均，再类别内船等权，再类等权。不同组的船和类别构成不同，组间数值不是控制全部混杂后的条件因果效应。高斯软分量相互重叠，能重建原图但各频带能量不要求相加为总能量。', '',
             '数据引导版本额外使用已有标注源船的未标注重复观测与MMSI关联。它没有新增船型标签，但不是与不使用同船信息的版本严格等信息量；不能把差异全部归为结构优势。', '',
             '![同船分频响应](source_same_vessel_bands.png)', '',
             '图中原始VH裁片与内部通道RMS只展示实际分解，不能从图直接指认某处为杂波。选例为SP_far候选中分量漂移总和居中的一对，不是正确率证据。', '',
             '## 分港结果与缺类误伤', '',
             table(ports.assign(arm=ports.arm.map(names)), ['port', 'arm']+cols), '',
             table(coverage.assign(arm=coverage.arm.map(names)), list(coverage.columns)), '',
             '缺类组只含4个任务；先在任务内平均缺类召回，再在任务间平均，不是总体BA。', '',
             table(stability.assign(arm=stability.arm.map(names)), list(stability.columns)), '',
             '## 是否真的进行了内部更新', '',
             table(update.assign(arm=update.arm.map(names)), list(update.columns)), '',
             'relative_update_rms基于支持特征；匹配源分支能量不保证训练后各模型的更新幅度完全相同。中频修正可令原非负特征的部分位置变负，本轮不额外截断。', '',
             '中频分支的直流为零，因此它在当前插入层不改变每个通道的空间平均值，只改变空间分布。这是已验证的算子性质，不代表后续卷积和非线性之后的最终特征均值或分类分数不变。它是否解释了较少的缺类误伤，还需与保留全部非直流频率的匹配残差比较，当前没有完成这项归因。', '',
             '## 验证与边界', '',
             '- 原13项测试及新增通道均值保持测试通过，包括空间维DCT/逆DCT、分量重建、中频直流为零、RMS匹配、零初态原路与梯度、已有支持头公式和缺类损失。',
             '- 完整原骨干输出的源头预测回放一致，9个任务的只适配头预测回放一致，详见REPLAYS.json。',
             f'- 完成{done["tasks"]}个任务、{done["adapter_fits"]}次适配，CPU单线程耗时{done["seconds"]:.1f}秒；未修改主模型或占用其GPU。',
             '- 这是开发阶段的小型冻结R18实验，支持/查询可能共享大景，预训练曝光未撤销，不是完整双路模型成绩或物理因果证明。', '']
    diagnostic = OUT/'sign_diagnosis'
    if (diagnostic/'COMPLETE.json').exists():
        sign = pd.read_csv(diagnostic/'summary.csv')
        energy = pd.read_csv(diagnostic/'energy_summary.csv')
        sign_coverage = pd.read_csv(diagnostic/'coverage_summary.csv')
        sign_ports = pd.read_csv(diagnostic/'port_summary.csv')
        lines += ['## 负响应是否是失效根因', '',
                  '固定已学到的残差系数，不再训练神经网络；分别原样前向、将更新后的特征截为非负，再用同一64标签重拟合相同分类头。18次原样预测回放全部一致。全频版本来不产生负值，截断是恒等操作，不是独立新增训练对照。', '',
                  table(sign, list(sign.columns)), '',
                  table(sign_coverage, list(sign_coverage.columns)), '',
                  '中频版截断后BA下降0.309个百分点，缺类召回从31.366%降至29.046%；源组合版BA仅上升0.067个百分点。结果不支持“只要消除负响应就能解决退化”的简单解释。', '',
                  table(energy, list(energy.columns)), '',
                  '原特征约54.0%的位置为零，因此负位置占比不能直接表示破坏程度。中频的负值能量约占更新能量10.12%，源组合约0.45%。中频原样更新的相对通道均值变化约4.49e-9，截断后变为0.0513；截断同时改变空间分布、均值与更新幅度，不能单独隔离负号本身的作用。', '',
                  table(sign_ports, list(sign_ports.columns)), '',
                  '该回放不是针对非负约束重新训练的模型，也不是物理因果干预。中频截断在Port Klang略好，在另两港变差；不按目标港挑选版本，不将事后诊断结果冒充预先确定的新方法。', '',
                  '## 收口判断与下一项可区分实验', '',
                  '本轮证明实现确实改变了内部空间响应，且中频相对全频少一些缺类误伤；没有证明提取了纯噪声、跨域因果因素，也没有得到超过只适配分类头的明确收益。源统计加权的版本不值得直接加入主模型。', '',
                  '下一项最小对照应区分两个解释：中频本身更有用，还是仅仅因为它不改通道均值。可在相同参数、能量归一化和训练预算下比较“保留全部非直流频率的残差”与当前中频残差。若前者相当或更好，应改为研究保均值的空间响应适配，而不是继续强化特定中频或因果滤波的叙事。这项对照尚未运行。', '']
    (OUT/'FINDINGS_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    print(means.to_string(index=False))
    print(ports.to_string(index=False))
    print(coverage.to_string(index=False))
    print(update.to_string(index=False))


if __name__ == '__main__':
    run()
