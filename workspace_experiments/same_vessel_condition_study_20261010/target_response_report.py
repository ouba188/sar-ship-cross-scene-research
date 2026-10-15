"""Separate target-head gains, internal adaptation gains and scene-prior gains."""

import json
import numpy as np
import pandas as pd
from channel_operator import BASE
from channel_operator_report import table

OUT = BASE/'target_response_adapter_balanced_20261010'


def run():
    done = json.loads((OUT/'COMPLETE.json').read_text())
    results = pd.read_csv(OUT/'results.csv')
    keys = ['port', 'draw']
    for ref, suffix in [('source_only', 'source'), ('head_only', 'head'), ('ordinary_adapter', 'ordinary')]:
        lookup = results[results.arm.eq(ref)][keys+['OA', 'BA']].rename(columns={m: m+'_'+suffix for m in ('OA', 'BA')})
        results = results.merge(lookup, on=keys)
        for metric in ('OA', 'BA'):
            results['delta_'+metric+'_vs_'+suffix] = results[metric]-results[metric+'_'+suffix]
    results.to_csv(OUT/'paired_results.csv', index=False)
    mean_columns = ['OA', 'BA', 'delta_OA_vs_source', 'delta_BA_vs_source',
                    'delta_OA_vs_head', 'delta_BA_vs_head', 'delta_BA_vs_ordinary']
    means = results.groupby('arm', sort=False)[mean_columns].mean().reset_index()
    means.to_csv(OUT/'summary.csv', index=False)
    ports = results.groupby(['port', 'arm'], sort=False)[mean_columns].mean().reset_index()
    ports.to_csv(OUT/'port_summary.csv', index=False)
    stability = results.groupby('arm', sort=False).agg(
        tasks=('draw', 'size'), positive_vs_head=('delta_BA_vs_head', lambda x: int((x > 0).sum())),
        negative_vs_head=('delta_BA_vs_head', lambda x: int((x < 0).sum())),
        positive_vs_ordinary=('delta_BA_vs_ordinary', lambda x: int((x > 0).sum())),
        negative_vs_ordinary=('delta_BA_vs_ordinary', lambda x: int((x < 0).sum()))).reset_index()
    stability.to_csv(OUT/'task_stability.csv', index=False)
    similarities = []
    for path in sorted(OUT.glob('*_models.npz')):
        with np.load(path) as model:
            a = np.concatenate([model['ordinary_adapter_log_gain'], model['ordinary_adapter_shift']])
            b = np.concatenate([model['scene_guided_adapter_log_gain'], model['scene_guided_adapter_shift']])
            similarities.append(dict(task=path.stem, relative_parameter_difference=float(np.linalg.norm(b-a)/max(np.linalg.norm(a), 1e-12)),
                parameter_cosine=float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b), 1e-12)),
                prediction_disagreements=int((model['ordinary_adapter_pred'] != model['scene_guided_adapter_pred']).sum())))
    similarities = pd.DataFrame(similarities)
    similarities.to_csv(OUT/'guidance_effect.csv', index=False)
    history = pd.read_csv(OUT/'training_history.csv')
    first = history[history.step.eq(1)][['task', 'arm', 'target_loo_loss']].rename(columns={'target_loo_loss': 'initial_loss'})
    last = history[history.step.eq(history.step.max())].merge(first, on=['task', 'arm'])
    last['relative_training_loss_drop'] = 1-last.target_loo_loss/last.initial_loss
    last.to_csv(OUT/'optimization_diagnosis.csv', index=False)
    designs = pd.read_csv(OUT/'support_design.csv')
    updates = pd.read_csv(OUT/'parameter_updates.csv')
    update_mean = updates.groupby('arm', sort=False)[['parameter_rms', 'guided_top64_update_fraction', 'support_LOO_BA']].mean().reset_index()
    text = ['# 少量目标标签监督内部响应适配', '',
        '## 本轮回答的问题', '',
        '不是继续删去场景敏感通道，而是允许同一适配程序用目标支持标签学习如何增强、减弱或偏移内部响应。必须分别判断：目标标签本身是否有用；内部适配是否超过只适配分类头；源域场景信息是否超过普通内部适配。', '',
        '## 实验设置', '',
        '- Santos、Port Klang、Los Angeles三个目标港，各3个固定支持抽样，每次64艘船各取一片；同一组标签被分类头与适配器共用，不增加预算。',
        '- 所有版本共用同一支持/查询划分。查询排除支持船舶的所有重复观测；支持与查询可能来自同一大景，因此不是目标域跨大景独立验证。',
        '- 骨干固定，内部适配器位于R18最后一个残差块之前。普通/场景引导适配均1024个参数，均从不改变网络的状态初始化，训练24步。',
        '- 场景引导只改变全类别共享的参数调整先验，先验强度均值与普通版本相同。没有船型专用门控，也不按目标港结果选择配置；各目标港通过相同程序从自己的64个支持标签学习适配参数。',
        '- 识别头保留源域八类先验；残差头使用同一支持集的类别平衡权重。支持集中未出现类不额外拟合负残差，其分数保留当前表征下的源域先验。',
        '- 支持留一损失、源域分数保持和初始状态约束一起训练；留一计算固定完整支持集的权重与出现类别掩码，不逐次重估，且用于优化，所以不是独立验证成绩。',
        '- 使用相同939个源域标签的算子级小试，不是当前完整双路模型，也不是所有港口或外部数据集成绩。旧预训练曝光及开发集重复使用的边界保留。', '',
        '## 总体结果', '',
        table(means, ['arm', 'OA', 'BA', 'delta_OA_vs_source', 'delta_BA_vs_source', 'delta_OA_vs_head', 'delta_BA_vs_head', 'delta_BA_vs_ordinary']), '',
        '百分数与百分点。先对每港3次支持抽样取平均，再对3港等权；这些抽样共享查询人群，不是9个独立数据集，也不是3个神经预训练种子。', '',
        '## 逐港结果', '',
        table(ports, ['port', 'arm', 'OA', 'BA', 'delta_BA_vs_head', 'delta_BA_vs_ordinary']), '',
        '## 抽样稳定性', '',
        table(stability, list(stability.columns)), '',
        table(designs, ['port', 'draw', 'support', 'query', 'support_classes', 'query_classes', 'removed_same_vessel']), '',
        '## 场景信息是否真的改变了适配', '',
        table(update_mean, ['arm', 'parameter_rms', 'guided_top64_update_fraction', 'support_LOO_BA']), '',
        table(similarities, list(similarities.columns)), '',
        '先验修改了更新位置，不等于额外提高了查询准确率。若与普通版本参数或预测接近，应报告引导作用有限，不能把普通适配的全部收益归给场景机制。', '',
        '## 训练是否在工作', '',
        table(last, ['task', 'arm', 'initial_loss', 'target_loo_loss', 'relative_training_loss_drop', 'source_preservation_loss', 'parameter_rms']), '',
        '训练损失下降而查询下降时，应区分少样本过拟合、表示/读出目标不一致和引导先验不可靠，不凭单个负结果判定目标适配不可能。', '',
        '## 已定位并修正的头部问题', '',
        '首版采用不加权残差头，且把支持集缺类也作为负证据更新，与类别平衡训练目标和闭集源域先验不协调。在完成Santos三个支持抽样后中止，原结果保留于 target_response_adapter_20261010；Port Klang首个任务不完整，不计入结果。', '',
        '修正版独立输出，统一对所有对照应用类别平衡与缺类先验保留；支持集、标签预算、训练步数和其他超参数不变。加权留一公式已与显式逐样本删除拟合核对，缺类残差系数严格为零。', '',
        f'本轮完整完成 {done["tasks"]} 个任务、{done["adapter_fits"]} 次内部适配。CPU单线程耗时 {done["seconds"]:.1f} 秒，未修改主线程模型或使用其GPU。', '']
    old_path = BASE/'target_response_adapter_20261010/results.csv'
    if old_path.exists():
        old = pd.read_csv(old_path)
        comparison = old[old.arm.eq('head_only')].merge(
            results[results.arm.eq('head_only')][['port', 'draw', 'OA', 'BA']],
            on=['port', 'draw'], suffixes=('_legacy', '_balanced'))
        comparison.to_csv(OUT/'head_correction_comparison.csv', index=False)
        text += ['## 同特征同标签的头部纠错', '',
                 table(comparison, ['port', 'draw', 'OA_legacy', 'OA_balanced', 'BA_legacy', 'BA_balanced']), '']
    coverage_path = OUT/'label_coverage_group_summary.csv'
    if coverage_path.exists():
        coverage = pd.read_csv(coverage_path)
        cases = pd.read_csv(OUT/'case_score_accounting.csv')
        text += ['## 标签覆盖与误伤', '',
                 table(coverage, list(coverage.columns)), '',
                 '先在各任务的标签覆盖组内平均类别召回，再平均可用任务组；不是总体BA的直接贡献，也不是独立样本估计。支持缺类组仅4个任务组。', '',
                 '支持未覆盖类的平均召回由33.503降至21.723，下降11.780个百分点。虽然缺类残差系数保持为零，内部适配仍改变其源分类头的输入；冻结权重不等于冻结判别函数。其他类别分数改变也会影响竞争结果。', '',
                 '## 逐样本分数分解', '',
                 table(cases, list(cases.columns)), '',
                 '上述24例是每港第0次抽样中各取前4个纠错与前4个误伤，不代表总体贡献比例。固定竞争类别后，分数间隔变化精确拆为：源先验变化、旧残差头下的特征变化、支持头重拟合。各例预测已回放核对。这是依赖分解顺序的计算归因，不是散射或杂波的物理因果证明。', '']
    crossover = OUT/'prior_kernel_crossover'
    if (crossover/'COMPLETE.json').exists():
        cross_summary = pd.read_csv(crossover/'summary.csv')
        cross_ports = pd.read_csv(crossover/'port_summary.csv')
        cross_coverage = pd.read_csv(crossover/'coverage_summary.csv')
        text += ['## 先验分数与支持关系交叉对照', '',
                 '复用全部9个任务的两种已训练适配器，不重新训练网络、不增加标签。第一字母为源分类头先验所用表征，第二字母为支持关系所用表征；F为原冻结表征，A为适配后表征。每格使用相同64个标签重新拟合平衡残差头。FF为仅适配头，AA为完整内部适配。', '',
                 table(cross_summary, list(cross_summary.columns)), '',
                 table(cross_ports, list(cross_ports.columns)), '',
                 table(cross_coverage, list(cross_coverage.columns)), '',
                 '两种适配器在所有任务的FF/AA对角线共36次回放均与原预测一致。FA/AF是对已按AA目标训练模型的诊断交换，不是针对各结构重新训练的公平最终模型对比，也不据此按港挑选部署版本。', '',
                 '场景引导版本在巴生港：AA相对FF的BA提高2.954点，AF提高3.049点，FA下降0.138点。桑托斯港：AA下降2.768点，AF下降2.738点，FA基本不变。洛杉矶：AA下降0.763点，AF下降0.701点，FA下降0.072点。普通适配也呈相同方向。', '',
                 '在这批固定模型中，主要收益与损失随适配后的先验分数路径保留，而不是随支持关系几何变化保留；不代表已证明先验路径单独训练必然最优。', '',
                 '支持缺类组在FF、FA、AF、AA的召回分别为33.503、33.170、21.723、21.723。保留原先验基本恢复缺类表现，但同时丢失巴生港的主要增益，说明不能仅靠冻结该路径同时解决所有问题。', '',
                 '## 本轮收口与下一步', '',
                 '已确认：内部适配确实改变决策；巴生港三次抽样均有BA正收益；其他两港没有同样受益。总体场景引导适配相对仅适配头OA提高1.445点、BA下降0.192点；场景引导相对普通适配的BA仅提高0.035点，尚无有意义的额外收益证据。', '',
                 '已定位：收益/损失主要沿源分类头在新表征上的分数响应传递；支持缺类的原有判别未被有效保护。尚未定位：具体散射因素、真正非语义噪声，以及源域场景敏感排序为何不能有效指导目标更新。标签覆盖分析是事后诊断，不用于制定类别专用规则。', '',
                 '建议下轮只检验一个方向：由源港模拟64标签支持任务，学习全类别共享的更新约束，使少量目标标签可校正分数响应，又能保护支持未覆盖的判别关系。与普通适配保持相同预算和容量；不按目标港结果选择结构或强度。该方案尚未实施，不能写成已有提升。', '']
    (OUT/'FINDINGS_ZH.md').write_text('\n'.join(text), encoding='utf-8')
    print(means.to_string(index=False))
    print(ports.to_string(index=False))


if __name__ == '__main__':
    run()
