"""Close out only after all 24 tasks have their three neural seeds."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
OUT = ROOT/'strong_single_multidata_20261010'
ARM = 'task_adapted_single_same_readout'
FROZEN = 'frozen_single_same_readout'
SEEDS = [20261010, 20261011, 20261012]


def closeout(neural, refs):
    r18 = pd.read_csv(OUT/'r18_control/summary.csv').set_index(['dataset', 'mode'])
    sar = pd.read_csv(ROOT/'strong_single_sarclip_20261010/summary.csv').set_index(['dataset', 'arm'])
    recon = pd.read_csv(ROOT/'dual_reconstruction_20261010/metrics.csv').groupby(['dataset', 'mode'])[['oa', 'ba']].mean()
    polar = pd.read_csv(ROOT/'strong_single_polarization_20261010/summary.csv').set_index(['dataset', 'mode'])
    oss = pd.concat([pd.read_csv(ROOT/f'strong_single_r50_task_20261010/seed_{s}/metrics.csv') for s in SEEDS])
    oss = oss.groupby(['seed', 'arm'])[['oa', 'ba']].mean().reset_index()
    oss['dataset'] = 'oss2'
    adapted = pd.concat([neural, oss], ignore_index=True).groupby(['dataset', 'arm'])[['oa', 'ba']].mean()
    lines = ['# 单骨干补齐与双路解释：总报告', '',
        '本轮24港中的23个留港任务与F_0921，共72个任务/神经种子组合已完成；原OSS2三种子结果单独复用。每次均为原5组64标签。另补齐25任务的R18单用/组合、输入差分控制及F的冻结SARCLIP。', '',
        '## 1. 结果先分清比较条件', '',
        '每格OA / BA（%）。24港按23个留港任务等权平均。以下是不同配置的成绩清单，不是每行都只改变一个因素；严格配对控制在各子报告中。', '',
        '| 配置 | 24港 | F_0921 | OSS2 |', '|---|---:|---:|---:|']
    rows = [('当前完整R50单路', refs, 'r50_late'),
            ('源端MLP重建第二表示，推理仍是单骨干', recon, 'mlp_late'),
            ('任务式R50，三个神经种子均值', adapted, ARM),
            ('已有任务训练R18单用，共同紧凑读出', r18, 'r18_task_trained'),
            ('两路拼接，共同紧凑分类头', r18, 'concat_shared_head'),
            ('两路紧凑头等权softmax', r18, 'equal_single_softmax'),
            ('当前完整双路', refs, 'dual_refit_late'),
            ('冻结SARCLIP，源端选共同读出', sar, 'sarclip')]
    for label, frame, mode in rows:
        values = [frame.loc[(d, mode)] for d in ('self24ports', 'f0921', 'oss2')]
        lines.append('| '+label+' | '+' | '.join(f'{v.oa:.4f} / {v.ba:.4f}' for v in values)+' |')
    lines += ['', '任务训练R50的标准差、全部正负港口与预算边界见`FINDINGS_ZH.md`和`TRAINING_AUDIT.csv`；历史双路不是三次新重训。冻结SARCLIP不能冒充充分任务适配的强模型。', '',
        '## 2. 双路究竟补充在哪里', '',
        '已有R18单独读出并未达到完整双路；同一源标签、同一64标签、同一紧凑算子下，两路组合在三组平均OA/BA均优于分别单用任一路。这支持现有两种表示的联合价值，不是维数增加就必然有效。B=A重复分支控制恢复单路核与分数。', '',
        '以前的完整模型干预已区分类别先验和支持关系：24港两条路径都有贡献，F主要来自类别先验，OSS2主要来自支持关系。详见`../dual_evidence_20261010/DUAL_BACKBONE_RATIONALE_ZH.md`。它们不能直接解释新任务单路与完整双路的全部差距。', '',
        '本轮用`explain/gap_bridge_summary.csv`明确分开“匹配完整配置的第二路增量”“完整与紧凑读出的差异”“任务训练影响”；用`crossover/`定位新单路先验/关系变化。救回、误改、类别贡献、跨神经种子一致性和实际案例均已落盘。', '',
        '## 3. 找到一个输入层面的具体限制，但不是统一根因', '',
        '当前冻结R50前段的VV、VH卷积核完全相同，因此只见两者之和；R18另看VH。固定差分保留适配不增加参数或前向次数，两个版本都重新提取源和目标并分别拟合同一读出：', '',
        '| 数据集 | 求和输入 OA / BA | 保留差分 OA / BA | BA变化(pp) |', '|---|---:|---:|---:|']
    for dataset in ('self24ports', 'f0921', 'oss2'):
        old, new = polar.loc[(dataset, 'sum')], polar.loc[(dataset, 'sum_difference')]
        lines.append(f'| {dataset} | {old.oa:.4f} / {old.ba:.4f} | {new.oa:.4f} / {new.ba:.4f} | {new.ba-old.ba:+.4f} |')
    lines += ['', '固定先验/关系交叉结果：24港收益来自关系路径，F主要来自先验路径，OSS2两条路径均退化；两种替换顺序及交互项完整保留。见`../strong_single_polarization_20261010/diagnosis/FINDINGS_ZH.md`。该对照解释了特定输入适配的功能变化，不把差分当标定后散射比。此前全参数CE单路已打破通道对称，因此这也不是全部单路试验的共同缺陷。', '',
        '## 4. 能否说单骨干做不到', '',
        '不能。24港单骨干MLP已追回既有双路BA增量的大部分；新任务训练也须逐港报告，不能用宏平均掩盖单路胜出的港口。共同难例只相对于固定已测集合：三个任务训练单路、冻结控制、完整R50、MLP替代和R18单用；不包括尚未穷尽的单骨干。', '',
        'R50与R18的输入、预训练和训练上下文不同，完整模块读出也尚未与所有强单骨干对齐。因此当前可写“具有可定位的互补收益”，不能写“双骨干不可替代”，也不能编造一个只看轮廓、另一个只看散射的物理分工。', '',
        '## 5. 收益是否值得成本', '',
        '既有实际像素配对测量中，B16额外R18每张增加0.618/0.839/2.418ms，对应Antwerp/F/OSS2增加10.88%/14.19%/37.41%。24港成本只测Antwerp，不能代表所有港口。新训练与特征提取期间没有重跑计时，以免资源争用污染测量。', '',
        '这是精度与代价的应用选择，不存在只凭涨点自动认定值得的统一阈值。单骨干重建与不同批量还有各自机会；最终部署选择须针对目标延迟与精度要求。', '',
        '## 6. 可追溯入口', '',
        '- `comparison.png`、`port_ba_gap.png`：新训练和逐港正负差距。',
        '- `explain/NUMERIC_FINDINGS.md`及CSV：净纠错、共同难例和差距桥接。',
        '- `error_modes/`：支持缺类、目标缺类与普通类别混淆。',
        '- `cases/`：真实VV/VH，纠错/误改/双方错误，64个支持项的分数贡献。',
        '- `r18_control/FINDINGS_ZH.md`：分别单用与同算子组合。',
        '- `../strong_single_sarclip_20261010/summary.csv`：补齐F后的冻结外部模型结果。', '',
        '所有结果仍是研发数据上的证据，历史无标签预训练接触范围保留；未新增OSS1、FUSAR或通用跨域任务。', '']
    effects = pd.read_csv(OUT/'crossover/effects.csv')
    errors = pd.read_csv(OUT/'error_modes/summary.csv')
    lines += ['## 7. 本轮三种子诊断数值', '',
        '任务训练相对同读出冻结控制的固定路径干预，单位为百分点。联合变化包含交互，不能简单相加前两列；完整条件效应见原CSV。', '',
        '| 数据集 | 指标 | 只换先验 | 只换关系 | 联合 | 交互 |', '|---|---|---:|---:|---:|---:|']
    for row in effects.itertuples():
        lines.append(f'| {row.dataset} | {row.metric.upper()} | {row.prior_only_pp:+.4f} | {row.kernel_only_pp:+.4f} | {row.joint_pp:+.4f} | {row.nonadditivity_pp:+.4f} |')
    kinds = dict(wrong_class_absent_from_target='错分成目标队列没有的类别',
                 wrong_class_without_support='错分成目标存在但支持集未覆盖的类别',
                 wrong_supported_class='错分成支持集已有的另一类')
    lines += ['', '双路相对任务单路的净BA优势来自哪些错误，单位为百分点；救回减去误改。类别存在性仅用于事后统计，没有给分类器目标类别清单。', '',
        '| 数据集 | 错误类型 | 救回贡献 | 误改损失 | 净变化 |', '|---|---|---:|---:|---:|']
    for (dataset, kind), part in errors.groupby(['dataset', 'error_kind']):
        values = part.set_index('event').ba_contribution_pp
        rescued, harmed = values.loc['dual_rescue'], values.loc['dual_harm']
        lines.append(f'| {dataset} | {kinds[kind]} | {rescued:.4f} | {harmed:.4f} | {rescued-harmed:+.4f} |')
    lines += ['', '这些表解释条件配置变化和实际纠错位置，不把单个数值当作独占物理机制，也不按目标查询选择交叉配置。', '']
    expanded = pd.read_csv(OUT/'commonhard_expanded/dataset_counts.csv')
    receipt = json.loads((OUT/'commonhard_expanded/RUN_RECEIPT.json').read_text())
    if not receipt['full_main']:
        raise RuntimeError('Closeout needs the complete expanded common-hard report')
    lines += ['## 8. 扩充单路集合后，还剩哪些独有纠错', '',
        '在原固定集合之外加入SARCLIP、源选R50共同读出、重新提取的求和输入R50及差分保留R50。原统计不覆盖，扩充版本单独保存。以下计数为查询/支持抽样曝光，不是独立样本或港口宏平均准确率。', '',
        '| 数据集 | 原集合共同难例中的双路救回 | 扩充后仍保留 | 被新增单路覆盖 | 双路错而扩充集合至少一个单路对 |', '|---|---:|---:|---:|---:|']
    for row in expanded[expanded.unit == 'draw_exposure'].itertuples():
        lines.append(f'| {row.dataset} | {row.old_dual_rescues_commonhard} | {row.expanded_dual_rescues_commonhard} | {row.old_dual_rescue_eliminated_by_any_added} | {row.expanded_dual_wrong_any_single_correct} |')
    all_draw = expanded[expanded.unit == 'unique_query_all_draw']
    lines += ['', '跨该样本所有有效支持抽样都持续出现的“扩充集合全错、双路正确”独有救回：'+
        '；'.join(f'{r.dataset}={r.expanded_dual_rescues_commonhard}' for r in all_draw.itertuples())+'。',
        '这说明独有纠错依赖具体支持集，不能当作已发现单路无法表达的固定物理线索。双路平均更准和存在持续独占信息，是两个不同命题。',
        '反向列使用知道真值后的任一单路正确并集，是描述性的oracle集合，不是可以部署的择优分类器；不能把它与某个实际模型直接作准确率排名。更符合目前证据的解释是：现有两种表示及其校正路径组合后，决策错误分布得到改善；信息本身是否只能由两个骨干获取仍未证明。',
        '完整逐查询、逐新增控制及去重计数见`commonhard_expanded/NUMERIC_FINDINGS.md`。', '',
        '## 9. 哪些船型受益或受损', '',
        '下面是双路相对任务式单路的BA差距贡献，不是某类召回率本身。逐类贡献求和还原本轮总BA差距；24港缺席类别不虚构分母。', '',
        '| 船型 | 24港贡献(pp) | F_0921贡献(pp) |', '|---|---:|---:|']
    classes = pd.read_csv(OUT/'explain/class_macro_ba_contributions.csv')
    classes = classes[classes.contrast == 'dual_vs_trained'].set_index(['dataset', 'class_id'])
    for c, name in enumerate(['散货船', '渔船', '一般货船', '成品油/化学品油轮', '集装箱船', '原油油轮', '拖轮', '海工供应船']):
        a = classes.loc[('self24ports', c), 'ba_contribution_pp']
        b = classes.loc[('f0921', c), 'ba_contribution_pp']
        lines.append(f'| {name} | {a:+.4f} | {b:+.4f} |')
    lines += ['', '例如F中集装箱船、散货船、成品油/化学品油轮贡献较大，但原油油轮和海工供应船反而受损。不能把总指标上升写成所有船型都受益。', '']
    (OUT/'CLOSEOUT_ZH.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    status = json.loads((OUT/'STATUS.json').read_text())
    if status['completed_task_seeds'] != 72 or status['tasks_per_dataset'] != {'f0921': 1, 'self24ports': 23}:
        raise RuntimeError('Final report requires all23 held ports and F, three seeds each')
    task = pd.read_csv(OUT/'task_seed_means.csv')
    for _, part in task.groupby(['dataset', 'task', 'arm']):
        if sorted(part.seed.tolist()) != SEEDS:
            raise RuntimeError('Missing or duplicated neural seed')
    neural = pd.read_csv(OUT/'neural_seed_means.csv')
    reference = pd.read_csv(ROOT/'backbone_ablation_20261010/metrics.csv')
    reference = reference[reference['mode'].isin(['dual_refit_late', 'r50_late'])]
    refs = reference.groupby(['dataset', 'mode'])[['oa', 'ba']].mean()
    ports_ref = reference[reference['mode'] == 'dual_refit_late'].groupby(['dataset', 'task'])[['oa', 'ba']].mean()
    table, numbers = [], {}
    for dataset, label in [('self24ports', '自建24港（23个留港任务）'), ('f0921', 'F_0921')]:
        stats = neural[neural.dataset == dataset].groupby('arm')[['oa', 'ba']].agg(['mean', 'std'])
        numbers[dataset] = stats
        for arm, name in [(FROZEN, '同读出冻结R50'), (ARM, '任务式训练R50')]:
            row = stats.loc[arm]
            table.append(dict(dataset=dataset, arm=arm, label=name,
                oa=row[('oa', 'mean')], oa_sd=row[('oa', 'std')],
                ba=row[('ba', 'mean')], ba_sd=row[('ba', 'std')], new_neural_seeds=3))
        for arm, name in [('r50_late', '当前完整R50单路'), ('dual_refit_late', '当前完整双路')]:
            row = refs.loc[(dataset, arm)]
            table.append(dict(dataset=dataset, arm=arm, label=name, oa=row.oa, ba=row.ba,
                              oa_sd=np.nan, ba_sd=np.nan, new_neural_seeds=0))
    pd.DataFrame(table).to_csv(OUT/'FINAL_COMPARISON.csv', index=False)
    curves = []
    for path in sorted((OUT/'targets').glob('*/*/seed_*/TRAINING_COMPLETE.json')):
        r = json.loads(path.read_text())
        curves.append(dict(dataset=path.parents[2].name, task=path.parents[1].name,
            seed=int(path.parent.name[5:]), selected_epoch=r['selected_epoch'],
            best_at_budget_boundary=r['best_at_budget_boundary'], seconds=r['seconds'],
            update_l2=r['update_l2'], prior_update_l2=r['prior_update_l2']))
    curves = pd.DataFrame(curves)
    curves.to_csv(OUT/'TRAINING_AUDIT.csv', index=False)
    means = task[task.arm == ARM].groupby(['dataset', 'task'])[['oa', 'ba']].mean()
    delta = means-ports_ref.loc[means.index]
    delta.to_csv(OUT/'TRAINED_MINUS_DUAL_BY_TASK.csv')
    lines = ['# 24港与F_0921：任务式单骨干补齐与双路解释', '',
             '日期：2026-10-10。72个任务/神经种子组合已完成；每个组合使用原5组64标签。', '',
             '## 结果', '', '| 数据集 | 配置 | OA（%） | BA（%） |', '|---|---|---:|---:|']
    for r in table:
        fmt = lambda metric: f'{r[metric]:.4f}' if np.isnan(r[f'{metric}_sd']) else f'{r[metric]:.4f} +/- {r[f"{metric}_sd"]:.4f}'
        lines.append(f'| {r["dataset"]} | {r["label"]} | {fmt("oa")} | {fmt("ba")} |')
    lines += ['', '训练单路的误差条是三个神经种子均值的样本标准差；每个种子先平均五次支持抽样，再等权平均港口。历史双路只有既有固定模型，不冒充三次双路重训。', '']
    for dataset, stats in numbers.items():
        gain = stats.loc[ARM][('ba', 'mean')]-stats.loc[FROZEN][('ba', 'mean')]
        gap = refs.loc[(dataset, 'dual_refit_late'), 'ba']-stats.loc[ARM][('ba', 'mean')]
        oa_gain = stats.loc[ARM][('oa', 'mean')]-stats.loc[FROZEN][('oa', 'mean')]
        lines.append(f'- {dataset}：任务训练相对同读出冻结控制，OA {oa_gain:+.4f}、BA {gain:+.4f}个百分点；双路减训练单路的BA差为{gap:+.4f}个百分点。')
    p = delta.loc['self24ports']
    lines += [f'- 23港中，训练单路的港均BA高于当前双路的港口有{int((p.ba > 0).sum())}个，低于的有{int((p.ba < 0).sum())}个；OA与BA同时高于的有{int(((p.oa > 0)&(p.ba > 0)).sum())}个。', '',
              '## 解释要分清三层', '',
              '1. 性能差距：分港、分类核对救回与误改；重复支持抽样不能当独立样本。',
              '2. 功能机制：先验/支持关系交叉，原始预测对角线复现；完整读出与紧凑读出的配置差异另列，不把所有差距算成第二骨干贡献。',
              '3. 表示来源：R18单独读出、同读出组合和既有训练阶段对照判断第二表示是否确有补充；不同预训练、输入和模型容量尚未完全隔离，不编造两条互斥物理机制。', '',
              '诊断表中交叉格只定位问题，不按目标查询成绩重新选择部署方案。', '',
              '## 训练边界', '',
              f'72次训练中有{int(curves.best_at_budget_boundary.sum())}次源验证最优点落在第6轮预算边界；{int((curves.selected_epoch == 0).sum())}次选择初始化。保留这些事实，不能仅凭固定轮数宣称充分收敛。',
              '本轮是既有尾层/源先验任务训练配方的跨数据集补齐；尚非完整模块和SAR预训练机会完全对等的架构因果比较。F及部分港域的历史无标签预训练接触边界继续保留。', '',
              '## 结果入口', '',
              '- `FINAL_COMPARISON.csv`：主比较，历史参考与新神经种子分开。',
              '- `TRAINED_MINUS_DUAL_BY_TASK.csv`：全部港口的正负变化。',
              '- `crossover/`：固定先验/核交叉及相互作用。',
              '- `explain/`：救回、误改、跨种子一致性、固定已测单路集合共同难例、差距桥接。',
              '- `error_modes/`：类别混淆与缺少支持类别的作用。',
              '- `r18_control/`：单独R18与统一读出组合控制；不是新神经训练。',
              '- `cases/`：真实VV/VH及所有64个支持项；有纠错，也有误改和双方错误。',
              '- `TRAINING_AUDIT.csv`：选中轮数、更新量和预算边界。', '']
    (OUT/'FINDINGS_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    colors = ['#777777', '#237ca3', '#b87d28', '#19885e']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    labels = ['Frozen R50', 'Task-trained R50', 'Current full R50', 'Current dual']
    for i, dataset in enumerate(['self24ports', 'f0921']):
        rows = [r for r in table if r['dataset'] == dataset]
        for j, metric in enumerate(['oa', 'ba']):
            ax = axes[i, j]
            values = [r[metric] for r in rows]
            error = [0 if np.isnan(r[f'{metric}_sd']) else r[f'{metric}_sd'] for r in rows]
            ax.barh(labels, values, xerr=error, color=colors, capsize=3)
            ax.set_xlim(0, 100)
            ax.set_xlabel(f'{metric.upper()} (%)')
            ax.set_title(dataset)
            for k, value in enumerate(values):
                ax.text(value+error[k]+1, k, f'{value:.2f}', va='center', fontsize=9)
    fig.suptitle('Three neural seeds; support draws averaged within each seed\nHistorical full-model references are not three new neural fits', fontsize=12)
    fig.savefig(OUT/'comparison.png', dpi=160)
    plt.close(fig)
    sd = task[(task.dataset == 'self24ports') & (task.arm == ARM)].groupby('task').ba.std()
    p = p.sort_index()
    fig, ax = plt.subplots(figsize=(9, 10), layout='constrained')
    ax.barh(p.index, p.ba, xerr=sd.loc[p.index], color=np.where(p.ba > 0, '#19885e', '#b34a4d'), capsize=2)
    ax.axvline(0, color='black', linewidth=.7)
    ax.set_xlabel('Task-trained single minus fixed current dual BA (percentage points)')
    ax.set_title('All 23 held ports; mean +/- neural-seed SD (not a confidence interval)', fontsize=11)
    fig.savefig(OUT/'port_ba_gap.png', dpi=160)
    plt.close(fig)
    closeout(neural, refs)
    print('Final72-task-seed report and figures written', flush=True)


if __name__ == '__main__':
    main()
