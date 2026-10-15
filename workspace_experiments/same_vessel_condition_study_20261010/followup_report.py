"""Human-readable evidence transitions and exact per-case contribution plots."""

import numpy as np
import pandas as pd

from followup import OUT
from study import GROUPS
from report import LABELS, PLOT_LABELS


def plots(funnel, swaps, relation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), layout='constrained')
    data = funnel[(funnel.cohort == 'damaged') & (funnel.minimum_ships_per_class == 1)].set_index('group').loc[list(GROUPS)]
    x = np.arange(4)
    for shift, name, label, color in [(-.23, 'r50_relation_worsened', 'R50 relation worsens', '#b9583e'),
            (0, 'r18_relation_positive_b', 'R18 correct-class relation', '#286a9e'),
            (.23, 'rescued_damage', 'Actually rescued by dual', '#22876f')]:
        axes[0].bar(x+shift, data[name], .22, label=label, color=color)
    axes[0].set_ylabel('Conditional rate (%)')
    axes[0].set_ylim(0, 100)
    axes[0].set_title('A. Pairs where R50 correct -> wrong')
    comparable = pd.read_csv(OUT/'relation_rescue_common_classes.csv')
    for shift, flag, label, color in [(-.18, False, 'R18 relation not positive', '#b9583e'),
                                    (.18, True, 'R18 relation positive', '#22876f')]:
        subset = comparable[comparable.r18_relation_positive_b.eq(flag)].set_index('group').reindex(GROUPS)
        axes[1].bar(x+shift, subset.rescue_rate, .34, label=label, color=color)
    axes[1].set_title('B. Rescue: common classes, >=5 ships / stratum')
    axes[1].set_ylabel('Dual rescue among damaged pairs (%)')
    axes[1].set_ylim(0, 100)
    swap = pd.read_csv(OUT/'swap_supported_summary.csv').set_index('group').loc[list(GROUPS)]
    for shift, name, label, color in [(-.18, 'r50_previous_net', 'Reuse earlier R50 feature', '#286a9e'),
                                    (.18, 'r18_previous_net', 'Reuse earlier R18 feature', '#b9583e')]:
        axes[2].bar(x+shift, swap[name], .34, label=label, color=color)
    axes[2].axhline(0, color='0.5', linewidth=.8)
    axes[2].set_title('C. Feature swaps: classes with >=5 ships')
    axes[2].set_ylabel('Accuracy change vs actual dual (pp)')
    for ax in axes:
        ax.set_xticks(x, PLOT_LABELS, fontsize=8)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle('Frozen diagnostic probes: identity-held; no refit; no physical counterfactual claim', fontsize=12)
    fig.savefig(OUT/'evidence_chain.png', dpi=170)
    fig.savefig(OUT/'evidence_chain.pdf')
    plt.close(fig)


def cases(obs, table, nearest):
    import matplotlib.pyplot as plt
    from PIL import Image
    chosen, references = [], []
    obs_by_row = obs.set_index('manifest_row')
    for kind in ('rescued_damage', 'harm'):
        fig, axes = plt.subplots(4, 3, figsize=(13, 13), layout='constrained',
                                 gridspec_kw={'width_ratios': [1, 1, 1.5]})
        for gi, group in enumerate(GROUPS):
            part = table[table.group.eq(group)]
            if kind == 'rescued_damage':
                part = part[part.rescued_damage]
            else:
                part = part[part.r50_pred_b.eq(part.class_id) & part.dual_pred_b.ne(part.class_id)]
            if part.empty:
                for ax in axes[gi]:
                    ax.axis('off')
                continue
            row = part.sort_values(['r50_raw_mse', 'pair_id']).iloc[len(part)//2]
            record = row.to_dict()
            record['kind'] = kind
            chosen.append(record)
            for col, oi in enumerate([int(row.a), int(row.b)]):
                item = obs.iloc[oi]
                image = str(item.image_relpath_vh)
                with Image.open(image) as img:
                    axes[gi, col].imshow(np.asarray(img.convert('L')), cmap='gray', vmin=0, vmax=255)
                suffix = 'a' if col == 0 else 'b'
                axes[gi, col].set_title(f'{item.port}\n{str(item.acq_time)[:10]} | class {int(item.class_id)}', fontsize=9)
                axes[gi, col].set_xlabel(f'R50 {int(row["r50_pred_"+suffix])} | R18 {int(row["r18_pred_"+suffix])} | Dual {int(row["dual_pred_"+suffix])}', fontsize=9)
                axes[gi, col].set_xticks([])
                axes[gi, col].set_yticks([])
                if col == 0:
                    axes[gi, col].set_ylabel(group, fontsize=10)
                for mi, model in enumerate(('r50', 'r18')):
                    for order, manifest_row in enumerate(nearest[oi, mi]):
                        ref = obs_by_row.loc[int(manifest_row)]
                        references.append(dict(kind=kind, pair_id=int(row.pair_id), endpoint=suffix,
                            model=model, order=order+1, reference_row=int(manifest_row),
                            reference_class=int(ref.class_id), reference_port=ref.port,
                            reference_id=ref.instance_id, reference_image=str(ref.image_relpath_vh)))
            ax = axes[gi, 2]
            values = [row.single_margin_b_fixed_rival, row.r50_reweight_term,
                      row.intercept_term, row.r18_added_term]
            base = 0.
            for index, value in enumerate(values):
                ax.bar(index, value, bottom=base, color='#26866f' if value >= 0 else '#bb5a41', width=.7)
                ax.text(index, base+value, f'{value:+.3f}', ha='center', va='bottom' if value >= 0 else 'top', fontsize=8)
                base += value
            ax.bar(4, row.dual_margin_b_fixed_rival, color='#286a9e', width=.7)
            extent = np.r_[0., np.cumsum(values), row.dual_margin_b_fixed_rival]
            pad = .18*max(float(np.ptp(extent)), .01)
            ax.set_ylim(float(extent.min())-pad, float(extent.max())+pad)
            ax.axhline(0, color='0.45', linewidth=.8)
            ax.set_xticks(range(5), ['R50\nmargin', 'R50\nreweight', 'Bias\nchange', 'R18\nevidence', 'Dual\nmargin'], fontsize=8)
            ax.set_title(f'Later observation: true {int(row.class_id)} vs rival {int(row.rival)}', fontsize=9)
            ax.set_ylabel('Logit margin; not probabilities', fontsize=9)
        fig.suptitle(f'{kind}: actual VH observations and exact fixed-rival score accounting', fontsize=12)
        fig.savefig(OUT/f'cases_{kind}.png', dpi=145)
        plt.close(fig)
    chosen = pd.DataFrame(chosen)
    chosen.to_csv(OUT/'case_accounting.csv', index=False)
    pd.DataFrame(references).to_csv(OUT/'case_nearest_references.csv', index=False)
    return chosen


def report(obs, table, funnel, swaps, relation, nearest):
    plots(funnel, swaps, relation)
    case_table = cases(obs, table, nearest)
    lines = ['# 同船条件变化：判别关系与第二路贡献', '',
        '2026-10-10。沿用上一轮三折身份留出的冻结探针；未重训、未调参、未改主模型。所有分类预测与参考关系均重放核对。', '',
        '## R50从正确变错误时发生了什么', '',
        '以下仅分析时间较早观测R50正确、较晚观测R50错误的配对。百分比先船内平均，再类内船等权，再类别等权；不同条件子集不一定具有相同类别构成。', '',
        '| 组别 | 退化观测对 | 船舶 | R50同类关系变差% | R50同类优势由正变非正% | R18后时相同类优势为正% | 双路实际救回% |',
        '|---|---:|---:|---:|---:|---:|---:|']
    selected = funnel[(funnel.cohort == 'damaged') & (funnel.minimum_ships_per_class == 1)]
    for row in selected.itertuples():
        lines.append(f'| {LABELS[row.group]} | {row.pairs} | {row.ships} | {row.r50_relation_worsened:.2f} | {row.r50_relation_lost:.2f} | {row.r18_relation_positive_b:.2f} | {row.rescued_damage:.2f} |')
    lines += ['', '同类关系定义：固定64参考案例（每类8艘）中，正确类别的平均点积相似度减去最强错误类别的平均相似度。它是描述性关系指标，不是这个岭分类头的实际参考案例权重，也不直接代表物理散射机制。', '',
        '## 关系仍有用时，双路是否更容易救回', '',
        '| 组别 | R18后时相同类优势为正 | 配对 | 船舶 | 双路救回率% |', '|---|---|---:|---:|---:|']
    for row in relation.itertuples():
        lines.append(f'| {LABELS[row.group]} | {row.r18_relation_positive_b} | {row.pairs} | {row.ships} | {row.rescue_rate:.2f} |')
    lines += ['', '这不是随机干预，也不是可直接部署的路由规则：判断“正确类别优势”用到了事后真值。这里仅用于验证关系指标与纠错是否吻合。',
        '上表两种关系状态的类别构成不同。下表仅在两种状态每类均至少5艘船的共同类别中比较，先类内船等权再共同类等权：', '',
        '| 组别 | 共同类别ID | 同类优势非正时救回% | 同类优势为正时救回% |', '|---|---|---:|---:|']
    comparable = pd.read_csv(OUT/'relation_rescue_common_classes.csv', dtype={'class_ids': str})
    for group, part in comparable.groupby('group'):
        q = part.set_index('r18_relation_positive_b')
        lines.append(f'| {LABELS[group]} | {q.loc[True,"class_ids"]} | {q.loc[False,"rescue_rate"]:.2f} | {q.loc[True,"rescue_rate"]:.2f} |')
    lines += ['', '共同类对照仍是关联，不是控制全部观测条件后的因果结论。若只剩一个类别，只能对该类子集作说明。', '',
        '## 分数贡献拆解', '',
        '固定同一个真实类别和竞争类别，双路相对R50的分数优势变化精确等于：', '',
        '`R50原有特征权重变化 + 截距变化 + 新增R18证据`。', '',
        '同船两个时相在同一双路分类器中的分数变化，则精确等于：`R50路径的时相变化 + R18路径的时相变化`，截距相同抵消。', '',
        '救回案例竞争类别取R50错误预测；误伤案例取双路错误预测。所有项单位均为logit，不是概率。净准确率贡献不能按这些项线性相加。', '',
        '| 组别 | 实际救回的退化配对数 | 将R18坐标置源PCA均值后不再救回的配对数 |', '|---|---:|---:|']
    for row in selected.itertuples():
        lines.append(f'| {LABELS[row.group]} | {row.raw_rescued_pairs} | {row.raw_removal_breaks} |')
    lines += ['', '置源均值是固定双路分类头下的计算干预，不是重新训练的单骨干对照，更不能证明任何单骨干都无法取得该收益。配对数有重复观测，不是独立船数。', '',
        '## 把某一路换回前一次观测会怎样', '',
        '保持后时相的分类器和另一条路径不变，把一条路径替换为同船前时相的真实特征。下表是后时相正确率相对实际双路的变化；正值表示替换帮助，负值表示替换损害。', '',
        '| 组别 | 换回前时相R50/pp | 换回前时相R18/pp |', '|---|---:|---:|']
    for row in swaps.itertuples():
        lines.append(f'| {LABELS[row.group]} | {row.r50_previous_net:+.3f} | {row.r18_previous_net:+.3f} |')
    lines += ['', '每类至少5艘船的子集结果如下；原全类结果保留。', '',
        '| 组别 | 类别数 | 换回前时相R50/pp | 换回前时相R18/pp |', '|---|---:|---:|---:|']
    supported = pd.read_csv(OUT/'swap_supported_summary.csv')
    for row in supported.itertuples():
        lines.append(f'| {LABELS[row.group]} | {row.classes} | {row.r50_previous_net:+.3f} | {row.r18_previous_net:+.3f} |')
    lines += ['', '混合时相特征不一定对应任何真实可成像状态，只能检验该分类器内部路径变化的影响。不是物理反事实、不是可直接部署的新模型，也没有用这项结果选择替换策略。', '',
        '## 图与案例', '', '![关系与贡献](evidence_chain.png)', '',
        '![退化后被救回](cases_rescued_damage.png)', '', '![双路误伤](cases_harm.png)', '',
        '每组各在退化救回或误伤候选中，按R50特征变化排序选中位位置的一对。后验说明性案例，不代表总体频率。两路各自最近3个参考案例的身份、类别、图像路径在case_nearest_references.csv。', '',
        '## 稀疏性与边界', '',
        '- transition_summary.csv同时保留所有类别和每类至少5艘船两种统计。后一种不替换原结果；缺失类别不能自动补零。',
        '- condition_subsets.csv将入射角明显变化、升降轨变化及相对方向可用的子集单列。这些只匹配已知条件，不是独立物理因子。',
        '- 全部工作仍是共有冻结R50/R18及诊断分类头，不是论文当前各港任务训练后的完整模型。',
        '- 训练与留出按身份隔离，但可共享港口与场景；既有预训练曝光未撤销。',
        '- 单个观测可出现在多个配对中，不能把观测对数当作独立样本量。', '']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
