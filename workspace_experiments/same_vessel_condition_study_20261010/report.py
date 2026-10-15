"""Post-freeze paired evidence, common-support comparisons and visual examples."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from study import OUT, GROUPS, MODES, SEED, write

LABELS = {'SP_near': '同港/条件近似匹配', 'SP_far': '同港/条件明显不同',
          'DP_near': '跨港/条件近似匹配', 'DP_far': '跨港/条件明显不同'}
PLOT_LABELS = ['Same port\nmatched metadata', 'Same port\nchanged metadata',
               'Different port\nmatched metadata', 'Different port\nchanged metadata']
PRIMARY = ['r50_raw_mse', 'r18_raw_mse', 'r50_flip', 'r18_flip', 'dual_flip',
           'r50_accuracy', 'r18_accuracy', 'dual_accuracy', 'dual_net',
           'dual_rescue', 'dual_harm', 'r50_margin_abs_change', 'r18_margin_abs_change',
           'r50_relation_flip', 'r18_relation_flip']


def truth_margin(values, truth):
    other = values.copy()
    other[np.arange(len(truth)), truth] = -np.inf
    return values[np.arange(len(truth)), truth]-other.max(1)


def summarize_pairs(obs, pairs):
    with np.load(OUT/'FROZEN_PREDICTIONS.npz') as z:
        np.testing.assert_array_equal(z['manifest_rows'], obs.manifest_row)
        scores, relations = z['scores'], z['relations']
    measurements = pd.read_csv(OUT/'pair_representation.csv')
    table = pairs.merge(measurements, on='pair_id', validate='one_to_one')
    a, b, y = table.a.to_numpy(), table.b.to_numpy(), obs.class_id.to_numpy()
    point = obs[['obs_id', 'manifest_row', 'mmsi', 'class_id', 'fold']].copy()
    predictions = scores.argmax(2)
    for mi, mode in enumerate(MODES):
        p = predictions[:, mi]
        correct = p == y
        ca, cb = correct[a], correct[b]
        margin = truth_margin(scores[:, mi], y)
        point[mode+'_prediction'] = p
        point[mode+'_margin'] = margin
        table[mode+'_pred_a'], table[mode+'_pred_b'] = p[a], p[b]
        table[mode+'_accuracy'] = 50*(ca.astype(float)+cb.astype(float))
        table[mode+'_flip'] = 100.*(p[a] != p[b])
        table[mode+'_correct_to_wrong'] = 100.*(ca & ~cb)
        table[mode+'_wrong_to_correct'] = 100.*(~ca & cb)
        table[mode+'_both_wrong'] = 100.*(~ca & ~cb)
        table[mode+'_margin_a'], table[mode+'_margin_b'] = margin[a], margin[b]
        table[mode+'_margin_abs_change'] = np.abs(margin[a]-margin[b])
        if mi < 2:
            rel = relations[:, mi]
            rmargin = truth_margin(rel, y)
            table[mode+'_relation_margin_a'], table[mode+'_relation_margin_b'] = rmargin[a], rmargin[b]
            table[mode+'_relation_flip'] = 100.*(rel[a].argmax(1) != rel[b].argmax(1))
    base = predictions[:, 0] == y
    for mi, mode in [(1, 'r18'), (2, 'dual')]:
        correct = predictions[:, mi] == y
        rescued, harmed = correct & ~base, ~correct & base
        table[mode+'_rescue'] = 50*(rescued[a].astype(float)+rescued[b].astype(float))
        table[mode+'_harm'] = 50*(harmed[a].astype(float)+harmed[b].astype(float))
        table[mode+'_net'] = table[mode+'_rescue']-table[mode+'_harm']
        np.testing.assert_allclose(table[mode+'_net'], table[mode+'_accuracy']-table.r50_accuracy)
    point.to_csv(OUT/'observation_predictions.csv', index=False)
    table.to_csv(OUT/'pair_metrics.csv', index=False)
    return table, point


def hierarchical_summary(table, grouping):
    cells = table.groupby(grouping+['class_id', 'mmsi'], observed=True)[PRIMARY].mean().reset_index()
    classes = cells.groupby(grouping+['class_id'], observed=True)[PRIMARY].mean().reset_index()
    return classes.groupby(grouping, observed=True)[PRIMARY].mean().reset_index()


def common_cell_comparison(table):
    counts = table.groupby(['group', 'class_id', 'gap_bin'], observed=True).mmsi.nunique().unstack('group')
    counts = counts.reindex(columns=GROUPS).fillna(0)
    common = counts.loc[counts.min(axis=1) >= 5]
    counts.to_csv(OUT/'class_time_cell_ships.csv')
    if common.empty:
        return pd.DataFrame(), common
    keep = pd.MultiIndex.from_frame(table[['class_id', 'gap_bin']]).isin(common.index)
    selected = table.loc[keep]
    cells = selected.groupby(['group', 'class_id', 'gap_bin', 'mmsi'], observed=True)[PRIMARY].mean()
    cells = cells.groupby(['group', 'class_id', 'gap_bin'], observed=True)[PRIMARY].mean().reset_index()
    summary = cells.groupby('group')[PRIMARY].mean().reset_index()
    summary.to_csv(OUT/'common_class_time_summary.csv', index=False)
    common.to_csv(OUT/'common_class_time_cells.csv')
    return summary, common


def paired_ship_contrasts(table):
    by_ship = table.groupby(['group', 'class_id', 'mmsi'])[PRIMARY].mean()
    contrasts = [('same_port_condition', 'SP_far', 'SP_near'),
                 ('port_under_near_metadata', 'DP_near', 'SP_near'),
                 ('cross_port_condition', 'DP_far', 'DP_near'),
                 ('port_under_far_metadata', 'DP_far', 'SP_far')]
    records, supported_records, class_records = [], [], []
    rng = np.random.default_rng(SEED+90)
    for name, candidate, reference in contrasts:
        left, right = by_ship.loc[candidate], by_ship.loc[reference]
        common = left.index.intersection(right.index)
        delta = left.loc[common]-right.loc[common]
        if delta.empty:
            continue
        class_counts = delta.groupby('class_id').size()
        for class_id, part in delta.groupby('class_id'):
            means = part.mean()
            for metric in PRIMARY:
                class_records.append(dict(contrast=name, class_id=int(class_id),
                    common_ships=len(part), metric=metric, delta=means[metric],
                    contribution_to_all_class_mean=means[metric]/len(class_counts)))
        supported = delta.loc[delta.index.get_level_values('class_id').isin(class_counts[class_counts >= 5].index)]
        if not supported.empty:
            means = supported.groupby('class_id').mean().mean()
            for metric in PRIMARY:
                supported_records.append(dict(contrast=name, candidate=candidate, reference=reference,
                    common_ships=len(supported), classes=supported.index.get_level_values('class_id').nunique(),
                    metric=metric, delta=means[metric],
                    minimum_ships_per_class=5, posthoc_sparsity_diagnostic=True))
        point = delta.groupby('class_id').mean().mean()
        boot = np.zeros((300, len(PRIMARY)))
        for _, part in delta.groupby('class_id'):
            values = part.to_numpy()
            for i in range(len(boot)):
                boot[i] += values[rng.integers(len(values), size=len(values))].mean(0)/delta.index.get_level_values('class_id').nunique()
        low, high = np.quantile(boot, [.025, .975], axis=0)
        for i, metric in enumerate(PRIMARY):
            records.append(dict(contrast=name, candidate=candidate, reference=reference,
                common_ships=len(delta), classes=delta.index.get_level_values('class_id').nunique(),
                metric=metric, delta=point[metric], descriptive_ci_low=low[i], descriptive_ci_high=high[i]))
    result = pd.DataFrame(records)
    result.to_csv(OUT/'common_ship_contrasts.csv', index=False)
    pd.DataFrame(supported_records).to_csv(OUT/'common_ship_supported_contrasts.csv', index=False)
    pd.DataFrame(class_records).to_csv(OUT/'common_ship_class_contrasts.csv', index=False)
    return result


def plots(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    s = summary.set_index('group').loc[list(GROUPS)]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), layout='constrained')
    xx = np.arange(4)
    for offset, mode, color in [(-.18, 'r50', '#2369a8'), (.18, 'r18', '#bf5a3c')]:
        axes[0].bar(xx+offset, s[mode+'_raw_mse'], .34, color=color, label=mode.upper())
    axes[0].set_ylabel('Mean squared feature change (source-standardized)')
    axes[0].set_title('A. Same vessel, different observations')
    for offset, mode, color in [(-.23, 'r50', '#2369a8'), (0, 'r18', '#bf5a3c'), (.23, 'dual', '#26866f')]:
        axes[1].bar(xx+offset, s[mode+'_flip'], .22, color=color, label=mode.upper())
    axes[1].set_ylabel('Prediction disagreement within pair (%)')
    axes[1].set_title('B. Changing prediction is not always wrong')
    axes[2].bar(xx-.18, s.dual_rescue, .34, color='#26866f', label='Rescue')
    axes[2].bar(xx+.18, s.dual_harm, .34, color='#bf5a3c', label='Harm')
    axes[2].plot(xx, s.dual_net, 'ko-', label='Net advantage')
    axes[2].axhline(0, color='0.6', linewidth=.8)
    axes[2].set_ylabel('Paired observation rate / difference (pp)')
    axes[2].set_title('C. Dual probe versus R50 probe')
    for ax in axes:
        ax.set_xticks(xx, PLOT_LABELS, fontsize=8)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle('Identity-held diagnostic; equal class / vessel weighting; partially matched conditions', fontsize=12)
    fig.savefig(OUT/'four_group_diagnostic.png', dpi=170)
    fig.savefig(OUT/'four_group_diagnostic.pdf')
    plt.close(fig)


def case_cards(table, obs):
    import matplotlib.pyplot as plt
    from PIL import Image
    cases = []
    for kind in ('rescue', 'harm'):
        fig, axes = plt.subplots(4, 4, figsize=(12, 13), layout='constrained')
        for gi, group in enumerate(GROUPS):
            part = table.loc[table.group.eq(group)]
            eligible = part.loc[part['dual_'+kind] > 0]
            fallback = eligible.empty
            if fallback:
                eligible = part
            row = eligible.sort_values(['r50_raw_mse', 'pair_id']).iloc[len(eligible)//2]
            cases.append(dict(kind=kind, group=group, pair_id=int(row.pair_id),
                selection='Median R50 feature distance among pairs with requested rescue/harm; fallback to all pairs',
                fallback=bool(fallback), a=int(row.a), b=int(row.b)))
            for col, (oi, pol) in enumerate([(int(row.a), 'vv'), (int(row.a), 'vh'),
                                            (int(row.b), 'vv'), (int(row.b), 'vh')]):
                item = obs.iloc[oi]
                ax = axes[gi, col]
                path = Path(item['image_relpath_'+pol])
                if path.exists():
                    with Image.open(path) as img:
                        ax.imshow(np.asarray(img.convert('L')), cmap='gray', vmin=0, vmax=255)
                else:
                    ax.text(.5, .5, 'Original image unavailable', ha='center', transform=ax.transAxes)
                endpoint = 'a' if col < 2 else 'b'
                angle = item.incidence_deg
                title = f'{item.port} | {pol.upper()}\n{str(item.acq_time)[:10]} | inc={angle:.1f}'
                ax.set_title(title, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                if col % 2 == 0:
                    ax.set_xlabel(f'Truth {item.class_id} | R50 {int(row["r50_pred_"+endpoint])}\nR18 {int(row["r18_pred_"+endpoint])} | Dual {int(row["dual_pred_"+endpoint])}', fontsize=8)
                if col == 0:
                    ax.set_ylabel(group+f'\n{row.gap_days:.1f} days', fontsize=9)
        fig.suptitle(f'Illustrative {kind} pairs (post-hoc selection, not frequency evidence)', fontsize=12)
        fig.savefig(OUT/f'cases_{kind}.png', dpi=140)
        plt.close(fig)
    pd.DataFrame(cases).to_csv(OUT/'case_selection.csv', index=False)


def render(summary, coverage, common, cells, contrasts, overall):
    lines = ['# 同船、跨港与观测条件：四组诊断', '',
        '日期：2026-10-10。已完成元数据分组、三折按MMSI留出的冻结表示诊断和正负案例。未修改主模型。', '',
        '## 先说清楚这轮测了什么', '',
        '- 每对真实重复观测使用相同的网络缓存、源拟合标准化/PCA、分类头及64个参考案例，避免把不同港口模型混入成像变化。',
        '- 使用共有的ImageNet R50配对VV/VH特征，以及源任务训练前的冻结SAR-SSL R18特征。不是当前各港分别训练过的R18专家，不是完整模型性能复测。',
        '- 三折按MMSI分离；标准化、PCA128/64和岭分类头只拟合另外两折。类别/船舶等权拟合；相同船的全部港口和时相只出现在一折。',
        '- 诊断分类头使用拟合折全部标签；64个参考案例用于邻近关系诊断，不是论文64目标标签适配协议。不存在新增64标签方法增益声明。',
        '- MMSI为身份代理，既有冻结预训练曝光未撤销。训练与留出折可能共享港口及采集场景；结果不是物理因果鉴定，也不是独立外部确认。', '',
        '## 配对覆盖', '', '| 组别 | 观测对 | 船舶身份 | 类别数 | 时间差中位数/天 | 相对方向可比对 | 方向差≤10° |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for group in GROUPS:
        row = coverage.loc[group]
        lines.append(f'| {LABELS[group]} | {int(row.pairs)} | {int(row.ships)} | {int(row.classes)} | {row.median_gap_days:.1f} | {int(row.direction_known)} | {int(row.direction_near)} |')
    lines += ['', '近似条件：同卫星平台、同升降轨方向、入射角差≤2°。明显不同：平台/升降轨变化或入射角差≥5°。未假定船体相对方向、背景、海况、载荷与航行状态一致。方向代理相近仍不等于所有条件相同。',
        '全部配对及未匹配组在pairs.csv；配对共享船舶/影像，不是独立样本。以下组均值先船内平均，再类内船等权，再类别等权。', '',
        '## 原始分组结果', '', '| 组别 | R50特征变化 | R18特征变化 | R50预测变化% | R18预测变化% | 双路预测变化% | R50配对正确率% | 双路配对正确率% | 双路净纠错pp |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary.itertuples():
        lines.append(f'| {LABELS[row.group]} | {row.r50_raw_mse:.3f} | {row.r18_raw_mse:.3f} | {row.r50_flip:.2f} | {row.r18_flip:.2f} | {row.dual_flip:.2f} | {row.r50_accuracy:.2f} | {row.dual_accuracy:.2f} | {row.dual_net:+.2f} |')
    lines += ['', '特征变化为拟合折标准化后的维均方差；更小不自动意味着保留了更多类别信息。预测变化包括纠错，不能全部当成退化。配对正确率按上述层级平均两端正确率，不是LOPO主表OA/BA。',
        '', '![四组诊断](four_group_diagnostic.png)', '', '## 时间与样本构成核对', '']
    if common.empty:
        lines.append('没有满足四组每个类别/时间格都至少5艘船的公共格，不能给出该标准化对照。')
    else:
        lines += [f'公共类别×时间间隔格：{len(cells)}个。每格四组均至少5个MMSI，格内先船等权，再公共格等权。此对照仍未控制所有船舶及观测因素。', '',
            '| 组别 | R50变化 | R18变化 | R50预测变化% | 双路净纠错pp |', '|---|---:|---:|---:|---:|']
        for row in common.itertuples():
            lines.append(f'| {LABELS[row.group]} | {row.r50_raw_mse:.3f} | {row.r18_raw_mse:.3f} | {row.r50_flip:.2f} | {row.dual_net:+.2f} |')
    lines += ['', '## 同一批船内比较', '',
        '只保留同时有两种配对情形的船：先分别求每艘船的组内均值，再相减，最后类别等权。区间为300次类内船舶重采样的描述性95%区间；不代表控制全部混杂后的因果效应。', '',
        '| 对照 | 共同船舶 | R50特征变化差 | R18特征变化差 | R50预测变化差pp | 双路净纠错差pp |', '|---|---:|---:|---:|---:|---:|']
    for name, part in contrasts.groupby('contrast', sort=False):
        q = part.set_index('metric')
        values = [f'{q.loc[k,"delta"]:+.3f}' for k in ['r50_raw_mse', 'r18_raw_mse', 'r50_flip', 'dual_net']]
        lines.append(f'| {name} | {int(part.common_ships.iloc[0])} | '+' | '.join(values)+' |')
    lines += ['', '完整区间见common_ship_contrasts.csv；公共格明细见common_class_time_cells.csv。稀疏子集只能作线索，不能推广到全部船型。',
        '实际发现部分共同船舶对照中的类别只有1—2个MMSI；该类重采样不能反映未观测船舶的变异，不能据此把原区间解释为总体显著性。', '',
        '### 稀疏类别根源检查', '',
        '以下是看到上述稀疏性后增加的解释性检查：只统计每类至少5艘共同船，原全类别结果不替换，不据此调模型。', '',
        '| 对照 | 共同船舶 | 保留类别 | R50特征变化差 | R18特征变化差 | 双路净纠错差pp |',
        '|---|---:|---:|---:|---:|---:|']
    supported = pd.read_csv(OUT/'common_ship_supported_contrasts.csv')
    for name, part in supported.groupby('contrast', sort=False):
        q = part.set_index('metric')
        values = [f'{q.loc[k,"delta"]:+.3f}' for k in ['r50_raw_mse', 'r18_raw_mse', 'dual_net']]
        lines.append(f'| {name} | {int(part.common_ships.iloc[0])} | {int(part.classes.iloc[0])} | '+' | '.join(values)+' |')
    lines += ['', '按类的贡献与船数见common_ship_class_contrasts.csv。条件组差异不是一个可自动解释为物理强弱的连续等级。', '',
        '## 完整观测集的诊断读出', '', '| 读出 | 观测OA% | 观测类别平均召回BA% |', '|---|---:|---:|']
    for row in overall.itertuples():
        lines.append(f'| {row.mode} | {row.oa:.3f} | {row.ba:.3f} |')
    lines += ['', '这是跨身份、混合港口的冻结表示探针，不是目标港完全留出的跨港准确率。不能与论文当前主表数值直接比较。', '',
        '## 可追溯案例', '',
        '每组分别选择存在双路救回/误伤的候选中，R50特征变化处于中位位置的一对；不是随机频率估计。选择规则、行号在case_selection.csv。原图不作重新成像或方向模拟。', '',
        '![救回案例](cases_rescue.png)', '', '![误伤案例](cases_harm.png)', '',
        '## 下一步解释边界', '',
        '- 若同港条件变化的关联在公共时间格、共同船舶中仍在，支持继续研究观测条件响应，而不是直接删除全部同船变化。',
        '- 跨港近似匹配仍缺相对方向等控制，剩余差异不能直接命名为港口物理效应。',
        '- 第二路更稳定与第二路有用是两件事，需联合净纠错及类别/案例关系变化判断；不能从网络名字分配轮廓或散射职责。',
        '- 若要归因当前任务训练后的R18，需要在共同且身份隔离的训练设定下提取配对两端；本轮不借用不同港口的不同权重作这种证明。', '']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines), encoding='utf-8')


def report(obs, pairs):
    if not (OUT/'PREDICTIONS_SAVED.json').exists():
        raise RuntimeError('Freeze all fold predictions before query readout')
    table, point = summarize_pairs(obs, pairs)
    selected = table.loc[table.group.isin(GROUPS)]
    summary = hierarchical_summary(selected, ['group'])
    summary.to_csv(OUT/'group_summary.csv', index=False)
    per_class = selected.groupby(['group', 'class_id', 'mmsi'])[PRIMARY].mean().groupby(['group', 'class_id']).mean()
    per_class.to_csv(OUT/'class_summary.csv')
    selected.groupby(['group', 'class_id']).agg(pairs=('pair_id', 'size'),
        ships=('mmsi', 'nunique')).to_csv(OUT/'class_coverage.csv')
    common, cells = common_cell_comparison(selected)
    contrasts = paired_ship_contrasts(selected)
    strict = selected.loc[selected.direction_near & selected.group.isin(['SP_near', 'DP_near'])]
    hierarchical_summary(strict, ['group']).to_csv(OUT/'direction_matched_summary.csv', index=False)
    angle = selected.loc[selected.angle_only_far]
    hierarchical_summary(angle, ['group']).to_csv(OUT/'incidence_only_far_summary.csv', index=False)
    overall = []
    for mode in MODES:
        correct = point[mode+'_prediction'].eq(point.class_id)
        overall.append(dict(mode=mode, oa=100*correct.mean(),
            ba=100*correct.groupby(point.class_id).mean().mean()))
    overall = pd.DataFrame(overall)
    overall.to_csv(OUT/'probe_overall.csv', index=False)
    coverage = pd.read_csv(OUT/'coverage.csv').set_index('group')
    plots(summary)
    case_cards(selected, obs)
    render(summary, coverage, common, cells, contrasts, overall)
    print(summary[['group', 'r50_raw_mse', 'r18_raw_mse', 'r50_flip', 'r18_flip', 'dual_net']].to_string(index=False), flush=True)
    print(overall.to_string(index=False), flush=True)


if __name__ == '__main__':
    obs = pd.read_csv(OUT/'observations.csv', dtype={'mmsi': str}, parse_dates=['acq_time'])
    pairs = pd.read_csv(OUT/'pairs.csv', dtype={'mmsi': str})
    report(obs, pairs)
