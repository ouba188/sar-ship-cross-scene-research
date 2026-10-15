"""Equal-port summaries and paired mechanism tables; no selection."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
LABELS = dict(raw='Original', count='Count anchoring', count_common='Common penalty',
              guard='Absent-class guard', confidence='Confidence anchoring',
              initial='Relation correction', initial_guard='Relation + guard',
              simple='Fixed scaling', prior_only='Expert prior only')


def boot(values):
    values = np.asarray(values)
    rng = np.random.default_rng(20261012)
    means = values[rng.integers(0, len(values), size=(4000, len(values)))].mean(1)
    return np.quantile(means, [.025, .975])


def table(frame, columns):
    lines = ['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
    for _, row in frame.iterrows():
        vals = [f'{row[col]:.4f}' if isinstance(row[col], (float, np.floating)) else str(row[col]) for col in columns]
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)


def main():
    pair = pd.read_csv(OUT/'paired.csv')
    supports = pd.read_csv(OUT/'supports.csv')
    ids = ['task', 'budget', 'layer', 'arm']
    metrics = ['oa', 'ba', 'delta_oa', 'delta_ba', 'delta_macro_brier']
    port = pair.groupby(ids).agg(**{k:(k,'mean') for k in metrics},
        delta_ba_sd=('delta_ba','std'), positive_draws=('delta_ba',lambda x:int((x>1e-10).sum())),
        negative_draws=('delta_ba',lambda x:int((x< -1e-10).sum()))).reset_index()
    composition = supports.groupby(['task','budget']).agg(n_query=('n_query','first'),
        query_classes=('query_classes','first'), support_classes=('support_classes','mean'),
        absent_query_classes=('absent_query_classes','mean'), support_imbalance=('support_imbalance','mean')).reset_index()
    port = port.merge(composition, on=['task','budget'], validate='many_to_one')
    port.to_csv(OUT/'port_summary.csv', index=False)
    rows = []
    for (b, layer, arm), frame in port.groupby(['budget','layer','arm']):
        ci = boot(frame.delta_ba)
        rows.append(dict(budget=b, layer=layer, arm=arm, ports=len(frame),
            **{k:frame[k].mean() for k in metrics},
            positive_ports=int((frame.delta_ba>1e-10).sum()),
            negative_ports=int((frame.delta_ba< -1e-10).sum()),
            all_draws_positive_ports=int((frame.positive_draws==5).sum()),
            delta_ba_port_sd=frame.delta_ba.std(), descriptive_ci_low=ci[0], descriptive_ci_high=ci[1]))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT/'summary.csv', index=False)
    port[port.n_query>=30].groupby(['budget','layer','arm']).agg(
        ports=('task','size'), delta_oa=('delta_oa','mean'), delta_ba=('delta_ba','mean'),
        delta_macro_brier=('delta_macro_brier','mean')).reset_index().to_csv(OUT/'query_size_sensitivity.csv', index=False)
    targeted = []
    keys = ['task','budget','draw','layer']
    for arm, ref in [('count',r) for r in ('guard','count_common','initial','simple')]+[('initial_guard','initial')]:
        one = pair[pair.arm==arm].merge(pair[pair.arm==ref], on=keys, suffixes=('', '_ref'), validate='one_to_one')
        for metric in ('oa','ba','macro_brier'):
            one['difference_'+metric] = one[metric]-one[metric+'_ref']
        one['candidate'], one['reference'] = arm, ref
        targeted.append(one[keys+['candidate','reference','difference_oa','difference_ba','difference_macro_brier']])
    target = pd.concat(targeted, ignore_index=True)
    target.to_csv(OUT/'targeted_pairs.csv', index=False)
    tp = target.groupby(['task','budget','layer','candidate','reference'])[
        ['difference_oa','difference_ba','difference_macro_brier']].mean().reset_index()
    tp.to_csv(OUT/'targeted_ports.csv', index=False)
    tp.groupby(['budget','layer','candidate','reference']).agg(
        delta_oa=('difference_oa','mean'), delta_ba=('difference_ba','mean'),
        delta_macro_brier=('difference_macro_brier','mean'),
        positive_ports=('difference_ba',lambda x:int((x>1e-10).sum()))).reset_index().to_csv(OUT/'targeted_summary.csv', index=False)
    cls = pd.read_csv(OUT/'classes.csv')
    cp = cls.groupby(['task','budget','layer','arm','class_id','class_name']).agg(
        recall=('recall','mean'), delta_recall=('delta_recall','mean'),
        delta_ba_contribution=('delta_ba_contribution','mean'),
        rescue_ba=('rescue_ba','mean'), harm_ba=('harm_ba','mean'), support_n=('support_n','mean'),
        query_n=('n','first')).reset_index()
    cp.to_csv(OUT/'class_ports.csv', index=False)
    cg = cp.groupby(['budget','layer','arm','class_id','class_name']).agg(
        ports_with_class=('task','size'), recall=('recall','mean'), delta_recall=('delta_recall','mean'),
        delta_ba_contribution=('delta_ba_contribution',lambda x:x.sum()/23),
        rescue_ba=('rescue_ba',lambda x:x.sum()/23), harm_ba=('harm_ba',lambda x:x.sum()/23),
        support_n=('support_n','mean')).reset_index()
    cg.to_csv(OUT/'class_summary.csv', index=False)
    np.testing.assert_allclose(cg.groupby(['budget','layer','arm']).delta_ba_contribution.sum().sort_index(),
        summary.set_index(['budget','layer','arm']).delta_ba.sort_index(), atol=1e-10)
    text = ['# 23港先验保护：完整数值附录',
        '先对每港五组支持抽样平均，再对23港等权。增量单位为百分点；宏Brier差负值表示改善。BA分母为本港查询中实际出现的类别数。',
        '## 完整融合总表', table(summary[summary.layer=='fusion'],
        ['budget','arm','oa','ba','delta_oa','delta_ba','delta_macro_brier','positive_ports','negative_ports','all_draws_positive_ports']),
        '## 数量规则逐港结果', table(port[(port.layer=='fusion')&(port.arm=='count')].sort_values(['budget','task']),
        ['budget','task','n_query','query_classes','oa','ba','delta_oa','delta_ba','delta_ba_sd','positive_draws']),
        '## 数量规则各类别贡献', table(cg[(cg.layer=='fusion')&(cg.arm=='count')],
        ['budget','class_name','ports_with_class','delta_recall','delta_ba_contribution','rescue_ba','harm_ba']),
        '类别召回变化先在有该类查询的港口平均；全局BA贡献则除以全部23港，缺类港贡献为零。两者口径不同。',
        '## 解释边界', '这是历史任务扩展，不是新采集确认或严格未见船验证。重复支持抽样不是独立训练种子。']
    (OUT/'TABLES_ZH.md').write_text('\n\n'.join(text)+'\n', encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    colors = ['#247d8f','#9b4e91','#d18137','#488742']
    for j, arm in enumerate(('count','count_common','guard','initial')):
        data = summary[(summary.layer=='fusion')&(summary.arm==arm)].sort_values('budget')
        for ax, metric in zip(axes, ('delta_oa','delta_ba')):
            ax.plot(np.arange(3), data[metric], marker='o', label=LABELS[arm], color=colors[j])
            ax.set_xticks(np.arange(3), ['16','64','128'])
            ax.set_xlabel('Labeled support budget')
            ax.axhline(0, color='#555555', linewidth=.8)
            ax.grid(axis='y', alpha=.2)
            ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('OA change (percentage points)')
    axes[1].set_ylabel('BA change (percentage points)')
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle('Frozen rules across 23 ports: equal-port mean')
    fig.savefig(OUT/'port_protection.png', dpi=160)
    plt.close(fig)
    print(summary[(summary.layer=='fusion')].to_string(index=False), flush=True)


if __name__=='__main__':
    main()
