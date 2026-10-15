"""Paired, class-aware summaries and a compact figure; no method selection."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
LABELS = dict(raw='原模型', initial='已有关系修正', simple='旧版固定缩放',
    class_strength_match='旧逐类强度匹配', prior_only='仅源专家先验',
    guard='仅保护缺席类', count='按类数量约束', count_common='同均值统一约束',
    confidence='按类源置信度约束', initial_guard='关系修正+缺席类保护')


def markdown(frame, columns):
    rows = ['| '+' | '.join(columns.values())+' |', '| '+' | '.join(['---']*len(columns))+' |']
    for _, row in frame.iterrows():
        parts = []
        for key in columns:
            value = row[key]
            if key in ('arm', 'reference'):
                value = LABELS.get(value, value)
            elif isinstance(value, (float, np.floating)):
                value = f'{value:.6f}'
            parts.append(str(value))
        rows.append('| '+' | '.join(parts)+' |')
    return '\n'.join(rows)


def main():
    frame = pd.read_csv(ROOT/'metrics.csv')
    summary = pd.read_csv(ROOT/'summary.csv')
    cls = pd.read_csv(ROOT/'classes.csv')
    subsets = pd.read_csv(ROOT/'coverage_groups.csv')
    ids = ['dataset','budget','draw','layer']
    contrasts = []
    for arm, reference in [('guard','raw'), ('initial_guard','initial'), ('count','guard'),
                           ('count','count_common'), ('confidence','guard'), ('count','initial'),
                           ('confidence','initial'), ('count','class_strength_match'),
                           ('confidence','class_strength_match')]:
        a = frame[frame.arm==arm]
        b = frame[frame.arm==reference]
        pairs = a.merge(b, on=ids, suffixes=('_a','_b'), validate='one_to_one')
        for _, row in pairs.iterrows():
            contrasts.append(dict(**{key:row[key] for key in ids}, arm=arm, reference=reference,
                delta_oa=row.oa_a-row.oa_b, delta_ba=row.ba_a-row.ba_b,
                delta_macro_brier=row.macro_brier_a-row.macro_brier_b))
    pairs = pd.DataFrame(contrasts)
    pairs.to_csv(ROOT/'targeted_pairs.csv', index=False)
    comparisons = pairs.groupby(['dataset','budget','layer','arm','reference']).agg(
        delta_oa=('delta_oa','mean'), delta_ba=('delta_ba','mean'), sd_ba=('delta_ba','std'),
        positive_draws=('delta_ba', lambda x:int((x>1e-10).sum())),
        delta_macro_brier=('delta_macro_brier','mean')).reset_index()
    comparisons.to_csv(ROOT/'targeted_summary.csv', index=False)
    grouped = subsets.groupby(['dataset','budget','layer','arm','absent']).agg(
        draws=('draw','nunique'), mean_classes=('classes','mean'), mean_queries=('queries','mean'),
        delta_recall=('delta_recall','mean'), delta_brier=('delta_brier','mean'),
        contribution=('contribution','mean')).reset_index()
    grouped.to_csv(ROOT/'coverage_group_summary.csv', index=False)
    fusion = summary[summary.layer=='fusion']
    sections = ['# 先验保护对照：完整数值附录',
        '所有OA/BA差为百分点；Brier差负数表示改善。SD描述五次支持抽样，不是独立训练重复。']
    for ds in ('f0921','oss2'):
        sections += ['## '+ds, markdown(fusion[fusion.dataset==ds],
            dict(budget='B',arm='方法',oa='OA%',ba='BA%',delta_oa='ΔOA',delta_ba='ΔBA',
                 delta_ba_sd='ΔBA的SD',positive_draws='BA正向抽样/5',delta_macro_brier='Δ宏Brier'))]
    sections += ['## 针对性替换比较',markdown(comparisons[comparisons.layer=='fusion'],
        dict(dataset='数据',budget='B',arm='方案',reference='参照',delta_oa='ΔOA',delta_ba='ΔBA',
             positive_draws='BA正向抽样/5',delta_macro_brier='Δ宏Brier'))]
    (ROOT/'TABLES_ZH.md').write_text('\n\n'.join(sections)+'\n',encoding='utf-8')

    fig, axes = plt.subplots(1,2,figsize=(11,4.4),sharey=True)
    arms = ['guard','count','confidence','initial']
    labels = ['Absent-class guard','Count anchoring','Confidence anchoring','Existing relation correction']
    colors = ['#4477AA','#228833','#CC6677','#AA4499']
    for ax, ds, title in zip(axes, ('f0921','oss2'), ('F_0921','OpenSARShip 2.0')):
        for j, (arm,label,color) in enumerate(zip(arms,labels,colors)):
            rows = fusion[(fusion.dataset==ds)&(fusion.arm==arm)].set_index('budget').loc[[16,64,128]]
            ax.bar(np.arange(3)+(j-1.5)*.18, rows.delta_ba, width=.17, color=color,
                   yerr=rows.delta_ba_sd,capsize=2,error_kw={'linewidth':.65},label=label)
        ax.axhline(0,color='#222222',linewidth=.7)
        ax.set_xticks(range(3),['16','64','128'])
        ax.set_title(title,fontsize=12)
        ax.set_xlabel('Labeled support budget')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.18)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Fusion BA change (percentage points)')
    fig.suptitle('Fixed prior-protection rules; mean and SD of five paired draws',fontsize=11)
    fig.legend(*axes[0].get_legend_handles_labels(),ncol=2,loc='lower center',frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,.13,1,.94))
    fig.savefig(ROOT/'prior_protection.png',dpi=160)
    plt.close(fig)
    print(fusion.to_string(index=False))


if __name__ == '__main__':
    main()
