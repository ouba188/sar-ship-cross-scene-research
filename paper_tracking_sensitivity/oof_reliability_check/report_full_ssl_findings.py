"""CPU-only findings from completed, consistently re-extracted full SSL readout."""

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from full_ssl_run import OUT, CACHE, TARGETS, MODES, EPOCHS, ETA, BETA, SEED


DATASET_NAMES = {'f0921': 'F_0921', 'oss2': 'OSS2'}
CONTRAST_NAMES = {
    'D_vs_G_single': 'D50-G50，单路',
    'D_vs_G_dual': 'D50-G50，固定D18双路',
    'dual_vs_single_G': '加入D18，G50固定',
    'dual_vs_single_D': '加入D18，D50固定',
    'c4_vs_raw': 'C4-原视图',
    'condition_vs_c4': '条件均值核-C4',
    'frequency_vs_condition': '频率核-条件均值核',
    'frequency_single': '频率增量，单路融合',
    'frequency_dual': '频率增量，双路融合',
}


def contrasts():
    """Each tuple is family, name, before(domain, mode), after(domain, mode)."""
    result = []
    for family, prefix in (('primary', 'late'), ('secondary', 'unified')):
        single, dual = f'{prefix}_single', f'{prefix}_dual'
        result.extend([
            (family, 'D_vs_G_single', ('G', single), ('D', single)),
            (family, 'D_vs_G_dual', ('G', dual), ('D', dual)),
            (family, 'dual_vs_single_G', ('G', single), ('G', dual)),
            (family, 'dual_vs_single_D', ('D', single), ('D', dual)),
        ])
    for domain in ('G', 'D'):
        for name, before, after in (
                ('c4_vs_raw', 'base_raw', 'base_c4'),
                ('condition_vs_c4', 'base_c4', 'base_condition'),
                ('frequency_vs_condition', 'base_condition', 'base_full'),
                ('frequency_single', 'late_single_no_frequency', 'late_single'),
                ('frequency_dual', 'late_dual_no_frequency', 'late_dual')):
            result.append(('stage', name, (domain, before), (domain, after)))
    return result


def error_flow(truth, before, after):
    """Exact OA and mean-present-class BA accounting, on query rows only."""
    old, new = before == truth, after == truth
    rescue, harm = ~old & new, old & ~new
    classes = np.unique(truth)
    per_class = []
    for c in classes:
        mask = truth == c
        n = int(mask.sum())
        r, h = int(rescue[mask].sum()), int(harm[mask].sum())
        old_recall, new_recall = 100*old[mask].mean(), 100*new[mask].mean()
        per_class.append(dict(class_id=int(c), n=n, recall_before=old_recall,
            recall_after=new_recall, delta_recall_pp=new_recall-old_recall,
            rescue_n=r, harm_n=h, rescue_ba_pp=100*r/n/len(classes),
            harm_ba_pp=100*h/n/len(classes), delta_ba_contribution_pp=100*(r-h)/n/len(classes)))
    result = dict(n=len(truth), classes=len(classes), rescue_n=int(rescue.sum()),
        harm_n=int(harm.sum()), both_correct_n=int((old & new).sum()),
        both_wrong_n=int((~old & ~new).sum()),
        oa_before=100*old.mean(), oa_after=100*new.mean(),
        ba_before=float(np.mean([r['recall_before'] for r in per_class])),
        ba_after=float(np.mean([r['recall_after'] for r in per_class])),
        rescue_oa_pp=100*rescue.mean(), harm_oa_pp=100*harm.mean(),
        rescue_ba_pp=sum(r['rescue_ba_pp'] for r in per_class),
        harm_ba_pp=sum(r['harm_ba_pp'] for r in per_class))
    result['net_oa_pp'] = result['rescue_oa_pp']-result['harm_oa_pp']
    result['net_ba_pp'] = result['rescue_ba_pp']-result['harm_ba_pp']
    return result, per_class


def collect(metrics, per_class):
    metric_index = metrics.set_index(['dataset', 'domain', 'mode', 'draw'], verify_integrity=True)
    class_index = per_class.set_index(['dataset', 'domain', 'mode', 'draw', 'class_id'], verify_integrity=True)
    paired, flows, class_deltas = [], [], []
    for ds, paths in TARGETS.items():
        with np.load(OUT/f'{ds}_design.npz', allow_pickle=False) as saved:
            design = {k: saved[k] for k in ('rows', 'ids', 'support', 'support_labels', 'eval_masks')}
        frame = pd.read_csv(paths['manifest'], usecols=['sample_id', 'class_id', 'class_name']).iloc[design['rows']]
        np.testing.assert_array_equal(frame.sample_id.to_numpy(str), design['ids'])
        truth = frame.class_id.to_numpy(int)
        np.testing.assert_array_equal(truth[design['support']], design['support_labels'])
        class_names = frame[['class_id', 'class_name']].drop_duplicates().set_index('class_id', verify_integrity=True).class_name.to_dict()
        predictions = {}
        for domain in ('G', 'D'):
            folder = CACHE/domain/ds
            receipt = json.loads((folder/'PREDICTIONS_SAVED.json').read_text(encoding='utf-8'))
            if receipt['modes'] != list(MODES):
                raise ValueError(f'Prediction mode order differs: {domain}/{ds}')
            predictions[domain] = np.load(folder/'predictions.npy', mmap_mode='r')
            if predictions[domain].shape != (5, len(MODES), len(truth)):
                raise ValueError(f'Prediction shape differs: {domain}/{ds}')
        for family, name, (bd, bm), (ad, am) in contrasts():
            for draw, mask in enumerate(design['eval_masks']):
                meta = dict(dataset=ds, family=family, contrast=name, before_domain=bd,
                            before_mode=bm, after_domain=ad, after_mode=am, draw=draw)
                if mask[design['support'][draw]].any():
                    raise ValueError(f'Support rows enter query metrics: {ds}/{draw}')
                old = predictions[bd][draw, MODES.index(bm), mask]
                new = predictions[ad][draw, MODES.index(am), mask]
                flow, changes = error_flow(truth[mask], old, new)
                before = metric_index.loc[(ds, bd, bm, draw)]
                after = metric_index.loc[(ds, ad, am, draw)]
                # A mismatch means the CSV and prediction archives are not one readout.
                np.testing.assert_allclose([flow['oa_before'], flow['ba_before'], flow['oa_after'], flow['ba_after']],
                                           [before.oa, before.ba, after.oa, after.ba], rtol=0, atol=1e-9)
                np.testing.assert_allclose([flow['net_oa_pp'], flow['net_ba_pp']],
                                           [after.oa-before.oa, after.ba-before.ba], rtol=0, atol=1e-9)
                paired.append(dict(**meta, n=flow['n'], oa_before=before.oa, oa_after=after.oa,
                    ba_before=before.ba, ba_after=after.ba, delta_oa_pp=flow['net_oa_pp'],
                    delta_ba_pp=flow['net_ba_pp']))
                flows.append(dict(**meta, **flow))
                for change in changes:
                    c = change['class_id']
                    cb, ca = class_index.loc[(ds, bd, bm, draw, c)], class_index.loc[(ds, ad, am, draw, c)]
                    np.testing.assert_allclose([change['recall_before'], change['recall_after'], change['n'], change['n']],
                                               [cb.recall, ca.recall, cb.n, ca.n], rtol=0, atol=1e-9)
                    class_deltas.append(dict(**meta, class_name=class_names[c], **change))
    paired = pd.DataFrame(paired)
    keys = ['dataset', 'family', 'contrast', 'before_domain', 'before_mode', 'after_domain', 'after_mode']
    grouped = paired.groupby(keys, sort=False)
    paired['draws'] = grouped.draw.transform('count')
    for metric in ('oa', 'ba'):
        paired[f'{metric}_positive_draws'] = grouped[f'delta_{metric}_pp'].transform(lambda x: int((x > 1e-10).sum()))
    if not paired.draws.eq(5).all():
        raise ValueError('Expected five paired support draws per contrast')
    return paired, pd.DataFrame(flows), pd.DataFrame(class_deltas)


def markdown_table(headers, rows):
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| '+' | '.join(map(cell, headers))+' |',
                      '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(cell, row))+' |' for row in rows])


def optional_summary(filename, title):
    path = OUT/filename
    if not path.exists():
        return ['', f'## {title}', '', f'待完成：尚无 {filename}，本报告不推断结果。']
    frame = pd.read_csv(path)
    rows = [[f'{value:.6g}' if isinstance(value, (float, np.floating)) else value
             for value in row] for row in frame.head(12).itertuples(index=False, name=None)]
    note = f'直接列示 {filename}；定义与测量范围以该实验协议为准。'
    if len(frame) > 12:
        note += f'这里只展示原文件前12行，全部{len(frame)}行见CSV，未按收益筛选。'
    return ['', f'## {title}', '', note, '', markdown_table(frame.columns, rows)]


def make_figure(metrics, paired):
    """Primary BA, paired effects, and base stages; dots are support draws."""
    datasets = list(TARGETS)
    colors = ('#0072B2', '#009E73', '#CC79A7', '#D55E00')
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titlesize': 12, 'axes.labelsize': 10, 'legend.fontsize': 9}):
        fig, axes = plt.subplots(3, len(datasets), figsize=(12, 11), squeeze=False, layout='constrained')
        for col, ds in enumerate(datasets):
            ax = axes[0, col]
            for i, (domain, mode) in enumerate((('G', 'late_single'), ('D', 'late_single'),
                                               ('G', 'late_dual'), ('D', 'late_dual'))):
                values = metrics[(metrics.dataset == ds) & (metrics.domain == domain) & (metrics['mode'] == mode)].sort_values('draw').ba.to_numpy()
                ax.scatter(i+np.linspace(-.1, .1, len(values)), values, color=colors[i], s=25, alpha=.65)
                ax.plot([i-.2, i+.2], [values.mean()]*2, color=colors[i], linewidth=3)
            ax.set_xticks(range(4), ['G50', 'D50', 'G50+D18', 'D50+D18'])
            ax.set_ylabel('Balanced accuracy (%)')
            ax.set_title(f'{DATASET_NAMES[ds]} | primary equal-softmax')
            ax.grid(axis='y', alpha=.2)
            ax = axes[1, col]
            labels = ['D-G, single', 'D-G, dual', 'Dual-single, G', 'Dual-single, D']
            for i, name in enumerate(list(CONTRAST_NAMES)[:4]):
                values = paired[(paired.dataset == ds) & (paired.family == 'primary') & (paired.contrast == name)].sort_values('draw').delta_ba_pp.to_numpy()
                ax.scatter(values, i+np.linspace(-.1, .1, len(values)), color=colors[i], s=25, alpha=.65)
                ax.scatter([values.mean()], [i], color=colors[i], marker='D', s=65, edgecolors='black', linewidths=.5)
            ax.axvline(0, color='#555555', linewidth=1)
            ax.set_yticks(range(4), labels)
            ax.invert_yaxis()
            ax.set_xlabel('Paired BA change (percentage points)')
            ax.grid(axis='x', alpha=.2)
            ax = axes[2, col]
            stages = ('base_raw', 'base_c4', 'base_condition', 'base_full')
            for domain, color, marker in (('G', colors[0], 'o'), ('D', colors[1], 's')):
                values = metrics[(metrics.dataset == ds) & (metrics.domain == domain)]
                means = [values.loc[values['mode'] == stage, 'ba'].mean() for stage in stages]
                ax.plot(range(4), means, color=color, marker=marker, label=f'{domain}50', linewidth=2)
            ax.set_xticks(range(4), ['Raw', 'C4', '+ condition', '+ frequency'])
            ax.set_ylabel('Mean base BA (%)')
            ax.set_title('Ordered readout interventions')
            ax.legend(frameon=False)
            ax.grid(axis='y', alpha=.2)
        fig.suptitle('Full readout integration: one neural seed, five paired support draws\nDots: support draws; diamonds / short lines: means; no confidence intervals', fontsize=13)
    return fig


def findings(metrics, paired, flows, protocol):
    means = metrics.groupby(['dataset', 'domain', 'mode'])[['oa', 'ba']].mean()
    rows = []
    for ds in TARGETS:
        for domain, architecture in (('G', 'single'), ('D', 'single'), ('G', 'dual'), ('D', 'dual')):
            primary = means.loc[(ds, domain, f'late_{architecture}')]
            secondary = means.loc[(ds, domain, f'unified_{architecture}')]
            label = f'{domain}50'+('+D18' if architecture == 'dual' else '')
            rows.append([DATASET_NAMES[ds], label, f'{primary.oa:.3f}', f'{primary.ba:.3f}',
                         f'{secondary.oa:.3f}', f'{secondary.ba:.3f}'])
    lines = ['# 通用/领域骨干：完整外部读出配平实验', '',
        '本轮比较 G50/D50 × 单路/固定同一D18双路。主结果为预先固定的等权 softmax 融合；'
        '统一核/先验读出为次要结果，不按目标成绩另选读出。以下均为原5组64标签支持抽取的均值。', '']
    for ds in TARGETS:
        subset = paired[(paired.dataset == ds) & (paired.family == 'primary')]
        a = subset[subset.contrast == 'D_vs_G_single']
        b = subset[subset.contrast == 'dual_vs_single_D']
        lines.append(f"- {DATASET_NAMES[ds]}：D50单路相对G50单路 ΔOA {a.delta_oa_pp.mean():+.3f}、"
                     f"ΔBA {a.delta_ba_pp.mean():+.3f} 个百分点（BA正向 {int(a.ba_positive_draws.iloc[0])}/5）；"
                     f"在D50上加入D18后 ΔOA {b.delta_oa_pp.mean():+.3f}、ΔBA {b.delta_ba_pp.mean():+.3f} "
                     f"个百分点（BA正向 {int(b.ba_positive_draws.iloc[0])}/5）。")
    lines += ['', '## 四组主比较与次要读出', '',
        markdown_table(['数据', '骨干', '主OA %', '主BA %', '统一OA %', '统一BA %'], rows), '',
        '单路也保留 base 与 original-view expert 两个头；不是删掉完整算子的弱单路。'
        '两组双路共用冻结的源域SSL D18。单路 retained/discarded 分别按128/2048维归一化，双路按192/2560维归一化。', '',
        '## 配对误差流', '',
        '下表均为“后者减前者”。救回=原错后对，误伤=原对后错；BA救回/误伤先在每个查询类别内除以其样本数，'
        '再对查询中出现的类别等权平均，因此每组抽取严格满足 ΔBA=救回BA-误伤BA。数量是每组抽取的均值，不是独立船舶数。', '']
    rows = []
    for ds in TARGETS:
        for name in list(CONTRAST_NAMES)[:4]:
            p = paired[(paired.dataset == ds) & (paired.family == 'primary') & (paired.contrast == name)]
            f = flows[(flows.dataset == ds) & (flows.family == 'primary') & (flows.contrast == name)]
            rows.append([DATASET_NAMES[ds], CONTRAST_NAMES[name], f'{p.delta_oa_pp.mean():+.3f}',
                f'{p.delta_ba_pp.mean():+.3f}', f'{int(p.oa_positive_draws.iloc[0])}/5',
                f'{int(p.ba_positive_draws.iloc[0])}/5', f'{f.rescue_n.mean():.1f}/{f.harm_n.mean():.1f}',
                f'{f.rescue_ba_pp.mean():.3f}/{f.harm_ba_pp.mean():.3f}'])
    lines += [markdown_table(['数据', '对照', 'ΔOA pp', 'ΔBA pp', 'OA正向', 'BA正向', '救回/误伤条数', '救回/误伤BA pp'], rows), '',
        '逐次配对值及正向抽取数见 paired_deltas.csv；主/次要与阶段对照的完整误差流见 errorflow.csv。'
        'perclassdelta.csv 保留元数据中的类别名称、逐类召回变化及其BA贡献，包含全部负变化。', '',
        '## 阶段与频率诊断', '']
    rows = []
    for ds in TARGETS:
        for domain in ('G', 'D'):
            stage_ba = [means.loc[(ds, domain, name), 'ba'] for name in ('base_raw', 'base_c4', 'base_condition', 'base_full')]
            nofreq = [means.loc[(ds, domain, f'late_{a}_no_frequency'), 'ba'] for a in ('single', 'dual')]
            gains = [means.loc[(ds, domain, f'late_{a}'), 'ba']-v for a, v in zip(('single', 'dual'), nofreq)]
            rows.append([DATASET_NAMES[ds], domain+'50', *[f'{v:.3f}' for v in stage_ba],
                         f'{nofreq[0]:.3f}', f'{nofreq[1]:.3f}', f'{gains[0]:+.3f}', f'{gains[1]:+.3f}'])
    lines += [markdown_table(['数据', '骨干', '原视图BA', 'C4 BA', '+条件BA', '+频率BA',
                             '无频率单路BA', '无频率双路BA', '频率Δ单路', '频率Δ双路'], rows), '',
        '阶段变化是按固定顺序干预的计算结果，可能含交互；不能当作相互独立的因果贡献。'
        '频率分支以base支持校正任务训练，不等于对最终双头融合进行了端到端优化。', '',
        '![主比较、配对变化与阶段诊断](full_ssl_results.png)', '', '## 实施与边界', '',
        f'- 源池为两套外部任务共同允许的 {protocol["source_rows"]:,} 行；PCA、尺度与全部分类先验重新拟合，'
        f'频率分支固定训练 {EPOCHS} 轮，eta={ETA}、beta={BETA}，源ridge1/支持ridge8。',
        '- 原视图/C4 R50及D18均在本次 consistent_replay 中重新提取，统一 deterministic=True、benchmark=False、TF32关闭；'
        '读取的是当前CACHE，未混用旧GAP/频率特征。旧尝试保留，旧SSL检查点仅作为冻结编码器初始化。',
        '- 已修复实现问题：源缓存map在Haar前由bf16经half转为float32，目标曾直接用bf16做Haar，'
        '导致训练/推理算术不一致。frequency_embedding现统一在Haar前转float32，目标与计时路径同步修正并加入回归测试。'
        '修正前G/OSS2特征归档于pre_haar_alignment，F未完成特征重新提取；没有已完成预测受影响。'
        '这是数值路径修正，不是已证实的模型增益或历史退化根因。',
        '- 输入为112像素 RGB=[VV,VH,均值]；G/D及单/双路输入权限配平。未复用历史F暴露R18，'
        '也没有重训历史监督layer4尾部。因此名称是“完整读出集成”，不是旧完整训练模型的等价复现。',
        f'- 神经种子仅 {SEED} 一个；5组支持抽取重叠，不能当成5个神经种子或独立重复。正向次数不是显著性检验。',
        '- 全部配置预测冻结后才读查询真值；目标数据已有开发历史，不是独立确认集。源episode的PCA共享允许源池，'
        '频率训练episode不是独立验证；共享SSL编码器也不能直接声称严格23港LOPO。',
        '- 历史模型的输入、训练与源池不同，跨历史表格相减不是受控增益。本报告不拼接历史成绩；'
        '任何双路收益只说明当前决策下的增量预测价值，不证明两路对应物理上互斥的线索或双骨干必需。', '']
    lines += optional_summary('path_summary.csv', '先验与核交叉诊断')
    lines += optional_summary('cost_summary.csv', '计算成本')
    lines += ['', '交叉组合是决策路径诊断，不自动等于可部署单骨干；计时不超出其实际覆盖范围解释，'
              '不将神经特征提取耗时称作端到端部署延迟。', '']
    return '\n'.join(lines)


def main():
    if not (OUT/'READOUT_COMPLETE.json').exists():
        raise SystemExit('READOUT_COMPLETE.json is absent; no results read or outputs written.')
    metrics = pd.read_csv(OUT/'metrics.csv')
    per_class = pd.read_csv(OUT/'per_class.csv')
    protocol = json.loads((OUT/'PROTOCOL.json').read_text(encoding='utf-8'))
    paired, flows, class_deltas = collect(metrics, per_class)
    paired.to_csv(OUT/'paired_deltas.csv', index=False, encoding='utf-8-sig')
    flows.to_csv(OUT/'errorflow.csv', index=False, encoding='utf-8-sig')
    class_deltas.to_csv(OUT/'perclassdelta.csv', index=False, encoding='utf-8-sig')
    figure = make_figure(metrics, paired)
    figure.savefig(OUT/'full_ssl_results.png', dpi=300, bbox_inches='tight')
    plt.close(figure)
    (OUT/'FINDINGS_ZH.md').write_text(findings(metrics, paired, flows, protocol), encoding='utf-8')
    print(f'Report written to {OUT}; {len(paired)} paired draw contrasts, {len(class_deltas)} class deltas.')


if __name__ == '__main__':
    main()
