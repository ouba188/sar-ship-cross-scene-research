"""Bounded root-cause diagnostics and closeout of the frozen adaptive gate."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'

import json
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

from adaptive_fusion_handoff_run import OUT, MODES, tasks, metadata, scores, load_npz
from adaptive_fusion_handoff_policy import THRESHOLDS, probability, select
from compact_fusion_policy import balanced_weights
from strong_single_multidata_explain import truth_for

DATASETS = ('self24ports', 'f0921', 'oss2')
NAMES = {'self24ports': '24 ports', 'f0921': 'F_0921', 'oss2': 'OpenSARShip 2.0'}


def selection_signal(y, base, fusion, p, groups):
    w = balanced_weights(y, groups)
    rescue, harm = (base != y) & (fusion == y), (base == y) & (fusion != y)
    informative = rescue | harm
    result = {}
    if rescue.any() and harm.any():
        result['auc'] = float(roc_auc_score(rescue[informative], p[informative]))
        result['ba_weighted_auc'] = float(roc_auc_score(rescue[informative], p[informative], sample_weight=w[informative]))
    else:
        result.update(auc=np.nan, ba_weighted_auc=np.nan)
    for name, mask in [('rescue', rescue), ('harm', harm)]:
        result[name + '_mass'] = float(w[mask].sum())
        result[name + '_mean_score'] = float(w[mask] @ p[mask] / w[mask].sum()) if mask.any() else np.nan
    total = w[informative].sum()
    result['fusion_win_share'] = float(w[rescue].sum()/total) if total else np.nan
    return result


def diagnose():
    signals, curves, bounds = [], [], []
    for folder in tasks():
        info, suffix = metadata(folder)
        model = json.loads((OUT/'source'/suffix/'MODEL.json').read_text())
        source = load_npz(OUT/'source'/suffix/'source_pairs.npz')
        pb, pe = softmax(source['base'], axis=1), softmax(source['expert'], axis=1)
        base, fusion = pb.argmax(1), ((pb+pe)*.5).argmax(1)
        for split, p in [('source_fit', probability(model, source['features'])),
                         ('source_oof', source['oof_probability'])]:
            signals.append(dict(dataset=info['dataset'], task=info['task'], split=split, draw=-1,
                **selection_signal(source['labels'], base, fusion, p, source['groups'])))
        data = load_npz(OUT/'targets'/suffix/'predictions.npz')
        y, _ = truth_for(info, data)
        for draw, ev in enumerate(data['eval_masks']):
            truth = y[ev]
            b, f = data['predictions'][draw, 0, ev], data['predictions'][draw, 1, ev]
            p = data['gate_probability'][draw, ev]
            signals.append(dict(dataset=info['dataset'], task=info['task'], split='target', draw=draw,
                **selection_signal(truth, b, f, p, np.zeros(len(truth), int))))
            ref = scores(truth, f)
            for t in THRESHOLDS:
                pred, accepted = select(b, f, p, t)
                metric = scores(truth, pred)
                curves.append(dict(dataset=info['dataset'], task=info['task'], draw=draw,
                    threshold=float(t), selected_source_threshold=model['threshold'],
                    **metric, delta_ba=metric['ba']-ref['ba'], delta_oa=metric['oa']-ref['oa'],
                    selection_rate=float(100*accepted.mean())))
    signal = pd.DataFrame(signals)
    signal.to_csv(OUT/'signal_diagnosis.csv', index=False)
    signal.groupby(['dataset', 'task', 'split']).mean(numeric_only=True).groupby(['dataset', 'split']).mean().to_csv(OUT/'signal_summary.csv')
    curve = pd.DataFrame(curves)
    curve.to_csv(OUT/'threshold_diagnostic.csv', index=False)
    for (ds, task), part in curve.groupby(['dataset', 'task']):
        table = part.groupby('threshold')[['delta_ba', 'delta_oa']].mean()
        best = table.delta_ba.idxmax()
        bounds.append(dict(dataset=ds, task=task, oracle_threshold=float(best),
            oracle_delta_ba=float(table.loc[best, 'delta_ba']),
            oracle_delta_oa=float(table.loc[best, 'delta_oa']),
            source_threshold=float(part.selected_source_threshold.iloc[0]),
            diagnostic_only_uses_target_labels=True))
    pd.DataFrame(bounds).to_csv(OUT/'threshold_oracle_bounds.csv', index=False)
    curve.groupby(['dataset', 'threshold'])[['delta_ba', 'delta_oa', 'selection_rate']].mean().to_csv(OUT/'threshold_diagnostic_summary.csv')


def md_table(frame, columns, decimals=4):
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---']*len(columns)) + ' |']
    for row in frame[columns].itertuples(index=False, name=None):
        cells = [f'{x:.{decimals}f}' if isinstance(x, (float, np.floating)) else str(x) for x in row]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def report():
    diagnose()
    summary = pd.read_csv(OUT/'summary.csv')
    diag = pd.read_csv(OUT/'diagnostic_summary.csv')
    signals = pd.read_csv(OUT/'signal_summary.csv')
    source = pd.read_csv(OUT/'source_summary.csv')
    draws = pd.read_csv(OUT/'draw_metrics.csv')
    thresholds = pd.read_csv(OUT/'thresholds.csv')
    oracle = pd.read_csv(OUT/'threshold_oracle_bounds.csv')
    table, flow, variability, causes = [], [], [], []
    for ds in DATASETS:
        s = summary[summary.dataset == ds].set_index('mode')
        d = diag[(diag.dataset == ds) & (diag['mode'] == 'adaptive')].iloc[0]
        table.append(dict(dataset=NAMES[ds], full_oa=s.loc['full_fusion', 'oa'], full_ba=s.loc['full_fusion', 'ba'],
            adaptive_oa=s.loc['adaptive', 'oa'], adaptive_ba=s.loc['adaptive', 'ba'],
            delta_oa=s.loc['adaptive', 'delta_oa'], delta_ba=s.loc['adaptive', 'delta_ba'],
            selection_rate=d.selection_rate))
        flow.append(dict(dataset=NAMES[ds], avoided_harm_ba=d.avoided_harms_ba_pp,
            lost_rescue_ba=d.lost_rescues_ba_pp, net_ba=d.delta_ba,
            acceptance_of_disagreements=d.acceptance_of_disagreements,
            gain_over_random_ba=d.delta_ba_vs_random))
        draw = draws[(draws.dataset == ds) & (draws['mode'] == 'adaptive')]
        variability.append(dict(dataset=NAMES[ds], positive_ba_draws=int((draw.delta_ba > 1e-10).sum()),
            positive_oa_draws=int((draw.delta_oa > 1e-10).sum()),
            delta_ba_std=float(draw.delta_ba.std()), delta_ba_min=float(draw.delta_ba.min()),
            delta_ba_max=float(draw.delta_ba.max())))
        sig = signals[signals.dataset == ds].set_index('split')
        src = source[source.dataset == ds].set_index('mode')
        causes.append(dict(dataset=NAMES[ds], source_fit_auc=sig.loc['source_fit', 'ba_weighted_auc'],
            source_oof_auc=sig.loc['source_oof', 'ba_weighted_auc'],
            target_auc=sig.loc['target', 'ba_weighted_auc'],
            source_nested_delta_ba=src.loc['adaptive_nested', 'ba']-src.loc['full_fusion', 'ba'],
            source_fusion_win_share=sig.loc['source_oof', 'fusion_win_share'],
            target_fusion_win_share=sig.loc['target', 'fusion_win_share'],
            target_oracle_threshold_gain=oracle[oracle.dataset == ds].oracle_delta_ba.mean()))
    table, flow, variability, causes = map(pd.DataFrame, (table, flow, variability, causes))
    table.to_csv(OUT/'COMPARISON_TABLE.csv', index=False)
    causes.to_csv(OUT/'ROOT_CAUSE_TABLE.csv', index=False)
    classes = pd.read_csv(OUT/'class_metrics.csv')
    classes = classes[classes['mode'] == 'adaptive'].groupby(['dataset', 'task', 'class_id']).mean(numeric_only=True).groupby(['dataset', 'class_id']).mean()
    classes.to_csv(OUT/'class_delta_summary.csv')
    text = '''# 自适应预算：基础模型与全量融合之间选择

## 结论

本轮25任务、125组支持抽样已完成。按源港选出的自适应阈值，没有超过最新匹配版本的全量等权融合：三个集合平均BA均下降，24港OA微升，两个外部数据集OA下降。F_0921部分抽样仍有正收益，不能据此否定所有自适应方案；但本版不能替代当前全量融合主方案。

这里的主对照是最新 `backbone_ablation_20261010/dual_refit_late`，不是此前外部数据集分数较低的旧融合版本。所有基础/融合逐样本预测及OA/BA精确复现。

## 实测结果

数值为百分数，差值为百分点。24港按23个有效留港任务等权，随后汇总五组支持抽样；外部数据集同五组64标签。selection_rate是全部查询中实际选择融合且与基础预测不同的比例。

'''
    text += md_table(table, list(table.columns)) + '\n\n'
    text += md_table(variability, list(variability.columns))
    text += '''

五组支持抽样不是五个神经训练种子，数据相互重叠；表中的标准差不是独立确认置信区间。

## 为什么没超过全量融合

精确核算式为：相对全量融合的增益 = 避免的融合伤害 - 丢失的融合救回。下表用每类等权的BA质量计算，不拿原始样本数冒充BA贡献。

'''
    text += md_table(flow, list(flow.columns))
    text += '''

本版并非完全随机：在相同分歧集合、相同接受比例下，比均匀随机选择的解析期望更好。但“比随机更会选”并不等于“比全部融合更准确”。三个集合均出现丢失救回的BA贡献大于避免伤害的贡献。OSS2尤其明显：门控只保留约35%的分歧样本进行融合，拒绝掉的正确融合太多。

随机对照为精确期望，不是另一个可部署的确定性模型。oracle_base_or_fusion_ba是用查询真值决定每条样本选谁的上限，不是实验方法性能。

## 排序与阈值诊断

AUC仅在基础/融合恰好一方正确的样本上计算；下表AUC保留源港/类别均衡权重。它衡量排序，不保证分数已校准，也不保证阈值可迁移。

'''
    text += md_table(causes, list(causes.columns))
    text += '''

- source_fit_auc与source_oof_auc区分拟合能力和源港留出排序能力。源港的嵌套评估同时留出了模型拟合与阈值选择，24港/F本身没有稳定正收益，不能把全部失效归咎于跨数据集变化。
- fusion_win_share表示在恰好一方正确时，融合获胜的类别均衡比例。源到目标比例变化可作为阈值不匹配的证据，但不能单独认定是唯一根因。
- 固定0.5阈值作为预先规定的诊断，也未在三个集合的平均BA上超过全量融合。不能把问题简单归结为当前源域阈值这一处设置。
- target_oracle_threshold_gain是读取目标标签后，在同一组21个阈值中、每个任务汇总五次抽样取最优的诊断上限，**不可用于部署、不替换主方案**。它用于区分“同一排序仅需改阈值”与“现有排序也不够”的可能性，仍会高估可实现收益。
- 当前证据解释的是功能层面的救回/伤害取舍、有限的排序能力与阈值适配，不是物理散射根因或因果发现。

## 模块与训练流程

1. 保持两路特征、各自64标签适配和等权softmax融合不变。对每个源港留出案例同时得到基础预测与融合预测。
2. 八个输入是基础/融合/专家的分数间隔、归一化熵，以及基础与专家对两个候选类别的偏好差。没有增加目标标签或新图像网络。
3. 用融合救回作为正事件、融合带错作为负事件。两者都错或都对不构成有方向的收益标签。先按完整源港/类别分布加权，再筛事件，避免筛选改变BA目标。
4. 用带固定L2正则的线性sigmoid学习偏好。源港三折OOF挑阈值，嵌套三折单独评估；最终源港全量拟合后冻结。
5. 目标上只有基础/融合预测不同且分数超过阈值时才选择融合。其余保留基础预测，实际交接比例自然产生。两种预测一致时无需改变输出。

全量融合参与所有查询的计算；selection_rate不是专家特征提取比例，也不是计算节省。本轮没有宣称减少端到端推理时间。

## 实现核对与边界

- 修复了测试阶段发现的事件筛选后标签未同步索引错误；11项相关测试通过后才开始正式实验。
- 外部源港专家头在与最新目标部署一致的joint-tail坐标拟合，留出该源港；23港复用逐项兼容的历史源案例。基础、支持集和评估掩码没有改变。
- 25任务的基础/最新融合预测逐项精确复现。所有目标预测先保存，再读目标查询标签生成指标；选择器未使用查询标签拟合或选阈值。
- `cases.csv`每个任务取原始顺序下最多两个避免伤害、丢失救回、保留救回、保留伤害案例，保留每项门控logit贡献，可重建分数；不是只展示成功案例。
- 源表征、几何仍共享历史源数据，嵌套留出仅评估新增选择器。外部数据历史研发暴露与F原SSL语料暴露边界不变。
- 保留全部负结果，不按数据集或支持抽样挑有利版本组成主方案，也不继续针对测试成绩搜索参数。

## 建议

当前主方案继续采用全量融合。本轮自适应交接作为已完成的负结果与机制边界保留：存在有限的选择信号，但没有转化为超过全量融合的净收益。若继续研发，先补足用于判断“融合是否有害”的样本级证据，再考虑门控，而不是继续寻找有利的预算比例。

## 文件

`PROTOCOL.json`、`COMPLETE.json`、`summary.csv`、`draw_metrics.csv`、`diagnostics.csv`、`source_nested.csv`、`signal_diagnosis.csv`、`threshold_diagnostic.csv`、`threshold_oracle_bounds.csv`、`class_metrics.csv`、`cases.csv`，以及各任务源模型和目标预测。
'''
    (OUT/'FINDINGS_ZH.md').write_text(text, encoding='utf-8')
    plot(table, flow)
    print(causes.to_string(index=False))
    print('Report:', OUT/'FINDINGS_ZH.md')


def plot(table, flow):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 180})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
    x = np.arange(len(table))
    axes[0].bar(x-.18, table.delta_oa, .36, color='#2878a0', label='OA')
    axes[0].bar(x+.18, table.delta_ba, .36, color='#ac425a', label='BA')
    axes[0].set_title('Adaptive selection minus full fusion')
    axes[0].set_ylabel('Change (percentage points)')
    axes[0].legend(frameon=False)
    axes[1].bar(x-.22, flow.avoided_harm_ba, .22, color='#27806c', label='Avoided harm')
    axes[1].bar(x, -flow.lost_rescue_ba, .22, color='#ac425a', label='Lost rescue')
    axes[1].bar(x+.22, flow.net_ba, .22, color='#555555', label='Net BA')
    axes[1].set_title('Exact BA gain accounting')
    axes[1].set_ylabel('BA contribution (percentage points)')
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.set_xticks(x, table.dataset)
        ax.axhline(0, color='#444444', lw=.8)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.savefig(OUT/'comparison.png')
    fig.savefig(OUT/'comparison.pdf')
    plt.close(fig)


if __name__ == '__main__':
    report()
