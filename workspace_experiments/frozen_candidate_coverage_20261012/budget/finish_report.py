"""Close the completed frozen-budget study without changing predictions."""

import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent


def main():
    for path in (OUT/'trained_predictions').glob('*/*/PREDICTIONS_FROZEN.json'):
        receipt = json.loads(path.read_text(encoding='utf-8'))
        if 'baseline_stage_seconds' not in receipt:
            receipt['baseline_stage_seconds'] = receipt.pop('seconds')
            receipt['seconds'] = None
            receipt['timing_note'] = 'New prediction time was not separately measured; inherited baseline time relabeled.'
            path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = pd.read_csv(OUT/'matched_summary.csv')
    paired = pd.read_csv(OUT/'matched_paired.csv')
    coverage = pd.read_csv(OUT/'coverage_final.csv')
    full = summary[summary.layer.eq('fusion')]
    figures = [('ports14_matched','14 ports (matched)'),('oss2_strong','OSS2'),('f0921_strong','F_0921')]
    colors = dict(initial='#26755c', simple='#ba7334', task='#3173a1', isolated='#a04362')
    fig, axes = plt.subplots(2,3,figsize=(13,7),sharex=True)
    for column, (scope, title) in enumerate(figures):
        rows = full[full.scope.eq(scope)]
        raw = rows[rows.family.eq('raw')].set_index('budget')
        for row, metric in enumerate(('oa','ba')):
            ax = axes[row,column]
            for family, color in colors.items():
                values = rows[rows.family.eq(family)].set_index('budget').sort_index()
                if not values.empty:
                    ax.plot(range(len(values)),values[metric]-raw.loc[values.index,metric],
                            marker='o',label=family,color=color,linewidth=1.6)
            ax.axhline(0,color='#777777',linewidth=.7)
            ax.set_xticks([0,1,2],['16','64','128'])
            ax.grid(axis='y',alpha=.2)
            ax.set_title(title if row==0 else '')
            ax.set_ylabel(metric.upper()+' change (pp)')
            if row==1:
                ax.set_xlabel('Support-label budget')
    axes[0,0].legend(frameon=False,fontsize=9,ncol=2)
    fig.tight_layout()
    fig.savefig(OUT/'budget_effects.png',dpi=170)
    plt.close(fig)
    kept = ['ports23_baselines','ports14_matched','ports_new11','oss2_strong','f0921_strong']
    display = full[full.scope.isin(kept)][['scope','budget','family','tasks','oa','ba','class_brier']]
    increments = paired[paired.layer.eq('fusion') & paired.scope.isin(kept) & paired.reference_family.eq('initial')][
        ['scope','budget','family','delta_oa','delta_ba','class_brier_gain']]
    cost = coverage[['new_affine_fits','new_optimizer_steps','source_prep_seconds','gpu_encode_seconds','fit_seconds']].sum().to_dict()
    completion = dict(status='complete_with_declared_scope', baseline_tasks=25,
        shared_isolated_tasks=15, new_training_tasks=12, budgets=[16,64,128], support_draws=5,
        source_fold_insufficient_tasks=10, oss1_budget_operator_available=False,
        reconstruction='historical_three_port_diagnostic_only', metrics_rows=4230,
        cost=cost, target_based_selection=False, new_backbone_training=False,
        prediction_runtime_measurement='Not separately measured for new trained predictions',
        all_experiment_sessions_exited=True)
    (OUT/'COMPLETE.json').write_text(json.dumps(completion,indent=2),encoding='utf-8')
    body = ['# 冻结预算修正：扩大覆盖结果', '',
        '2026-10-12。预测与评价完成，未改既有候选、训练配方或超参数；无新骨干训练，无查询成绩选checkpoint。', '',
        '## 结论', '',
        '- 原修正器在F_0921的B16有值得保留的正信号：完整OA+0.2179、BA+0.8684个百分点，五组支持BA四正一负，类别宏Brier风险也改善。B64却OA-0.1179、BA-0.0434，因此不是所有预算通用。',
        '- 23港B16的源选简单衰减提高BA约2.0678，但OA降低0.3575；OSS2 B16同一方案OA/BA降低3.2035/1.1886。它是强对照和范围内权衡，不是统一获益方法。',
        '- 同14港B64拆分训练较raw的BA+0.1572，两个训练seed均正；但原修正器已贡献+0.1556，新训练仅另加+0.0016，且该范围类别宏Brier风险变差。不能把继承收益包装成新增训练创新。',
        '- OSS2 B128原修正器OA/BA+0.0756/+0.0563，拆分训练相对原修正器仅另加约0.0001 BA。当前优先保留有作用范围的既有修正，新增共享/拆分训练不升格为主突破。', '',
        '## 完成范围', '',
        'raw/initial/simple为23港+OSS2+F共25任务；shared/isolated为14港+OSS2共15任务。旧三港使用自己的历史权重；新11港和OSS2按相同配方在任务自身坐标与允许源队列内训练，共96个仿射修正器、4320步。重构损失仅保留旧三港历史诊断。', '',
        '另9港与F原三折船数不足每折128支持+32查询，本轮不改变配方；OSS1缺少本路线完整分频基础算子。不是所有历史候选已经全数据验证。coverage_final.csv为逐任务最终清单。', '',
        '## 完整模型表', '', display.to_markdown(index=False,floatfmt='.4f'), '',
        'OA/BA为%，同一B内各臂同支持与公共查询。23港与14港均值不能交叉作增量；两个训练seed先在同任务同draw平均，不扩增任务数。外部主表均用当前较强post-tail先验；旧先验结果单列external_version_summary.csv。', '',
        '## 新训练真正追加多少', '', increments.to_markdown(index=False,floatfmt='.6f'), '',
        '增量为百分点；class_brier_gain正数为风险改善。与raw比较和与initializer比较回答不同问题，两张表都保留。', '',
        '## 范围与成本', '',
        '- 全23港B64原修正器BA仅+0.0013，不等于同14港+0.1556。新增11港B64原修正器+0.0890。Mombasa公共查询仅7行，去除后的13港B64原修正器+0.1676；去除统计只是描述性补充，不取代原等港主结果。',
        '- 源准备183.30秒、GPU冻结编码59.32秒、CPU拟合245.62秒，都是分项累计，不是端到端墙钟或部署耗时。新增查询预测未单独计时；原来继承的seconds已改标baseline_stage_seconds，未编造新计时。',
        '- 五组嵌套支持在B16/64/128使用固定公共查询，不能直接与另一接法路线的原查询绝对分数混算，不能把两路线收益相加。',
        '- 当前是反复开发数据上的范围扩展，不是独立新数据确认；历史行级支持排除不改称严格未见船验证。结果支持局部有效，不要求每港每类全胜，也不自动把退化归为随机。', '',
        '## 验证', '',
        '原三港预算结果按既有支持/查询和预测重放；所有港B64核对历史专家类别，强外部B64基础参照的专家/融合类别均一致。新训练初始真实求解器重放、同seed同预算episode顺序、按折排船先验均已记录；每个评价单元的BA救回减误伤还原配对差值。没有改动旧模型或历史结果。', '',
        '![预算效应](E:/临时会话/frozen_candidate_coverage_20261012/budget/budget_effects.png)', '']
    (OUT/'FINDINGS_ZH.md').write_text('\n'.join(body),encoding='utf-8')
    print(json.dumps(completion,indent=2))


if __name__=='__main__':
    main()
