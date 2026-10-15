"""Plain-language interpretation of the matched unified-head experiment."""

import json
import numpy as np
import pandas as pd
from unified_head_run import OUT

DATASETS = dict(self24ports='自建24港（23个任务）', f0921='F_0921', oss2='OpenSARShip 2.0')
LABELS = dict(base='仅基础头', expert='仅专家头', late_softmax_fixed='原双头等权概率融合',
    late_logits_fixed='双头等权分数融合', unified_fixed='统一头：等权、正则8',
    unified_selected='统一头：源域选参', late_softmax_selected='双头概率融合：同等源域选参',
    late_logits_selected='双头分数融合：同等源域选参', scratch_selected='统一头无源域先验（诊断）')


def table(frame):
    lines = ['| '+' | '.join(map(str, frame.columns))+' |',
             '| '+' | '.join(['---']*len(frame.columns))+' |']
    lines += ['| '+' | '.join(map(str, row))+' |' for row in frame.itertuples(index=False, name=None)]
    return '\n'.join(lines)


def main():
    complete = json.loads((OUT/'COMPLETE.json').read_text())
    summary = pd.read_csv(OUT/'summary.csv').set_index(['dataset', 'mode'])
    draws = pd.read_csv(OUT/'draw_metrics.csv').set_index(['dataset', 'draw', 'mode'])
    fits = pd.read_csv(OUT/'fit_diagnostics.csv')
    choices = pd.read_csv(OUT/'source_choices.csv')
    comparison = []
    for mode, label in LABELS.items():
        row = {'方法': label}
        for dataset, name in DATASETS.items():
            v = summary.loc[dataset, mode]
            row[name+' OA / BA'] = f'{v.oa:.4f} / {v.ba:.4f}'
        comparison.append(row)
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(OUT/'COMPARISON_TABLE.csv', index=False, encoding='utf-8-sig')
    text = ['# 相同两路信息，一个统一分类头还是两个独立分类头', '',
        '## 结论先行', '',
        '在预先固定的等权、正则8设置下，统一适配头相对原双头概率融合：OA分别+0.0927、+0.1834、+0.3043个百分点，BA分别-0.2747、-0.3968、-0.2470个百分点（24港、F_0921、OSS2）。因此统一头可以使用这两路信息，但本轮没有得到OA与BA同时更好的替代方案。',
        '严格改成相同分数读出的比较，统一头BA仍分别下降0.2744、0.4114、0.5996个百分点；不是完全由softmax读出造成。若当前仍以BA为主要指标，暂不替换双头；不能据此声称所有单模型都不如双模型。',
        '源域选参版本的负结果完整保留。OSS2源域选择rho=0.25、正则64后BA为27.9566；后续仅作诊断的两角点对照表明，固定rho=0.25、正则改为8时BA为34.1478。强正则压弱目标适配是该大幅退化的直接控制变量之一，不应全部归因于统一头结构。诊断没有替换正式源域选参结果。', '',
        '## 核心结果', '',
        f'已完成{complete["tasks"]}个任务、{complete["draws"]}次支持集抽样、{complete["rows"]}条评估记录。所有选择仅使用源域，目标预测全部冻结后才读取目标评估标签。', '',
        '下表为OA / BA（%）；24港采用23个有效任务等权汇总，不用样本量大的港口压过小港口。', '',
        table(comparison), '', '## 直接回答研究问题', '']
    contrasts = []
    for dataset, name in DATASETS.items():
        text += [f'### {name}', '']
        for candidate, reference, explanation in [
            ('unified_fixed', 'late_logits_fixed', '固定相同核权重、相同正则、相同源域先验，比较联合拟合与独立拟合'),
            ('unified_fixed', 'late_softmax_fixed', '不新增选参的统一头与原概率融合'),
            ('unified_selected', 'late_softmax_fixed', '源域选参后的统一头与原概率融合'),
            ('unified_selected', 'late_softmax_selected', '同样9组源域参数机会下，统一头与双头概率融合')]:
            a, b = summary.loc[dataset, candidate], summary.loc[dataset, reference]
            delta = [draws.loc[(dataset, d, candidate), 'ba']-draws.loc[(dataset, d, reference), 'ba'] for d in range(5)]
            positive = sum(x > 1e-10 for x in delta)
            contrasts.append(dict(dataset=dataset, candidate=candidate, reference=reference,
                delta_oa=a.oa-b.oa, delta_ba=a.ba-b.ba, positive_ba_draws=positive, draws=5))
            text += [f'- {explanation}：OA {a.oa-b.oa:+.4f}、BA {a.ba-b.ba:+.4f}个百分点，BA为正的抽样{positive}/5。']
        f = fits[fits.dataset == dataset]
        text += [f'- 同设定统一头的支持集正则化训练目标均值 {f.unified_objective.mean():.4f}，独立拟合后合成的目标均值 {f.late_objective.mean():.4f}。',
            '  该数值只检查联合求解是否符合设计，不用训练误差替代目标测试效果。', '']
    pd.DataFrame(contrasts).to_csv(OUT/'paired_comparisons.csv', index=False)
    text += ['## 统一分类头究竟做了什么', '',
        '原来：基础特征拟合一套系数，专家特征拟合另一套系数，最后融合两路输出。',
        '现在：两路完整特征关系一起进入一个分类器，用同一批64个目标标签、一次联合求解得到同一套64×8对偶系数。此处的64是支持样本数，不是特征维数。', '',
        '```text',
        '冻结基础表征 -> 基础核 Kb --+',
        '                           +-> 联合核 (1-rho)Kb + rho Ke -> 一次联合拟合 -> 8类分数',
        '冻结专家表征 -> 专家核 Ke --+',
        '               既有源域先验作为初始化，64个目标支持标签作联合校正',
        '```', '',
        '核的加权和等价于拼接 sqrt(1-rho) 倍的基础特征映射与 sqrt(rho) 倍的专家特征映射后，训练一个分类头。条件RBF与分频表示仍完整保留，不先丢掉它们再比较。',
        '公式：K=(1-rho)Kb+rho Ke；f0=(1-rho)f0b+rho f0e；f(x)=f0(x)+K(x,S) alpha。alpha只联合求解一次。',
        '主对照保留已有两路源域先验，合为固定预测起点，再对两路特征的联合核训练一个校正头；没有把统一头降格成只用64标签从零学习。另列无先验版本，仅用于观察源域知识的作用。这里没有重新联合训练源域先验或骨干网络。', '',
        '## 公平性与验证', '',
        '- 三个数据集、相同64目标支持标签、相同五次支持集抽样，特征提取网络保持冻结。',
        '- 固定对照为rho=0.5、正则8；源域选择只搜索rho={0.5,0.25,0.75}与正则={8,1,64}的9组组合。统一头和双头融合获得相同选择机会。',
        '- 源域分类头按原有留港协议生成；外部专家用原始pre-tail特征拟合留港线性头，在冻结joint-tail特征上预测。表征仍共享源域，不声称独立表征验证。',
        '- 原基础头、专家头和等权概率融合的目标预测、OA/BA必须精确复现，否则实验停止。',
        '- 新统一核解与独立实现的显式拼接特征原始空间解一致；同时检查端点回退、同核等价性、联合优化目标和分数变化的精确分解。', '',
        '## 不能过度推论', '',
        '1. 本轮比较的是一个还是两个目标适配分类头，不是重新端到端训练一个单骨干网络。两个特征提取器以及已有源域预训练知识仍然保留。',
        '2. 若统一头不差，说明这些信息不必依赖两个独立目标分类头才能有效；不等于某一路特征本身可以删除。',
        '3. 若统一头更差，先看固定对照和训练目标验证：优化解正确但泛化下降，不足以直接断言某类物理证据被压制，也不证明所有单模型方案无效。',
        '4. 固定分数融合与固定概率融合的差异属于读出方式；不能全部记到“两个独立模型”的贡献上。',
        '5. 数据均为历史研发数据，五次支持抽样不是五次网络种子；不按目标表现挑选各数据集的最优版本拼成主模型。',
        '6. 一次目标适配求解替代两次求解不等于端到端算力减半，本轮没有删掉特征提取网络。', '',
        '## 结果文件', '',
        '- `metrics.csv` / `summary.csv`：各抽样和汇总OA、BA、救回与误改。',
        '- `class_metrics.csv`：逐港逐类召回与对平均BA的贡献。',
        '- `source_choices.csv`：各任务仅由源域确定的权重与正则。',
        '- `fit_diagnostics.csv`：训练目标、专家重建误差和联合/独立输出变化。',
        '- `paired_comparisons.csv`：匹配比较与五次抽样的一致性。', '']
    # Contributions keep missing classes as zero contributions, not absent tasks in the denominator.
    cls = pd.read_csv(OUT/'class_metrics.csv')
    denom = pd.read_csv(OUT/'metrics.csv').groupby('dataset').task.nunique()*5
    contributions = cls.groupby(['dataset', 'mode', 'class_id']).ba_contribution.sum().reset_index()
    contributions['ba_contribution'] /= contributions.dataset.map(denom)
    contributions.to_csv(OUT/'class_ba_contributions.csv', index=False)
    for (ds, mode), total in contributions.groupby(['dataset', 'mode']).ba_contribution.sum().items():
        np.testing.assert_allclose(total, summary.loc[ds, mode].delta_ba, atol=1e-10)
    choices.groupby(['dataset', 'family', 'ratio', 'strength']).size().to_csv(OUT/'selection_counts.csv')
    (OUT/'FINDINGS_ZH.md').write_text('\n'.join(text), encoding='utf-8')
    diagnostic = OUT/'oss2_diagnosis/summary.csv'
    if diagnostic.exists():
        d = pd.read_csv(diagnostic)
        strength = pd.read_csv(OUT/'oss2_diagnosis/correction_strength.csv').groupby(['ratio', 'strength']).mean(numeric_only=True)
        s = pd.read_csv(OUT/'source_validation.csv')
        s = s[(s.dataset == 'oss2') & (s.family == 'unified')].groupby(['ratio', 'strength'])[['oa', 'ba']].mean()
        report = ['# 统一分类头：失效定位', '',
            '## 已经排除的实现问题', '',
            '5项针对性测试通过，包括显式拼接原始空间解与联合核对偶解的一致性。25任务的原基础、专家、融合预测及OA/BA精确复现；源域专家与上一轮缓存逐样本匹配。统一头的正则化支持集目标在全部目标抽样中不高于独立拟合后的合成解。',
            '这些检查支持“本次联合求解符合设计、不是没有优化起来”，不等于证明任何实现问题都不可能存在。', '',
            '## 固定设置：拟合更好不等于BA更好', '',
            '三组的联合支持集拟合损失与正则化目标均下降，OA略升，BA略降。严格对比分数融合后，BA下降仍在，说明它不只是softmax与原始分数的读出差异。',
            '24港固定设置的源域BA：联合58.4756、双头分数58.8354；目标BA：联合55.7126、双头分数55.9870。F_0921在源域联合更好、目标反而更差。因此不能把三个集合的退化都简单归为同一个原因。',
            '当前证据说明：单一联合拟合的目标与最终类别均衡识别之间存在差异，且部分源域收益未迁移。没有证据把它解释成某类具体物理线索被抹掉。', '',
            '## OSS2选参失效：权重与正则拆分', '',
            '以下2×2表中，(.5,8)和(.25,64)为正式实验已存在的两个点；其余两个为结果之后追加的定位对照，不参与主模型选择。', '',
            table(d.round(6)), '',
            '固定rho=0.25，源域BA偏好正则64：',
            f'- 正则8：源域BA {s.loc[(.25, 8.), "ba"]:.4f}；目标BA 34.1478。',
            f'- 正则64：源域BA {s.loc[(.25, 64.), "ba"]:.4f}；目标BA 27.9566。',
            f'- 校正分数RMS：{strength.loc[(.25, 8.), "correction_rms"]:.4f}降至{strength.loc[(.25, 64.), "correction_rms"]:.4f}。',
            f'- 预测不同于源域先验的比例：{strength.loc[(.25, 8.), "departure_from_prior"]:.2f}%降至{strength.loc[(.25, 64.), "departure_from_prior"]:.2f}%。',
            '- 同权重下，源域先验本身目标BA只有22.1817。强正则让有限标签的校正减弱，更多保留了这个在目标上较弱的起点。',
            '- 在rho=0.5下，正则8到64也使目标BA从33.6677降至28.6177；不是只在一个权重点偶然出现。', '',
            '所以该大幅退化的可定位机制是：源域选出的强正则在目标上压弱了有效适配。为什么源域BA偏好与目标偏好不一致，仍涉及域差异与支持集组成；本轮没有进一步把它定位为某种物理成像因素。', '',
            '## 决策边界', '',
            '本轮不追加参数网格，不用诊断结果偷偷替换正式源域选择。保留统一头的OA正信号，但若BA仍是主指标，目前双头融合更适合保留。这个结论仅限现有表征和有限标签适配头，不外推为单骨干端到端模型必然更差。', '']
        (OUT/'ROOT_CAUSE_ZH.md').write_text('\n'.join(report), encoding='utf-8')
    print(comparison.to_string(index=False))


if __name__ == '__main__':
    main()
