"""Matched factorial results, including missing-class recall and score changes."""

import json
import pandas as pd
from channel_operator_report import table
from target_loss_repair import OLD, OUT


def run():
    done = json.loads((OUT/'COMPLETE.json').read_text())
    rows = pd.read_csv(OUT/'results.csv')
    for ref, suffix in [('head_only', 'head'), ('ordinary_adapter', 'old')]:
        baseline = rows[rows.arm.eq(ref)][['port', 'draw', 'OA', 'BA']].rename(
            columns={m: m+'_'+suffix for m in ('OA', 'BA')})
        rows = rows.merge(baseline, on=['port', 'draw'])
        for m in ('OA', 'BA'):
            rows['delta_'+m+'_vs_'+suffix] = rows[m]-rows[m+'_'+suffix]
    cols = ['OA', 'BA', 'delta_OA_vs_head', 'delta_BA_vs_head', 'delta_OA_vs_old', 'delta_BA_vs_old']
    means = rows.groupby('arm', sort=False)[cols].mean().reset_index()
    ports = rows.groupby(['port', 'arm'], sort=False)[cols].mean().reset_index()
    rows.to_csv(OUT/'paired_results.csv', index=False)
    means.to_csv(OUT/'summary.csv', index=False)
    ports.to_csv(OUT/'port_summary.csv', index=False)
    class_rows = pd.read_csv(OUT/'by_class.csv')
    old_class = pd.read_csv(OLD/'label_coverage_class_detail.csv')
    old_class = old_class[old_class.arm.eq('head_only')].rename(columns={'coverage_group': 'coverage'})
    coverage = pd.concat([class_rows, old_class], ignore_index=True).groupby(
        ['port', 'draw', 'arm', 'coverage']).agg(recall=('recall', 'mean'), classes=('class_id', 'size')).reset_index()
    coverage.groupby(['arm', 'coverage']).agg(recall=('recall', 'mean'), task_groups=('draw', 'size')).reset_index().to_csv(
        OUT/'coverage_summary.csv', index=False)
    coverage_summary = pd.read_csv(OUT/'coverage_summary.csv')
    absent = class_rows[class_rows.coverage.eq('absent')].groupby(['port', 'draw', 'arm']).agg(
        recall=('recall', 'mean'), mean_true_prior=('mean_true_prior', 'mean')).reset_index()
    absent.to_csv(OUT/'absent_class_detail.csv', index=False)
    designs = pd.read_csv(OLD/'support_design.csv')
    wide = rows.pivot(index=['port', 'draw'], columns='arm', values='BA').reset_index()
    wide = wide.merge(designs[['port', 'draw', 'support_classes']], on=['port', 'draw'])
    wide['mask_without_source'] = wide.mask_only-wide.ordinary_adapter
    wide['mask_with_source'] = wide.mask_and_source-wide.source_supervision
    wide['source_without_mask'] = wide.source_supervision-wide.ordinary_adapter
    wide['source_with_mask'] = wide.mask_and_source-wide.mask_only
    wide['interaction'] = wide.mask_and_source-wide.mask_only-wide.source_supervision+wide.ordinary_adapter
    wide['support_coverage'] = wide.support_classes.map(lambda n: 'all8' if n == 8 else 'missing')
    effects = ['mask_without_source', 'mask_with_source', 'source_without_mask', 'source_with_mask', 'interaction']
    wide.to_csv(OUT/'factorial_by_task.csv', index=False)
    effect_mean = wide.groupby('support_coverage')[effects].mean().reset_index()
    effect_mean.to_csv(OUT/'factorial_summary.csv', index=False)
    stability = rows[rows.arm.isin(['mask_only', 'source_supervision', 'mask_and_source'])].groupby('arm').agg(
        positive_vs_old=('delta_BA_vs_old', lambda x: int((x > 1e-8).sum())),
        equal_vs_old=('delta_BA_vs_old', lambda x: int((abs(x) <= 1e-8).sum())),
        negative_vs_old=('delta_BA_vs_old', lambda x: int((x < -1e-8).sum()))).reset_index()
    stability.to_csv(OUT/'stability.csv', index=False)
    name = dict(ordinary_adapter='原内部适配', mask_only='仅屏蔽缺类直接损失',
                source_supervision='仅增加源监督', mask_and_source='两项一起', head_only='仅适配分类头', source_only='仅源模型')
    lines = ['# 缺类训练路径纠错：固定预算四格对照', '',
             '## 结论先行', '',
             '修到一个实际训练问题，但尚未解决全部跨港退化。联合修正相对原普通内部适配：总体OA提高0.024点、BA提高0.422点；相对只适配分类头：OA提高1.472点、BA提高0.195点。三港平均的小正信号，不是各港均优。', '',
             '支持缺类组召回由原内部适配21.723%恢复至30.558%，仍低于只适配头33.503%。单独屏蔽恢复至28.549%，单独源监督为23.701%，显示缺类直接损失路径确实影响误伤，而非仅有代码层面的推测。', '',
             '联合修正相对原适配，桑托斯BA提高1.502点、洛杉矶提高0.478点、巴生港下降0.713点；相对只适配头仍分别为-1.202、-0.456、+2.243点。不能按目标港事后挑选版本组成虚假的统一模型。', '',
             '## 总体结果', '',
             table(means.assign(arm=means.arm.map(name)), ['arm']+cols), '',
             'OA、BA为百分数，差值为百分点。对三个港、每港三个固定支持抽样等权平均；支持抽样不是独立查询人群或神经预训练种子。', '',
             '## 实际改了什么', '',
             '- 使用上一轮完全相同的939个源标签、每任务64个目标标签和查询划分；无新增标签。',
             '- 所有四格均保持1024个通道仿射参数、冻结R18和源分类头、24步Adam、学习率0.01、岭系数8、原0.25源分数保持项及0.1初始状态约束。',
             '- 屏蔽开关：只删去目标训练损失中支持未出现类的误差项；仍除以8，已出现类的直接梯度不放大。缺类的残差头仍为零，与上一轮一致。',
             '- 源监督开关：在同一批每步32个、八类各4个源回放样本上，额外加入0.25权重的真实标签均方误差；不替换原有分数保持项。正负类编码与原残差头一致，为+1/-1。',
             '- 四格及系数预先固定，没有查询集选强度、按港选最优或特定船型规则。全部使用普通适配，不混入上一轮收益极小的场景排序引导。',
             '- 对全部八类均有支持的任务，屏蔽与不屏蔽的目标完全相同，复用对应模型；这不是额外独立训练证据。', '',
             '## 分港结果', '',
             table(ports.assign(arm=ports.arm.map(name)), ['port', 'arm']+cols), '',
             '## 两项改动分别起什么作用', '',
             table(effect_mean, list(effect_mean.columns)), '',
             'mask_without_source=仅屏蔽减原适配；mask_with_source=联合减仅源监督；source_without_mask=仅源监督减原适配；source_with_mask=联合减仅屏蔽。interaction是两项作用的非加性差值。', '',
             table(wide, ['port', 'draw', 'support_classes']+effects), '',
             table(stability, list(stability.columns)), '',
             '## 支持标签覆盖与误伤', '',
             table(coverage_summary.assign(arm=coverage_summary.arm.map(name)), list(coverage_summary.columns)), '',
             '先在任务内对相应覆盖组的类别召回等权平均，再平均可用任务组。缺类仅4个任务组，不能当作整个测试集BA或总体因果效应。', '',
             table(absent.assign(arm=absent.arm.map(name)), list(absent.columns)), '',
             'mean_true_prior是缺类真实样本自身类别的源头分数均值，不是概率；保住它也不保证召回，因为竞争类别分数仍可能改变。', '',
             '## 实现核对与边界', '',
             '- 8项测试通过，包括缺类直接梯度移除、其余类别梯度不缩放、满类等价、源回放八类正监督和原残差头留一公式。',
             '- 首任务关闭两个新开关后重新训练，与旧版参数核对；全部九任务的旧版查询预测均回放核对。具体误差见REPLAYS.json。',
             '- 屏蔽直接损失不保证缺类分数不变：其他类别更新仍经共享特征影响缺类；本轮就是检验这条直接路径是否为主要可修正来源。',
             '- 源监督可能改善或损伤目标表现；仅凭源训练损失下降不能认定跨港机制有效。',
             '- 这是固定R18小规模开发实验，非完整双路模型；支持/查询可共享大景，历史预训练曝光未撤销，未识别物理噪声。', '',
             f'完成9个任务，新适配训练{done["new_adapter_fits"]}次，另有1次旧训练回放；CPU单线程耗时{done["seconds"]:.1f}秒。未使用主线程GPU或修改其文件。', '']
    score_path = OUT/'missing_class_score_summary.csv'
    if score_path.exists():
        scores = pd.read_csv(score_path)
        receipt = json.loads((OUT/'MISSING_SCORE_REPLAY.json').read_text())
        lines += ['## 缺类分数路径的回放诊断', '',
                  table(scores.assign(arm=scores.arm.map(name)), list(scores.columns)), '',
                  '本表recall是0到1比例，不是百分数；分数不是概率。先对任务内各缺类平均，再对4个任务等权，与覆盖组召回口径一致。delta均相对只适配分类头。', '',
                  '原内部适配：缺类自身分数平均下降0.155815，竞争类最高分平均增加0.020404，间隔恶化0.176220。联合修正：自身分数仅下降0.001489，竞争类最高分仍增加0.015653，间隔仍恶化0.017142。', '',
                  '因此，缺类原有分数被整体压低是一个可干预的实际误伤来源；去除对应训练项并加入源监督后，该平均效应明显减弱。剩余误伤不能都归为自身分数下移，还包括竞争分数变化和样本间不均衡。均值恢复不保证每个样本恢复。', '',
                  f'回放涉及{len(receipt["replays"])}个任务、{receipt["sample_exposures"]}次样本出现、{receipt["unique_observations"]}张唯一裁片、{receipt["unique_vessels"]}个唯一MMSI，旧头预测均一致。查询标签只用于事后诊断，不用于拟合或选参数。', '',
                  '这支持训练目标层面的失效解释，不证明已分离港口噪声或物理散射因素；也不说明未覆盖类别的所有负监督都应在任意任务中删除。巴生港的整体损失是必须同时保留的反证。', '',
                  '## 下一步范围', '',
                  '保留本轮修正与原版全部结果，暂不把小试直接并入完整模型。下一轮优先用源港模拟64标签支持/独立查询任务，学习既适配新港又保留稀缺类别判断的更新约束；目标查询不参与配置选择。不增加船型特定规则，不进入物理去噪宣称，也不扩展无关数据集。', '']
    (OUT/'FINDINGS_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    print(means.to_string(index=False))
    print(ports.to_string(index=False))
    print(coverage_summary.to_string(index=False))


if __name__ == '__main__':
    run()
