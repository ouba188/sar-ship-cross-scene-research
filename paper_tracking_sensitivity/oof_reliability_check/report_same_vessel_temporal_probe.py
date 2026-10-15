"""Report repeat-vessel directions and their actual frozen-model utility."""

import pickle

import numpy as np
import pandas as pd

from run_same_vessel_temporal_probe import OUT, PARENT, SOURCE, latent, energy_fraction


def target_class_contrasts():
    frame = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id','port'])
    y = frame.class_id.to_numpy()
    rows = []
    for folder in sorted((OUT/'ports').iterdir()):
        with np.load(folder/'directions.npz') as f:
            frozen = {k:f[k] for k in ('class_centers','temporal','class_protected')}
        with (PARENT/'ports'/folder.name/'geometry.pkl').open('rb') as f:
            g = pickle.load(f)
        with np.load(folder/'policies.npz') as p:
            ids, masks = p['query_rows'], p['eval_mask']
        bank = np.load(PARENT/'ports'/folder.name/'target_bank.npy', mmap_mode='r')
        z = latent(bank, g)
        for draw, ev in enumerate(masks):
            classes = [c for c in range(8) if np.sum(ev & (y[ids]==c)) >= 5]
            if len(classes) < 2:
                continue
            target = np.stack([z[ev & (y[ids]==c)].mean(0) for c in classes])
            source = frozen['class_centers'][classes]
            a, b = source-source.mean(0), target-target.mean(0)
            sc, tc = a.T @ a / len(classes), b.T @ b / len(classes)
            for mode in ('temporal','class_protected'):
                d = frozen[mode]
                rows.append(dict(port=folder.name, draw=draw, mode=mode, eligible_classes=len(classes),
                    source_matched_class_fraction=energy_fraction(sc,d),
                    target_class_fraction=energy_fraction(tc,d)))
    result = pd.DataFrame(rows)
    result.to_csv(OUT/'target_class_contrasts.csv', index=False)
    pm = result.groupby(['port','mode']).mean(numeric_only=True).drop(columns='draw').reset_index()
    pm.to_csv(OUT/'target_class_contrast_port_means.csv',index=False)
    return pm.groupby('mode').mean(numeric_only=True)


def make_plot(geometry, scores):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    modes = ['temporal','class_protected']
    names = ['Largest repeated-vessel changes', 'Class-center protected changes']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    x = np.arange(2)
    for offset, field, label, color in [(-.2,'held_temporal_fraction','Held-vessel change captured','#197d80'),
                                       (.2,'class_center_contrast_fraction','Source class contrast overlap','#a63b63')]:
        axes[0].bar(x+offset, [100*geometry.loc[m,field] for m in modes], .38, label=label, color=color)
    axes[0].set_xticks(x, ['Temporal', 'Protected'])
    axes[0].set_ylabel('Energy fraction (%)')
    axes[0].set_title('Different vessels from direction fitting')
    axes[0].legend(fontsize=8)
    view = scores[(scores['group']=='all8_target12') & (scores.arm=='joint')].set_index('mode')
    axes[1].bar(x, [view.loc[m,'full_ba_delta_pp'] for m in modes], color=['#197d80','#a63b63'])
    axes[1].set_xticks(x, ['Temporal', 'Protected'])
    axes[1].axhline(0, color='#777777', linewidth=.8)
    axes[1].set_ylabel('Full-model BA change (pp)')
    axes[1].set_title('12 targets containing all eight classes')
    fig.savefig(OUT/'temporal_directions_and_utility.png', dpi=170)
    plt.close(fig)


def main():
    contrasts = target_class_contrasts()
    geometry = pd.read_csv(OUT/'geometry_means.csv').set_index('mode')
    scores = pd.read_csv(OUT/'coverage_group_means.csv')
    checks = pd.read_csv(OUT/'verification.csv')
    source = pd.read_csv(OUT/'held_source_readout_means.csv')
    names = {'reference':'不干预','temporal':'同船变化最大8方向','class_protected':'类别中心保护的8方向'}
    arms = {'matched_baseline':'同流程基线','stable_only':'显式稳定读出','joint':'联合版本'}
    lines = ['# 同船多时相：哪些变化值得抑制', '', '日期：2026-10-09。原23港、五组64标签；本轮识别方向并诊断，不重训或替换主模型。', '',
        '## 方法与真实边界', '',
        '先在每个外层源划分中，保留有效9位MMSI、同一身份源标签一致、可解析采集时间的记录。同港、同身份、同一采集时刻的重复记录先取特征均值，再将同港同船相邻时相配对。不同港口之间不直接配对。',
        '源样本沿用原目标身份/产品排除；实际断言确保目标MMSI未进入源观测库。MMSI仍是身份代理，未逐一确认物理身份。一次同船差分也可能包含姿态、状态及成像变化，不自动等于纯噪声。',
        '按船身份随机固定划分约2/3拟合、1/3留出，同一MMSI在不同港的观测也只能落入一侧。不是按影像随机拆分。',
        f'各外层源划分有{int(checks.train_pairs.min())}至{int(checks.train_pairs.max())}对拟合观测、{int(checks.held_pairs.min())}至{int(checks.held_pairs.max())}对留出观测。各划分重叠，不能累加为独立样本量。',
        '方向搜索只在已有源标准化R50的[Qs,Qd]联合正交空间中进行，维数为39至43；不是在原始影像或全部2048维中挖掘。它可以发现原候选稳定空间内部仍存在的重复观测变化。',
        '差分二阶矩为0.5*E[delta delta^T]，不按天数除法、不减去平均差分。类别等权、类内港等权、港内船等权、船内相邻对等权，避免拍摄次数最多的船主导。',
        '两种固定候选：直接取二阶矩最大的8个正特征方向；或者先投影掉拟合船8类中心差异张成的7维空间，再取8个方向。类别中心按每类港口等权，港内聚合观测等权。后一种只保护源类中心差异，不保证保护全部判别信息，也不保证保护目标类中心。',
        'Qs/Qd和既有源分类头仍来自全部外层源域，包含本轮留出船。留出船结果是给定现有表示的方向重现诊断，不是几何和分类器完全独立的内层OOF验证。', '',
        '## 提取了什么', '',
        '| 候选8方向 | 拟合船时相变化覆盖 | 留出船时相变化覆盖 | 拟合源类别中心差异重合 | 25%抑制后留出时相能量比 | 方向能量落在Qs的比例 |',
        '|---|---:|---:|---:|---:|---:|']
    for mode, g in geometry.iterrows():
        lines.append(f'| {names[mode]} | {100*g.train_temporal_fraction:.2f}% | {100*g.held_temporal_fraction:.2f}% | {100*max(0., g.class_center_contrast_fraction):.2f}% | {g.held_latent_temporal_energy_ratio:.4f} | {100*g.stable_span_fraction:.2f}% |')
    lines += ['', '这些比例是特征几何的能量比例，不是准确率贡献率。留出船捕获率高说明这些变化并非只出现在拟合船中；不能由此确认具体物理因素。类别中心保护的源重合接近零是构造保证，不是独立实证发现。', '',
        '## 作用到完整八输出识别', '',
        '固定减弱25%，不搜索强度。查询特征沿候选方向向拟合源类中心的总均值收缩，同时改变基础与专家对应的R50分数路径。支持样本、64标签适配系数、R18及门控参数均保持不变；重算八列分数、置信度、交接资格及5%排序。',
        '所有23港策略先冻结，再统一读取查询标签。没有依据目标结果挑方向、类别或强度。这是计算干预，不是协调重训后的新模型性能；不能把一次负结果解释成该方向永远无用。', '',
        '| 已有模型 | 干预 | 全23港OA | 全23港BA | OA变化pp | BA变化pp |', '|---|---|---:|---:|---:|---:|']
    for _, row in scores[scores['group']=='all23'].iterrows():
        lines.append(f'| {arms[row.arm]} | {names[row["mode"]]} | {100*row.full_oa:.4f}% | {100*row.full_ba:.4f}% | {row.full_oa_delta_pp:+.4f} | {row.full_ba_delta_pp:+.4f} |')
    lines += ['', '全23港包含12个八类齐全港、8个三类港、2个七类港、1个六类港；模型始终输出八类。BA按查询实际出现类别平均召回，再按港等权。下面单列查询均具备八类的12港，不按表现筛选。', '',
        '| 已有模型 | 干预 | 八类齐全12港OA | 八类齐全12港BA | OA变化pp | BA变化pp |', '|---|---|---:|---:|---:|---:|']
    for _, row in scores[scores['group']=='all8_target12'].iterrows():
        lines.append(f'| {arms[row.arm]} | {names[row["mode"]]} | {100*row.full_oa:.4f}% | {100*row.full_ba:.4f}% | {row.full_oa_delta_pp:+.4f} | {row.full_ba_delta_pp:+.4f} |')
    lines += ['', '## 变化变小是否等于分类变好', '',
        '下表只看留出源船的冻结源先验，不是64标签适配后的完整模型。时相分数能量先去掉八类分数的公共偏移，以免无关的共同logit变化主导。BA按留出聚合观测实际类召回平均，时相能量按上述层级权重；二者不是同一种加权。', '',
        '| 联合版本分支 | 干预 | 时相分数能量比 | 源先验BA变化pp |', '|---|---|---:|---:|']
    for _, row in source[source.arm=='joint'].iterrows():
        lines.append(f'| {row["head"]} | {names[row["mode"]]} | {row.temporal_centered_score_energy_ratio:.4f} | {row.ba_delta_pp:+.4f} |')
    lines += ['', '### 源类别保护能否推广到目标', '',
        '仅作事后解释：对各查询集每类至少5条的类别计算中心（至少两类），并与相同类别集合的拟合源中心比较。目标标签不用于拟合或调整候选方向。', '',
        '| 候选方向 | 相同类别集合的源中心差异重合 | 目标中心差异重合 |', '|---|---:|---:|']
    for mode, row in contrasts.iterrows():
        lines.append(f'| {names[mode]} | {100*row.source_matched_class_fraction:.2f}% | {100*row.target_class_fraction:.2f}% |')
    lines += ['', '目标中心包含有限样本波动，重合不等于该比例的有效判别信息会丢失，更不是某种物理因素贡献率。它用于检查“只保护源均值就足以保护目标类别差异”的假设是否可靠。', '',
        '## 文件和验证', '',
        f'17项同船配对/权重/已知方向测试通过。各港原分数重放最大误差{checks.max_score_replay_error.max():.3g}，任意联合空间扰动的解析响应与直接原始特征变换差异最大{checks.max_raw_perturbation_response_error.max():.3g}；原门控分数重放差异最大{checks.max_gate_replay_error.max():.3g}。',
        '每港保存train_pairs.csv、held_pairs.csv、directions.npz、policies.npz和FROZEN.json；所有模式及逐港负结果保留。主目录有geometry/directions/held_source_readout/metrics/class_recalls/port_means/coverage_group_means等CSV。',
        '后续结论与处置见同目录DECISION_ZH.md。本报告不把重复观测变化称为纯噪声、物理因果因子或已验证新模块。', '',
        '![方向与完整识别影响](temporal_directions_and_utility.png)']
    (OUT/'FINDINGS_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    make_plot(geometry, scores)
    print(geometry.to_string())
    print(contrasts.to_string())


if __name__ == '__main__':
    main()
