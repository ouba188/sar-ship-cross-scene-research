"""Summarize the bounded trial and localize changes without new training."""

from full_teacher_data import *


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str, r))+' |' for r in rows])


def main():
    assert (OUT/'COMPLETE.json').exists()
    metrics = pd.read_csv(OUT/'metrics.csv')
    classes = pd.read_csv(OUT/'class_metrics.csv')
    manifest = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id', 'mmsi'])
    summary = metrics.groupby(['endpoint', 'arm'])[['oa', 'ba']].mean()
    ports = metrics[metrics.endpoint == 'fusion'].groupby(['port', 'arm'])[['oa', 'ba']].mean()
    paired = metrics[metrics.endpoint == 'fusion'].pivot(index=['port','draw'], columns='arm', values='ba')
    paths = pd.read_csv(OUT/'prior_kernel_interventions.csv').groupby(['arm','path'])[['oa','ba']].mean()
    extra_events, replay, student, gradient, teachers = [], [], [], [], []
    for port in PORTS:
        dest = OUT/port
        knowledge = dict(np.load(dest/'knowledge_teacher/predictions.npz'))
        equal = dict(np.load(dest/'equal_teacher/predictions.npz'))
        original = dict(np.load(dest/'no_training/predictions.npz'))
        hist = np.load(ROOT/'frozen_comparison_20261010/targets/self24ports'/port/'policies.npz')
        old = hist['predictions'][:, list(hist['modes']).index('softmax_average_all')]
        changes = int(np.sum(old != original['fusion'].argmax(2)))
        assert changes == 0
        replay.append(dict(port=port, historical_full_fusion_argmax_changes=changes))
        y = manifest.iloc[knowledge['rows']].class_id.to_numpy()
        for draw, ev in enumerate(knowledge['eval_masks']):
            before = equal['fusion'][draw].argmax(1)
            after = knowledge['fusion'][draw].argmax(1)
            for j in np.flatnonzero(ev & (before != after)):
                rid = knowledge['rows'][j]
                extra_events.append(dict(port=port, draw=draw, row=int(rid),
                    mmsi=str(manifest.iloc[rid].mmsi), class_id=int(y[j]),
                    before=int(before[j]), after=int(after[j]),
                    rescue=bool(before[j] != y[j] and after[j] == y[j]),
                    harm=bool(before[j] == y[j] and after[j] != y[j])))
        teachers.append(dict(port=port, **json.loads((dest/'TEACHER_DIAGNOSTIC.json').read_text())))
        for arm in ('no_training', *ARMS):
            student.append(dict(port=port, arm=arm, **json.loads((dest/arm/'STUDENT_DIAGNOSTIC.json').read_text())))
            if arm != 'no_training':
                gradient.append(dict(port=port, arm=arm, **json.loads((dest/arm/'GRADIENT_DIAGNOSTIC.json').read_text())))
    events = pd.DataFrame(extra_events)
    events.to_csv(OUT/'knowledge_vs_equal_events.csv', index=False)
    pd.DataFrame(student).to_csv(OUT/'student_diagnostics.csv', index=False)
    pd.DataFrame(gradient).to_csv(OUT/'gradient_diagnostics.csv', index=False)
    write(OUT/'HISTORICAL_FUSION_REPLAY.json', replay)
    names = dict(no_training='原模型，不续训', continued='A 同预算任务续训',
                 equal_teacher='B 加等权同船教学', knowledge_teacher='C 加知识加权同船教学')
    primary = summary.loc['fusion']
    improvement = primary.loc['knowledge_teacher','ba']-primary.loc['continued','ba']
    additional = primary.loc['knowledge_teacher','ba']-primary.loc['equal_teacher','ba']
    student_frame = pd.DataFrame(student).groupby('arm')[['equal_mse','knowledge_mse']].mean()
    reduce = 100*(1-student_frame.loc['equal_teacher','equal_mse']/student_frame.loc['continued','equal_mse'])
    rescue, harm = events[events.rescue], events[events.harm]
    text = f'''# 完整模型同船教学接入试验

日期：2026-10-12。状态：完成。9 次真实 R18 layer4 短续训，非只重算教师或分类头。

## 结论先说

- 接入完整模型后有小幅正信号：知识加权方案相对同预算任务续训，三港平均融合 BA **+{improvement:.6f} 个百分点**。
- 主要贡献来自同船教学本身。知识加权相对等权只有 **+{additional:.6f} 个百分点**，不足以声称运动知识已形成稳定的新核心模块。
- 不是因果因素已分离出来。当前证据是训练约束影响了专家的支持匹配和最终融合；没有识别纯因果空间。
- 原主方案未覆盖。保留等权同船教学作为小收益候选，运动加权暂不升格。此轮没有追加参数搜索。

## 1. 实际接法

```text
训练端：源港 SAR -> 原 R18 前缀 -> 可训练 layer4 -> 当前完整专家算子
                                          |         |
                   同船其他整景的读出教师 --+      原64支持集任务CE
                   等权 / 运动知识加权     |
                                   中心化类别分数MSE

预测端：单张SAR -> 原R50/分频基础分支 ---------+
            -> 更新后的R18 -> 重新提特征    |
                            -> 当前专家    +-> 等权概率融合 -> 八类预测
```

预测不读取 AIS、MMSI、同船历史、SOG 或 COG。保留原模型的 SAR 采集入射角侧信息；这不等于严格“只给图像像素”。没有新增推理网络或额外元数据查询。

专家读出包括原保留坐标、PCA互补坐标、R50候选稳定坐标、保护稳定空间的漂移衰减和入射角核项。
此轮让训练梯度经过上述当前算子，不再用缺少后加项的旧训练算子。固定 R50、分频基础分支、PCA/Qs/Qd/标准化尺度和 BN；只更新 R18 layer4。
训练期间源端分类头固定，训练后重新提取源/目标R18特征、更新坐标与支持核，并在新坐标下重新拟合源先验。该头重拟合不是可微双层优化。

三组同初始化、同源港episode、同样本顺序、同64步、同学习率1e-5、同最终checkpoint。B/C仅增加0.5倍中心化分数MSE。A也执行匹配的教师前向，但损失系数为0。
每个目标港五次相同的64标签支持抽样，基础模型和专家共享标签，不增加目标标签。一个训练顺序种子31415；五组支持抽样不是五个神经网络种子。

## 2. 最终完整模型结果

以下均为三个指定开发港各五组支持抽样的等权平均，不是完整23港平均。OA/BA单位为%，差值单位为百分点。

{table(['方案','融合OA','融合BA','相对A的BA'], [[names[a],f'{primary.loc[a,"oa"]:.6f}',f'{primary.loc[a,"ba"]:.6f}',f'{primary.loc[a,"ba"]-primary.loc["continued","ba"]:+.6f}'] for a in ('no_training',*ARMS)])}

{table(['港口','原模型BA','A BA','B BA','C BA','C-A','C-B'], [[p,*[f'{ports.loc[(p,a),"ba"]:.6f}' for a in ('no_training',*ARMS)], f'{ports.loc[(p,"knowledge_teacher"),"ba"]-ports.loc[(p,"continued"),"ba"]:+.6f}',f'{ports.loc[(p,"knowledge_teacher"),"ba"]-ports.loc[(p,"equal_teacher"),"ba"]:+.6f}'] for p in PORTS])}

知识加权相对A在15个港口-支持抽样单元中，BA正向{int((paired.knowledge_teacher > paired.continued).sum())}、负向{int((paired.knowledge_teacher < paired.continued).sum())}、持平{int((paired.knowledge_teacher == paired.continued).sum())}。这些单元共享样本和网络，不是独立重复。
提升主要集中在洛杉矶。巴生港虽然比续训A好，仍低于不续训原模型；不能只展示A对照掩盖这一点。

## 3. 知识究竟多带来了什么

C相对B总计改变{len(events)}个“样本-支持抽样”预测事件，其中救回{len(rescue)}、损害{len(harm)}。救回涉及{rescue.row.nunique()}个不同样本、{rescue.mmsi.nunique()}艘不同船；损害涉及{harm.row.nunique()}个不同样本、{harm.mmsi.nunique()}艘船。详见 `knowledge_vs_equal_events.csv`，不把重复支持抽样算成独立船只。

教师可靠性读出使用4维已限定运动特征：u=SOG/(1+SOG)、u平方、u平方*cos(2theta)、u平方*sin(2theta)，theta是COG与原生GRD地距轴的相对角。没有把它称为真实雷达径向速度或船体朝向。
按源船三折拟合本轮完整专家的教师先验，重新学习质量读出，没有直接移植简化探针系数。当前整景完全排除；每个其他整景先等权聚合；缺失知识的供体仍保持等权总质量，已知供体相对权重最多2:1。
每港1296个源端教学候选、525艘船，1040行有同船其他整景；真正可以做知识重权的只有364行、70艘船。

{table(['对应完整专家','等权教师BA','知识教师BA','差值','救回/损害'], [[t['port'],f'{t["equal"]["ba"]:.6f}',f'{t["knowledge"]["ba"]:.6f}',f'{t["knowledge"]["ba"]-t["equal"]["ba"]:+.6f}',f'{t["rescues"]}/{t["harms"]}'] for t in teachers])}

这里是重复开发的源端教师诊断，不是目标港识别结果。运动加权在三个完整专家中的两个降低教师BA，说明之前简化探针的正信号没有直接迁移到当前评分环境。历史骨干/几何见过源船，只有线性读出按船排除；质量读出的上游头未做嵌套重训，因此不宣称完全未见船的独立验证。

## 4. 收益通过哪里发生

固定替换实验只改变先验或支持核，不新增训练，也不按目标结果选择组合。表中仍为最终融合BA。

{table(['教学方案','仅换先验-A','仅换支持核-A','两者同时换-A'], [[names[a],*[f'{paths.loc[(a,k),"ba"]-paths.loc[(a,"control"),"ba"]:+.6f}' for k in ('prior_only','kernel_only','both')]] for a in ('equal_teacher','knowledge_teacher')])}

结论：主要正向路径是更新后的支持匹配核，单换先验略负。联合结果有交互，不能把两项差值直接相加，也不能把核的改善解释为某个已确认的物理机制。
相对同预算A，B在源端固定教师读出下的中心化教学误差降低{reduce:.3f}%。但B/C误差仍高于不续训原模型，符合“约束任务续训偏移”的表现，不是无条件地提纯表示。
第一批样本中教学梯度约为任务梯度的9%-20%，余弦为负；教学并非没有接入，且确实与任务更新存在局部冲突。只测首批梯度，不能推断全部训练过程。
专家单独的平均BA：A {summary.loc[("expert","continued"),"ba"]:.6f}，B {summary.loc[("expert","equal_teacher"),"ba"]:.6f}，C {summary.loc[("expert","knowledge_teacher"),"ba"]:.6f}。C略低于B，却在融合后略高，表明额外收益与基础分支的分数配合有关，不能只看专家单独top1。

## 5. 已检查与限制

- 可微核映射与原NumPy算子对齐，分数误差约1e-15，R18梯度非零。每港记录 `OPERATOR_PARITY.json`。
- 新提取的初始源特征与旧缓存逐元素相同。初始专家分数误差小于1e-6，专家top1不变；三港五组完整融合预测与历史主方案全部一致。
- 同船聚合小测试覆盖整景排除、重复裁片和未知知识供体质量保持。全部运行已结束。
- 保留所有负结果和每类召回，不更换测试集、不扩大目标标签预算。原模型、论文结论均未覆盖。
- 仅64步、一个训练种子、三个历史开发港。不是已收敛的完整训练，也不是23港/跨数据集稳定验证。没有用此轮测试结果回选超参数。

## 6. 下一步判断

值得保留的是“同船教学约束完整专家的支持匹配”这条小收益路线。运动知识的独立增量只有少数样本，且教师本身多数港退化，暂不支持直接扩成重型模块。
后续若继续，先按同一固定协议补第二个训练顺序种子，并检查主要获益船是否重复；不要先加新模块。若要继续发展运动知识，先解决“完整专家下教师排序为何变差”，而不是增大教学权重硬推收益。

关键文件：`metrics.csv`、`port_means.csv`、`class_metrics.csv`、`error_events.csv`、`knowledge_vs_equal_events.csv`、`prior_kernel_interventions.csv`、`student_diagnostics.csv`、`PROTOCOL.json`。
'''
    (OUT/'FINDINGS_ZH.md').write_text(text, encoding='utf-8')
    print(events.to_string(index=False))
    print('REPORT', OUT/'FINDINGS_ZH.md')


if __name__ == '__main__':
    main()
