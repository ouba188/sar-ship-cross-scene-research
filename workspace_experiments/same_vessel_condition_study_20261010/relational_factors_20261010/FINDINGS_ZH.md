# 同船与同景关系监督的双因子小试

## 本轮结论

关系监督产生了候选分工信号，但没有提高跨港分类：稳定分支目标BA由同结构对照33.763%降至33.503%，加条件校正33.310%，64标签微调校正后33.371%。原始特征加分类头基线为38.304%，本轮新网络不加入主模型。
在未参与表示训练的302艘源船上，条件分支场景排序正确率50.391%升至68.164%，但仍可用来判船型（BA30.072%），不是纯场景或纯噪声。稳定分支同船余弦距离0.635降至0.364，但后续去整体均值诊断发现，这一降幅部分来自公共偏移，不可全部解释为跨条件不变性增强。

## 本轮实际管线

冻结R18的512维池化特征 -> 源训练数据标准化 -> 稳定候选MLP128维 / 条件候选MLP32维；条件向量经线性层和0.1*tanh生成128维修正；重建头从稳定+条件重建输入，分类头同时监督稳定及校正后的表示。
表示训练：1744张、605艘、446景；诊断：823张、302艘、336景。身份无重叠，但有290个大景共用，因此诊断是新船检验，不是新大景检验。源训练采用4670对同船异景观测，含缺少细粒度条件元数据的配对；它们不是4670次已知物理干预。
源训练固定400步后冻结两分支。目标阶段以原939标签拟合源先验，再用原64标签拟合残差分类头；额外适配版只训练128个条件修正增益（相对源修正幅度上下25%），沿用缺类屏蔽及源回放。

## 源域未参与表示训练的船舶诊断

| arm | factor | OA | BA | same_ship_cosine_distance | semantic_gap | semantic_triplet_accuracy | scene_gap | scene_triplet_accuracy | coordinate_variance | same_ship_pairs | scene_triplets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| task_only | stable | 46.294 | 38.043 | 0.635 | 0.227 | 69.531 | -0.147 | 40.430 | 1.002 | 1492 | 512 |
| task_only | condition | 37.910 | 36.916 | 0.732 | 0.110 | 61.914 | 0.000 | 50.391 | 1.206 | 1492 | 512 |
| task_only | corrected | 45.808 | 37.865 | 0.630 | 0.232 | 70.117 | -0.150 | 39.453 | 1.027 | 1492 | 512 |
| relations | stable | 46.416 | 37.504 | 0.364 | 0.248 | 70.508 | -0.182 | 35.156 | 0.745 | 1492 | 512 |
| relations | condition | 30.012 | 30.072 | 0.450 | 0.007 | 52.344 | 0.169 | 68.164 | 0.623 | 1492 | 512 |
| relations | corrected | 46.416 | 37.672 | 0.351 | 0.240 | 71.094 | -0.176 | 34.961 | 0.746 | 1492 | 512 |

same_ship_cosine_distance越小表示同船更一致；semantic_gap/scene_gap及对应排序正确率越大越好。条件排序正例为同景异类异船，负例为同类异景异船。所有诊断在留出身份上进行，但部分场景在表示训练中出现过；不能称为独立新场景确认。

## 原3港9组64标签目标评估

| arm | mode | OA | BA |
| --- | --- | --- | --- |
| relations | adapted | 33.296 | 33.371 |
| relations | corrected | 33.293 | 33.310 |
| relations | stable | 33.392 | 33.503 |
| task_only | adapted | 32.930 | 33.618 |
| task_only | corrected | 32.960 | 33.620 |
| task_only | stable | 33.135 | 33.763 |

| port | arm | mode | OA | BA |
| --- | --- | --- | --- | --- |
| Los Angeles | relations | adapted | 36.776 | 34.763 |
| Los Angeles | relations | corrected | 36.776 | 34.763 |
| Los Angeles | relations | stable | 37.127 | 35.304 |
| Los Angeles | task_only | adapted | 36.605 | 34.929 |
| Los Angeles | task_only | corrected | 36.612 | 34.869 |
| Los Angeles | task_only | stable | 36.902 | 35.204 |
| Port Klang | relations | adapted | 30.988 | 29.898 |
| Port Klang | relations | corrected | 30.958 | 29.888 |
| Port Klang | relations | stable | 30.838 | 29.758 |
| Port Klang | task_only | adapted | 31.068 | 29.672 |
| Port Klang | task_only | corrected | 31.108 | 29.706 |
| Port Klang | task_only | stable | 31.128 | 29.583 |
| Santos | relations | adapted | 32.125 | 35.453 |
| Santos | relations | corrected | 32.146 | 35.280 |
| Santos | relations | stable | 32.211 | 35.447 |
| Santos | task_only | adapted | 31.116 | 36.253 |
| Santos | task_only | corrected | 31.159 | 36.284 |
| Santos | task_only | stable | 31.374 | 36.501 |

| arm | mode | coverage | recall | task_groups |
| --- | --- | --- | --- | --- |
| relations | adapted | absent | 29.538 | 4 |
| relations | adapted | multiple | 32.250 | 9 |
| relations | adapted | singleton | 32.751 | 5 |
| relations | corrected | absent | 29.538 | 4 |
| relations | corrected | multiple | 32.250 | 9 |
| relations | corrected | singleton | 31.918 | 5 |
| relations | stable | absent | 29.538 | 4 |
| relations | stable | multiple | 32.352 | 9 |
| relations | stable | singleton | 33.018 | 5 |
| task_only | adapted | absent | 29.892 | 4 |
| task_only | adapted | multiple | 32.490 | 9 |
| task_only | adapted | singleton | 31.936 | 5 |
| task_only | corrected | absent | 29.892 | 4 |
| task_only | corrected | multiple | 32.499 | 9 |
| task_only | corrected | singleton | 31.936 | 5 |
| task_only | stable | absent | 29.892 | 4 |
| task_only | stable | multiple | 32.689 | 9 |
| task_only | stable | singleton | 31.669 | 5 |

stable为稳定候选分支；corrected增加源域学到的条件残差；adapted再用64标签调整该残差的128个通道系数。三个版本都重拟合相同形式的源分类头和目标残差头。

## 对照与边界

- task_only与relations网络、初始化、源样本、训练步数相同，区别仅为是否加入同船一致性、同类跨场景排序和同景异类排序约束。
- 两版均有船型监督、重建任务和条件残差。不能将两版差异归因为凭空增加分支或额外标注。稳定/条件为设计名称，是否成功分工要看诊断，不能依名称认定因果因素。
- 源表示训练只用fold1/2的源船；fold0用于诊断，不用于选超参数或轮次。目标源头重拟合使用原939个允许源标签，故诊断和目标源头的训练范围不同。
- 重复观测使用已知源MMSI船型标签继承，没有新增人工船型标注，但增加了观测及身份关系信息。与历史原始特征基线比较并非等训练量。
- 冻结R18的512维池化特征上学习轻量网络，不更新骨干空间算子；无法据此判定全部SAR信息可分离或物理根因。
- 源域共景不表示船只局部观测条件完全一致。场景排序只提供共享条件代理；条件分支仍可能携带船型信息。
- 固定400步源训练与24步目标适配；CPU1，无目标查询拟合，不按港选版本。支持与查询可共享大景，历史预训练曝光保留。

## 为什么更相似却没有分得更好

| arm | factor | source_centered | same_ship_distance | same_type_other_ship_distance | ship_to_type_distance_ratio | within_port_scene_gap | within_port_scene_accuracy | scene_eligible_anchors | same_ship_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| task_only | stable | False | 0.635 | 0.698 | 0.910 | -0.215 | 31.836 | 528 | 1492 |
| task_only | stable | True | 0.759 | 0.818 | 0.928 | -0.216 | 33.789 | 528 | 1492 |
| task_only | condition | False | 0.732 | 0.820 | 0.892 | -0.047 | 46.875 | 528 | 1492 |
| task_only | condition | True | 0.810 | 0.896 | 0.904 | -0.061 | 44.531 | 528 | 1492 |
| relations | stable | False | 0.364 | 0.420 | 0.866 | -0.238 | 29.688 | 528 | 1492 |
| relations | stable | True | 0.672 | 0.749 | 0.897 | -0.310 | 32.031 | 528 | 1492 |
| relations | condition | False | 0.450 | 0.520 | 0.866 | 0.093 | 56.445 | 528 | 1492 |
| relations | condition | True | 0.825 | 0.958 | 0.861 | 0.115 | 57.031 | 528 | 1492 |

去除源训练均值后，稳定分支同船距离为0.759 -> 0.672，同类异船距离也从0.818降至0.749。同船/同类异船距离比0.928 -> 0.897，说明还存在相对同船一致性信号，但远小于未经居中时的表面降幅。不能只用整体余弦距离下降宣称找到域不变因素。
条件排序限制为同一港口内比较，并去除源均值后，从44.531%升至57.031%。因此场景关系信号不只来自港口差别，但仍是弱监督代理，可能包含船只状态及局部观测差异。512个抽样三元组会重复锚点，不是512个独立确认样本。

| representation | prior | kernel | OA | BA |
| --- | --- | --- | --- | --- |
| pca128 | new | new | 36.835 | 36.404 |
| pca128 | new | raw | 35.983 | 36.055 |
| pca128 | raw | new | 38.024 | 38.223 |
| pca128 | raw | raw | 37.195 | 38.304 |
| relations | new | new | 33.392 | 33.503 |
| relations | new | raw | 32.730 | 32.813 |
| relations | raw | new | 37.344 | 37.466 |
| relations | raw | raw | 37.195 | 38.304 |
| task_only | new | new | 33.135 | 33.763 |
| task_only | new | raw | 32.786 | 32.916 |
| task_only | raw | new | 37.814 | 38.085 |
| task_only | raw | raw | 37.195 | 38.304 |

raw表示原冻结特征；new表示对应的新特征。prior替换源分类分数，kernel替换支持案例关系，所有组合仍重拟合同一64标签残差头。原始/原始及两个已训练模型的对角预测共27次回放完全一致。交叉组合用于定位路径，不作为按目标成绩挑选的新部署模型。
原始/原始BA38.304%；关系模型只替换源先验降至32.813%，只替换案例关系37.466%，两者都换33.503%。因此大部分直接损害沿新源先验路径出现，存在路径交互，不能把降幅简单相加。只换回旧先验仍未超过基线，不是现成修复。
PCA128对照在同一表示训练源集无监督拟合：全替换BA36.404%，优于两个学习型128维表示但仍低于原始512维。128维压缩可能参与损失，却不足以解释学习型表示的全部退化；同结构无关系约束版也只有33.763%，大幅下降并非全由关系损失引起。

## 已定位与未定位

- 已定位：关系损失确实改变两分支的关系偏好；部分表面稳定性来自公共偏移；主要直接分类损害与新表示的源分类分数有关。
- 尚未定位：哪些SAR散射变化被条件分支编码；为何新源分类分数跨港变差的唯一根因；可分离因子能否在未见大景上保持同样分工。
- 实现检查：三项针对性测试通过；两分支均有梯度；预测路径回放一致。Pandas只读标签转Tensor曾提示警告，但没有标签原地修改；已改为显式复制，未据此认定训练错误或重跑结果。
- 下一步设计重点：保留原512维识别主路，用条件因子约束有限修正，而不直接让128维候选因子替换整个判别通路；同时用源域未见船舶任务的分类效果约束分离，不能继续把“更相似”当作唯一成功条件。这是建议，尚未运行。
