# 2026-11-06：同步实际研究中的 M1/M2 代码

## 本机与仓库是否一致

同步前并不完全一致：

- `workspace_experiments/sar_m123_current_20261031/sar_m123/` 正式工程快照的已跟踪Python源码与本机一致，本次未覆盖。
- 11月1日新版M1、位置/调参/覆盖实验，以及11月6日M1类别响应引导M2区域读出实验，仓库原先没有。
- 当前公共实验实际导入的 `sar_m123_standalone/sar_m123` 研究运行库也有未归档的源文件。本次补齐，并保留其被调用的几何/匹配依赖。

**现在同步的是实际使用的代码及实验记录，不是把全部候选升级为正式模型。** 本次依据用户明确要求执行；旧报告中的“不推送Git”描述的是当时状态。

## 从这里看代码

详细入口：[M1/M2代码与版本对应](workspace_experiments/m1_guided_m2_20261106/CODE_MAP_ZH.md)。

| 代码 | 用途 |
|---|---|
| [新版M1算子](workspace_experiments/m1_covariance_20261101/operators.py) | `RelationFeedback`：均值支路、类别竞争响应、二阶关系及残差调节 |
| [M1骨干接入](workspace_experiments/m1_covariance_20261101/run.py) | 当前源719公共载体使用layer2[3]的conv2之后、conv3之前，VV/VH独立模块参数 |
| [M1覆盖载体](workspace_experiments/m1_coverage_20261101/experiment.py) | 源719/913、匹配控制、检查点恢复与数据组织 |
| [M2原区域池化](workspace_experiments/sar_m123_standalone/sar_m123/pool_replacement.py) | 当前研究载体的区域池化实现 |
| [新M1到M2接口](workspace_experiments/m1_guided_m2_20261106/study.py) | 在几何区域内使用M1候选类别响应聚合layer2特征，补充原区域摘要 |
| [配对分析及图集生成](workspace_experiments/m1_guided_m2_20261106/analyze.py) | mean/content/class、关闭/均匀/置乱、原图及参考实例 |

这里的M1/M2候选没有显式频域分支。新M1到M2接口是区域读出补充，不是新增所有bottleneck内回写。公共载体缺少私有正式模型的layer2[2]M2，报告已明确这一范围。

## 最新结果没有被隐藏

OSS2四分类，源719、开发228、历史评价378、固定32源参考，三次新增读出训练种子：

| 方法 | 评价OA | 评价BA |
|---|---:|---:|
| 原冻结模型 | 62.434 | 59.718 |
| 类别引导读出＋更新分类头 | 62.346 | 59.781 |
| 本轮类别引导读出＋冻结分类头 | 60.229 | 57.641 |

更新分类头版本比其配套均值对照OA/BA +0.882/+1.548个百分点，但未形成相对原模型的明确整体提升。本轮冻结分类头反而退化，不替换正式版本。

核查还发现统一梯度裁剪与分类头冻结耦合，改变了读出梯度缩放，所以该轮不能单独证明“分类边界协同”就是根因。负结果、实例和这一局限一起保存。

- [类别引导区域读出报告](workspace_experiments/m1_guided_m2_20261106/FINDINGS_ZH.md)
- [冻结分类头对照报告](workspace_experiments/m1_guided_m2_20261106/frozen_head/FINDINGS_ZH.md)
- [冻结头图集](workspace_experiments/m1_guided_m2_20261106/frozen_head/index.html)
- [M1覆盖与配套协方差消融](workspace_experiments/m1_coverage_20261101/FINDINGS_ZH.md)

## 同步边界

同步六个相关M1研究目录的源码、协议、轻量结果、诊断图和配对记录，以及当前运行需要的源代码依赖。没有重新训练其他候选，没有改变正式模型权重，也没有提交工作区中无关修改。

不上传原始数据集、权重、特征缓存、完整样本清单、凭据或训练日志。大于1MiB的CSV按原归档约定压缩为`.csv.gz`。运行仍需本机合法数据和检查点；历史脚本保留原路径，异机需调整。正式工程继续保留原版本。
