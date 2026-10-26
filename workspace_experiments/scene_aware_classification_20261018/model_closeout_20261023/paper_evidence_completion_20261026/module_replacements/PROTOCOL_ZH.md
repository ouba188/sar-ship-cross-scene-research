# M1/M2 条件匹配头对照协议

2026-10-26。只训练四个预先指定的头，每个源集三个种子，共24次拟合；没有新骨干、reader、温度、M3、参考集或目标域训练。

## 数据与冻结输入

- 源域：`oss1_m123_replication_20261026`、`oss2_m123_replication_20261026`。
- 源分数：`public_m123_cross_dataset_20261026/normalization_repair_20261026/source_oss{1,2}/seed{17,29,43}/predictions.npz`，读取 `frontend_base/m1/s12`。
- cross分数：相同 repair 根目录下 `oss1-to-oss2`、`oss2-to-oss1` 对应 seed 缓存。
- 源摘要：各源目录的 `frontend/summaries.npz`；cross摘要：原 transfer 根目录各方向的 `summaries/summaries.npz`，使用源 reader/温度已有提取，不重新提特征。
- 摘要按 obs_id 对齐。修复后的 source predictions 没有 obs_id，其生成脚本按 stack_table 顺序写出；本次核对原 frontend scores 的 obs_id/行数与该表一致，并保留此顺序假设，未伪称缓存自带 ID。
- 源 fit/validation 分别为 OSS1 405/89、OSS2 478/228；只从这两种角色读取训练/选点标签。utility_train、selection、evaluation 的 class_id 在训练表中置 -1。
- cross训练阶段仅读取 obs_id/sample_id/mmsi/pass_group/role，不读取目标标签；两个方向全部预测保存完成后再统一评价。

源 fit 四类样本数（bulk/general/chemical/container）：OSS1为270/33/33/69，OSS2为138/47/95/198；验证为66/8/9/6与87/23/29/89。源角色隔离复用原 `load_table` 检查，cross与所有源监督角色的船只及pass不交叉。

## 四个固定臂

| 臂 | 冻结起点 | 输入改动 | 架构/名义参数 | 输出 |
| --- | --- | --- | --- | --- |
| m1_both_refit | corrected frontend_base | 原 mean3+hb3 | 原 AttentionHead，266 | s1 |
| m1_mean_only | 同上 | hb3全零，再构造原15维raw | 完全相同双分支，266 | s1 |
| m2_identity_refit | corrected原m1 | 原三区域摘要与质量 | 原69→12→1，853 | s12 |
| m2_summary_merged | 同一个corrected原m1 | 摘要与质量合并、复制三槽 | 原69→12→1，853 | s12 |

M1 mean-only 仍保留 local 分支，它继续接收原base和极化有效性列；只是分布输入及其中心化列为零。是同名义参数/架构的输入信息对照，不是删去分支，也不是第一阶段全源监督普通区域MLP。

M2两臂**均冻结原m1**，不把本轮 M1 refit 接上M2，不进行跨臂组合训练。原修复缓存的 `original_m1`、`original_s12` 另列，不被覆盖。这里“原full”指原M2终点s12，完整M3并未配套重拟合。

## M2 合并的精确定义

M2区域顺序为 body/end/side，区别于M3 token顺序。每极化只保留 `valid && quality.flag>0 && area>0` 的区域；令有效面积为a，w=a/sum(a)。

将每类三个统计量（均值及两项分布摘要）分别按w求和，所得summary复制到三个区域槽。质量字段复制为：

1. area：有效区域面积总和；
2. coverage：有效面积加权的原coverage均值；
3. min_neff：有效面积加权的原min_neff均值；
4. valid：任一有效区域为1，否则0。

无有效区域时summary/quality均为零；geometry不变。这些coverage/neff是本摘要控制的汇总代理，**不是合并像素后重新计算的真实coverage或有效样本量**。本对照同时丢弃分区摘要与分区质量身份，不能只归因于其中某一部分。

非线性注意力摘要的加权平均一般不等于像素先合并再池化；因此报告使用“summary-merged”，不称“无区域像素池化”或独立区域物理机制实验。

## 同一拟合与选择规则

直接调用原 `oss1.../train_frontend.py::train_head` 和 `upstream.AttentionHead/RegionHead`。
每臂先固定相同seed17/29/43；对应两臂初始化一致。全batch200steps，AdamW lr=.003、weight_decay=.01；损失为0.5样本平均CE+0.5类平均CE。不是船只加权训练，也不是全栈OOF。

标准化仅以源fit raw沿样本与类别轴估计mean/std，std下限1e-8；所有refit统一采用 `ptp(source_fit_raw)==0` 的严格恒定维屏蔽，恒定维在所有角色与cross均置零。保存mask到checkpoint与scaler NPZ。实际both/identity/merged无恒定列；mean-only的8个分布列恒零，故屏蔽只去掉这8列，没有另外偷偷改变原full输入。

每5步（含0）只在fit_validation按精确有理数J=(OA+BA)/2选点；严格变好才替换，平局保留最早。全部跑满200步，保存选中checkpoint与完整曲线。原train_head已经校验零初始残差、checkpoint恢复以及分批预测的选点一致性。

## 交付与解释限制

- `metrics.csv`：source各角色及cross逐seed分数；`metrics_summary.csv`：三seed均值与seed间标准差，非独立目标重采样置信区间。
- `selections.csv`、各方向selection JSON与curves CSV：所有选点及训练曲线。
- 各方向 `seed*/source_predictions.npz`、`cross_predictions.npz`：四臂及原控制分数，另带obs_id/role。
- 各方向 `models/`：所选头、初始状态、fit/validation ID、scaler/mask。
- `paired_changes.csv`、`stratified_changes.csv`：救回/误伤、类别/pass分层；跨seed计数是重复评价同一批查询，不能称独立样本数。
- `data_receipt.json`：实际观测/船只/pass数、源类别组成、极化/区域有效条件。
- `rawcases/`：固定seed17代表性原生VV/VH与对应分差证据，不涉及M3参考。

这是 **conditional matched head comparison**：冻结前段模型、reader与摘要后比较输入内容及重新拟合的头。名义容量匹配不消除优化、选点和统计估计差异；没有独立物理因果结论，也不是完整M3系统替换结果。使用既定历史队列，不宣称新的盲测。

## 执行

```powershell
python -B test_replacements.py -v
python -B run_replacements.py all
python -B review_replacements.py
```

本轮3项针对性测试通过（有效面积合并/质量、source恒定维、M1输入/架构）；未重跑此前9项局部匹配测试。所有写入均限定本目录，除用户额外授权的 `../stage1_analysis/WITHIN_CASE_REVIEW_ZH.md`；没有Git操作或新实验。
