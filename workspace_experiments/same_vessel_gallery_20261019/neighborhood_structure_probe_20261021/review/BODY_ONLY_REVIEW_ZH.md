# Body-only 定向审阅

日期：2026-10-21。结论：**在本次指定范围内，未发现实质问题。** 新实现关闭了 frozen baseline 的框外真实像素旁路，重新训练了与 body baseline 配套的原始 R reference；源端匹配、选步、保存和 `analyze.py --body` 的读取逻辑正确。

## 依据

1. **基线确实在卷积前隔离框外像素。**
   `crude_input_controls_20261021/input_controls.py:74` 先构造 `source = x * target`，随后才旋转、固定 ImageNet 标准化并进入 stem（第 75–77 行）。`prepare_view` 只做原生 crop/pad 和 `/255`，没有根据全图强度计算归一化参数；target 来自 OBB 二值栅格。`body_global` 虽在 valid footprint 上池化，输入中的框外值已经是常数零，不会重新引入框外真实像素。旧 `input_checks.csv` 记录 `outside_change_cannot_enter_body_global = 0.0`，这是已有检查结果，本次未重跑。

   `body_only.py:18,24–28` 明确读取 `body_global` features 和同名 fold head，使用保存的 mean/std/dimension/coef/intercept 重建分数，并与旧 baseline 的 test 分数对照。没有调用 whole-chip 的 `frozen_scores`。OBB 长宽与 valid-footprint 几何仍保留，符合协议；这里关闭的是给定 PNG 与掩膜下的框外像素值通路，并非消除所有几何信息。R 分支及原始 R mean 仍按设计读取邻域，不应将结论表述成整个模型完全不读取框外信息。

2. **没有复用 whole residual。**
   `body_only.py:35–38` 每个 fold/seed 先处理 `original`，复用的是独立提取的 centered-R 特征，随后在第 49–65 行新建并训练 residual，损失中的 baseline 已替换为 body 分数。第 72–80 行恢复源端选出的权重，再设置 `reference_head/reference_stats/reference_scores/reference_step`。后续 fixed 干预使用这组 reference 及其归一化（第 42–43 行）。后续 arm 会创建新的 head/optimizer，不会继续更新 reference；脚本没有加载旧 whole residual checkpoint。

3. **源端划分、匹配训练与选步正确。**
   只读比较旧 body 基线 `partitions.csv` 与当前沿用的 `neighborhood_level_split_20261021/partitions.csv`，按 fold/partition/obs_id 比较，均为 **18,888 行，顺序完全一致，差异 0**。直接检查 `relation.load_data` 和 `prior.indices`，训练/验证来自源端，test 未参与参数或步数选择。

   `body_only.py:37,41` 的特征标准化只用 tr；fixed 使用 original 的源端统计。五个 arm 重置相同初始化种子与采样 RNG，训练预算、优化器和权重规则一致；`weights[take]` 与 `ii = tr[take]` 对齐。第 53–60 行只以 val 的 BA 优先、Brier 次优选取 0/50/100/150，严格比较使完全并列时保留较早步。Residual 没有 dropout/BatchNorm，未调用 `eval()` 不会改变此处预测行为。

4. **保存的是选中模型及相应分数。**
   `body_only.py:67–71` 单独记录 terminal 指标，第 72 行先恢复选中 state，第 73–85 行再保存 head、选中模型的 train/val/test 指标和 test 分数。`source_constant` 只取 tr residual 均值。original reference 与后续 arm 的比较参考均属于当前 fold/seed；结果写入独立 `body_only/`，完成标志在循环结束后写入。

5. **`analyze.py --body` 读取正确。**
   第 156–158 行在所有分析调用前将 ROOT 切到 `body_only/`。`evaluate` 第 17–26 行从该目录的 reference 文件分别读取 `base` 和 `scores`，fixed/matched 也读取该目录；配对差异和案例预测承接这些结果。`verify` 第 133–146 行在 BODY 模式包含新训练的 `original`，其特征用旧 centered-R cache，其余 arm 用干预 cache，并读取 body 目录的 selection/head/scores。验证排序 BA 降序、Brier 升序、step 升序，与训练选步一致。

   `cases` 使用 `exp.ROOT/features.npz` 的共享 usable 和原始 PNG 是正确的输入来源，不是误读 whole-model 预测。图中第一列仍是明确标注的原始 chip，**不是 `x * target` 的实际基线输入展示**；`fixed D`、`rescue_vs_D` 等沿用旧命名，但数值参考已是 body reference。解释图表时应明确这一点，不构成本次训练或评估逻辑错误。

## 审阅边界

仅静态阅读两个目标脚本及其直接依赖，读取已有划分与输入检查记录；未重复审阅四种干预本身。未 import torch、未执行目标脚本、未运行 GPU/训练、未读取模型执行预测、未执行 git。没有将代码中的断言或计划执行的 `verify()` 视为本次已通过的运行时验证，也没有认定正在进行的 45 fits 已全部完成。仅新增本报告，未修改实验代码或产物。
