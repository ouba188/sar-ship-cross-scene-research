# 两候选固定匹配器 M3 读出重训

完成日期：2026-10-26。两候选 × 两方向 × seed17/29/43，共12组已完成；训练与图库生成的全部 exec session 已结束。本支线仅写入 downstream/，未执行 Git，未扩展第三候选或追加实验。

## 结论与范围

严格保留旧 M1/M2 的 s12 加分锚定，并未形成相对原 rawMLP 的双向优势；重新训练 M3 也没有解决这一限制。raw_summary 的反向 bare OA 更高，但其本质是用新分类分数替换 M2 输出，**不再保留旧 M1 加分路径**，不能称为原 M1/M2 全保留模型的提升。

- anchored_raw_summary：以旧 s12 为加法锚定，是预定下游候选。
- raw_summary：用 raw+summary 分类器输出替换 s12；是在看到本轮中间结果后追加的开发诊断，不是新盲测，不按 target 结果挑选或修改超参。
- 固定历史 PairMetric、32个源参考及 local 缓存，只重训 JointHead、VV、centeredVH、alpha 和 frequency gate。未重训 M2、匹配器或特征提取器。
- utility_train 优化并拟合 scaler，selection 选 checkpoint/alpha；head200步、gate100步、seed17/29/43。JointHead 是平行比较臂，最终 frequency 路径为 bare 加 VV/VH 加权残差，并不串联 JointHead。

## 跨域三 seed 均值

OA/BA 单位为百分比；最后两列是 final 相对本候选 bare 的百分点变化。final 固定指 frequency，不是 target 端最优臂。

| 候选 | 方向 | bare OA/BA | final OA/BA | ΔOA | ΔBA |
|---|---|---:|---:|---:|---:|
| anchored | OSS1→OSS2 | 52.8104 / 49.7897 | 52.8516 / 49.8589 | +0.0412 | +0.0693 |
| anchored | OSS2→OSS1 | 60.9482 / 52.0954 | 60.9121 / 50.9398 | -0.0362 | -1.1556 |
| raw_summary | OSS1→OSS2 | 53.9016 / 52.4061 | 54.1075 / 52.5581 | +0.2059 | +0.1520 |
| raw_summary | OSS2→OSS1 | 62.9388 / 51.6921 | 60.8759 / 51.9651 | -2.0629 | +0.2730 |

原 rawMLP 对照的正/反向 OA/BA 为53.7781/51.0819、60.0796/50.8147。不能据此宣称超过所有强简单方法；主任务的等容量及 ridge 对照由主报告统一解释。

## 配对与作用阶段

下表依次列 seed17/29/43 的救回数/误伤数，比较均为 bare→final。

| 候选 | 方向 | seed17 | seed29 | seed43 |
|---|---|---:|---:|---:|
| anchored | OSS1→OSS2 | 0/0 | 0/0 | 3/1 |
| anchored | OSS2→OSS1 | 39/43 | 0/0 | 4/1 |
| raw_summary | OSS1→OSS2 | 0/0 | 10/9 | 38/29 |
| raw_summary | OSS2→OSS1 | 13/66 | 24/12 | 24/40 |

anchored 正向 seed17/29 的两个 polar head 均选零步；seed43 只有小幅改善。反向 seed17 净少4例正确，bulk净增11，但 general/chemical/container 分别净减6/3/6，解释了 OA 小降而 BA 明显下降。其源 selection 选择 VV 为主、frequency100步；该选择未在跨域保持逐类收益。

raw_summary 反向三 seed 平均 bulk recall 下降3.2593pp，general/chemical分别提高3.2864/1.3889pp，container下降0.3236pp。目标921例中bulk675例，因而少数类改善与总体OA下降并存。seed17选择VH、净损53例，其中bulk净损55例；seed29选择VV并净增12例。反向 VV 单臂均值63.4093/51.7464，final为60.8759/51.9651；这是源端选中的融合未保留该OA收益的证据，不构成事后改选VV的授权或盲测结论。

## 验证、原图与产物

12组source/cross预测全部保存后才执行target评价。60条限定验证记录覆盖读出零步恒等、gate零步等于B_fixed、候选obs_id顺序、重算base字段和utility scaler、冻结local及非base gate字段；全部通过。源修复缓存没有obs_id，使用其对应combined table的source original_obs_id核对原stack顺序。没有额外全审计。

8张anchored seed17面板均已通过实际view_image查看，包含原VV/VH及真实同类/混淆类参考。正向无翻转，按原规则补选2个CC和2个WW；反向各选1个救回/误伤/CC/WW。事实、分差和作用阶段见 CASE_REVIEW_ZH.md；不把图像相似性或分差变化命名为物理原因。

完整结果：metrics.csv、summary.csv（均值/标准差及NLL/Brier）、per_class.csv、per_class_summary.csv、paired.csv、paired_summary.csv、composition.csv、selections.csv、validation.json。每臂目录保留checkpoint、曲线、scaler及全部预测；cases.json记录案例ID、原图路径、预处理、真实参考与四类分数。没有未完成训练、未查看面板或未结束exec session。
