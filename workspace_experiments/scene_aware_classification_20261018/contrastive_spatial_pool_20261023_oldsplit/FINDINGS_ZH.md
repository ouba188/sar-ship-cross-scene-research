# 旧 split 固定 layer3 std 复验

已完成。新 split 的 std 净 +19 没有在旧 split 复现为 OA 正收益；旧 split 净 -23，但 BA 与 balanced NLL 仍小幅改善。不能称为全面退化，也不能据此宣称稳定提升或局部离散信息无用。

## 固定协议与结果

同 5695 条、4312 MMSI、163 pass，沿用旧 EXP 六折。固定 layer3，不采用 source-selected stem 混合 A。六折 alpha 为 [1,1,10,1,1,1]，与历史 source validation 选择完全一致。历史逐例 A 分数重建误差至多 8.88e-16，top1 全不变，测试 A 保留原 CSV 精确值。

| 指标 | 原 fixed-layer3 A | A + std |
|---|---:|---:|
| OA | 81.510097 | 81.106234 |
| BA | 80.129739 | 80.205558 |
| macro F1 | 79.832502 | 79.791765 |
| NLL | 0.529658 | 0.530744 |
| balanced NLL | 0.549560 | 0.542236 |
| Brier | 0.286728 | 0.289542 |
| balanced Brier | 0.300456 | 0.298043 |

救回 60，误伤 83，净 -23。OA 差 -0.403863 pp，BA 差 +0.075820 pp。固定预测、2000 次 cluster bootstrap、seed20261023：pass OA 95% CI [-0.947279,+0.124743] pp，BA [-0.558969,+0.691743]；MMSI OA [-0.830316,+0.034906]，BA [-0.407794,+0.575831]。区间均跨零；不包含重新训练/重新选择不确定性，不宣称统计显著。与新 split 使用同一队列，不是独立新数据验证。

六折 source-only 选择 lambda=[1,.5,0,.25,.25,.5]，逐折净 [10,3,0,-6,-8,-22]。网格 [0,.1,.25,.5,1] 未修改，fold2 的零选择精确保留 A，无 test 选择或后补调参。源 balanced NLL 目标并不等于 pooled OA，结果保留二者的差异。

## 数据层面的效果

| 类别 | N | 救回 | 误伤 | 净 | 原 recall | 新 recall |
|---|---:|---:|---:|---:|---:|---:|
| bulk | 3170 | 7 | 60 | -53 | 85.4259 | 83.7539 |
| fishing | 1147 | 1 | 20 | -19 | 87.1840 | 85.5275 |
| general | 1378 | 52 | 3 | +49 | 67.7794 | 71.3353 |

这次总体变化是明确的类别间收益/损失重分配，而不是所有类别证据同时变差。较大的 bulk 类损失使 OA 下降，general 的 recall 增幅使 BA 略升；这是计数解释，不等于证明语义离散机制或静态偏置机制。

按采集港口，Rotterdam +12（15 救回/3 误伤），Mombasa +3；Shanghai -12（11/23），Qingdao -10（12/22），Port Klang -5（5/10）。其余采集及所有 pass 明细见 acquisition_effects.csv。不能把类别组成不同的港口净计数解释为港口因果效应。主线新 split 的 constant/amplitude 控制未在本支线重跑，其结论不能自动覆盖旧 split。

## 原窗回看

按每个 outcome 中首 instance_id 固定抽取，不根据外观选图。已使用 view_image 实际查看四张双极化面板；它们恰好均为 bulk，是抽样规则的限制，不代表三类总体。各原图路径及分差在 review_cases.csv，显示仅最近邻放大，原尺寸单独标注，绿色为记录 OBB。

- rescued，pair_0031216，Mombasa，fold0，VV/VH 均 64x64：框内横斜向紧凑白亮块；VV 的框外纹理比 VH 更明显。0 对原最强错类 2 的分差 -0.408468 -> +0.108977，std 加入 +0.517445，2 -> 0。可确认离散读出在此提供了 GAP 不同的类别排序，不能由亮块外观直接确认其类别依据。
- harmed，pair_0030918，Melbourne，fold3，VV64/VH72：细长白亮目标，两极化均见横向延伸亮纹，VV 更显著。原正确分差只有 +0.067231；delta 对类2更大，加权差 -0.257939，分差变 -0.190709，0 -> 2。可确认附加离散证据推翻了原小正边界；不能据亮纹断言它就是错误来源。
- CC，pair_0030906，Melbourne，fold4，VV72/VH80：较直立长亮块，VV 背景颗粒更明亮，VH 相对暗。增量同样偏向错类2（-0.204841），但原 margin +1.204591 足够大，仍正确且 margin +0.999750。说明持续正确也不等于模块改善了证据。
- WW，pair_0031200，Mombasa，fold0，VV80/VH64：倾斜长亮块，VH 内部强度更不均，VV 窗外有较亮颗粒。margin -0.461552 -> -0.635817，std 没有纠正原混淆，反而增强类2相对优势。

两极化窗口尺寸保持各自记录值，不强行统一画布。粗 layer3 map 和卷积 RF 不能精确定位 body/bg；上述视觉事实不等于物理噪声/纯船体机制。保留的条件性证据是：空间 std 可以改变 GAP 的类别分差，但其方向并不保证正确，较小的原边界尤其可能被推翻。本复验不提供新的调参建议，不追加消融。

## 验证、成本与产物

- 仅重建 6 个 source View、18 个候选 Ridge 以取得固定 W；无新 attention 或可训练池化模块。原层特征、标签、split 均未改。
- 11390 张已有 native PNG，9 个尺寸桶，716 个小 batch；未重读整景、未检测。GPU 峰值已分配 216366592 bytes（约206 MiB）。
- fresh GAP 对旧 layer3 缓存最大误差 7.03e-6；fresh 原通道 affine 对 A 最大误差 4.58e-6，top1 变化 0。delta 全有限，完整覆盖；输出逐例 score=base+lambda*delta 重放通过。
- 主执行计时 52.61 秒，其中 W/基线准备 3.64 秒，原窗提取/验证 48.59 秒，源选择/汇总 0.21 秒；分析脚本约1.33秒，不含人工回看与进程导入开销。
- 代码/协议：run_oldsplit.py、analyze_oldsplit.py、PROTOCOL_ZH.md。
- 缓存/核对：fold*_reader.npz、delta_features.npz、validation_maps.npz、baseline_verification.csv、historical_baseline.csv、gap_verification.csv、reader_verification.csv、invariants.csv。
- 结果：alpha_candidates.csv、lambda_candidates.csv、selections.csv、predictions.csv、metrics.csv、pooled.csv、class_effects.csv、acquisition_effects.csv、paired_intervals.csv、review_cases.csv、case_*.png、summary.json、analysis_summary.json、run.log。

任务已结束，本分支启动的计算进程均已退出。只写独立 review/oldsplit 目录；父文件未编辑，无 Git 操作。由主代理统一归档。
