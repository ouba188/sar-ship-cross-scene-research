# 审阅入口与输出Schema

供主线程转达Heisenberg；本任务不修改任何父目录或其他代理文件。

- `PROTOCOL_ZH.md`：运行前固定三臂、T、lambda候选、数据/精度/分析范围。
- `run_pool.py`：提取和确定性源验证选lambda，不调用fit/backward/optimizer。
- `analyze_pool.py`：固定预测汇总、配对区间、原图复核。
- `delta_features.npz`：`instance_ids (5695,)`、`arms (3,)`、`delta (6,5695,3,3)` float64，轴依次fold/target/arm/class。arm顺序class_residual、contrast_residual、contrast_std；class顺序bulk/fishing/general。六fold方向均来自各自保存outer W。
- `validation_maps.npz`：仅前次四例各极化各C4的本测试fold v/r/三臂delta及输入尺寸，无全量F；供独立重放。
- `gap_verification.csv`：每图每极化重提GAP与缓存差异。
- `reader_verification.csv`：每fold重提GAP解析A的误差、top1差异；最终A仍用原缓存。
- `invariants.csv`：各批/旋转的均值、uniform、common-field、joint-permutation、constant响应检查；包含独立NumPy回放。
- `lambda_candidates.csv`、`selections.csv`：每fold×arm×lambda候选balanced NLL及18个源val选择。
- `predictions.csv`：每arm5695行，字段含instance_id/fold/arm/selected_lambda/class_id/prediction/base_prediction/outcome/net/port/pass_group/mmsi、s0..2/b0..2/delta0..2。
- `metrics.csv`、`pooled.csv`、`class_effects.csv`、`acquisition_effects.csv`、`paired_intervals.csv`：逐折/全体/类别/采集及固定预测区间。
- `sources.csv`、`timing.csv`、`summary.json`：source数据路径、时间、样本数与检查摘要。
- `diagnose_pool.py`及`class_common_diagnosis.csv`、`validation_map_diagnosis.csv`、`residual_diagnosis.csv`：固定W类公共分量是否已近零，以及各真实类别/结局的残差分差变化；无新拟合。
- `review_cases.csv`及`case_*.png`：三臂四种结局与前次四例；无某结局如实记录。

审阅重点：原map先线性投影再每类空间中心化，contrast再每位置类中心化；每pol/C4分别非线性池化后求和/平均；不含h0项；std为ddof0；A独立不变；lambda只源val选择并含0；位置置换不变说明不是几何排列模型。
