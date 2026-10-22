# Heisenberg审阅入口

由主线程转达。本目录独占写入，不动父目录。

`run_transfer.py`实现唯一定义，先读父crop_pad/masks_for/pool，三fold源W直接读取，无fit/backward。`analyze_transfer.py`只做固定预测分析和原图。

- `std_features.npz`: instance_ids(N,),valid(N,2),mean_pol(3,N,2,8),delta_pol(3,N,2,8),delta(3,N,8)，N=15251。fold/target/pol/class轴；delta=sum_pol(delta_pol)，每个pol内部已C4均值。mean_pol=W投影的有效面积均值，不含截距；invalidpol两项0。
- `window_coverage.csv`: 每目标/极化原尺寸、裁补位置、像素保留数、validflag、输入footprint比例。
- `mean_delta.csv`: 每fold/目标/极化8维mean及8维delta，无标签。
- `source_statistics.csv`: 每fold原train/val/test索引上的mean/std/分位等无标签统计，非选参用特征筛选。
- `lambda_candidates.csv`,`selections.csv`: 3fold×5lambda验证风险与3个选择，含0。
- `predictions.csv`: 原test4744行，fold/instance_id/row_index/真类/A与std类别/s0..7/b0..7/delta0..7/所选lambda/采集信息。
- `metrics.csv`,`pooled.csv`,`class_effects.csv`,`acquisition_effects.csv`,`paired_intervals.csv`: A与std固定比较。
- `gap_verification.csv`,`reader_verification.csv`,`invariants.csv`,`summary.json`: 原weightedmean重放、原A与线性W一致性、mask加权不变性及成本。
- `validation_maps.npz`: 少数原样本响应v/r/权重，不存全量F。
- `replay_checks.py`,`independent_map_replay.csv`: 独立NumPy验证map/mask联合置换、类公共场、常量、按类不同的零权重cell扰动；后者专门避免“扰动被类中心化抵消”造成的弱检查。
- `review_cases.csv`,`case_*.png`,`FINDINGS_ZH.md`: 实际原图回看及结果边界。
