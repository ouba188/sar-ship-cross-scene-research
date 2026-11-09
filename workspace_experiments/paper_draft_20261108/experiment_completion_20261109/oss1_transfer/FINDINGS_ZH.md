# 当前工作版OSS2到OSS1冻结迁移

原图前向；不训练、不更改源参考、不使用目标标签选择检查点。三次结果是末端读出训练重复，共用上游seed17。
historical921沿用已有队列；另一子集排除与OSS2评价集同MMSI的71条。两者嵌套，不是两次独立验证。

| 队列 | 方法 | 数量 | OA | BA | NLL |
|---|---|---:|---:|---:|---:|
| historical921 | class | 921 | 59.862 ± 0.125 | 49.186 ± 0.246 | 1.2252 |
| historical921 | mean | 921 | 59.645 ± 0.332 | 49.005 ± 0.201 | 1.2280 |
| historical921 | content | 921 | 59.573 ± 0.332 | 49.397 ± 0.398 | 1.2336 |
| historical921 | no_m3_refit | 921 | 59.537 ± 0.125 | 48.620 ± 0.261 | 1.2193 |
| no_oss2_evaluation_ship_overlap | class | 850 | 58.902 ± 0.068 | 48.958 ± 0.223 | 1.2675 |
| no_oss2_evaluation_ship_overlap | mean | 850 | 58.706 ± 0.311 | 48.789 ± 0.264 | 1.2703 |
| no_oss2_evaluation_ship_overlap | content | 850 | 58.588 ± 0.235 | 49.187 ± 0.389 | 1.2756 |
| no_oss2_evaluation_ship_overlap | no_m3_refit | 850 | 58.588 ± 0.118 | 48.400 ± 0.314 | 1.2612 |

继承上游单次模型的OA/BA为60.695/50.610。当前末端组合未超过该上游，不能把相对局部替换的收益写成相对所有旧版本的整体升级。

mean/content保留其他模块；no_m3_refit为源端末阶段配套移除对照。它们不是Plain R50，也不是完整流程重新训练消融。
评价标签仅用于本表与后验案例整理；未用于预测。原图与源参考见case_records.json及index.html。