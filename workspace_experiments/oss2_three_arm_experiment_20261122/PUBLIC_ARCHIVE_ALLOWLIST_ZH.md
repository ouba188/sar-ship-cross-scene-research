# 本轮公开归档与代码 ZIP 白名单

状态：脚本支持增量归档；已有运行中代码包保留，最终结果按显式白名单补齐。`archive_public_experiment.py` 默认只预览；真正复制需要 `--apply`，新建代码包另传 `--zip`。已有 ZIP 一律拒绝覆盖。

## 目标目录

仓库根为 `E:/临时会话/_sar_ship_cross_scene_incremental_git`。同步位置固定为：

- `workspace_experiments/run_on_d.py`、`workspace_experiments/STORAGE_LAYOUT_ZH.md`：D 盘启动器和存储说明；保持启动器与实验目录的相对层级。
- `workspace_experiments/port_prototype_model_20261121/`：代码根 `D:/SAR_Experiments/cross_port_sar_ship_ATR/port_prototype_model_20261121` 的显式源码清单。
- `workspace_experiments/oss2_three_arm_experiment_20261122/`：本轮公开实验工具、文档及聚合结果。
- `workspace_experiments/oss2_h_tuning_20261122/`：独立 H 融合验证调优工具、计划及聚合结果。
- `workspace_experiments/oss2_disentangle_tuning_20261122/`：分类路径衔接诊断、有限辅助权重对照及聚合结果。
- `workspace_experiments/seed29_diagnosis_20261122/`：种子29错误分层、优化审查和优先试跑工具。
- `workspace_experiments/two_stage_minimal_20261122/`：两份现有设计文档原归档目录。

脚本不修改源代码树或私有实验目录；不删除文件、不复制整个目录、不执行 Git add/commit/push。目标文件若存在不同的未提交修改则拒绝覆盖；其他仓库改动保持原状。新 ZIP 由内存中的同一白名单文本组成，不能通过扫描已有仓库目录把历史私有文件带入。

当前实验源根为 `D:/SAR_Experiments/cross_port_sar_ship_ATR`，由归档脚本自身位置推导；两份设计文档也从该根目录读取。启动器与存储说明通过同样的真实 ID 和文本内容检查。迁移 receipts、路径映射、备份目录及旧 C 盘副本不加入白名单。

## 代码包必需清单

源码文件名全部静态列在脚本 `MODEL_MODULES`、`TWO_STAGE_MODULES`、`TEST_MODULES`、`MODEL_FILES` 和 `TEMPLATE_CONFIGS` 中；文件夹新增内容不会自动进入归档。

- 根文档/包定义：`pyproject.toml`、`README_ZH.md`、`REFERENCE_NOTES.md`、`TWO_STAGE_README_ZH.md`、`TWO_STAGE_VERIFICATION_ZH.md`、`VERIFICATION_ZH.md`。
- `sar_port/*.py`：`__init__`、`backbone`、`cli`、`data`、`disentangle`、`exchange`、`geometry`、`matching`、`model`、`priors`、`protocol`、`prototypes`、`region_ops`、`selection`、`supervision`、`teacher`。
- `sar_port/two_stage/*.py`：`__init__`、`bank`、`cache`、`cli`、`contracts`、`data`、`geometry`、`h`、`stage1`、`stage1_losses`、`training`。
- `tests/*.py`：`test_cli_contract`、`test_disentangle`、`test_geometry_exact_coverage`、`test_matching`、`test_prototypes`、`test_supervision`、`test_teacher`、`test_two_stage_bank`、`test_two_stage_cache`、`test_two_stage_comparison`、`test_two_stage_data`、`test_two_stage_empirical`、`test_two_stage_encoder`、`test_two_stage_h`、`test_two_stage_h_batch`、`test_two_stage_pipeline`。
- `scripts/*.py`：`check_cli`、`check_model`、`check_native`、`make_ablation_configs`。
- 公开模板 JSON：`configs/{two_stage_native,baseline,two_layer,full,bootstrap}.json`，以及 `configs/ablations/` 下脚本静态列举的 25 个模板。这些是代码分发模板；本轮含真实查询/供体 ID 的 `oss2_three_arm_experiment_20261122/configs/` 完全排除。

## 主实验必需清单

文档：`EXPERIMENT_PROTOCOL_ZH.md`、`CONTROLLED_QUALIFICATION_REVIEW_ZH.md`、`DATA_COMPOSITION_ZH.md`、本清单。

工具：`prepare_experiment.py`、`run_comparison.py`、`progress.py`、`download_official_weights.py`、`smoke_real_supervision.py`、`make_intervention_gallery.py`、`record_gallery_review.py`、`summarize_data_composition.py`、`collect_results.py`、`collect_subgroups.py`、`branch_diagnostics.py`、`verify_finished_comparison.py`、`review_cases.py`、`archive_public_experiment.py`。

现有聚合：`aggregate_data_composition.json`、`aggregate_intervention_audit.json`。

预检摘要只做指定字段投影，生成 `preflight_summary_public.json`。原 `preflight_summary.json` 的逐样本排除清单含真实 ID，不复制；公共版本只保留排除总数、原因计数、划分/类别/画布计数、监督覆盖、实验范围和预算等明确字段。

## 主实验可选结果

以下结果尚可能不存在。脚本逐项列为 `missing_optional`，不会伪造或以空结果填补：

- `FINDINGS_ZH.md`。
- `aggregate_training_integrity.json`、`aggregate_branch_diagnostics.json`、`aggregate_h_margin_diagnostics.json`。
- `aggregate_validation_subgroups.json`、`aggregate_test_subgroups.json`。
- `collected/validation/aggregate_summary.json`、同目录 `aggregate_summary.csv`。
- `collected/test/aggregate_summary.json`、同目录 `aggregate_summary.csv`。
- `real_smoke_summary.json`、`real_smoke_original_loop.json`（标注为运行/梯度检查，不能充当正式性能结果）。

## 独立调优目录

必需：`TUNING_PLAN_ZH.md`、`scan_validation.py`、`test_scan_validation.py`、`collect_fusion_scan.py`、`evaluate_selected_gamma.py`。

可选：`FINDINGS_ZH.md`、`selected_gamma.json`、`aggregate_fusion_scan_summary.json`、`fusion_candidate_table.csv`，以及 `final_seed17/`、`final_seed29/`、`final_seed43/` 下各自的 `aggregate_validation_fusion_scan.json`。公共 JSON 不得含原检查点绝对路径；检查点哈希和相对公共扫描文件名可以保留。未来调优测试汇总需另行显式加入，不按通配符自动收录。

设计文档：从工作区根同步 `STAGE1_MULTISCALE_DISENTANGLEMENT_DESIGN_ZH.md` 和 `STAGE2_H_MULTIPORT_EVIDENCE_DESIGN_ZH.md` 到原 `two_stage_minimal_20261122/` 目录。

## 排除项与内容检查

种子29目录必需文件为 `FINDINGS_ZH.md`、`OPTIMIZATION_REVIEW_ZH.md`、`aggregate_optimization_review.json`、`aggregate_error_diagnosis.json`、`analyze_errors.py`、`run_seed29_pilot.py`；可选结果为 `aggregate_seed29_aux_pilot.json`。`cases_local/`原图、图板、样本列表和队列重排前状态均排除。

补充必需工具 `run_post_training_diagnostics.py`：只公开串行队列程序；队列状态、日志与锁留在本地。

解耦调优目录必需文件为 `TUNING_PLAN_ZH.md`、`STAGE1_HANDOFF_DIAGNOSIS_ZH.md`、`aggregate_stage1_handoff_diagnosis.json`、`run_aux_scale_ablation.py`、`test_aux_scale_ablation.py`、`analyze_head_reuse.py`。可选文件为 `test_head_reuse.py`、`aggregate_aux_scale_validation.json`、`AUX_SCALE_RESULTS_ZH.md`、`aggregate_head_reuse_validation.json`。新对照的真实配置、冻结计划、运行状态、逐图缓存与训练产物保持本地。

排除真实实验 configs、manifest、文件名/目录名含 `_local` 的内容、原图、Q 掩码、图库、权重、缓存、逐样本预测、逐样本资格记录、donor field、运行状态和 ledger、`CASE_REVIEW_LOCAL.md`，以及所有未列出的文件。只接受白名单文本后缀；不递归搜集模型文件。

真实样本 ID 检查使用 `pub2_opensarship2_` 后接数字的正则。代码中用于检查的前缀字面量本身允许存在，不会误判为真实 ID。JSON 还递归拒绝逐样本身份/预测/供体索引字段；文本脚本引用私有输入路径不等于把这些输入归档。遇到身份、原检查点绝对路径或必需文件缺失，整个操作在写入前停止。

## 完成后的运行方式

只预览：`python archive_public_experiment.py`。

主代理确认最终结果后，执行 `python archive_public_experiment.py --apply --zip <全新的代码包路径.zip>`。输出包含每项路径、SHA256、已有目标是否相同、可选文件缺失和阻断原因。ZIP 内增加 `PUBLIC_ARCHIVE_CONTENTS.json` 记录实际清单与缺失项；不会覆盖以前交付的代码包。
