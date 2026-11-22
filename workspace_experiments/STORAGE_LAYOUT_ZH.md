# OSS2 实验存储位置

本轮实验于 2026-11-22 迁入 `D:/SAR_Experiments/cross_port_sar_ship_ATR`。所有后续实验输出、日志、缓存、检查点、分析结果和代码交付包均写到该目录。

| 目录 | 内容 |
| --- | --- |
| `oss2_three_arm_experiment_20261122/` | 三组对照、配置、训练记录、检查点和评估结果 |
| `oss2_h_tuning_20261122/` | 独立的 H 验证集调参及案例分析 |
| `data_qualification_audit_20261122/` | 数据资格核验、Q 掩码和开发划分 |
| `port_prototype_model_20261121/` | 当前完整模型代码、配置、测试、代码说明和既有验证产物 |
| `STAGE1_MULTISCALE_DISENTANGLEMENT_DESIGN_ZH.md` | 第一阶段设计文档 |
| `STAGE2_H_MULTIPORT_EVIDENCE_DESIGN_ZH.md` | 第二阶段 H 设计文档 |
| `literature_review_20261122/`、`atom_coverage_audit_20261122/`、`native_canvas_audit_20261122/` | 两份设计文档引用的文献复核与几何统计 |
| `model_weights/` | 已校验的官方初始化权重和下载记录 |
| `temp/`、`python_cache/`、`torch_cache/`、`cuda_cache/`、`matplotlib_cache/` | 进程临时文件及运行缓存 |
| `migration_originals_local/` | 路径迁移前的清单、配置和实验账本 |

模型源代码、两份设计文档及其相关统计依据现统一以本 D 盘目录为准。训练入口和辅助脚本从同级目录定位模型，后续代码和设计修改均在这里进行。通过 D 盘包装器运行实验，避免依赖库默认将缓存写入用户 C 盘目录：

```powershell
& 'D:/Program_files/anaconda3/python.exe' 'D:/SAR_Experiments/cross_port_sar_ship_ATR/run_on_d.py' 'oss2_three_arm_experiment_20261122/run_comparison.py' --train-only
```

迁移时已复制并逐文件校验 2628 个实验文件。1318 条清单记录只改存储路径，样本、标签、划分及训练参数保持原值；11 个已完成步骤的输出哈希一致。2263 份训练/验证所需缓存经输入字节和张量精确校验后重新建立路径键。迁移中断的 `seed29/disentangle_stage1` 单独保留原记录，并从该阶段起点重跑，其余已完成步骤直接沿用。

随后按用户要求迁移了 354 个模型目录文件，以及 107 个设计依据文件。第二个种子的解耦训练与分类头验证完成后，在刚开始的 H 阶段切换代码根目录；14 个已完成步骤继续复用。模型 Python 文件保持原字节，训练入口只改默认代码路径。两份设计文档和代码说明中的相关链接已更新。

原 E 盘代码文件已移入 `relocation_source_backups/port_prototype_model_20261121_before_relocation/` 作为迁移前备份；旧 E 盘目录因进程占用暂留空目录。实验不依赖该旧目录。公共 Git 归档工作副本仍在原 E 盘仓库位置，公开归档从 D 盘活动代码读取。

自动审批审查拒绝删除已迁移的 C 盘副本，返回原因为策略阻止。因此旧实验目录和旧权重副本暂时保留，后续运行使用 D 盘路径。历史日志和旧检查点中的 C 盘路径是原始运行记录；活动清单、配置、账本和输出目录已切换到 D 盘。

完整迁移证据见本地 `migration_receipt_local.json`、`relocation_rebase_receipt_local.json`、`weights_migration_local.json` 及缓存重建报告。它们包含本机路径或私有数据映射，不加入公共代码归档。
