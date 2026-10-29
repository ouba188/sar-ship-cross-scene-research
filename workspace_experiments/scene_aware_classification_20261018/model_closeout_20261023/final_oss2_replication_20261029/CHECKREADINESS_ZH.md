# 准备完成，等待主代理 GPU 检查

2026-10-29。本目录内独立 OSS2 runner 已实现；复用私有分支 model.py，但未修改或运行私有分支文件，也未使用其正在训练的权重。

已实际执行并通过：

- 5 个标准库 unittest：raw OBB 的原 width/height/angle、不按文件存在推断有效极化、四类重复船补偿权重与标签拒绝、Python语法、runner/check模块禁止 torch/numpy/sar_m123 导入检查。
- converter：1519条 stack 转 current manifests；source_fit478、source_adapt241、validation228、selection194、evaluation378、references32。
- checkreadiness：核对 sample_id 与各级索引、四类映射、原角色、MMSI/pass/scene、极化计数、五角色 id/MMSI/pass 分离、32参考来自fit且每类8个独立MMSI、权重和与160/96步预算；仅抽样检查 PNG 路径存在，不打开原图。

未执行：torch 导入、权重反序列化、numpy 大数组读取、cache构建、GPU检查、训练、推理、原图案例分析、Git操作。`cpu_readiness.json` 的 passed 仅指轻量 CPU 元数据检查，不代表模型检查通过。

主代理审查重点：`run.py` 的 cache/train_one/select_one/evaluate，`tests_gpu.py` 的实际四类权重切片/梯度/保存/native对照，以及 `PROTOCOL_ZH.md` 的命令和目标边界。父目录创建已由共享 save 与本地 write 明确处理。

GPU空闲后执行 `python -B run.py all` 可一次完成 cache→check→公开6链；也可先分别运行 cache/check，再按臂/seed/阶段启动。两个开发角色仅保留可用信息，主评价固定378条，不选checkpoint。所有待执行检查失败时应定位具体接口问题，不能将本准备收据替代 GPU check。
