# OSS2 固定权重 OBB 鲁棒性实验：准备状态

**完成更新：31条件、三个种子已完成推理与原图审阅，辅助审计列也已修复。当前结论见 `FINDINGS_ZH.md`；下文为保留的启动前历史记录，不代表实验仍在等待执行。**

2026-11-11。**原23条件的 CPU 几何准备和小批模型检查已完成。按最新要求新增独立长/宽8条件，代码及增量合并检查已通过，尚未生成这8组几何。共31条件的三种子主实验待主代理明确释放 GPU 后执行，本轮不跑 pipeline。**

## 当前状态与边界

- 仅写入本目录；未修改正式模型、其他实验或 Git；未训练模型、未使用 GPU。
- 资源收紧通知前启动过 CPU 几何准备；随后立即停止自有 PID 11528，保留当时的 10 组文件。侧边 PID 61160 结束后，主代理明确允许继续 CPU 准备，遂完成剩余工作；本次 CPU 进程 PID 37052 已正常退出。
- 全体 378 查询、726 有效极化视图的 zero mask 与原始 `slots`、`local_slots` 缓存逐元素一致，最大误差均为 0。23 组几何和 50,094 条逐区域记录全部完成，见 `protocol.json`、`geometry/`、`mask_audit.csv`。
- CPU seed17 前4条查询的干净分数最大复现误差为 1.2397766e-5，预测完全一致。浅层、内部、深层 M2 的 VV/VH 六个 hook 均记录到变化；横移20%测试的分数最大变化为0.6404228。M3 输入、图像特征对象、可用性、footprint 和参考 slots 保持不变。见 `smoke.json`、`smoke_zero_replay.json`。这是小测，不是三种子全测试集复现。
- 标准库小型几何检查和 Pillow 小图预览也完成，相关进程未导入 torch。`light_checks.json`、`light_native_audit.csv`、`light_grid_audit.csv` 保存证据；三张原图预览已查看。
- 当前 GPU 归主代理参数共享/专属六条五阶段训练及分析。只有主代理发出释放通知后，才运行本实验；此目录没有自动等待/抢占队列。
- 新增 `length_m10/p10/m20/p20`、`width_m10/p10/m20/p20`。`prepare` 自动复用已有完整几何及审计，只生成缺失条件；每生成一组就合并保存审计，保留原条件所有字段值，失败/中断重试不重复追加同条件。`extension_checks.json` 验证实际计划为复用23、追加8，并验证尺寸隔离、合并和重试；此检查未导入 torch、未生成新几何、未运行推理。

## 已核实的接口

`staged.Network.forward` 从 `c['local_slots']` 取得 20×20 区域，从 `a['slots']` 取得 10×10 区域。前者用于浅层 M2 池化以及最终指导读出 mask，后者用于内部 M2 写回和深层区域池化。

M3 使用独立的 `a['desc']`、`a['mass']`、`a['write']` 和固定源参考。M2-only 在每个种子上先从干净输入生成并冻结 `fixed_delta`；随后仅替换查询的两组 slots，保持 M3、M1 输入、图像特征、有效观测、参考和标签不变。forward hook 小测试已确认低层、内部、深层三处实际收到改变后的区域。

原始 `input.prepare_view` 的 OBB 参与原生裁剪定位，因此所有扰动均在原图按原 OBB 完成裁剪/补零之后发生。只改 native160 坐标中的 geometry，不重新调用变更 OBB 后的裁图过程，不调整置信度或可用性。VV/VH 分别按自身原 OBB 轴施加相同符号和比例，可能仅一支可用，保留实际可用性。

## 固定条件

主实验 31 条件，全部 query-only、M2-only：

- zero。
- 沿长轴平移 ±5%、±10%、±20% 原始长度。
- 沿横轴平移 ±5%、±10%、±20% 原始宽度。
- 长宽同时等比缩放 ±10%、±20%，中心和角度不变。
- 中心旋转 ±5°、±10°、±20°，长度和宽度不变。图像坐标 x 向右、y 向下，正角度为顺时针。
- 独立长度 ±10%、±20%，宽度、中心和角度不变。
- 独立宽度 ±10%、±20%，长度、中心和角度不变。两种独立尺寸扰动均沿原始 OBB 坐标轴，不因边长变化而重新选择长轴。

总计 31 × 3 种子 × 378 = **35,154 条查询推理**，含各自零误差。种子 17、29、43，原始 Full 最终权重 `readout/final.pt`，32 个原始源参考。构造网络时预训练参数设为 None，然后严格恢复完整 checkpoint，不下载或重新加载 ImageNet 权重。主代理提供的现有预训练文件无需用于本任务的完整 checkpoint 加载。

**本轮明确不运行 pipeline。** 既有可选实现保留，但不列入执行命令、预算和本轮待完成项。M3 始终固定为原始干净查询与参考产生的 delta。

## 小船检查与已有证据

726 有效视图的原始 OBB 中位长度 22.0933 px，中位宽度 7.7320 px。横轴 10% 约 0.7732 px、5% 约 0.3866 px；沿长轴 5% 约 1.1047 px。全部使用连续浮点几何和原模型的软 mask/像素面积积分，不整数取整。

轻量检查选取最窄视图 `002753/VH`、中位宽度视图 `000214/VV`、最长视图 `002744/VH`，不是代表总体比例的随机样本。每例 23 条件；非零条件均有非零原生 body 软区域 L1 变化。已有部分真实 10×10/20×20 缓存也检查了三例的 5% 位移，例如最窄船横移 5% 时，10×10 body 只有 3 个格子改变，L1≈0.00290，而20×20 body 的 L1≈0.10067。不能把“变化小”或图上不明显视为没扰动，也不能把完全零变化视为鲁棒。

轻量检查本身没有调用 M2；之后完成的独立 CPU 小测验证了实际 M2 接入。全部样本的 native 面积、原区域覆盖、soft IoU、越出 footprint 面积、网格 L1/变化格数已写入 `mask_audit.csv`，按 body/end/side、VV/VH 记录。非零条件中，body 在两级网格同时无变化的视图数为0；+5° body soft IoU 中位数0.87813。这里的覆盖相对原始 OBB，不是分割真值或真实船体像素标签；不能由此推断鲁棒性能。

## 运行命令

必须在主代理明确通知资源释放后手动执行。PowerShell：

```powershell
$obbPython = 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe'
$obbRoot = 'E:/临时会话/paper_draft_20261108/obb_robustness_20261111'

# 放行后只补8个缺失条件并合并审计；原23组和既有小测复用。
& $obbPython -B -X utf8 "$obbRoot/experiment.py" --prepare --resources-released

# 仅在主代理明确释放GPU后执行；默认不运行附加pipeline。
& $obbPython -B -X utf8 "$obbRoot/experiment.py" --run --gpu-released --batch 12

# 汇总三种子配对结果与原始VV/VH/真实参考图。
& $obbPython -B -X utf8 "$obbRoot/analyze.py" --resources-released

```

低内存下可运行的检查（不导入 torch）：

```powershell
& 'C:/Python314/python.exe' -B -X utf8 "$obbRoot/light_geometry_check.py"
& 'C:/Python314/python.exe' -B -X utf8 "$obbRoot/make_preview.py"
& 'C:/Python314/python.exe' -B -X utf8 "$obbRoot/check_extension.py"
```

入口采用延迟导入，`--help` 或未携带放行参数时不会导入 torch。放行参数仅记录主代理授权，不自动推断 GPU 空闲即获准运行。

## 资源预估与待完成项

主运行批量12；暂估主机 RAM 2–4 GB、GPU 1–3 GB，受运行时库和 matcher 临时张量影响。CPU 几何准备改为只加载 mask/footprint，避免加载全体深层图像缓存；23条件重建实测227.9秒（不含导入和数据准备）。CPU小测只加载4查询+32参考，matcher及三次前向合计1.74秒。GPU耗时尚未标定，不以CPU小测冒充GPU预算；首个完整条件可用于估时。

待完成：放行后补8组几何（新增17,424行审计，完整31组共67,518行）；三种子全体zero分数复现；31条件M2-only推理；逐样本 scores、OA/BA/逐类及区域变化汇总；六类扰动的救回/误伤/持续对/持续错原图与实际源参考人工查看和解释。`analyze.py` 已扩为六幅条件曲线并为独立长度、宽度选择案例，但尚未在完整结果上执行；运行后仍标记待视觉核查，不自动宣告实验完成。零收益/负收益和失败复现均应保留，不据此修改正式模型或筛除样本。

当前预览：`preparation.html` 和三张 `preview_*.png`。它们仅展示真实原图和几何尺度，不是性能结果。最终结果入口为 `index.html`，原始分数为 `seed_{seed}/m2_{condition}.npz` 和 `scores.csv`，汇总为 `RESULTS.md`、`summary.csv`、`per_class.csv`、`stratified_changes.csv`、`mask_summary.csv`，案例记录为 `cases_local.json`。
