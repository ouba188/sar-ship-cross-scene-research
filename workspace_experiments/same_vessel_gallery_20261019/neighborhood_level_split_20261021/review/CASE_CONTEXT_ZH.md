# 4002 / 22543 案例来源核查

2026-10-21，独立只读核查。只读取旧报告、CSV、NPZ 和现有图像；未训练模型。

## 结论

4002 是旧弱诊断 full_global 相对 body_global 的救回，22543 是同一弱诊断比较的误伤。它们不能移植为强 baseline 被新方法救回的证据。旧强 baseline 三个种子中，4002 已全部判对，22543 全部判错；新方法若仍判对 4002，只能记持续正确。若将来救回 22543，必须由新旧匹配分数确认。

## 旧弱诊断头

来源：`../../crude_input_controls_20261021/FINDINGS_ZH.md`、`case_index.csv`。真实类别都是 5（原油轮），0 为散货，4 为集装箱。分差为真实类分数减最大错误类分数，不是概率。

| 观测 | 身份折 | full_global 预测 / 分差 | body_global 预测 / 分差 |
|---|---|---|---|
| 3999 | 0 | 5 / +0.144578 | 5 / +0.077651 |
| 4002 | 0 | 5 / +0.282870 | 4 / -0.084717 |
| 22542 | 1 | 0 / -0.346743 | 0 / -0.185519 |
| 22543 | 1 | 0 / -0.063557 | 5 / +0.206310 |

这些是分别拟合的 260 维冻结 stem + Ridge 诊断头，没有目标支持集。旧报告 full_global OA 35.139%，强 baseline OA 40.964%；不得跨读出能力将差异记为模块收益。案例由 `analyze_results.py:case_panels` 按诊断头对错及分差排序选取，属于事后示例，不是独立确认集。

## 直接读取旧强 baseline

读取 `../../paired_relation_probe_20261020/scores_f{fold}_baseline_s{seed}.npz` 中 `rows` 和 `scores`；以 `../../conditional_evidence_pilot/cohort.csv` 的零起始行号映射 obs_id，逐个种子 argmax 并重算分差。不是拿 obs_id 直接当数组下标。3999/4002 对应行 1089/1092、MMSI 249838000；22542/22543 对应行 7454/7455、MMSI 447143000。

| 观测 | 对应折 | seed 0 预测 / 分差 | seed 1 预测 / 分差 | seed 2 预测 / 分差 |
|---|---|---|---|---|
| 3999 | 0 | 5 / +0.664782 | 5 / +0.549947 | 5 / +0.540063 |
| 4002 | 0 | 5 / +0.585619 | 5 / +0.826375 | 5 / +0.704779 |
| 22542 | 1 | 5 / +0.043814 | 5 / +0.192020 | 5 / +0.146303 |
| 22543 | 1 | 0 / -0.710024 | 0 / -0.565945 | 0 / -0.543686 |

4002 同船两次在强 baseline 都正确；22543 与同船 22542 构成强 baseline 的正确到错误变化。三个种子不等于三份独立采集证据。

## 实际查看的图像事实

已查看旧 `case_context_rescue.png`、`case_context_harm.png` 两张原生像素面板，以及下列四张原始 PNG。面板采用 160 像素裁剪/补零、无缩放、固定 0–255 显示。下列原 PNG 均为 128×128，黑色补零不是观测到的背景。

- 4002，Jebel Ali，2025-12-21 14:23:48：VV 为近水平的亮长条主体，周围有明显纹理；主体左侧附近可见向上、向下延伸的响应。VH 主体仍亮，周围整体更暗，对应向上延伸仍可见。严格框内图删去大部分延伸；框外图保留部分跨框结构。与面板中的 3999 相比，4002 的 VV 周围更亮，但两次方向、图像内容和地点同时不同。
- 22543，Jebel Ali，2025-10-25 02:14:38：VV 近水平亮主体附近有较长的近竖直斜线穿过/邻接，另有较弱近水平延伸，均越出目标 OBB；VH 主体清楚，跨框延伸较弱。框内输入留下紧凑亮主体，框外输入仍能看见长条结构。
- 这些只能命名为亮度、纹理和延伸形状。不能据图确认尾迹、海况、真实雷达散射变化、船艏艉或错误成因；OBB 不是船体精确分割。PNG 来源为逐景逐极化显示拉伸，不能当作定标 backscatter。

- 4002 VV: `E:\SAR_AIS_T_shipchip_dataset\v5_context_dynamic_64_128_return_20260915\images\Jebel Ali\crude_oil_tanker\S1C_IW_GRDH_1SDV_20251221T142348_20251221T142417_005551_00B113_F32B_VV_UTM_8bit__r01_c02__b45be7939ce91bf3.png`
- 4002 VH: `E:\SAR_AIS_T_shipchip_dataset\v5_context_dynamic_64_128_return_20260915\images\Jebel Ali\crude_oil_tanker\S1C_IW_GRDH_1SDV_20251221T142348_20251221T142417_005551_00B113_F32B_VH_UTM_8bit__r01_c02__a58ddec49ecb93c9.png`
- 22543 VV: `E:\SAR_AIS_T_shipchip_dataset\v5_context_dynamic_64_128_return_20260915\images\Jebel Ali\crude_oil_tanker\S1C_IW_GRDH_1SDV_20251025T021438_20251025T021503_004712_0094F6_EC36_VV_UTM_8bit__r01_c02__b3248b618360b672.png`
- 22543 VH: `E:\SAR_AIS_T_shipchip_dataset\v5_context_dynamic_64_128_return_20260915\images\Jebel Ali\crude_oil_tanker\S1C_IW_GRDH_1SDV_20251025T021438_20251025T021503_004712_0094F6_EC36_VH_UTM_8bit__r01_c02__a4240c054362ecc5.png`

## 对新实验的约束

新 Braw / Ccentered / Dcentered+mean 如使用这些案例，必须重新读取同一 fold/seed 的实际预测。图中结构可用来提出周围空间结构与均值拆分假设；尚不能证明均值导致误判，也不能证明中心化必会修复。只有匹配控制的输出能界定哪些信息被模型使用及其收益，且仍需区别显示域作用与物理机制。

