# 两表、OSS1留空与加速计时修订

日期：2026-11-10。本轮仅修改论文与排版，不训练、不修改模型，不触碰侧边24港全量实验。

## 正文改动

- 引言加入表I：影响Sentinel-1船舶表征的主要因素。
- 数据集介绍加入表V：源产品、像素表示和预处理。因内容较多采用双栏宽表，其余表仍靠近对应正文。
- 表格按首次出现顺序编号，共15张；正文引用同步。
- OSS2仅介绍参与实验的IW-GRDH，删除未使用产品分支及其数量。
- 划分名称统一为训练集、验证集、测试集；数据划分与实验协议本身不变，历史开发用途没有被改写为新的盲测。
- OSS1旧分类结果、抽样数量、原有干预结果及摘要/结论中的相应增益撤下。主结果表保留方法行，数值用破折号留空，等待用户重做。原始实验记录未删除。
- 原图9包含旧OSS1结果，当前稿保留图9占位，不改动作者原图像素。需要在OSS1新结果完成后更新该图。
- 图12改单栏；图4维持双栏、图7维持单栏。
- 完整模型延迟采用加速后49.37 ms，正文不再出现129.47 ms或其他旧延迟。保留FP32、32源参考、稳态计时及初始化范围。

## 表V事实核对

1. 24港总库平台为Sentinel-1A和1C，不是仅1C：分别32,184/5,907次观测，698/143个产品，共38,091次观测、841个产品。
   - `dataset_profile_counts.json`
   - `E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv`
2. 约10 m × 10 m是标称像素间距，不是空间分辨率。公开产品XML中的方位向间距存在小数差异。
3. OSS1的Patch为float32幅度，Patch_Cal为float32辐射定标值，随包说明明确为sigma0；Patch_Uint8为增强灰度图。32位数据并非本轮分类输入。
   - `E:/临时会话/ports_root_cause_20261013/dataset_audit/BUNDLED_README_TEXT.md`
   - `E:/临时会话/oss1_joint_geometry_20261017/geometry.csv`
4. OSS2提供幅度、定标和灰度分支，当前分类使用灰度。未把本地定标TIFF尚未闭合的单位推断为已核验的线性sigma0。
   - `E:/临时会话/scene_aware_classification_20261018/oss2_distribution_transfer_20261023/radiometric_source_inspection/LITERATURE_AND_UNIT_REVIEW_ZH.md`
5. 24港地理处理存在不同批次：早期程序有bilinear，服务器分支使用near。故表中不统一写双线性。
   - `E:/临时会话/v5_server_toolkit_20260912/scripts/sentinel1_geocode_manifest.py`
   - `E:/临时会话/asf_ais_first_screen_20260913/server2_zip_pipeline.sh`
   - `E:/临时会话/sentinel_geocoder_extract/disassembly.txt`
6. 已追溯服务器分支直接处理measurement、未应用定标LUT，使用分极化整景2%/98%拉伸后裁片；不能推广为全部841产品都无定标。正文限定为“已追溯服务器分支”。
   - `E:/临时会话/same_vessel_gallery_20261019/crude_input_controls_20261021/radiometry_trace/FINDINGS_ZH.md`
   - 同目录`CODE_EVIDENCE.md`

用户DOCX被占用，采用用户完整粘贴内容并与以上来源核对，未声称已打开该DOCX。

## 加速成本口径

来源：`m3_latency_optimization_20261110/FINDINGS_ZH.md`。
完整前向49.37 ms，batch8整批63.87 ms、125.3张/s；三组各10次。普通骨干/SE沿用三组各15次计时。CUDA Graph仅捕获固定形状的60步transport，不改变精度、权重和迭代数；378条预测类别一致。
加速后完整路径的总峰值显存没有独立记录，不能把旧执行方式的174.0 MiB直接写成新计时配置的测量，因此该格暂留空。未为本次文稿修改新跑性能测试。

## 未变更项

已完成OSS2数值、18组公式、当前模块及训练方法不变；旧实验记录与旧PDF继续保留。作者图中已发现的符号问题仍见`FIGURE_UPDATE_20261110.md`，不擅自重绘。

## 构建验证

IEEE双栏12页，审阅版22页；15表、11幅实际图片及图9占位、18组编号公式、36篇文献。HTML、LaTeX和PDF同步更新。渲染检查无缺图、缺字、公式缺失和页面越界；双栏页面缩略图及第1、8、11页大图已查看。新增宽表与地图使旧H表格超出剩余栏高，已让数据集表VI–VIII在本节内按序浮动，解决越界，未缩小正文或图中文字。
