# 当前完整流程的有限代码追溯

日期：2026-10-23。范围：只读实际调用代码及既有检查记录，不重新提取、训练、推理或重跑历史缓存。本文件区分“代码加载了什么”“已有缓存记录验证了什么”与尚未独立验证的历史来源。

## 1. 可直接用于主报告的结论

1. 当前强基线不是新 layer3 GAP Ridge，也不是历史 R18、64支持或5%支持集模型。旧 **A = 轴向浅层 Ridge + keep_refs 五点采样残差**；旧 **D = separated 剩余区域上下文残差**。它们使用冻结 ImageNet ResNet50 的 **conv1→BN→ReLU，64通道**，不是完整 R50 最后层。
2. 令 U=A+D。强 **B=U+T_res(Standardize(U−mean_class(U)))**。T_res 是 **8→16 GELU→8、280参数**的普通分数校准残差；不是标量温度，不是把A/D拼成16维后融合。当前取用的是 `full_model_integration` 已选 `score_only` 头。
3. 默认 moments 路线是 **B + 类别中心化的“均值支路 + sigma/kappa分布支路”**。attention 对照保留均值输入及双支路结构，用 **h/b 替换 sigma/kappa**，重新训练该读出头；不是在原 moments 模型分数上再叠加 attention。
4. 最新 OBB 实验的 `s_keep` 明确来自已保存 **two_sided_attention** 的分数。新 OBB 使用同一 R50 权重的 **layer1** 区域摘要，再加一个类别中心化残差。代码没有把本轮 OBB 接到原默认 moments 输出上。
5. 全局 attention 的辅助类别方向来自 **layer3，单极化1024维、双极化2048维**；不是 layer4。新 OBB 的辅助方向重新拟合在 **layer1，单极化256维、双极化512维**，不是截取旧 layer3 W。

## 2. 最重要的输入区别

**不能把新 `native_targets` 的160画布直接写成旧A的输入来源。** 二者虽然都调用160尺度的原像素裁切/补零流程，但来源和区域组织不同：

| 路线 | 已核实的实际输入与处理 | 不能混同的部分 |
|---|---|---|
| 旧A/D | `same_vessel_gallery_20261019` 的 cohort、`view_snapshot.json`、原view图像/检测区域；T目标、O其他检测、R剩余区域；调用原尺度 `crop_pad(...,160)`；VV/VH分别处理，沿旧对齐状态禁用不可用VH | 不是本轮aux的 `native_targets` 缓存；A并非只含目标内部像素，也不是纯GAP分类器 |
| 新全局aux / attention | 八类 `prepare_features` / `extract_moments` 的 native目标窗口、160 crop/pad及有效像素footprint；各极化独立，四个90度旋转；layer3加权pool与类别响应读出 | 160是辅助读取画布，不代表旧A与aux逐像素完全相同；并非从旧A浅层描述直接升级到layer3 |
| 新OBBaux | 本轮 `common.py` 原图/几何接口，160画布；真实crop offset、C4区域变换；layer1 stride4/origin0，按采样中心Voronoi面积积分 | 不等于旧layer3的area等分mask；不能把名义相同160当成同一空间读出 |

旧A入口及C4见 [native_axis/probe.py:65](E:/临时会话/same_vessel_gallery_20261019/native_axis_probe_20261020/probe.py:65)、[probe.py:84](E:/临时会话/same_vessel_gallery_20261019/native_axis_probe_20261020/probe.py:84)；原像素裁切/补零定义见 [native_input.py:6](E:/临时会话/same_vessel_gallery_20261019/local_condition_module/native_input.py:6)。aux处理见 [prepare_features.py:35](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/prepare_features.py:35)。新坐标约定见 [OBB/common.py:24](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/obb_dynamic_readout/common.py:24)、[cell_mass:138](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/obb_dynamic_readout/common.py:138)。本次没有重做旧view与新native PNG的全量像素同一性检查。

## 3. 旧A、D及T的实际定义

### A：轴向Ridge加五点残差

- 权重文件为 `C:/Users/25791/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth`，见 [prepare.py:19](E:/临时会话/same_vessel_gallery_20261019/local_condition_module/prepare.py:19)。实际轴向提取只保留 `conv1,bn1,relu`，再额外avg_pool2；每极化灰度复制成RGB并用ImageNet均值/标准差。四个整90度旋转分别完成读出后平均，[probe.py:65](E:/临时会话/same_vessel_gallery_20261019/native_axis_probe_20261020/probe.py:65)。这不是layer1，也没有使用ImageNet最终分类层。
- 轴向输入包含T/O/R区域均值、几何/面积描述、船轴三段的中部/两端均值/两端绝对差。使用 `base+axis`，合计796维；源标准化后再除以维数平方根，`RidgeClassifier` 输出8类分数。定义与选择见 [probe.py:121](E:/临时会话/same_vessel_gallery_20261019/native_axis_probe_20261020/probe.py:121)、[probe.py:132](E:/临时会话/same_vessel_gallery_20261019/native_axis_probe_20261020/probe.py:132)。T/O/R构造见 [masks_for:22](E:/临时会话/same_vessel_gallery_20261019/local_condition_module/prepare.py:22)。
- 五点 `keep_refs` 分支在同一64通道浅层特征上，按船轴/雷达参照做中心及两方向正负偏移采样；保留参照，合并非目标O/R角色。其60个64维描述槽不是60个支持样本，64是通道数。特征域 `grid_sample` 有双线性采样，不能把“原图不resize”扩大为整个流程没有插值。见 [sampled:28](E:/临时会话/same_vessel_gallery_20261019/sampling_preservation_20261020/run.py:28)、[describe:47](E:/临时会话/same_vessel_gallery_20261019/sampling_preservation_20261020/run.py:47)、[readout:91](E:/临时会话/same_vessel_gallery_20261019/sampling_preservation_20261020/run.py:91)。
- 五点头为共享64→8 GELU后展开60×8→8，4368参数，末层零初始化。[Residual:168](E:/临时会话/same_vessel_gallery_20261019/sampling_preservation_20261020/run.py:168)。真正被载入的是 `sampling_preservation_20261020/mse_aligned/head_f{fold}_keep_refs_s{oldseed}.pt`；`frozen_scores` 返回轴向Ridge分数加该头输出，见 [axial_strip/run.py:184](E:/临时会话/same_vessel_gallery_20261019/axial_strip_sequence_20261020/run.py:184)。**调用经过axial_strip目录不等于使用其strip网络。**

### D：separated上下文残差

- 当前D来自 `neighborhood_level_split_20261021` 的 `separated`。读取R剩余区域，而不是邻船类别分数；R排除T和其他检测O。区域像素先减本区域均值，再编码为 `.5+.5*(x−mean_R)*R`，由冻结R50浅层stem提取，四旋转读出平均。见 [encode/descriptor:33](E:/临时会话/same_vessel_gallery_20261019/neighborhood_level_split_20261021/run.py:33)、[extract:50](E:/临时会话/same_vessel_gallery_20261019/neighborhood_level_split_20261021/run.py:50)。
- 每极化均值池化+最大池化得到128维；双极化256维，加2个原区域显示均值与2个valid标记，共260维。头为260→32 GELU→8，8616参数，零末层，直接加到A。源标准化/结构见 [standardize:117](E:/临时会话/same_vessel_gallery_20261019/neighborhood_level_split_20261021/run.py:117)、[Residual:147](E:/临时会话/same_vessel_gallery_20261019/neighborhood_level_split_20261021/run.py:147)。旧D训练使用加权MSE对±1类别目标，不能写成当前200步jointCE；见 [run.py:176](E:/临时会话/same_vessel_gallery_20261019/neighborhood_level_split_20261021/run.py:176)、[run.py:209](E:/临时会话/same_vessel_gallery_20261019/neighborhood_level_split_20261021/run.py:209)。
- `geometry_evidence_gate_20261021/inputs_f{fold}_s{oldseed}.npz` 保存的 `base=A,residual=D` 在后续几何门控训练之前生成；当前加载两者直接相加，**没有加载该目录训练出的几何gate**。见 [geometry/run.py:80](E:/临时会话/same_vessel_gallery_20261019/geometry_evidence_gate_20261021/run.py:80)。

### T：A+D的8维分数校准，不是温度

`full_model_integration/run_integration.py` 的 `load_old` 按obs/instance/class映射旧数据：outer读取上述geometry输入的base+residual；源训练读取 `held_vessel_gate_20261022/oof_f{fold}.npz` 的base+residual。旧seed0/1/2对应新seed17/29/43。见 [run_integration.py:20](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/calibrated_signed_transfer/polarization_readout/full_model_integration/run_integration.py:20)、[load_old:44](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/calibrated_signed_transfer/polarization_readout/full_model_integration/run_integration.py:44)。

T输入仅是U的逐行类别中心化分数，再按源训练逐类均值/标准差标准化；`score_only` 为8→16 GELU→8，末层零初始化、无tanh上限，输出不强制类别中心化。见 [inputs:27](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/calibrated_signed_transfer/polarization_readout/full_model_integration/run_integration.py:27)、[Head:24](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/calibrated_signed_transfer/run_training.py:24)。当前源端恢复函数明确载入该目录 `fold{fold}_score_only_seed{seed}.pt` 并执行 `raw+model(x,None)`，见 [source_baseline:63](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/calibrated_signed_transfer/polarization_readout/joint_distribution_readout/component_isolation/run_components.py:63)。后续强B不是先加某个local/signed候选再校准。

## 4. layer3辅助reader与两条替代读出

aux完整R50向前执行到layer3，160画布产生1024通道10×10图；VV/VH分别提取、有效footprint加权GAP、C4均值后拼2048维。source拟合 `StandardScaler→PCA128(randomized,seed17,iter4)→post-StandardScaler→加权Ridge(onehot8)`，alpha只在源val的balancedNLL上从1/10/100选，logit scale=3。保存精确线性反投影W和intercept；每外折1outer+4inner，共15reader。见 [baseline/run_baseline.py:197](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/baseline/run_baseline.py:197)、[linear_cache:154](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/baseline/run_baseline.py:154)、[View:26](E:/临时会话/scene_aware_classification_20261018/context_strong_reader_20261022/run_strong.py:26)。此reader的预测不是强B；它提供辅助类别方向。

对单pol、单C4方向，令 `z_c(p)=W_pol,c F_pol(p)`，`w`为归一化有效footprint权重，`mu_c=Σw z_c`，`v_c=z_c−mu_c`，`r_c=v_c−mean_class(v)`，再数值重中心化使Σw r_c=0。intercept不分摊进空间图，mu来自未减空间均值的z。见 [weighted_moments:24](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/calibrated_signed_transfer/extract_moments.py:24)、[W与layer3:75](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/calibrated_signed_transfer/extract_moments.py:75)。

- 原默认分布摘要：`sigma=sqrt(Σw r²)`，`kappa=Σw r|r|/sigma`（代码对零分母作数值保护）。这里sigma是类别证据离散度，不是SAR物理sigma0。
- attention替代摘要：`alpha±=softmax_p(log(w)±r/tau)`，`q±=Σalpha±r`，`h=(q+−q−)/2`，`b=(q++q−)/2`。零weight位置排除。tau每fold/pol固定为合法源OOF训练sigma在样本及类别上的中位数，不按test选。见 [attention:10](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/calibrated_signed_transfer/polarization_readout/joint_distribution_readout/matched_attention_readout/extract_attention.py:10)、[tau:89](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/calibrated_signed_transfer/polarization_readout/joint_distribution_readout/matched_attention_readout/extract_attention.py:89)。
- 两种摘要均每pol每C4分别完成非线性统计，再平均四旋转，不先平均响应图。attention显式layer3前向见 [extract_attention.py:131](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/calibrated_signed_transfer/polarization_readout/joint_distribution_readout/matched_attention_readout/extract_attention.py:131)。

令C(x)=x−mean_class(x)，所有输入标准化均只在相应源训练描述上fit。两条候选的最终分数分别为：

```text
s_mom = B + C( f_mu(B, mu_VV/VH, valid) + f_mom(B, sigma_VV/VH, kappa_VV/VH, valid) )
s_att = B + C( f_mu_new(B, mu_VV/VH, valid) + f_att(B, h_VV/VH, b_VV/VH, valid) )
```

均值支路7→12→1，分布/attention支路11→12→1，共266参数；输入含对应统计的原值及类别中心化值。`f_mu_new`与attention支路共同训练，不是保留原moments头的均值支路权重不动。见 [run_joint.py:19](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/calibrated_signed_transfer/polarization_readout/joint_distribution_readout/run_joint.py:19)、[inputs:49](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/calibrated_signed_transfer/polarization_readout/joint_distribution_readout/run_joint.py:49)。attention训练直接以B为base，原full模型只恢复作对照；test也取full记录的b列而非s列，见 [run_attention.py:62](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/calibrated_signed_transfer/polarization_readout/joint_distribution_readout/matched_attention_readout/run_attention.py:62)、[run_attention.py:178](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/calibrated_signed_transfer/polarization_readout/joint_distribution_readout/matched_attention_readout/run_attention.py:178)。文件名 `signed_features.npz` 不代表attention仍输入kappa；这里复用mu，另以sigma确定固定tau。

## 5. 最新OBB串联及缓存证据

`obb_dynamic_readout/run_heads.reference()`读取attention的source/test_predictions，只保留 `arm=two_sided_attention`；按fold/seed/role/instance_id匹配并核对obs/mmsi/class，返回 **s0..s7**。这些变成OBB预测表的 **b0..b7**。见 [run_heads.py:103](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/obb_dynamic_readout/run_heads.py:103)。

新reader从相同R50 checkpoint提取layer1的256×40×40图，双pol512维，重新进行source PCA128/Ridge及15reader拟合；不是换新骨干，也不是旧layer3 W直接复用。见 [extract_regions.py:27](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/obb_dynamic_readout/extract_regions.py:27)、[layer1:83](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/obb_dynamic_readout/extract_regions.py:83)、[fit_readers.py:26](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/obb_dynamic_readout/fit_readers.py:26)。目前是分阶段缓存提取，**不能把同权重称作已实现单次完整R50前向共享所有旧新分支**，也没有完整端到端成本实测。

动态body/end/side区域各读mu/h/b，固定模板对照保留实际轴但L/W取各reader源fit中位数；所有臂共同输入实际动态质量及几何32维。区域有效条件为area>0且min-C4 neff≥4。merged按有效域等权分别平均mu/h/b。dynamic/fixed输入69维、69→12→1，853参数；global/merged45维，565参数。见 [run_heads.py:34](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/obb_dynamic_readout/run_heads.py:34)、[raw_inputs:76](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/obb_dynamic_readout/run_heads.py:76)。

```text
s_keep = s_att
s_dynamic = s_keep + C( g_dynamic(C(s_keep), layer1三域mu/h/b及其类别中心值, 共同quality/geometry) )
```

新头末层零初始化，0步严格等于s_keep。layer1 tau来自新layer1合法源OOF整图sigma，不是沿用layer3 tau数值；各区域/各臂共用本层tau。见 [extract_regions.py:151](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/obb_dynamic_readout/extract_regions.py:151)。

本次实际读取的缓存证据：

- attention [checks.csv](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/calibrated_signed_transfer/polarization_readout/joint_distribution_readout/matched_attention_readout/checks.csv) 的9组 `baseline_error=0,full_replay_error=0`，源选点重放通过；[test_checks.csv](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/calibrated_signed_transfer/polarization_readout/joint_distribution_readout/matched_attention_readout/test_checks.csv) 9组full重放误差0、无新fit。这是已有运行记录，本次未重新计算。
- OBB [implementation_receipt.json](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/obb_dynamic_readout/implementation_receipt.json) 记录15新reader、36新head、旧stage拟合0、test新fit0；零步等价、source选点重放、共同quality/scaler一致通过。线性reader重放最大误差3.56e-14；区域global均值对W-GAP误差1.34e-6。
- [CACHE_SCHEMA.json](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/obb_dynamic_readout/CACHE_SCHEMA.json) 明确summary顺序为 `[mu,h,b]`、reader W为8×512、baseline为历史two_sided_attention。summary文件内的sigma字段是未使用零占位；真实定温sigma在独立scale文件，不能误读成OBB仍输入sigma/kappa。

## 6. OOF与尚未核实的边界

- 旧A/D源训练分数来自 `held_vessel_gate` 的三包MMSI交叉拟合；源码对fit/held身份隔离并重新拟合旧读出，缓存保存base/residual。其内分包按MMSI/类，不是新aux的四包pass划分；不能统一写成“所有层都按相同四包pass OOF”。见 [inner_partition:93](E:/临时会话/same_vessel_gallery_20261019/held_vessel_gate_20261022/run.py:93)、[oof保存:197](E:/临时会话/same_vessel_gallery_20261019/held_vessel_gate_20261022/run.py:197)。本次不追完每个历史选点及训练时序。
- 新layer3和layer1辅助W/源训练摘要使用合法inner reader；源val/test用outer reader。但T及attention头本身是在源训练集上拟合，再产生该集预测；OBB继承这些预测。**reader-crossfitted不等于T→attention→OBB整条流水线OOF。**
- 已证实的是当前脚本的加载链及已有检查表所记录的缓存一致性；没有重跑所有历史checkpoint，没有证明旧view与新aux输入逐像素一致，没有追溯检测器最初训练数据，也没有独立验证物理散射量。不能据目录名称或“同R50”补齐这些未知。
- 当前默认moments候选与attention替代实验必须分别保留表述；最新OBB只证明已接入attention路线，不能据此声称原moments+OBB完整模型已经评估。是否保留OBB应由主报告的实际效果/对照决定，不由本代码追溯替它作结论。
- 本文件不作“统计量/attention/MLP首创”判断，也不把网络多层串联本身当成创新证据；主报告可据上述真实信息流讨论类别坐标分布读出与区域组织的贡献，而不误写为旧支持集机制或层级堆叠。
