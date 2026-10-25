# 最终 frequency 缓存实际继承的 M1 / M2 代码溯源

日期：2026-10-25。范围：读取代码、NPZ、checkpoint，CPU 无梯度重放已有小头；没有训练、重新提取图像、修改模型或操作 Git。最终 M3 固定为 `anchored_polar_adjustment_20261025/frequency`，未改成 spatial。M3 详细推导和整合入口由主线程负责。

## 1. 结论与必须保留的限制

**最终 base 确实继承重新拟合的 M1 双支路注意力与 M2 三区域池化流程，而非旧 4744 队列的读出权重。** 当前 `native_validation/table.csv` 的 `up_index` 是 `regional_utility_pilot/upstream/table.csv` 的行位置；最终 base 等于 `strategies_seed{seed}.npz['logits'][up_index,2].astype(float32)`。第三列的实际 `strategy_names` 是 `full`。三 seed、1741 行的最终输入缓存及逐级 base 缓存与此式最大绝对差均为 **0**。

**均值不是只存在于参数名。** `AttentionHead.forward()` 实际调用 `self.mean(...)` 和 `self.local(...)`，相加后作类别中心化；`attention_raw()` 的均值输入实际来自 `summaries.npz['mean3']`。但源验证选择允许第 0 步，实际 checkpoint 必须逐 seed 区分：

| seed | M1 attention 选中步 | M2 full 选中步 | 当前 485 evaluation 行均值支路中心化输出最大绝对值 | 同组 M2 增量最大绝对值 |
|---|---:|---:|---:|---:|
| 17 | 0 | 15 | 0 | 0.962088287 |
| 29 | 70 | 0 | 3.275860786 | 0 |
| 43 | 10 | 0 | 0.244430453 | 0 |

因此准确表述为：**保留均值+双侧注意力、再接区域池化的重建结构；源端选点后，seed17 的 M1 整个残差为零，seed29/43 的 M2 整个残差为零。** 不能写成三个 seed 都有非零 M1 与非零 M2，也不能把 seed17 的零均值解释为“代码删掉均值支路”。当前三个配置甚至没有一个同时保留非零的 M1、M2 残差；结构串联与每个配置的实际生效情况要分开。

这些数值是本次只读检查点重放结果，不是新增分类收益实验。下文旧 4744 文档仅作溯源起点，不用于填补当前模型的训练事实。

## 2. 从最终缓存反查到重建入口

为避免目录名误导，下面逐步列出真正读取的值：

1. 最终 [anchored train.py:19](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/evidence_preserving_redesign/anchored_polar_adjustment_20261025/train.py:19) 继承前版 `Data`，第 23 行继承 `inputs`；第 59 行 `prediction()` 使用 `base + g_VV*delta_vv + g_VH*delta_vh`。本文件只追这个 base，不把它与该实验的 alpha 锚点 B 混称。
2. `polar_preference_20261025/train.py:28` 的 `inputs()` 委托给 [query_polar_gate/train.py:69](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/evidence_preserving_redesign/query_polar_gate_20261025/train.py:69)，实际读取 `query_polar_gate_20261025/results/features_{seed}.npz` 的 `base`。
3. 此缓存由 [query_polar_gate/extract.py:82](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/evidence_preserving_redesign/query_polar_gate_20261025/extract.py:82) 生成，第 86-88 行保存 `base=ss['base']`；`common.scores()` 在 [common.py:33](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/evidence_preserving_redesign/query_polar_gate_20261025/common.py:33) 读取 hybrid 结果。
4. [hybrid train.py:58](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/evidence_preserving_redesign/hybrid_polar_fusion_20261025/train.py:58) 的 `read_scores()` 从 null_centered 读取 base，第 113 行原样保存；[null_centered train.py:143](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/evidence_preserving_redesign/null_centered_readout_20261025/train.py:143) 将 polar_readout 的 base 作为比较项原样继承，第 159 行保存。
5. [polar_readout train.py:85](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/evidence_preserving_redesign/polar_readout_diagnosis_20261025/train.py:85) 的 `prepare()` 实例化 `prior.Reader('pair_joint',seed)`，以 `reader.ids` 建映射，再按 `data.up` 重排，取 `reader.base[ix]`；第 99 行写入 `inputs_{seed}.npz`，第 220、237 行保存 base 比较项。这里不是直接把 compact reader 行号当作 upstream 行号。
6. `prior` 导入关系：polar_readout `train.py:30-31` → intervention `train.py:27-28` → [anchored_m3 train.py:34](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/evidence_preserving_redesign/anchored_m3_20261025/train.py:34) 的 `pair_conditioned_joint_metric/train.py`。后者 [Reader:42](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/local_evidence_transport/pair_conditioned_joint_metric/train.py:42) 继承 spatial_frequency_joint_pooling 的 Reader；后者 [train.py:31](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/local_evidence_transport/spatial_frequency_joint_pooling/train.py:31) 继承 frequency_conditioned_matching 的 Reader，未改写 base。
7. 最底层实际赋值在 [frequency_conditioned_matching/train.py:58](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/local_evidence_transport/frequency_conditioned_matching/train.py:58)：`self.base=np.load(UP/f'strategies_seed{seed}.npz')['logits'][ids,2].astype(np.float32)`；`UP` 在第 16 行明确指向 `regional_utility_pilot/upstream`。第 35-36 行的 ids 来自 compact 表 `table_index`。

当前 `Data` 的直接定义在 [polar_readout train.py:35](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/evidence_preserving_redesign/polar_readout_diagnosis_20261025/train.py:35)：读取 native_validation 表，`up=table.up_index`，前 1741 行分为 utility_train 644、selection 612、evaluation 485。仅复用这个表不等于采用 native_validation 的新卷积模型。

### 真正的重建函数与检查点

所有重建函数位于 [regional_utility_pilot/upstream/pipeline.py](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/regional_utility_pilot/upstream/pipeline.py)。以下相对名均以该文件所在的 **upstream 绝对目录**为根。

| 函数 / 行号 | 实际动作 | 保存的模型或缓存 |
|---|---|---|
| `fit_base():257` | 读历史无监督描述，重新拟合 axis Ridge、five、neighborhood、calibration | `models/axis_ridge.npz`、`models/five_seed{seed}.pt`、`models/neighborhood_seed{seed}.pt`、`models/calibration_seed{seed}.pt`、`base_seed{seed}.npz` |
| `fit_aux():317` / `fit_reader():229` | 以新 fit_train 重拟合 layer3 与 layer1 类别 reader | `models/reader_layer3.npz`、`models/reader_layer1.npz` |
| `extract_source.main():37` | 用新 reader 从冻结 R50 特征读 sigma 定温和实际摘要 | `scale_sigma.npz`、`scales.json`、`summaries.npz` |
| `fit_strategies():361` | 从 `base_seed*.npz['B']` 重训 AttentionHead，再分别训练 body/full RegionHead | `models/attention_seed{seed}.pt`、`models/body_seed{seed}.pt`、`models/full_seed{seed}.pt` |
| `fit_strategies():388` | 按 `s0,body,full` 顺序保存分数 | `strategies_seed{seed}.npz` |

模型写出位置由 `train_head():202-210` 明确规定，并立即从保存 checkpoint 恢复再生成分数。当前 M1/M2 的具体权重路径是：

```text
E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/regional_utility_pilot/upstream/models/attention_seed17.pt
E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/regional_utility_pilot/upstream/models/attention_seed29.pt
E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/regional_utility_pilot/upstream/models/attention_seed43.pt
E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/regional_utility_pilot/upstream/models/full_seed17.pt
E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/regional_utility_pilot/upstream/models/full_seed29.pt
E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/regional_utility_pilot/upstream/models/full_seed43.pt
```

**没有加载旧 matched_attention_readout 的 fold/seed attention checkpoint，也没有加载旧 obb_dynamic_readout 的区域头 checkpoint。** 旧特征缓存与几何实现得到复用，监督读出权重重新拟合。旧文档里的“15 个 outer/inner reader”“合法 OOF sigma”不能直接用于描述本次重建。

## 3. 输入、骨干与基础 B

- 像素入口为 [obb_dynamic_readout/common.py:94](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/obb_dynamic_readout/common.py:94) 的 `native()`：读 `image_vv/image_vh` PNG，转灰度，使用各自 view/OBB；`crop_pad(...,160)` 保留原像素尺度，通过裁切与补零放入 160×160 画布。裁切位置显式加回 OBB 中心，见 `geometry():76-91`。不是把船旋转对齐后 resize。
- [native_input.py:6](E:/临时会话/same_vessel_gallery_20261019/local_condition_module/native_input.py:6) 的 `crop_pad()` 只做数组拷贝，目标过大直接报错；`place` 保存真实 crop/pad offset 和 `input_scale=1.0`。画布相同不表示所有历史分支的像素来源完全相同。
- 输入是显示域灰度 PNG `/255`，复制为 RGB，按 ImageNet mean/std 标准化。VV/VH 分开；无效极化跳过。有效性最初来自 [prepare_features.py:59](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/prepare_features.py:59) 的 crossview_alignment 禁用 VH 和第 83 行目标存在判定；重建 `fit_aux():320-323` 继承该 valid。
- 冻结权重实际为 `C:/Users/25791/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth`，定义在 [prepare.py:19](E:/临时会话/same_vessel_gallery_20261019/local_condition_module/prepare.py:19)，本次确认文件存在。重建提取 [extract_source.py:52](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/regional_utility_pilot/upstream/extract_source.py:52) 加载它并 `eval().requires_grad_(False)`。
- C4 是 0/90/180/270 度四次输入旋转；每次分别算响应、非线性摘要和区域权重，最后平均摘要，见 `extract_source.py:111-153`。不是先平均响应图再做 attention，也不是随机训练增强。
- 当前重建的基础仍是 `B=U+Calibration(Standardize(C(U)))`，`U=A+D`。`A=axis Ridge+Five`，`D=Neighborhood`。旧缓存入口在 `pipeline.py:274,288,291`，分别为 native_axis descriptors、keep_refs.npy、neighborhood features；没有使用这些旧目录的监督 head 权重。Five 是 60×64 → 共享 64→8 GELU → 480→8；Neighborhood 为 260→32 GELU→8；Calibration 为 8→16 GELU→8，实际类定义见 `pipeline.py:96-126`。

本次没有重新跑图像到特征的 R50 提取；像素来源和空间公式由实际生成代码确认。M1/M2 当前在 `extract_source` 一次旋转前向里从 z1 继续得到 z3，但旧 A/D 和 M3 仍有各自缓存流程，不能宣称整套系统已统一为单次前向。

## 4. M1：真实均值支路与双侧注意力

### 类别坐标与空间摘要

`extract_source.py:115-116`：layer1 为 256×40×40，继续 layer2/layer3 得到 layer3 的 1024×10×10。layer3 全有效 footprint 池化得到每极化 1024 维、拼接 2048 维，供 source reader 拟合。当前 reader 的 `W.shape=(8,2048)`，拆成每极化 `W_p∈R^(8×1024)`；本次读取的 ridge alpha=1。

对单极化、单旋转，用类别方向映射 `z_c(u)=W_{p,c}F_p(u)`。reader 的 intercept 不加进空间响应图。当前 layer3 权重由有效 footprint 的 16×16 像素块均值得到 10×10；归一化后记为 w。真实代码在 [extract_source.py:13](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/regional_utility_pilot/upstream/extract_source.py:13) 的 `centered_response()`、第 20 行 `attention()` 及第 95、132、144 行：

```text
mu_c = sum_u w(u) z_c(u)
v_c(u) = z_c(u) - mu_c
r_c(u) = v_c(u) - mean_class(v(u))
r_c(u) <- r_c(u) - sum_u w(u) r_c(u)  # M1 recenter=True 的数值重中心化
alpha_c^+(u) = softmax_u(log w(u) + r_c(u)/tau_p)
alpha_c^-(u) = softmax_u(log w(u) - r_c(u)/tau_p)
q_c^+ = sum_u alpha_c^+(u) r_c(u)
q_c^- = sum_u alpha_c^-(u) r_c(u)
h_c = (q_c^+ - q_c^-)/2
b_c = (q_c^+ + q_c^-)/2
```

零权重位置被排除。每个旋转的 mu、h、b 分别累加四分之一；不是省略 mu。`mean3` 形状为 `(3387,2,8)`，`hb3` 为 `(3387,2,8,2)`，输出写于 `extract_source.py:214`。

tau 使用新 reader 的 fit_train 有效极化样本：先各旋转计算 `sigma=sqrt(sum w*r²)`，再 C4 平均，再对样本与八类取中位数，见 `extract_source.py:138-141,206-211`。这是 strategy_fit 内的 in-sample 定温，**不是早期 reader OOF 定温**。三 seed 共用固定值，当前 `scales.json` 的 layer3 `[VV,VH]=[0.8670898382097123,0.8466651969624001]`。

### 输入维数与实际前向

[pipeline.py:330](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/regional_utility_pilot/upstream/pipeline.py:330) `attention_raw()` 每类输入 15 维：

```text
0: C(B)
1..6:  mu_VV, C(mu_VV), h_VV, C(h_VV), b_VV, C(b_VV)
7..12: mu_VH, C(mu_VH), h_VH, C(h_VH), b_VH, C(b_VH)
13..14: valid_VV, valid_VH
```

`C` 表示沿八类中心化。源 fit_train 上按样本和类别两个轴统计每个输入字段的 mean/std；标准差下限 1e-8，见 `scaled():82`。这不是把空间 mu 减掉后永远丢弃。

[AttentionHead:129](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/regional_utility_pilot/upstream/pipeline.py:129) 精确实现：

```text
f_mean = Linear(7,12) -> GELU -> Linear(12,1)
         x[..., [0,1,2,7,8,13,14]]
f_att  = Linear(11,12) -> GELU -> Linear(12,1)
         x[..., [0,3,4,5,6,9,10,11,12,13,14]]
s0 = B + C(f_mean + f_att)
```

两支路共同训练，合计 266 参数；不保留早期 moments 模型的 mean 权重。checkpoint 的 `mean.0.weight=(12,7)`、`mean.2.weight=(1,12)` 均被前向读取。seed29/43 中均值支路对最终 evaluation 行有非零类别差分，已由本次重放确认；seed17 为零是源端选择初始 checkpoint 的结果。

## 5. M2：目标相对三区域池化

### 几何与权重

当前 M2 使用冻结 layer1 的 256×40×40 图及新拟合 `reader_layer1.npz`，`W.shape=(8,512)`，每极化 8×256；实际 ridge alpha=100。没有把 layer3 W 截成 layer1 W。

[common.py:76](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/obb_dynamic_readout/common.py:76) `geometry()` 读取目标检测角点，要求 shape_usable 且 confidence≥0.25，最长边为 L，相邻边为 W，角度取模 pi。对 OBB 中心相对坐标旋转得到 u,v；[soft_regions():108](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/obb_dynamic_readout/common.py:108) 定义一像素软边界矩形：

```text
R(l,w)=clip(l/2+0.5-|u|,0,1) * clip(w/2+0.5-|v|,0,1)
body = R(L,W)
end  = max(R(1.5L,W)-body,0)
side = max(R(L,2W)-body,0)
```

两端合并、两侧合并，保留 body/end/side 三个区域身份；没有识别船首船尾。区域乘有效 footprint 后，每个 C4 重新算 `cell_mass()`；[cell_mass():138](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/obb_dynamic_readout/common.py:138) 对 piecewise-constant 像素作精确积分，对应 stride4、origin0 的 feature-center Voronoi 单元，边界 `[-0.5,2,6,...,154,159.5]`。它不是简单 4×4 均匀面积缩小，也不是在特征点上硬判 OBB 内外。

`region_quality():151` 给出 `area`、`coverage=有效面积/planned_mass`、`neff=(sum mass)^2/sum mass²`；使用四旋转 neff 的最小值。`extract_source.py:99-107` 的区域有效条件为 `area>0 && min_C4_neff>=4`。几何缺失或不满足条件的区域摘要清零。

### 摘要与头

每个区域使用该区域归一化 mass 作 w，重新算区域 mu，并对区域内相对响应算 h/b；公式与 M1 相同，但 M2 的 `attention(...,recenter=False)` 不额外执行数值重中心化。区域在旋转内先计算非线性统计，再 C4 平均。实际调用在 `extract_source.py:147-153`。

tau 来自同一新 layer1 reader 的 **整有效 footprint** sigma，而非每个区域独立选温度；当前 `[VV,VH]=[15.109352724203534,4.095158346606487]`。缓存 `summary1.shape=(3387,2,3,8,3)`，最后维顺序 `mu,h,b`；另有 `quality=(3387,2,3,4)`、`geometry=(3387,2,4)`、`region_valid=(3387,2,3)`。

[regional_raw():339](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/regional_utility_pilot/upstream/pipeline.py:339) 的每类 69 维输入为：

- 1 维 `C(s0)`；
- 36 维：2 极化×3 区域×3 摘要×原值/类别中心值；
- 32 维共同质量几何：每极化 `log1p(L),log1p(W),confidence,geometry_valid`，加每区域 `log1p(area),coverage,log1p(min_neff),region_valid`。

[RegionHead:142](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/regional_utility_pilot/upstream/pipeline.py:142) 为共享 `69→12 GELU→1`，853 参数，输出类别中心化：`s12=s0+C(g(Standardize(regional_raw)))`。当前使用 full 三区域；body 是另训对照头，不与 full 相加。`strategy_names=['s0','body','full']` 与保存次序由 `fit_strategies():372-391` 共同确定。

所以当前 M2 是**固定几何区域定义上的、类别响应均值与双侧统计池化，加可学习残差读出**。它不是新的可学习空间卷积，也不是自适应分组池化。

## 6. 当前训练协议与历史协议差异

`pipeline.load():39` 从 `indices.npz` 取 fit_train / fit_validation，两者均属于 strategy_fit，按 MMSI 和 pass 隔离。实际 checkpoint 记录 fit_train=1205、fit_validation=296。上游表共3387行；当前 M3 使用其中通过 up_index 映射的1741行。M1/M2 不在这1741行的644/612上重新训练。

`fit_reader():229-251`：StandardScaler → PCA128(randomized,random_state=17,iterated_power=4) → StandardScaler → 加权 Ridge(onehot8)，alpha 从1/10/100以 fit_validation balanced NLL 选择，输出乘3，精确反投影得到 W 和 intercept。权重函数 `weight():49` 按类别船数及船内观测数平衡。两层各拟合一个 reader，不是早期三外折×五reader。

`train_head():162` 对 M1/M2 使用 full-batch AdamW(lr=.003,weight_decay=.01)，最多200步，每5步源验证；loss 为 `0.5*普通CE均值+0.5*各类别CE均值的平均`，函数 `joint_loss():73`。按精确 `(OA+BA)/2` 严格改善才更新最优状态，平分保留更早 checkpoint；第0步允许入选。末层零初始化，因此初始输出精确退化到上游。先 M1，再以冻结的 s0 训练 M2；不端到端更新 R50，不三模块联合反传。

Five/Neighborhood 的重建训练仍是150步加权 MSE，见 `train_head():176-185`，不要把它们也写成200步CE。基础 calibration 使用同一 CE 分支。

checkpoint 明确记录 `upstream_training_predictions_oof=False`（`pipeline.py:205`）：当前重建没有 strategy_fit 内整个堆叠的 nested OOF。其尺度、tau、reader 与头训练都受各自源角色约束，但不能改写成“全流水线严格OOF”。同样，后续 M3 的612行历史选点不改变这一事实。

## 7. 本次只读缓存验证

为检出“旧权重冒充重建”“mean 参数存在但未参与”“最终缓存接错列”等具体失败，本次直接读取保存状态，用 CPU `Linear/GELU` 运算重放，没有执行任何训练入口或写缓存。

| seed | 重放 `B+C(mean+local)` 对保存 s0 最大误差（3387行） | 重放 `s0+C(region)` 对 full 最大误差 | query features base 对 full[up_index] 最大误差 |
|---|---:|---:|---:|
|17|0|5.96046448e-8|0|
|29|1.90734863e-6|0|0|
|43|1.90734863e-6|0|0|

微小头重放差异来自批量浮点路径；与当前上游 full float32 的缓存连接则逐元素完全相等。独立核对 `polar_readout_diagnosis`、`null_centered_readout`、`hybrid_polar_fusion`、`anchored_polar_adjustment` 四处 `results/base_{seed}.npz`：up_index 相同，score 与重建 full float32 最大差全部0。

当前1741行上，mean 分支中心化输出绝对值均值依次为0、0.664116204、0.051529851；在485 evaluation行上依次为0、0.628135800、0.049368154。作为调用连通性检查，在完整3387行冻结头中移除 mean 分支会改变 seed29/43 的405/27个 M1 argmax，seed17为0。**这不是性能增益或因果有效性结论**，也不作为新的消融结果表；只是证明非零 mean 分支实际影响保存模型的计算。

本次没有重跑R50、重算所有区域摘要或验证原始检测器的训练来源；上述空间生成公式来自实际代码，权重到分数及最终缓存继承则经过本次独立只读计算核实。

## 8. 未被当前 M1/M2 采用的历史候选

| 历史实现 | 真实代码位置 | 当前是否采用 |
|---|---|---|
| 原 sigma/kappa 分布支路 | [extract_moments.py:24](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/calibrated_signed_transfer/extract_moments.py:24) | 否。当前 M1 输入 mu/h/b；sigma仅用于定tau，kappa未进入当前头。 |
| 旧4744 attention和OBB监督读出权重 | [run_attention.py:62](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/calibrated_signed_transfer/polarization_readout/joint_distribution_readout/matched_attention_readout/run_attention.py:62)、[run_heads.py:103](E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/obb_dynamic_readout/run_heads.py:103) | 未加载这些旧head。复用的是冻结特征/几何接口，实际监督权重位于本报告的upstream/models。 |
| RoleConv与GroupPool | [m2_operator_validation/operators.py:14](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/m2_operator_validation/operators.py:14)、第44行；`Candidate`第127行，第149/154行才执行卷积/学习分配 | 否。当前 base 的生成链没有实例化 Candidate，也没有加载其checkpoint。 |
| 原生分尺度 AxisPairConv、可合并 moment_pool | [redesign/operators.py:31](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/evidence_preserving_redesign/operators.py:31)、第73/102行 | 否。当前 M2 仍来自 layer1 区域 mu/h/b 的69维头，不能把新卷积名称套用到旧收益。 |
| native_validation mean/moment/grid_moment/axis_moment、warm/hybrid/projection后续 | [train_native.py:163](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/evidence_preserving_redesign/native_validation/train_native.py:163)，第170行选择卷积，第333行区分old_full | 不作为最终 base。当前 Data 复用 native_validation/table.csv，不读取这些新模型输出；已验证最终base与重建full完全一致。 |
| M2 body-only、固定尺寸/merged等旧对照 | [pipeline.py:374](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/regional_utility_pilot/upstream/pipeline.py:374) 的body/full独立训练；历史对照见obb目录 | 当前取full列，不叠加body头，不取旧fixed/merged头。 |

M3 中名为 `spatial_frequency_joint_pooling` 的目录仍被导入作为 descriptor/Reader 基类；这不代表 M2 采用该目录的 Pool。当前 pair-conditioned Reader 覆盖了 forward，见 `pair_conditioned_joint_metric/train.py:45`。这类导入关系应按实际调用解释，不能靠目录里出现 pooling 一词归类。

## 9. 给主线程的整合口径

最终缓存重放可以明确使用：`s12=upstream strategies['logits'][native_table.up_index,2]`，然后接用户指定 frequency 的冻结 M3 结果；必须保留实际 up_index 对齐与三seed选择。M1写“均值保留的双侧类别响应注意力”，M2写“OBB相对三区域池化读出”，同时注明源选点后的零残差情况。

不能使用旧4744提升数字证明当前1741/485配置下各模块独立有效，也不能为让三个模块看起来都生效而替换检查点、改选点、重新训练或改用 spatial。本报告仅纠正实际继承与生效范围，未作任何此类变更。
