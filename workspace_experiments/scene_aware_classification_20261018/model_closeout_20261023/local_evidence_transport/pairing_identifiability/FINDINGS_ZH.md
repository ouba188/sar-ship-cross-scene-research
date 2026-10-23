# 配对可辨识性：有界源端诊断

## 当前结论

**utility644 中存在可测、带类别相关性的真实配对二阶信息；现有 shuffle 确实大幅破坏这部分信息，而原 mixed 读出仍高度相似。该诊断不等于完整模型有效：主代理最新 pairing_excess 已完成但没有超过 anchor。**

四组顺序统一为 VV_side / VV_end / VH_side / VH_end。排除同船、同pass图对后，真实中心化核的宏平均同类减异类差 / kernel RMS 为0.1788/0.1232/0.1427/0.0663；三个shuffle的对应值范围为 0.0020~0.0368 / -0.0272~-0.0065 / -0.0159~0.0133 / -0.0267~-0.0007。真实中心化核与shuffle核的图对相关接近0（-0.0133~0.0100），但既有mixed类别中心化读出相关仍为0.9773~0.9963。这支持区分配对信息与既有读出保留的共同成分，不能据此认定其失效根因。

body-neighbor 原始配对余弦均值仅0.108/0.047/0.028/-0.088，不是普遍近似重复token。每图配对余弦与边际均值点积的跨图相关却为0.976/0.987/0.965/0.983；差分向量能量中图内均值差占71.0%/75.6%/72.9%/78.0%。另一方面，body均值核与neighbor均值核的图对相关只有0.422/0.204/0.354/0.213，不能把上述两种相关混为一谈。

精确置换下，真实crosscov范数平方相对null均值的图级中位比为1.363/1.155/1.345/1.138，超过各自精确置换q95的图比例为58.5%/20.8%/61.9%/23.7%。侧边二阶结构较明确；端边较弱且逐类有负值。单图强依赖不能替代跨图类别方向与分类收益。

仅以 utility_train644 量化下一步设计。没有训练、模型选择、新分类器、Git 或目录外写入。全表仅用于重放既有顺序随机流；固定 bank64 仅检查置换集合。existing_evaluation_NOT_design 单独转录既有结果，不参与计算或选择。这是 token/算子诊断，不是一次新分类实验，未新增原图物理归因。

## 实际 shuffle

`edge_difference/run.py:edges` 每次用一个 default_rng(seed)，按 group、table row 顺序为每图合法边抽取独立的邻居置换。同一完整输入顺序和种子重放一致；不是随 query-reference 对重抽。learned Reader 在构造时缓存整个表，reference_relative 继承同一实现。每张 reference 的映射被所有 query 共享，每张 query 在所有类别/参考中固定；不同图没有强制共享位置置换，但也不是按 instance_id 派生的顺序无关 RNG。若先切子集或重排表再调用 edges，同一图的随机映射一般会改变。这不影响当前完整表用法。

shuffle 保留原合法边交集中的 body 集合、neighbor 多重集、正值 mask 和等权边质量；它不恢复因为另一端无效而原已排除的单端区域，也不保留真实连接关系。side/end 分开置换，不跨组。随机置换允许固定点和整图恒等，不能称每条边都被破坏。下面列出可达的排除计数；不是 shuffle 新增的缺失。

| group | valid_body_excluded_by_paired_neighbor | valid_neighbor_excluded_by_paired_body | missing_images |
|---|---:|---:|---:|
| VV_side | 0 | 0 | 0 |
| VV_end | 0 | 0 | 0 |
| VH_side | 0 | 0 | 83 |
| VH_end | 1 | 0 | 83 |

## 精确数学与统计口径

对一图一组的 m 条合法边，b_i,n_i 是原 PCA32 单位 token；原 mass 仅用于合法性，所有统计等权。mu_b=mean_i b_i，mu_n=mean_i n_i，B_c[i]=b_i-mu_b，N_c[i]=n_i-mu_n。

`C = B_c^T N_c / m = mean_i b_i n_i^T - mu_b mu_n^T`。精确枚举全部 m! 邻居双射（含恒等，不是只用3次随机采样）；令 C_pi=B_c^T P_pi N_c/m，则 `E_pi C_pi=0`，`E_pi ||C_pi||_F^2=||B_c||_F^2 ||N_c||_F^2/[m^2(m-1)]`（m=1 为0）。范数分布由小 Gram 矩阵精确枚举；mean、q95、midrank 不是独立图/边显著性检验。midrank=(小于真实值的个数+0.5*相等个数)/m!，浮点平局容差1e-12。

`||b-n||` 的平均平方恰分解为 `||mu_b-mu_n||^2 + mean_i ||B_c[i]-N_c[i]||^2`。因此 difference_image_mean_energy_fraction 是真正的能量分解；joint_outer_centered_component_energy_ratio 的两项不正交，不能叫解释方差比例。pooled additive `[mu_b,mu_n]/sqrt(2)` 与 pooled difference `(mu_b-mu_n)/sqrt(2)` 在任何邻居置换下恒定。原 Reader 却先逐边 logsumexp 再平均，所以不能由均值不变推出原 mixed 分数不变。

| group | paired_cosine_mean | paired_cosine_gt_0p8_fraction_mean | independent_pair_cosine_mean | paired_cosine_correlation_with_independent_cosine | difference_image_mean_energy_fraction_mean |
|---|---:|---:|---:|---:|---:|
| VV_side | 0.107615 | 0.0128106 | -0.00264647 | 0.976134 | 0.709973 |
| VV_end | 0.046586 | 0.0372671 | -0.029879 | 0.986561 | 0.756252 |
| VH_side | 0.027862 | 0.0026738 | -0.098715 | 0.96457 | 0.729445 |
| VH_end | -0.0880012 | 0.0115865 | -0.158699 | 0.982845 | 0.779833 |


| group | crosscov_norm2_mean | exact_permutation_norm2_mean_mean | crosscov_norm2_over_exact_null_mean_p50 | crosscov_norm2_exact_midrank_p50 | crosscov_norm2_above_exact_q95_mean |
|---|---:|---:|---:|---:|---:|
| VV_side | 0.0271788 | 0.0195895 | 1.36348 | 0.969494 | 0.585404 |
| VV_end | 0.0349573 | 0.0293826 | 1.15493 | 0.8125 | 0.208075 |
| VH_side | 0.0292197 | 0.0215488 | 1.34495 | 0.973648 | 0.618538 |
| VH_end | 0.0349654 | 0.0299899 | 1.13795 | 0.770833 | 0.237077 |

## 是否存在类别相关的配对方向

仅在644内部，构造描述性图间核 `K(q,s)=<F_q,F_s>`；排除同 MMSI 和同 pass_group 的图对，缺组排除。每张图分别求各类参考核均值，gap=同类均值-7个异类均值的平均，再按查询类宏平均。除以全部可用有向图对 kernel RMS 只用于尺度比较，不是拟合。未做原型分类、调参或新的模型/预测。它是开发集描述，不是独立泛化估计；正的 gap 不是涨点承诺。

| group | additive_mean | difference_mean | outer_independent | centered_crosscov | centered_shuffle101 | centered_shuffle211 | centered_shuffle307 |
|---|---:|---:|---:|---:|---:|---:|---:|
| VV_side | 0.391307 | 0.0978297 | 0.506964 | 0.178835 | 0.00204072 | 0.0249536 | 0.0367625 |
| VV_end | 0.324444 | 0.0794608 | 0.468803 | 0.123193 | -0.0179472 | -0.00647224 | -0.0271741 |
| VH_side | 0.292755 | 0.0683978 | 0.445822 | 0.142686 | 0.0133323 | 0.000341805 | -0.0159091 |
| VH_end | 0.234186 | 0.0667925 | 0.39213 | 0.0662522 | -0.0203607 | -0.00065241 | -0.0266591 |

centered_exact_permutation_mean 是每张图先对全部置换求平均，再计算图间核。对不同图独立均匀置换，双线性核的精确期望也为0；这不等于单次随机置换 C_pi 为0，也不等于对所有置换训练后模型的平均性能。outer_independent=mu_b mu_n^T 保留边际均值乘积。

| group | body_neighbor_mean_kernel_correlation |
|---|---:|
| VV_side | 0.422479 |
| VV_end | 0.203974 |
| VH_side | 0.353973 |
| VH_end | 0.213391 |

## 既有读出的配对敏感性（utility，未拟合）


### mixed_shuffle101

| group | class_centered_correlation_real | class_centered_relative_l2_change |
|---|---:|---:|
| VV_side | 0.986922 | 0.161648 |
| VV_end | 0.992584 | 0.146615 |
| VH_side | 0.983098 | 0.183103 |
| VH_end | 0.983299 | 0.245386 |

### mixed_shuffle211

| group | class_centered_correlation_real | class_centered_relative_l2_change |
|---|---:|---:|
| VV_side | 0.982 | 0.190826 |
| VV_end | 0.990309 | 0.164404 |
| VH_side | 0.977263 | 0.214334 |
| VH_end | 0.992152 | 0.125143 |

### mixed_shuffle307

| group | class_centered_correlation_real | class_centered_relative_l2_change |
|---|---:|---:|
| VV_side | 0.98406 | 0.179837 |
| VV_end | 0.993283 | 0.138893 |
| VH_side | 0.981502 | 0.192964 |
| VH_end | 0.996328 | 0.0892995 |

## 纯配对交互与主代理新算子的数学评估

中心化 outer-product 核是 `K_C(q,s)=<C_q,C_s>_F = mean_ij[(b_qi-mu_bq)^T(b_sj-mu_bs)]`[(n_qi-mu_nq)^T(n_sj-mu_ns)]`。它隔离线性特征下的 joint 相对 independent-product 的交叉二阶矩，不是检验所有形式统计独立性的完备核；C=0 不能推出真实图像无依赖，C非零也不推出类别有效或物理因果。

主代理提议：固定一个 query/reference 和组，合法边数为m/r，`B_ij=b_qi^T W b_sj/(2*tau)`、`N_kj=n_qk^T W n_sj/(2*tau)`，`L_ik=log[(1/r) sum_j exp(B_ij+N_kj)]`，`S=tr(L)/m-sum_ik L_ik/m^2`。因每个(i,k)恰在(m-1)!个双射中出现，`E_pi mean_i L_i,pi(i)=mean_ik L_ik` 严格成立。S 消去所有 `u_i+v_k` 项，m=1、query neighborconstant、reference r=1 时均为0；reference neighborconstant 也使 L 可分离从而为0。m=0/r=0 必须按已有合法性规则跳过，不能对空集合取logmeanexp。

高温：令 B=t*B0，N=t*N0，t->0，则 `L_ik=t*(mean B0_i+mean N0_k)+t^2/2*Var_j(B0_ij+N0_kj)+O(t^3)`。对角减全均值后行列项消失，`S=t^2*mean_ij[(H_m B0 H_r)_ij*(H_m N0 H_r)_ij]+O(t^3)`。H_d=I-11^T/d；若直接用已经含1/(2*tau)的 B,N，首项就是两者双中心化乘积的均值，无额外1/2。等价首项为 `<C_q, W C_s W^T>_F/(4*tau^2)`。若“centered(B)”只减参考轴均值，必须再减 `mean_j mean_i(BH_r)_ij * mean_i(NH_r)_ij`；否则仍含查询边际项。

该S严格隔离的是给定真实参考边对时的query配对偏离，不是自动给出同时对query和reference独立置换后的非线性joint-product差。query端全置换平均S必为0；reference端的log池化之后再平均置换一般不能与log交换，不能直接声称也为0。均匀参考图平均保留前述query-null及退化性质。相对I的S_W-S_I也保留这些性质。side与end的r=8/4不使高温近似自动成立；本协议tau=1/sqrt32且W可改变logit幅度，二阶式应作为明确简单对照而不是有限温度的精确替代。当前脚本不实现或训练该新算子。

## 实现边界与检查

映射重放、合法集合、多重集及精确代数最大误差=2.84e-15。穷举检查用于识别中心化、归一化和索引错误；当前通过。未发现当前完整表路径的shuffle实现错误。可达解释风险包括：随机置换留固定点；m=1不可辨识；当前边合法交集不等于全部单端有效区域；pooled difference不敏感不能误写成log池化也不敏感；仅参考轴中心化不等于新算子的高温极限。

现有三个阶段evaluation均值已原样记录于CSV的existing_evaluation_NOT_design，仅作历史背景；无evaluation拟合、选组、选阈值。后续是否接入和拟合由主代理决定；本报告不作因果或novelty结论。

复现：`D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe -X utf8 -B run.py`。仅生成diagnostic.csv与本报告，不保存大型缓存。

## 主代理最新已完成结果（只读转录）

来源：pairing_excess/summary.csv，evaluation485均值，OA/BA为百分数；未用于本诊断选择或拟合。
| arm | OA (%) | BA (%) |
|---|---:|---:|
| anchor | 59.038 | 40.556 |
| covariance | 58.351 | 40.373 |
| excess | 58.419 | 40.491 |
| excess_shuffle101 | 59.038 | 40.556 |
| excess_shuffle307 | 59.038 | 40.556 |
| joint | 57.732 | 40.075 |

真实配对信号存在，但excess、covariance及joint本次均未超过anchor；不能以两shuffle回退anchor来宣称真实算子有效。本子任务到此结束，新算子拟合与原图复核属于主代理。