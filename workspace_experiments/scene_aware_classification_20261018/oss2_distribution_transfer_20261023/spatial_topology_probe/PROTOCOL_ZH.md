# 固定source-only空间连续性九头

2026-10-23，首次提取/训练前锁定。仅源train1542、validation279；3臂×seed17/29/43，共9头。无test推理/评价、无新划分、无参数搜索。随机图null仅保留设计文件，最终臂为marginal/complete/local。

## 输入与数学

224保全PNG、相同R50 layer3/C4/VV/VH/footprint。只用每个样本合法source W：train按4inner held，validation outer。重新提取一次源1821图以取得每C4 map，复核mu/sigma/kappa与现有source_moments；不重拟合W/B。每seed使用各自原calibrated/fold0_seedXX的train_base及validation base。

类中心证据r使用原weighted_moments。有效节点w_i>0，p_i=w_i/sum(w)。v_c=sum_i p_i*r_ci²；原代码r加权均值为0。合法边为14x14网格的水平/垂直四邻接，无向各计一次，两端均w>0。alpha_ij=w_i*w_j，fractional权重不二值化。

E_local,c=sum_edges alpha_ij*(r_ci-r_cj)²/(2*sum_edges alpha_ij)。E_all,c同式但遍历全部i<j，精确等于v_c/(1-sum_i p_i²)。新增R=E/v按每pol每C4计算，再平均4视图。不是先平均图或把缓存平均sigma平方作v。完整图边数不同，是边缘分布/有效数对照，不称纯shuffle。

v恰为0时该class/pol/view的R置0，并保存zero_variance布尔诊断；不加epsilon构造信号、不排除对象、不增加输入维度。无合法边同样显式记录，但覆盖统计没有此情况。非零v的complete R=1/(1-1/n_eff)，n_eff=(sum w)²/sum(w²)。两个极化的n_eff共同进入所有臂，排除只让complete掌握计数；它不能保证消除所有native尺寸关联。

非同构数学例：等权2x2 stripes和checkerboard均mu0/v1/kappa0、Eall4/3，Elocal分别为1和2。仅证明可辨识性，不当作实图收益。该R属于Geary/Dirichlet类量，不声称新统计。

## 同容量控制

共同原joint.inputs15槽：centered B；每pol的mu、sigma、kappa及各自减类别均值，共12槽；validVV/VH两槽。追加每pol n_eff两槽，再追加每pol R两槽，共19。前三臂共同mu/sigma/kappa/B/valid/n_eff逐元素一致；marginal的末两槽原始置0，complete填完整图R，local填四邻接R。不得将统计冒称新sigma/kappa，不增加signed统计、heading或原h/w输入。

共享19→12 GELU→1，253参数，末层0，输出减八类均值后加B；同seed相同初始化。每臂仅源train全部class tokens拟合逐槽scaler，验证复用。AdamW .003/.01，200步jointCE=.5普通CE+.5八类均衡CE，每5步源val精确J含0，严格提高才更新，平局最早。每seed对应自己的冻结B。

## 交付与判断

保存源预测/完整分数、训练loss与验证曲线、selected checkpoint/scaler、覆盖及零方差、onepoint map重放、显式完整图和解析式一致性。保存源validation三臂配对OA/BA/NLL/Brier、类及结果图。固定seed17 local相对B四结局首ID，缺失明确记录；图展示实际四邻接/完整图定义及原VV/VH，不把RF格当船体物理定位。

主要比较local与同容量complete及marginal，避免把幅度/计数函数的收益叫连续性收益。验证已用于选点，仅source pilot，非独立确认。若选0/负结果同样保留，不延长epoch，不换图/网络、不扩test。小支持结果按已有覆盖描述，不能事后择阈值剔除。主流程/Heisenberg独立审阅分析，不改其文件。完成停止、不操作Git。
