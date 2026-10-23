# 八类强校准基线与有符号局部增量

2026-10-23，在用户授权继续研究的范围内，由主代理预先固定以下配置。仅本目录写入，不修改原A/W/split/labels及旧八类A+D；子代理不执行Git，由主流程阶段归档，无新增网格或其他头。

原15251观测、3outer/12inner、4744互斥test。完整复用baseline/fold*_inputs.npz和models下保存W/intercept/fit/held。源训练每条恰有一次合法inner-held读出；所有fit覆盖8类。三折train/validation/test大小分别4392/310/1561、4419/286/1632、4418/319/1551。验证tug最小7条，J存在小样本分辨率；不调整类别或划分。

## 固定提取

复用prepare_features的crop_pad/masks_for、原R50权重和ImageNet预处理，原尺度160crop/pad不resize。各pol各C4独立前向，footprint用area映射到10x10。v=W(F-有效空间均值)，r=v-类均值，并去除数值残留加权空间均值。sigma=sqrt(E_a r^2)，kappa=E_a[r abs(r)]/max(sigma,1e-12)。先每pol每rotation非线性池化，再C4均值、pol求和。invalid VH为0，保留VV；不修改A中的缺失观测证据。全队列VH缺失1468，test缺失308。

一次图像前向同时供15个W使用，不整景解码、不detect、不拟合Reader。不从已有std推断kappa。复现已有3outer sigma、原weighted GAP及outer/inner-held的均值logits。保留各pol矩/有效flags和少量验证map，不存全量F。

## 两阶段36轨迹

第一阶段9个C：3fold×seed17/29/43，score-only 8→16→8 GELU，280参数，零末层初始化。输入为A类中心化分数，源训练按各分数轴标准化。训练基础A_OOF，验证/测试A_outer。200步AdamW(.003,.01)，jointCE=.5普通CE+.5八类CE均值；每5步source J=.5OA+.5BA，含0，精确有理数比较，平局最早。全部9个C选择保存后才统一报告A→B。

第二阶段冻结对应fold/seed的C及其预处理，B=C(A)。27轨迹：signed_only113参数（共享5→16→1，tanh乘原kappa），conditional_std81参数（共享3→16→1，tanh乘sigma），score_only280参数（8→16→8）。descriptor以B替换A，矩仍来自原W。signed/std源标准化在样本与类别轴共享，score按分数轴；均只用train。相同seed/优化器/步数/J规则，零末层精确回B。三臂同一B、同一OOF/outer矩和有效flags，新增模型不更新C。

C已在相同源训练标签上拟合，C(A_OOF)并不是二级校准全栈OOF。这是冻结源分类器后的普通监督残差学习，不称样本外错误监督。源验证先选C再选残差有阶段性复用，原队列已反复开发。参数/函数约束不同，如实比较，不宣称严格同容量。测试不参与选择或改协议，不因第一阶段测试结果改变第二阶段。

## 输出与查看

两阶段分别输出选点/轨迹/逐fold与pooled/逐类/VV-VH覆盖/采集/预测。报告新增头相对B，并比较signed对额外score/std；固定预测pass和MMSI配对区间不覆盖训练与选择不确定性。

汇总时核实：三折validation共915次折次暴露、816个唯一观测，不能解释为915个独立样本；source J仍在各自validation内计算，未改变划分。test为4744个互斥观测。提取分别保存moments_pol各极化结果和moments求和结果，本轮模型只用sum。

固定seed17，每阶段/每臂按instance_id首例选rescued/harmed/CC/WW；缺少翻转则补持续对错并报告缺失。原VV/VH、实际160输入与footprint分别显示并实际view_image查看，注明原窗、valid及分差。不凭外观或粗map命名物理机制，不把案例比例当总体。

提取验证完成和训练数值先回报，再完成FINDINGS。无Jacobian重投影、参考CDF、新region组合或额外ablation。
