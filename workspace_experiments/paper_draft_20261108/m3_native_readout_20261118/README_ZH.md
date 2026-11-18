# M3原生浅层取证候选

代码修复与验证入口，不覆盖正式模型或原检查点。结果以FINDINGS_ZH.md和原始记录为准。

## 改了什么

legacy沿用原路径。fine_anchor保留layer1的原生40×40特征，在40网格上直接做槽位积分；不先降到20×20。fine_live在此基础上，将512维部分改成M1后、当前layer2末block输出；不再使用anchor末端投影生成这一部分。

两路特征仍拼成每槽768维，VV/VH独立取证。layer2从20到40的插值仅做坐标对齐，不称为新增真实细节。新增的真实高分辨率信息来自layer1。20槽数量、区域身份、原mass、合法参考条件、匹配器和layer3末端10×10回写均不变，未新增模型参数。

CCREL/PCEA的后段摘要/更新不被冒充为本次的新细特征源。本轮也未增加动态门控、首尾预测或替换为尚未训练的轴向匹配。槽内平均及深层回写依然存在，因此修的是“取证前不必要的空间降采样和M1旁路”，不是声称已恢复一切局部信息。

## 代码

- model.py：NativeReads从原生160掩膜生成40网格；Model.extract执行两种修正读出。
- run.py：重放、梯度、可辨别细节检查，同预算源端续训，以及同权重切回旧读出的诊断。
- PROTOCOL_ZH.md：预先固定的研究范围与对照。
- smoke.json：接口验证结果。旧路径重放及关闭M3的差异为0；新增读出可以区分旧空间降采样核的零空间变化；fine_live的描述梯度直接抵达M1。

沿用工作区已有ClearSAR环境，不新增依赖。命令示例（先smoke后train；已完成训练不重跑）：

```powershell
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' 'E:/临时会话/paper_draft_20261108/m3_native_readout_20261118/run.py' --phase smoke
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' 'E:/临时会话/paper_draft_20261108/m3_native_readout_20261118/run.py' --phase train
```

训练脚本默认只运行已固定的两个seed17短续训。已有continue_s17对照直接复用，并核对训练顺序、起点、有效参数组、步数和选模队列。验证缓存用于恢复同一运行，不适用于修改配置后的实验；修改设计应使用新输出目录。

图像、私人样本标识、逐例预测、源划分和权重仅本地保存；只同步代码、协议、汇总与分析结论。
