# 强分数校准后的最小局部增量协议

2026-10-23，训练前锁定。遵循最新18头授权，覆盖先前27头及校准crossfit建议。仅本目录写入，不改父文件/旧头、不Git。

## 冻结起点与源数据

S为旧A+D，T为 `full_model_integration/models/fold{fold}_score_only_seed{seed}.pt` 已选残差变换（含原scaler）：T(S)=S+g(center(S))，强度固定1，B=T(S)。不另训T，不用test选弱化强度。

oldseed0/1/2对应seed17/29/43；三折源train4392/4419/4418、val310/286/319、test1561/1632/1551均保持。复用既有load_old映射处理obs31237及3103行位置偏移。

训练B来自T(oldAD_OOF)，局部矩来自新支自身inner-held OOF。val/test取T(oldAD_outer)及outer矩。旧分数3包、新矩4包独立身份OOF合法，不要求fit完全相同；组合不是全链过境OOF。

**T已在这批源OOF分数及标签上训练，T本身非OOF。** 两臂共享普通两阶段源监督及其乐观性，不能称全栈样本外错误监督。旧OOF D与最终全源D的激活差异、源val复用也保留；不能把后果全部归于跨域。

## 两臂与训练

| arm | 冻结B之外的新残差 | 参数 |
|---|---|---:|
| local | standard_controls.Head('free_shared')，11→16GELU→1；descriptor以中心化B替换分数项，其余真实VV/VH sigma/kappa/flags保持；输出去类公共项，不乘kappa、不tanh | 209 |
| extra_score | run_training.Head('score_only')，8→16GELU→8；只输入中心化B | 280 |

3fold×3seed×2臂=18头，各200步full-batch AdamW(.003,.01)，jointCE=.5普通CE+.5八类均衡CE。scaler仅用本折源训练B与OOF矩拟合；不更新T的scaler。末层零初始化严格回到B。sourceval J=.5OA+.5BA，每5步且含0，精确比较、tie最早。18个选点全部保存后才test。

不扩aggregate、强度网格、W/Jacobian或新attention。新W是独立辅助分类方向，不是旧AD权重分解；旧AD不含全部历史支持适配。该实验检验局部证据在强分数参照后是否仍可兑现增量，不把诊断或普通残差监督冒称新算子。

## 验证与分析

逐fold/seed精确重放full_model_integration的score_only验证/测试分数及top1，记录浮点CSV往返误差；零头、选点重放、有限值、只优化新增头。保存18checkpoint、完整曲线及预测。

相对B、local相对extra_score分别报告OA/BA/NLL/Brier，逐seed/折/类/港/过境/VH；源val与test分开，val915折暴露另给816去重描述。固定预测下2000次pass/MMSI聚类配对区间，seed20261023；不把3seed当独立样本。

原图按每臂seed17、test四结果rescued/harmed/CC/WW的first instance ID取8例；缺翻转则记录并补CC/WW，不另挑seed。实际看VV/VH及native160 crop/pad、记录原路径、valid、oldrival与finalrival/finalwinner、实际残差分差。这里oldrival属于冻结B，不是未校准旧AD。

训练完成先报关键数据，再完成分析与报告。计时区分缓存训练/评价与已准备输入的小头前向；不将缓存运行耗时称端到端或新增部署成本。无完整端到端测量时明确未测。
