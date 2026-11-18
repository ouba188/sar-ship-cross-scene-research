# PCEA梯度路由修正独立审阅

2026-11-18。初始范围为新分析脚本、run.py/PROTOCOL的只读核对与smoke记录；全部训练及评价完成后追加了结果表、现有NPZ的CPU复核与原图回看。未运行模型、GPU或训练；未改run.py、历史实验、正式模型或论文。实际结果见末节。

## 结论

未发现阻断本轮三次有界训练的实现或变量控制问题。允许检验的结论是：在同一配方下，限制额外纠错/保护损失的直接梯度注入范围，是否改善完整模型结果。不能提前认定旧梯度路径是全部失败原因，也不能把本轮称为完全隔离共享表示的因果实验。净收益不成立时保留负结果，不自动增加种子、倍率、学习率或结构搜索。

## 路由与变量控制

- `run.py:44`先反传off分类项，再计算固定起点教师和同增强on路径。教师在no_grad下运行，off/teacher分差目标继续沿用旧correction中的detach。单极化或无有效回写区域样本的额外项为零，CE仍覆盖全部样本。
- `run.py:59`按实际参数名前缀`ccrel.`、`exchange.`选出目标，`autograd.grad(multiplier*extra, targets, retain_graph=True)`只返回这些参数的额外梯度。随后正常反传on分类项，再将返回梯度加到目标`.grad`；不是全局detach输入，也没有冻结骨干。
- 共享reader所有权已追溯：`ccrel_pcea_joint_20261116/model.py:20`将guide.value做deepcopy，CCREL参数不与原guide.value别名共享；同文件`:70`删除exchange旧value。`pcea_class_local_20261117/model.py:90`仅在net.ccrel注册reader，`:91`使用object.__setattr__让exchange引用它，不重复注册子模块。优化器groups_for和路由targets都来自net.named_parameters()；本地PyTorch实现默认remove_duplicate=True。源参考、query摘要与局部detail复用同一CCREL参数时，各图路径贡献正常求和，参数本身只入组/手工加extra一次，未发现重复计数。
- 每个12样本微批的分类项系数为0.5*multiplier，extra系数为multiplier；正式训练multiplier=0.5，两个微批组成24样本后才更新。与旧guided的损失权重和平均口径一致，没有多乘或漏乘一次微批系数。
- `run.py:65`的before快照位于两路CE反传完成之后、手工加入extra之前。逐参数比较的是目标CE+extra、非目标CE，审计基准正确。
- **CE-only限定于裁剪前的直接梯度来源。** `run.py:190`仍执行原有全局clip5；触发裁剪时，目标梯度会影响公共缩放系数。后续CE联合训练也会改变共享表示。因此不能宣称骨干最终更新或后续表示完全不受额外分支间接影响。此项是保留的原配方耦合，不是新增实验变量。
- 样本顺序、增强生成、240更新、学习率/调度、AdamW和参考重编码沿用原配方。runtime复核旧Full/generic及paired_ce/guided的相应配置和训练ID。共同起点仍为正式selected_s17；17/29/43是续训顺序/增强重复，不是三个独立预训练起点。
- 继续按source-inner BA/OA/-NLL选点，step0可选。`run.py:219`先设置新ROOT与仅routed的ARMS，再复用评价器；其三份receipt门槛与新缓存位置相符，最终status为3新fits、12复用fits、12外部评价条件。没有依据外部结果重选点或扫描倍率。

上述行号对应审阅时的本目录run.py；固定教师是起点分类器，不是像素干预教师。

## 已读取的smoke证据

来源为本目录`smoke.json`，并非本审阅者重新运行：

- 初始on/off/固定教师replay最大误差0。
- 两步的target与non-target路由逐参数最大误差均为0；teacher/off目标detach，无active时额外损失0。
- 第一步reader额外梯度范数0、exchange为0.000197205；零输出初始化下不能据此判定reader断路。
- 第二步reader额外梯度范数0.000139492、exchange为0.000199055；共享骨干和原模块分类梯度仍非零。
- 峰值显存4.792618 GiB。该smoke支持接入与局部梯度来源正确，不证明240步全程隔离、外部收益或唯一失败原因。

## 分析交付与执行

新增`analyze.py`只导入旧CPU分析工具，不导入run.py、torch或模型。先等待三份routed receipt，并核对240步完成及全部所需缓存，之后才读预测并输出结果。缺少availability缓存、恢复审计或split不一致时明确失败，不推测、不自动补推理。

```powershell
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -B 'E:/临时会话/paper_draft_20261108/pcea_routed_correction_20261118/analyze.py'
```

- `aggregate/metrics.csv`：五组source-inner、160、198逐种子OA/BA/固定8类macro-F1/NLL，共78行。旧Full/generic没有的inner-off不伪造或重新计算。
- `aggregate/paired.csv/json`：routed分别对guided、paired_ce、generic、Full及自身off，三个split逐种子，共45组；记录救回、误伤、持续对错、预测变化、分差及指标差值。类别/港口统计不含样本ID。
- `aggregate/source_inner_on_off.csv`：paired_ce/guided/routed各三种子，使用on选出的同一checkpoint读取inner/inner_off，不为off单独选点。
- `aggregate/availability_paired.csv/json`：三split、五种对照的双/单极化分层，共90行；源inner单极化组为空时n=0、OA=null。160按原native availability加确认恢复destination，198仅读取既有native NPZ的available，源inner同样按原cache加恢复审计。强制检查160为151/9、198为171/27、inner为1000/0。无像素解码或模型推理。
- `aggregate/summary.json`：三种子均值/样本SD、样本组成、分层结果和案例覆盖。种子重复不当作额外独立船只；双极化不等同于必有有效回写区域。
- `gallery/`与`cases_local.json`：最多12个不同corpus query，仅外部样本；优先routed对guided/paired_ce及自身off，尽量覆盖救回、误伤、持续正确和持续错误。没有翻转仍保留稳定案例；使用native160 VV/VH和实际固定16源参考库，保留输入恢复来源与参考合法性。生成图片不等于已完成实际图像查看，需后续view_image复核。
- 仅`aggregate/`可作为公开结果；gallery、单样本预测/ID、路径和参考身份留在本地。未读取3075测试预测或像素。

## 验证与解释边界

训练前独立CPU合成检查通过：语法与无torch导入、78条件缓存路径、45配对/90分层、9条源内on/off摘要、空子组处理、最多12个不同外部query及四类结果覆盖、receipt门槛先于metadata和任何输出。当时未运行真实分析或提前查看新实验结果；训练后的追加核查如下。

source-inner是曾被旧编码器见过的开发选模集；160是历史开发数据，198是已分析过的同历史过境新船，不是独立新场景。恢复旧guided误伤不自动等于超过paired_ce的新收益。跨模型比较与同检查点on/off分别报告；单极化跨模型变化可能来自共同网络训练，不能归因于直接PCEA回写。原图外观或参考示例不能单独证明物理机制或特定参考的因果作用。

## 最终结果独立核查

三份receipt的source_only均为true，最佳步为180/210/240；各receipt计时合计2220.384222秒，约37.006分钟。主代理完成78行指标、45组配对、90行可用性分层及12张原图面板。本审阅者读取实际结果表，并直接对现有NPZ核对ids/label、预测正确数与关键三例分差，没有再次运行推理或训练。

| 队列 | routed逐种子正确数（17/29/43） | 对paired_ce净正确数 | 对旧guided净正确数 | routed on/off救回/误伤 |
|---|---|---|---|---|
| source-inner1000 | 613 / 616 / 613 | 0 / +1 / 0 | 0 / −3 / 0 | 0/0，1/1，1/1 |
| validation160 | 72 / 70 / 71 | 0 / 0 / 0 | 0 / 0 / +2 | 0/0，0/0，1/0 |
| expansion198 | 123 / 123 / 123 | 0 / +1 / 0 | 0 / +1 / 0 | 0/0，0/0，0/0 |

- 160：routed平均OA44.3750、BA38.821274、固定8类F1为32.354085，与paired_ce/generic逐种子top1完全一致。对旧guided的均值改善OA+0.416667、BA+0.694444个百分点来自seed43恢复两例旧损失，不是超过paired_ce的新增准确率。对Full则seed43新增一例，与已有generic/paired_ce的成功一致。
- 198：routed平均OA62.121212、BA58.562339、F1为54.809080；对paired_ce/generic/Full分别均为OA+0.168350、BA+0.092593、F1+0.061923个百分点。仅seed29新增一例集装箱正确，没有新增误伤；另外两种子无相对paired_ce翻转。NLL对paired_ce略差（1.195751 vs 1.195654），不属于所有指标全面改善。
- 直接贡献与训练差异必须分开：198的唯一新增正确在同一个routed checkpoint关闭PCEA时已经正确，开启后margin反而略降；198三个种子全部on/off top1一致。160 seed43存在一例真实on/off直接救回，但paired_ce已能正确判别。另一例旧guided误伤在routed on/off均正确。因此不能把“恢复旧损失”“超过分类对照”“当前回写直接救回”合并为同一条因果证据。
- 已读取的可用性计数为160=151双/9单，198=171双/27单，inner=1000双/0单。外部正确性翻转均在双极化组；外部单极化on/off最大logit差均为0。源内仍有误伤，不以外部无误伤推断所有数据均无代价。
- 图00–05已由独立审阅者逐图view_image，06–11由主代理逐图查看并提供文字，随后核对cases数值。12图为3个不同救回、5个持续正确、4个持续错误，详细ID、实际参考、可见事实和精确margin仅保存在本地DATA_REVIEW_LOCAL.md。本审阅者未重复查看主代理另行完成的4个源内案例。

### FINDINGS一致性

已核对FINDINGS的10行外部指标表、救回/误伤表、源内均值、最佳步及约37分钟计时，数字与结果表/缓存一致。核对时发现一处类别笔误：“新增正确的原油轮”应为“新增正确的集装箱船（类别4，先前误判拖轮）”，已通知主代理纠正；这不改变统计数字，但影响案例解释。本审阅未修改FINDINGS。

最终判断：没有新增实现阻断。路由修正支持保留为开发候选，表现为旧guided退步被修复及单个种子单例外部新增；尚不足以证明稳定额外交互收益或把原失败唯一归因于梯度路由。保持正式模型不变，本轮停止于已批准三fits，不据此自动继续调参。
