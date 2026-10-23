# OBB动态区域：独立源数据与坐标核对

## 可执行结论

源端8750条观测已完成几何观察；6例12张VV/VH原窗及native160覆盖面板已由本agent实际view_image查看。rhoL=.25、rhoW=.5、1原像素线性软边没有造成普遍外围缺失，具备开展既定配对的可读支持；**这证明区域可生成和有覆盖，不证明分类收益或纯船体/背景分离。** 主流程随后已冻结上述系数及 `min_neff=4` 并放行训练，本支线未训练、不读取test分类成绩、不更改成员或标签。

没有发现需要阻塞当前实现的裁补偏移或Voronoi积分错误。源端最紧裁例的轴向外围仅保留约45.3%，必须保留覆盖量；不扩大输入来弥补。layer1较layer3更密，但RF35仍大于多数目标短轴以及许多目标长轴，区域名称只指读出权重位置。

## 实际路径与字段

以下ROOT为 `E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023`，OLD为 `E:/临时会话/same_vessel_gallery_20261019/local_condition_module`，不是scene_aware目录下另一个同名路径。

|内容|文件及字段|本轮用途|
|---|---|---|
|队列|ROOT/cohort.csv：instance_id、obs_id、image_vv/vh、class_id、mmsi、product、time|图像路径与观测身份；不按过滤前行号跨表拼接|
|划分|ROOT/baseline/fold{0,1,2}_inputs.npz：train/validation/test、instance_ids|沿用实际外层身份/日期划分；源端按索引并集去重|
|每极化几何|OLD/view_snapshot.json，obs_id→images[pol]→priors.detections[target_index]|corners、xywhr、confidence、shape_usable；每pol自己的目标框|
|关联情况|priors.match_status；OLD/crossview_alignment.csv|区分缺框、邻近关联和跨极化冲突，不把所有valid视为确认身份|
|原尺度变换|OLD/native_input.py：crop_pad；ROOT/coverage.csv|crop_x/y、crop_width/height、pad_left/top、valid、disabled_alignment、input_scale|
|真实footprint|OLD/prepare.py：masks_for(view)三mask之和，经同一crop_pad|和为原图有效范围，padding为0；不是只有目标框有效|
|旧C4接口|ROOT/prepare_features.py、calibrated_signed_transfer/extract_moments.py|像素与mask同步rot90，每view先非线性读出再平均|

角点导出长轴单位向量和相邻短轴，L≥W；不直接假定xywhr中的w就是长轴。角点重建最大误差约4.09e-5像素，属于浮点记录量级。源端所有原窗均≤160，当前 **crop_x=crop_y=0，额外裁剪损失为0**，偏移实际来自pad；通用关系仍为 `x160=x_native-crop_x+pad_left`，y同理。不得因当前无裁剪就省掉变换或共享两pol偏移。

## 队列与OBB可用率

|范围|源train并集|源validation并集|源并集|历史test并集|
|---|---:|---:|---:|---:|
|独立观测数|7934|816|8750|4744|

源和test无观测交集。每fold train为4392/4419/4418、validation为310/286/319；不能将它们求和当独立船数。源端图像原窗范围：VV64至152像素，VH64至128像素。

|源端极化|目标OBB可用|继承有效|禁用的跨极化冲突|缺目标匹配|
|---|---:|---:|---:|---:|
|VV|8750/8750|8750/8750|0|0|
|VH|7799/8750|7774/8750，88.85%|25|951|

VV目标匹配状态为8743条inside_unique_box、7条near_anchor_not_containing；VH为7784条inside_unique_box、14条near_anchor_not_containing、1条inside_multiple_boxes_ambiguous、951条no_target_match。沿用历史选择，不在本轮自行清理这些行。

有效VH7774条中，已有跨极化核对状态target为5077、unknown为2595、outside为102。因此“合法可用率”指继承既有标志，不是7774条都已证实同一目标身份。25条other冲突继续禁用，未用新区域暗中恢复；unknown/outside也不自动改成新排除规则。

## 固定候选覆盖

下表来自独立 `source_region_coverage.csv`：覆盖比例是连续硬OBB矩形及扩展差集与真实footprint相交面积/计划面积；软权重采用与当前实现相同的1px过渡。它与主实现按软mask计划质量算的覆盖定义略不同，不能要求末位完全相同。

|极化/区域|有效记录|覆盖<90%|覆盖<50%|零覆盖|软质量中位数(px)|原朝向n_eff中位数|
|---|---:|---:|---:|---:|---:|---:|
|VV body|8750|9|0|0|231.39|18.15|
|VV end|8750|30|8|0|115.67|11.27|
|VV side|8750|11|0|0|231.48|21.99|
|VH body|7774|5|0|0|227.60|17.96|
|VH end|7774|20|5|0|113.66|11.10|
|VH side|7774|6|0|0|227.64|21.84|

独立表的n_eff仅原朝向，不冒称四旋转最小值。主实现 `source_coverage_summary.json` 逐C4最小n_eff中，body最小4.991、side最小6.067，end有2条低于4：

- returned_local_knowledge_0016658 / VV / end：3.612838。
- returned_local_knowledge_0015887 / VH / end：3.966449。

已冻结门槛会将相应区域标为无效，不是删除观测或整体VV。近小目标的end最薄，不能把增加grid密度解释成获得独立新船体细节。框边平滑只减少离散跳变，也不修复检测定位误差。

## 六例实际视查

案例只按源VV几何分位数与首ID确定，未读分类结果。6例为6个MMSI，除紧裁例外五个大小/比例条件覆盖源train，紧裁例为2025-09-01源validation；没有test案例。选择明细在 `selected_cases.csv`，各pol路径与裁补字段在 `review_cases.csv`。

|面板|实例|选择条件与可见事实|
|---|---|---|
|case1.png|pair_0005058|面积p05附近；VV框12.6×7.1，64窗。body覆盖中央短亮块，end/side有小但非零支持，未落入padding。|
|case2.png|returned_local_knowledge_0005328|面积p95附近；VV88窗、VH80窗，分别pad36/40。亮区向框外延伸，几何外围会读到这些响应，不可叫纯背景。|
|case3.png|returned_local_knowledge_0018363|窄宽p01附近；VV10.6×6.1，画面还有其他分离亮团。窄侧条有支持但小于RF，不能宣称精细船体分割；当前区域没有覆盖所有邻亮团。|
|case4.png|returned_local_knowledge_0022188|长宽比p99附近；128窗，VV50.6×13.1、VH63.5×15.3。两pol检测长度不同，必须独立建区；长条主体与两端/两侧可见分开。|
|case5.png|returned_local_knowledge_0018976|最紧端区；64窗，亮结构延伸到右图边，OBB也越出原窗。VV end覆盖约45.31%，只剩一端的主要可见支持；不把归一化后读出称完整双端。|
|case6.png|returned_local_knowledge_0018362|近方形p01；VV10.9×9.9，VH10.3×9.5。近方形长轴方向的语义稳定性较弱，两pol侧向区域方向也不同；这是检测几何边界，不据外观命名首尾。|

原图显示当前rho参数能够形成不同读出位置，且没有普遍退化为无覆盖；因此支持固定候选进入配对训练。是否优于普通浅层或固定模板，必须由后续同信息对照回答。几何可见性不预先保证涨点。

## 实际网络及实现核对

本地 `D:/Program_documents/Anaconda_envs/envs/ClearSAR/Lib/site-packages/torchvision/models/resnet.py` 的实际ResNet50层结构加一次零输入shape前向确认：layer1输出256×40×40，stride4，理论RF35，按像素中心索引首中心0。未做预训练图像特征提取、未使用GPU训练。

中心0,4,...,156对应canvas上的Voronoi边界 `[-.5,2,6,...,154,159.5]`。像素x=2横跨首两cell边界，各分一半质量；与旧整齐4×4分箱中心1.5不同。区域权重是分片常数像素mask在这些cell内的积分，不是认为卷积只读取该cell。独立积分质量守恒最大误差3.82e-11。

已只读审阅新 `common.py` 的geometry/native/soft_regions/cell_mass/rotate_geometry：

- 按每pol角点中心加pad-crop偏移，长短轴取角点边，正确。
- 1px过渡及1.5L/2W外框减body与本候选一致，正确。
- 逆/正C4遵从图像索引：一次rot90为(x,y)→(y,159-x)，角度减pi/2；mask同步，正确。
- cell_mass的边界端点及半像素积分与origin0一致，正确。每旋转应重新积分，不能假设旧weights旋转即可精确等价。

六例12张图的crop_pad实际返回值逐字段对照原coverage，全部一致。当前未发现该坐标模块的可达阻塞；reader/提取/训练后续针对性审阅另记，不把尚未检查的实现称已通过。

## 交付与范围

`review_data.py`、`source_geometry.csv`、`source_region_coverage.csv`、`coverage_summary.csv`、`fold_membership_counts.csv`、`selected_cases.csv`、`review_cases.csv`、`summary.json`、case1至case6.png均只在data_review目录。脚本约33.20秒，属于几何覆盖和制图成本，不是新模型推理成本。无新训练、无test分类成绩访问、无父目录修改、无Git。本轮不重复覆盖统计，转做针对性实现审阅及后续固定案例准备。

## 后续授权阶段已完成

上述“无test分类成绩访问”限最初源覆盖阶段。主流程完成36头和冻结test后，本支线按后续明确授权读取已选预测、完成实现补审与六个test案例，不改源配置。结果见 `IMPLEMENTATION_REVIEW_ZH.md`、`CASE_REVIEW_ZH.md`：4例相对fixed、2例相对B；六图已实际查看，固定区域摘要干预完成，无新训练。不要将相对fixed保护已正确样本写成相对B的新救回。
