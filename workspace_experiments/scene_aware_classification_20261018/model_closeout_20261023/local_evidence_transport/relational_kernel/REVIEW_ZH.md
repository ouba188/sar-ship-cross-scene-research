# 区域关系kernel限定审阅

2026-10-23。只读`run.py`、其调用的`match_readout.py`和`train_readout.py`，未修改代码、未训练、未新增测试。结论：在要求的广播、端部mask、availability与kernel实现范围内，未发现阻塞性逻辑错误。下列是实现事实与必须保留的解释边界，不是更广创新认证。

## 实现对照

- `compute`：query从[B,P,20,D]扩展为[B,1,1,P,20,D]，reference为[B,C,K,P,20,D]。`dot`的省略号广播保留类别/参考/极化维，结果[B,C,K,P,8,8]；VV/VH没有串接误匹配。
- body仅用前8token求一次带null的T，各臂body score原样共用。side/end只替换T上的value，没有用关系kernel重新求T。
- ordered确为`<b,b'><n,n'>`；role_contrast确为该项减`<b,n'><n,b'>`，两次交叉dot的i/j方向正确。四个向量来自同一个源fit PCA32坐标，交叉内积有一致定义。
- end以`END=[16,17,16,17,18,19,18,19]`映射到8body位置，并乘`IS_END=[1,1,0,0,0,0,1,1]`。因此重复占位的内部槽不会贡献；不是把4end token重复计数成8个独立端部。
- `pair_score`用两侧区域mass>0形成keep，返回未重新归一的`sum(T*value)`及`sum(T)`。anchored/ordered/role_contrast完全共用该mass。此处end质量是body T经过端部/有效性mask的子集，不声称与body总质量相同。
- availability按body两侧存在且本区域两侧存在计算，在reference维K计数归一；只剔除结构缺失，不因为低匹配质量剔除参考。全缺失计数0时结果0。主流程报告的availability-anchor重放误差4.47e-7与这一读取路径一致；本次没有重跑该断言。
- mispaired仅将side特征与mass一起按每观测/极化固定排列，body与T不动，end直接复用role_contrast。side总有效性不变，故共用availability计数合法；局部有效槽位置可改变，具体继承匹配mass不保证不变。它不是“只改内容、保持每一条局部权重”的纯干预，也没有破坏end依附。
- 局部12槽是2pol×3region×(score,mass)，实际head输入另有3个base比较量，共15维；复用15→12→1、205参数共享head。各臂同初始化种子、零末层、200步、同优化器；scaler只拟合utility_train。selection选J，含0步且严格大于才替换。`curves.csv`本实现从step5记录，未记录epoch0曲线行，但0步确实参与选择，不应把它写成未参加。
- external路径复用源投影、参考、head/scaler；标签仅在`evaluate`计算指标，不用于fit/选步。这里只确认代码数据流，不替代全链历史开发/独立性审计。

## 数学与解释边界

令`Phi(b,n)=b n^T`，则ordered是其Frobenius内积。令`Psi(b,n)=(b n^T-n b^T)/sqrt(2)`，则role_contrast恰为`<Psi(b,n),Psi(b',n')>`。这是已有张量积及反对称二阶特征的内积，不是新wedge或新kernel数学；值可负，并不意味着负可见性质量或错误运输。

同侧b=n时Psi为0；交换一侧b/n时局部kernel变号，两侧同时交换不变。这些是**固定局部输入下的kernel性质**，不是完整分类器交换body/context仍满足同一性质：完整算法的T本来由body建立。坐标pi重新标号时须同时重排body、side、end、mass和T；公式对一致重标号成立，不等于实际R50对图像旋转严格等变。

RF重叠仅说明输入可能混合，不推出描述子复制。主流程源统计side内积均值VV .108/VH .028，>.8仅1.28%/.27%，不支持“普遍body/context共线，wedge已去除重复信号”。wedge会压低接近平行的二阶表示，但这种分量可能也有类别价值；不能称物理去噪或因果背景剥离。

## 最近先例与本轮阅读

已有CrossTransformer的key/value分工和bilinear关系表示足以限制宽泛的新意表述。为核实后者，本轮通过requests+pypdf实际读取Lin等《Bilinear CNN Models for Fine-grained Visual Recognition》，ICCV2015，官方PDF第3–4页，§2.1–2.2及训练部分：位置特征外积、跨位置sum pooling、signed square root/L2及可反传训练。未声称它使用当前固定body T或反对称区域差。

一手PDF：`https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Lin_Bilinear_CNN_Models_ICCV_2015_paper.pdf`。Web工具本轮仍未返回可用正文；实际内容来自上述PDF内存解析，无大型文件保存。没有额外获取DMN4，也不把主流程二手描述当一手方法。

这里实际不同的是在已有body匹配后，用带明确body/context角色的局部二阶value替换普通context余弦，随后仍由原T汇总；不是B-CNN整体模型复现，也不是发明二阶学习。现阶段只是有解释的固定算子候选。

## 与已完成结果一致的收束

主流程已完成结果通报：外部role_contrast OA44.8672/BA39.4438，valid anchored 44.7913/39.2491，ordered 44.8455/39.2542，mispaired 44.9214/39.7334。以上为主流程报告数值，本次未再次复算或用其改配置。

这不支持附着身份必要性，也不足以证明反对称约束整体优于普通乘积；源端直接side BA也是ordered39.20高于role_contrast36.84，不能挑外部小增量讲成机制闭合。mispaired同时改变side内容与有效性依附且保留end，需按其具体范围解释。

建议保留本轮全部结果，不改公式、不续搜手工kernel。若继续发展，应回到真正的局部比较表示学习及其相对于body-only/独立背景的条件增量检验，而不是以新名称包装当前wedge；尚未开始该训练。
