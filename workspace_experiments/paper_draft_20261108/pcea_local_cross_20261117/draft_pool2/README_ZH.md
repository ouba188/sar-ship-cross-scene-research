# 修正前草稿归档

此目录保留本轮最初的2x2区域降采样实现与运行记录。该映射和既有origin0 writer网格不完全一致，发现后停止队列，统一修正后在父目录重启B/C/D六次拟合。

- independent、cross_mean各两档学习率，四个完整fit。
- cross_local/base完成epoch1后中断，没有完整receipt，不是有效完成的D对照。
- 本目录的checkpoint路径/selection按原始记录保留，移动后不作为父目录选优入口。
- model.py/run.py是修正前源代码快照，其相对依赖路径按原目录保存，不是可在本子目录直接执行的独立工程。
- 权重和预测NPZ只保留本地。不得将草稿与修正后成绩混合选优，也不能把中断当作收敛负结果。
