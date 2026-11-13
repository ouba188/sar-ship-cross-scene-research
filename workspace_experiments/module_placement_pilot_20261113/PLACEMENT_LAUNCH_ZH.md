# 插入位置实验启动记录

2026-11-13，已启动OSS2七组位置开发筛选。

- 服务端目录：`/root/autodl-tmp/sar_server_experiments_20261113`。
- 进程PID：5203；PID文件：`placement_pilot.pid`；日志：`placement_pilot.log`。
- 命令：`python -u placement_study.py train --dataset oss2 --seed 17 --epochs 12`。
- 原生输入和固定划分不变，七组具有共同初始化、相同参数量、相同训练次序与预算。
- CPU/CUDA均已验证当前位置等价、七组实际插入位置及梯度。
- 新代码位于本地实验包根目录，已作为小文件增量上传；原1.39GiB压缩包未重打包。
- 状态以服务器 `runs/placement_pilot/status.json`、日志和各组receipt为准。此记录不是训练完成声明。
- 本轮不评估测试集、不自动替换正式模型、不自动扩至24港；共用输入缓存可能预处理测试像素，但测试标签和测试预测不用于训练或选点。

完成后须比较配对验证结果及曲线、查看救回/误伤/持续正确/持续错误原始VV/VH，之后冻结候选再补多种子及当前完整训练流程验证。
