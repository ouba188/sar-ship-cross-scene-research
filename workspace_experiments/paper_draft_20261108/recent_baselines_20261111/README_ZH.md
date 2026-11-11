# CPINet强SAR基线独立复现

状态：OSS2四分类三种子已完成。结论见FINDINGS_ZH.md，完整数字见RESULTS_ZH.md，原图入口为本地index.html。当前Full OA更高、CPINet BA更高，不写成全面胜出。

本目录对应当前OSS2四分类跨场景协议。先读PROTOCOL_ZH.md和CPINET_IMPLEMENTATION_SPEC_ZH.md，区分原文结构、公开FBC组件补全和本任务适配选择。不是作者官方代码，不复用历史e207自拟模型。

## 执行顺序

在E:/临时会话工作目录，使用现有ClearSAR环境：

```powershell
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -X utf8 paper_draft_20261108/recent_baselines_20261111/data.py
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -X utf8 paper_draft_20261108/recent_baselines_20261111/train.py --smoke
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -X utf8 paper_draft_20261108/recent_baselines_20261111/train.py
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -X utf8 paper_draft_20261108/recent_baselines_20261111/evaluate.py
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -X utf8 paper_draft_20261108/recent_baselines_20261111/gallery.py
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -X utf8 paper_draft_20261108/recent_baselines_20261111/summarize.py
```

训练只读取训练/验证像素，测试缓存准备不涉及模型选择；evaluate须等选定LR的三次300epoch训练全部完成。已有完整训练receipt会跳过对应运行；中断从每10epoch的resume.pt续跑。旧C/16偏差目录和optimization_probe均不得作为正式结果。

## 依赖与数据

复用本地numpy、torch、Pillow、pandas、matplotlib；不需要新增依赖。需要既有m1_coverage_20261101/manifest.json、groups.json及原图路径；data.py生成native_cache。评估需staged_ablation_20261109中的selected.json及Full/Plain/SE逐seed evaluation.npz，不重新训练这三组。

Git只归档代码、协议、文献核查、曲线、训练receipt及汇总表；原始chip、缓存、权重、逐样本预测与本地路径不上传。
