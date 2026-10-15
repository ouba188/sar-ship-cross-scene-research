# 候选隐向量使用说明

本轮真正导出了显式 32 维向量，不再只是八类分数残差。它是相对稳定的任务候选表示，尚不是经过因果识别证明的纯因果因素。

## 文件

- `reliable_31415_vectors.npz`、`reliable_27182_vectors.npz`：可靠关系加权的同船稳定约束版本。
- `uniform_*_vectors.npz`：普通同船稳定约束对照。
- `relation_*_vectors.npz`：可靠类别关系约束，但不强制整个向量同船靠近。
- `task_*_vectors.npz`：同容量任务学习对照。
- `pca_*_vectors.npz`：未训练的源域 PCA 初始化；两份种子文件内容相同，不算两次独立实验。
- `INPUT_NORMALIZATION.npz`：原始 512 维 R18 特征进入本轮模型前的归一化参数。
- `{arm}_{seed}.pt`：含 selected 和 last 两个检查点。本文结果使用源验证选出的 selected，不根据目标结果挑选。

每份向量文件包含：

| 键 | 含义 |
|---|---|
| z | 37893×32，观测对应的实际候选向量 |
| obs_ids | 对应研究目录 observations.csv 的零起始行号 |
| manifest_rows | 原始特征清单中的行号 |
| basis | 512×32 正交投影基 Q |
| scale | 32 个源训练投影标准差 |
| input_center | 原归一化特征在源训练集上的中心 |

37893 包括已缓存的全部观测，不是 37893 个训练样本。本轮训练仍只有源训练 1296 张；目标观测只在训练结束后导出。

## 从原始特征得到向量

设原始 512 维特征为 f，原归一化参数为 c0、s0，文件内中心为 c，投影标准差为 d：

```text
x = (f - c0) / s0
u = x - c
z = (u @ Q) / d / sqrt(32)
```

可以得到输入特征空间中对应的保留部分与剩余部分：

```text
u_kept = (z * d * sqrt(32)) @ Q.T
u_remaining = u - u_kept
```

Q 的列正交，因此这两部分是该输入坐标系下的正交分解。**剩余部分不能直接叫纯噪声或非因果部分**，其中可能保留有用类别信息。

读取示例，当前工作目录设为本文件所在目录：

```python
import numpy as np

space = np.load('reliable_31415_vectors.npz')
normalization = np.load('INPUT_NORMALIZATION.npz')
z_all = space['z']
obs_ids = space['obs_ids']

# raw_features must use the same frozen R18 extraction and original preprocessing.
def encode(raw_features):
    x = (raw_features - normalization['raw_center']) / normalization['raw_scale']
    return ((x - space['input_center']) @ space['basis']) / space['scale'] / np.sqrt(32)
```

这些坐标没有逐维物理命名。旋转等价的基可以表示相同子空间，不要把“第 7 维”直接解释成固定物理因素。
