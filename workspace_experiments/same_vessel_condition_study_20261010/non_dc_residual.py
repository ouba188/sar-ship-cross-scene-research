"""Matched non-DC residual control; preserve previous spectral experiments."""

from spectral_residual import (
    BASE, OUT as PREVIOUS, PRIOR, OLD, SEED, SpectralResidual,
    build_blocks, encode, fit, phi, residual_head, metrics, save_json,
)
import time
import numpy as np
import pandas as pd
import torch
from channel_operator_report import table

OUT = BASE/'non_dc_residual_20261010'


def remove_dc(h):
    return h-h.mean(axis=(-2, -1), keepdims=True)


class NonDCBranch:
    def __init__(self, maps, scale):
        self.maps, self.scale = maps, scale

    def __getitem__(self, rows):
        h = np.array(self.maps[rows])
        return remove_dc(h)*self.scale[None, :, None, None]


def report():
    old = pd.read_csv(PREVIOUS/'results.csv')
    new = pd.read_csv(OUT/'results.csv')
    rows = pd.concat([old, new], ignore_index=True)
    for arm in ('head_only', 'middle'):
        ref = rows[rows.arm.eq(arm)][['port', 'draw', 'BA', 'OA']]
        rows = rows.merge(ref.rename(columns={c: c+'_'+arm for c in ('BA', 'OA')}), on=['port', 'draw'])
        for metric in ('BA', 'OA'):
            rows['delta_'+metric+'_vs_'+arm] = rows[metric]-rows[metric+'_'+arm]
    cols = ['OA', 'BA', 'delta_OA_vs_head_only', 'delta_BA_vs_head_only', 'delta_BA_vs_middle']
    summary = rows.groupby('arm')[cols].mean().reset_index()
    ports = rows.groupby(['port', 'arm'])[cols].mean().reset_index()
    classes = pd.concat([pd.read_csv(PREVIOUS/'by_class.csv'), pd.read_csv(OUT/'by_class.csv')])
    coverage = classes.groupby(['port', 'draw', 'arm', 'coverage']).recall.mean().reset_index().groupby(
        ['arm', 'coverage']).agg(recall=('recall', 'mean'), task_groups=('draw', 'size')).reset_index()
    for name, data in [('paired_results', rows), ('summary', summary), ('port_summary', ports), ('coverage_summary', coverage)]:
        data.to_csv(OUT/(name+'.csv'), index=False)
    updates = pd.read_csv(OUT/'update_diagnosis.csv')
    paired = rows[rows.arm.eq('non_dc')]
    wins = int((paired.delta_BA_vs_middle > 0).sum())
    loss = int((paired.delta_BA_vs_middle < 0).sum())
    lines = ['# 去直流与中频残差的匹配对照', '',
             '## 结论', '',
             '去直流版BA为37.982%，低于中频版38.344%和只适配分类头38.304%。相对中频低0.362个百分点；三港平均BA分别低0.722、0.110、0.254个百分点。',
             f'9组任务相对中频{wins}组更好、{loss}组更差。结果不支持“中频优势只是保住了通道均值”的充分解释；但抽样并不独立，没有据此作统计显著性或物理因果判断。',
             '去直流的缺类召回30.919%，比全频26.689%好，但仍低于中频31.366%和只适配分类头33.503%。保均值可能限制部分误伤，却不足以提高总体BA；改善的来源仍可能涉及更新形态和优化过程，不能直接称为港口噪声被消除。', '',
             '## 总体结果', '', table(summary, list(summary.columns)), '',
             '百分数与百分点；同3港、同9次64标签抽样，查询身份与支持身份排除。只新增non_dc的9次训练，其他结果原样引用前轮。', '',
             '## 本轮改变了什么', '',
             '- non_dc分支为每张特征图减去其空间均值，再按源939样本逐通道RMS匹配原特征能量。等价于只清零空间DCT的直流系数、保留全部其余系数。',
             '- h_new=h+gate*scale*(h-mean_spatial(h))，不丢弃原主路。它保持插入层的通道均值，只缩放空间起伏；不是任意空间变形或新注意力。',
             '- 与中频版同256参数、同零初始化、同24步优化、同源回放、同缺类损失修正、同插入层；没有新阈值搜索或目标查询选模。',
             '- 中频同样保持通道均值，但对非直流频率施加不同权重。本轮区分这种频率选择是否优于统一调整全部非直流成分。', '',
             '## 分港与类别覆盖', '', table(ports, list(ports.columns)), '',
             table(coverage, list(coverage.columns)), '',
             '缺类组仅4个任务。抽样共享查询人群，9组不是9个独立数据集。', '',
             '## 实际更新与边界', '', table(updates, list(updates.columns)), '',
             'RMS归一化使源分支初始能量可比，但学习后的系数、实际更新大小与负响应比例仍可能不同；该比较不能单独鉴定物理噪声或因果因素。',
             '实际去直流更新RMS约为原特征的17.79%，中频为18.16%；去直流插入层相对通道均值变化仅约5.72e-9。结果不是因为去直流分支未启用，或均值保持根本没有实现。',
             '两项新增测试通过：直接去均值与DCT仅清零直流的一致性；能量归一化及残差更新后的通道均值保持。9次适配全部完成，用时294.3秒，未重新训练旧版本。',
             '只研究冻结R18的小型诊断，不代表完整双路模型。支持/查询可能共享大景，历史预训练曝光未撤销。', '',
             '## 本轮决定', '',
             '不将去直流残差加入主方案，也不把中频命名为已验证的因果去噪模块。保留中频作为较少误伤的候选及本轮负结果；中频相对只适配分类头仅有0.040个百分点BA优势，目前不足以支撑继续围绕固定频带反复调参。', '']
    (OUT/'FINDINGS_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    print(summary.to_string(index=False), flush=True)
    print(coverage.to_string(index=False), flush=True)


def run():
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    OUT.mkdir(exist_ok=True)
    if (OUT/'COMPLETE.json').exists():
        raise ValueError('Preserve completed experiment')
    start = time.perf_counter()
    obs = pd.read_csv(BASE/'observations.csv')
    with np.load(PRIOR/'split.npz') as z:
        train = z['train_obs_ids']
    labels_source = obs.iloc[train].class_id.to_numpy()
    maps = np.load(PREVIOUS/'prefix_maps.npy', mmap_mode='r')
    h = np.array(maps[:len(train)])
    raw = remove_dc(h)
    rms = np.sqrt(np.mean(h.astype(np.float64)**2, axis=(0, 2, 3)))
    branch_rms = np.sqrt(np.mean(raw.astype(np.float64)**2, axis=(0, 2, 3)))
    scale = np.divide(rms, branch_rms, out=np.zeros_like(rms), where=branch_rms > 1e-6).astype(np.float32)
    pd.DataFrame(dict(channel=np.arange(256), original_rms=rms, raw_branch_rms=branch_rms,
                      multiplier=scale, matched_branch_rms=branch_rms*scale)).to_csv(OUT/'source_scale.csv', index=False)
    del h, raw
    residuals = NonDCBranch(maps, scale)
    blocks = build_blocks()
    with np.load(sorted(OLD.glob('*_models.npz'))[0]) as z:
        state = {k: z[k] for k in z.files}
    original = encode(blocks, SpectralResidual(), maps, maps, np.arange(len(train)))
    with torch.no_grad():
        _, source_scores = phi(torch.from_numpy(original), state)
    save_json(OUT/'PROTOCOL.json', dict(
        arm='non_dc', reference=str(PREVIOUS), train_steps=24, parameters=256, tasks=9,
        operator='h+gate*source_rms_scale*(h-spatial_mean(h))',
        selection='Fixed final step; no query labels in fitting or model selection; previous results reused unchanged',
        resource='CPU1; compute branch on demand; no new large cache; no parent changes'))
    records, classes, traces, updates = [], [], [], []
    for path in sorted(OLD.glob('*_split.npz')):
        tag = path.stem.removesuffix('_split')
        with np.load(path) as z:
            support, query, labels, query_ids = [z[k] for k in
                ('support_map_rows', 'query_map_rows', 'support_labels', 'query_obs_ids')]
        with np.load(OLD/f'{tag}_models.npz') as z:
            state = {k: z[k] for k in z.files}
        model, history = fit(blocks, maps, residuals, support, labels, labels_source, source_scores, state, tag, 'non_dc')
        traces.extend(history)
        features = encode(blocks, model, maps, residuals, np.r_[support, query])
        with torch.no_grad():
            ph, pri = phi(torch.from_numpy(features), state)
            beta, _, _ = residual_head(ph[:64], pri[:64], torch.from_numpy(labels), balanced=True)
            score = pri[64:]+ph[64:]@beta
        pred = score.argmax(1).numpy()
        gate = model.gate.detach().numpy()
        h = np.array(maps[support])
        delta = residuals[support]*gate[None, :, None, None]
        updates.append(dict(task=tag, gate_rms=float(np.sqrt(np.mean(gate**2))),
            relative_update_rms=float(np.linalg.norm(delta)/np.linalg.norm(h)),
            relative_channel_mean_shift=float(np.linalg.norm(delta.mean((2, 3)))/np.linalg.norm(h.mean((2, 3)))),
            negative_fraction=float(((h+delta) < 0).mean())))
        truth = obs.iloc[query_ids].class_id.to_numpy()
        port, draw = obs.iloc[query_ids].port.iloc[0], int(tag.rsplit('draw', 1)[1])
        records.append(dict(port=port, draw=draw, arm='non_dc', **metrics(pred, truth)))
        for c in np.unique(truth):
            count = int((labels == c).sum())
            classes.append(dict(port=port, draw=draw, arm='non_dc', class_id=int(c), support_count=count,
                coverage='absent' if count == 0 else 'singleton' if count == 1 else 'multiple',
                recall=float(100*(pred[truth == c] == c).mean())))
        np.savez_compressed(OUT/f'{tag}_models.npz', gate=gate, scale=scale, predictions=pred,
                            scores=score.numpy(), prior=pri[64:].numpy(), query_obs_ids=query_ids)
        for name, data in [('results', records), ('by_class', classes), ('training_history', traces), ('update_diagnosis', updates)]:
            pd.DataFrame(data).to_csv(OUT/(name+'.csv'), index=False)
        print('FIT_DONE', tag, records[-1], flush=True)
    save_json(OUT/'COMPLETE.json', dict(tasks=len(records), adapter_fits=len(records),
        seconds=time.perf_counter()-start, query_labels_used_for_training=False))
    report()


if __name__ == '__main__':
    run()
