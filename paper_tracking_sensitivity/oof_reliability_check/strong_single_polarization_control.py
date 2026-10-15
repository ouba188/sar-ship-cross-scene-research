"""Bounded input-observability control: preserve VV+VH, add the lost VV-VH path."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from strong_single_r50_task import (ROOT, SOURCE, CACHE as PIXELS, SingleR50, geometry,
    pixel_tensor, fit_source_prior, support_scores, target_design, write_json, metrics)

OUT = ROOT/'strong_single_polarization_20261010'
CACHE = Path('C:/Users/25791/AppData/Local/Temp/SAR_polarization_control_20261010')
TASKS = ROOT/'learned_frequency_residual_20261009/tasks'
MODES = ('sum', 'sum_difference')


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    protocol = OUT/'PROTOCOL.json'
    if not protocol.exists():
        write_json(protocol, dict(date='2026-10-10', reason='Verified identical frozen VV/VH first-convolution weights in current R50',
            modes=list(MODES), common_weight='0.5*(W_R+W_G+W_B)',
            contrast_weight='0.5*(W_R-W_G)',
            new_weights='VV=common+contrast; VH=common-contrast',
            equivalent_native_RGB_input='VV,VH,mean(VV,VH), shared .5/.25 normalization',
            preserved='Common-mode VV+VH response; architecture; parameter count; four C4 forwards; source rows; supports',
            changed='Fixed first-layer access to differential polarization only; source prior refitted per representation',
            input_units='Contrast of the encoded/normalized VV and VH inputs, not a calibrated physical backscatter ratio',
            source_alpha=1., residual_lambda=8., support_draws=5, support_labels=64,
            hyperparameter_search=False, neural_training=False, query_label_selection=False,
            source_features='Both modes freshly encoded; no historical cache precision mixed into source fit',
            scope='23 held ports + F_0921 + OSS2; existing fixed dual reference',
            boundary='Tests this fixed ImageNet-derived contrast path, not optimal polarimetric learning or physical causality'))


def pixels_for(dataset):
    if dataset == 'f0921':
        path = ROOT/'f0921_detector_recrop_20261007/inputs128.npy'
        ids = pd.read_csv(ROOT/'f0921_detector_recrop_20261007/target_manifest.csv', usecols=['sample_id']).sample_id.to_numpy(str)
    else:
        path = PIXELS/f'{dataset}_inputs128.npy'
        ids = np.load(PIXELS/f'{dataset}_ids.npy')
    pixels = np.load(path, mmap_mode='r')
    if pixels.shape != (len(ids), 2, 128, 128):
        raise ValueError('Input cache ordering/shape mismatch')
    return pixels, ids, path


@torch.no_grad()
def audit():
    prepare()
    torch.set_num_threads(2)
    a, b = SingleR50(polarization='sum').eval(), SingleR50(polarization='sum_difference').eval()
    wa, wb = a.prefix[0].weight.detach().double(), b.prefix[0].weight.detach().double()
    torch.manual_seed(20261010)
    d = torch.randn(4, 1, 32, 32, dtype=torch.float64)
    antisymmetric = torch.cat((d, -d), dim=1)
    old = F.conv2d(antisymmetric, wa, stride=2, padding=3)
    new = F.conv2d(antisymmetric, wb, stride=2, padding=3)
    same_input = torch.cat((d, d), dim=1)
    old_same = F.conv2d(same_input, wa, stride=2, padding=3)
    new_same = F.conv2d(same_input, wb, stride=2, padding=3)
    np.testing.assert_allclose(new_same.numpy(), old_same.numpy(), atol=2e-6, rtol=1e-6)
    if old.abs().max() > 1e-12 or new.square().mean().sqrt() < 1e-3:
        raise AssertionError('Polarization null-space intervention is not wired correctly')
    if sum(p.numel() for p in a.parameters()) != sum(p.numel() for p in b.parameters()):
        raise AssertionError('Parameter budget changed')
    # Directly check the RGB interpretation instead of inferring it from labels.
    from torchvision.models import ResNet50_Weights
    path = Path(torch.hub.get_dir())/'checkpoints'/Path(ResNet50_Weights.IMAGENET1K_V2.url).name
    rgb = torch.load(path, map_location='cpu', weights_only=True)['conv1.weight'].double()
    pair = torch.randn(4, 2, 32, 32, dtype=torch.float64)
    native = torch.cat((pair, pair.mean(1, keepdim=True)), dim=1)
    equivalent = F.conv2d(native, rgb, stride=2, padding=3)
    actual = F.conv2d(pair, wb, stride=2, padding=3)
    np.testing.assert_allclose(actual.numpy(), equivalent.numpy(), atol=2e-6, rtol=1e-6)
    result = dict(old_channel_weight_difference_max=float((wa[:, 0]-wa[:, 1]).abs().max()),
        new_channel_weight_difference_max=float((wb[:, 0]-wb[:, 1]).abs().max()),
        old_antisymmetric_response_max=float(old.abs().max()),
        new_antisymmetric_response_rms=float(new.square().mean().sqrt()),
        common_mode_response_difference_max=float((old_same-new_same).abs().max()),
        native_RGB_equivalence_max_error=float((actual-equivalent).abs().max()),
        parameters=sum(p.numel() for p in a.parameters()), unchanged_parameters=True,
        old_prefix_frozen=all(not p.requires_grad for p in a.prefix.parameters()),
        conclusion='Old adapter has a differential-polarization null space; accuracy contribution not established by this audit')
    ce_path = ROOT/'completion_single_finetune_20261010/r50_final.pt'
    ce = torch.load(ce_path, map_location='cpu', weights_only=False)['state']['conv1.weight']
    result['earlier_full_CE_finetune_channel_difference_max'] = float((ce[:, 0]-ce[:, 1]).abs().max())
    result['earlier_full_CE_finetune_checkpoint'] = str(ce_path)
    result['scope_exception'] = 'Earlier all-parameter CE R50 can break the symmetry; the frozen-prefix limitation does not apply to every prior single-backbone trial'
    write_json(OUT/'OBSERVABILITY.json', result)
    print(json.dumps(result), flush=True)


@torch.no_grad()
def extract(datasets):
    prepare()
    torch.set_num_threads(2)
    for mode in MODES:
        net = SingleR50(polarization=mode).cuda().eval()
        for dataset in datasets:
            receipt = OUT/f'{dataset}_{mode}_EXTRACTED.json'
            if receipt.exists():
                continue
            pixels, ids, path = pixels_for(dataset)
            target = CACHE/f'{dataset}_{mode}.npy'
            result = np.lib.format.open_memmap(target, mode='w+', dtype=np.float32, shape=(len(ids), 2048))
            started, last = time.perf_counter(), time.perf_counter()
            for lo in range(0, len(ids), 32):
                rows = np.arange(lo, min(lo+32, len(ids)))
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    z = net.encode(pixel_tensor(pixels, rows, torch.device('cuda')))
                result[lo:lo+len(rows)] = z.cpu().numpy()
                if time.perf_counter()-last >= 30:
                    print(f'{dataset}/{mode}: {lo+len(rows)}/{len(ids)}', flush=True)
                    last = time.perf_counter()
            result.flush()
            np.save(CACHE/f'{dataset}_ids.npy', ids)
            write_json(receipt, dict(dataset=dataset, mode=mode, rows=len(ids), feature_dimension=2048,
                pixels=str(path), output=str(target), seconds=time.perf_counter()-started,
                source_and_target_refit_required=True, actual_forwards_per_chip=4))
            del result, pixels
            gc.collect()
        del net
        torch.cuda.empty_cache()


def task_folders():
    return [p.parent for p in sorted((TASKS/'self24ports').glob('*/PREPARED.json'))]+[TASKS/'f0921', TASKS/'oss2']


@torch.no_grad()
def predict(datasets=('source', 'f0921', 'oss2')):
    prepare()
    torch.set_num_threads(2)
    frame = pd.read_csv(SOURCE/'manifest.csv', usecols=['instance_id', 'class_id'])
    source_labels = frame.class_id.to_numpy(np.int64)
    np.testing.assert_array_equal(np.load(CACHE/'source_ids.npy'), frame.instance_id.to_numpy(str))
    for folder in task_folders():
        info = json.loads((folder/'PREPARED.json').read_text())
        dataset, name = info['dataset'], info['task']
        pool = 'source' if dataset == 'self24ports' else dataset
        if pool not in datasets:
            continue
        dest = OUT/'targets'/dataset/name
        if (dest/'PREDICTED.json').exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        data = target_design(folder/'target.npz')
        np.testing.assert_array_equal(np.load(CACHE/f'{pool}_ids.npy')[data['rows']], data['ids'])
        predictions = np.empty((5, 2, len(data['ids'])), np.int8)
        for ai, mode in enumerate(MODES):
            if not all((OUT/f'{d}_{mode}_EXTRACTED.json').exists() for d in ('source', pool)):
                raise ValueError('Finish the specified representation extraction before prediction')
            source = np.load(CACHE/f'source_{mode}.npy', mmap_mode='r')
            target = np.load(CACHE/f'{pool}_{mode}.npy', mmap_mode='r')
            weights = torch.as_tensor(fit_source_prior(source, source_labels, data['source_rows']), dtype=torch.float32)
            np.save(dest/f'{mode}_source_prior.npy', weights.numpy())
            for draw, support in enumerate(data['support']):
                s = geometry(torch.from_numpy(np.array(target[data['rows'][support]])))
                ps = F.linear(s, weights)
                labels = torch.as_tensor(data['support_labels'][draw]).long()
                for lo in range(0, len(data['ids']), 1024):
                    rows = data['rows'][lo:lo+1024]
                    q = geometry(torch.from_numpy(np.array(target[rows])))
                    pq = F.linear(q, weights)
                    scores = support_scores(s, labels, q, support_prior=ps, query_prior=pq)
                    predictions[draw, ai, lo:lo+len(rows)] = scores.argmax(1).numpy()
        np.savez_compressed(dest/'predictions.npz', predictions=predictions, modes=np.array(MODES), **data)
        write_json(dest/'PREDICTED.json', dict(dataset=dataset, task=name, source_rows=len(data['source_rows']),
            no_neural_training=True, no_target_selection=True, matched_information_input_control=True))
        print('PREDICTED', dataset, name, flush=True)
        gc.collect()


def report():
    records, flows, classes = [], [], []
    for folder in task_folders():
        info = json.loads((folder/'PREPARED.json').read_text())
        dataset, name = info['dataset'], info['task']
        dest = OUT/'targets'/dataset/name
        if not (dest/'PREDICTED.json').exists():
            raise ValueError('Report only after all25 fixed controls finish')
        with np.load(dest/'predictions.npz') as saved:
            data = {k: saved[k] for k in saved.files}
        path = SOURCE/'manifest.csv' if dataset == 'self24ports' else ROOT/('f0921_detector_recrop_20261007' if dataset == 'f0921' else 'opensarship2_transfer_20261007')/'target_manifest.csv'
        idcol = 'instance_id' if dataset == 'self24ports' else 'sample_id'
        frame = pd.read_csv(path, usecols=[idcol, 'class_id'])
        np.testing.assert_array_equal(frame[idcol].to_numpy(str)[data['rows']], data['ids'])
        y = frame.class_id.to_numpy()[data['rows']]
        with np.load(ROOT/'backbone_ablation_20261010/targets'/dataset/name/'predictions.npz') as saved:
            for key in ('rows', 'ids', 'support', 'eval_masks'):
                np.testing.assert_array_equal(saved[key], data[key])
            dual = saved['predictions'][:, list(saved['modes']).index('dual_refit_late')]
        for draw, mask in enumerate(data['eval_masks']):
            truth = y[mask]
            for ai, mode in enumerate(MODES):
                pred = data['predictions'][draw, ai, mask]
                records.append(dict(dataset=dataset, task=name, draw=draw, mode=mode, **metrics(y[mask], pred)))
            records.append(dict(dataset=dataset, task=name, draw=draw, mode='current_dual_reference', **metrics(y[mask], dual[draw, mask])))
            old, new = data['predictions'][draw, 0, mask], data['predictions'][draw, 1, mask]
            rescue, harm = (old != truth)&(new == truth), (old == truth)&(new != truth)
            gains = []
            for c in np.unique(truth):
                subset = truth == c
                n = int(subset.sum())
                gain = 100*float((rescue[subset].sum()-harm[subset].sum())/n)
                gains.append(gain)
                classes.append(dict(dataset=dataset, task=name, draw=draw, class_id=int(c), n=n,
                    support_count=int((data['support_labels'][draw] == c).sum()),
                    old_recall=100*float(np.mean(old[subset] == c)),
                    new_recall=100*float(np.mean(new[subset] == c)), recall_delta_pp=gain,
                    rescue=int(rescue[subset].sum()), harm=int(harm[subset].sum())))
            delta_oa, delta_ba = 100*float((rescue.sum()-harm.sum())/len(truth)), float(np.mean(gains))
            np.testing.assert_allclose(delta_ba, metrics(truth, new)['ba']-metrics(truth, old)['ba'], atol=1e-10)
            flows.append(dict(dataset=dataset, task=name, draw=draw, query_n=len(truth),
                rescue=int(rescue.sum()), harm=int(harm.sum()), oa_delta_pp=delta_oa, ba_delta_pp=delta_ba))
    table = pd.DataFrame(records)
    table.to_csv(OUT/'metrics.csv', index=False)
    per_task = table.groupby(['dataset', 'task', 'mode'])[['oa', 'ba']].mean()
    per_task.to_csv(OUT/'task_means.csv')
    summary = per_task.groupby(['dataset', 'mode']).mean()
    summary.to_csv(OUT/'summary.csv')
    pd.DataFrame(flows).to_csv(OUT/'rescue_harm.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'per_class.csv', index=False)
    lines = ['# 输入差分可观测性：固定单R50控制', '',
        '全部25任务、原5组64支持标签。两个输入适配版本均从像素重新提取源/目标，并分别拟合相同源头；无新增神经训练、无目标选超参。', '',
        '| 数据集 | 配置 | OA（%） | BA（%） |', '|---|---|---:|---:|']
    for (dataset, mode), row in summary.iterrows():
        lines.append(f'| {dataset} | {mode} | {row.oa:.5f} | {row.ba:.5f} |')
    lines += ['', '## 能解释什么', '',
        '旧冻结前段的VV/VH卷积权重相同，数学上只能观察两通道之和。新控制引入预先固定的差分响应，参数量与C4前向次数不变；两种输入处理均使用同样的bf16推理。数学共同模等价不意味着有限精度逐位一致。',
        '以上正负变化衡量这个特定输入适配与匹配读出的效果，不等于所有极化信息的价值，也不是两种互斥物理机制。这里的差分是编码/归一化输入的差，不能冒充标定后的物理后向散射比。',
        '此前全参数CE单R50已打破相同权重约束（实际检查点通道差最大0.04833943），因此不能用冻结前段的盲区解释全部历史单路负结果。', '',
        '## 对照边界', '',
        '当前完整双路为历史固定参考，包含其他模块与不同预训练。输入控制之间是同骨干、同读出的直接比较；它们与完整双路的差距不是纯架构效应。',
        '旧任务训练使用历史C4缓存初始化先验，本控制使用新提取源特征；不要把本表sum臂与旧冻结单路数字强行视作同一预测。',
        '源头alpha1、支持lambda8、差分系数均事前固定。没有看到目标结果后选择不同数据集的最佳输入方案。五个支持抽样不是五次神经训练。', '',
        '## 文件', '',
        '- `OBSERVABILITY.json`：代码/实际权重可观测性与早期全参数试验的反向证据。',
        '- `metrics.csv` / `task_means.csv`：完整正负结果。',
        '- `per_class.csv`：逐类召回、支持覆盖与救回/误改。',
        '- `rescue_harm.csv`：净纠错与OA/BA增量逐项闭合。', '']
    (OUT/'FINDINGS_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(OUT/'COMPLETE.json', dict(tasks=25, support_draws=5, actual_neural_training_runs=0,
        difference_path_fixed_before_query_readout=True, source_and_target_encoded_consistently=True,
        accuracy_causality_scope='Specified input adapter and refitted fixed readout, not physical mechanisms or universally optimal single backbone'))
    print(summary.to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['audit', 'extract', 'predict', 'report'])
    parser.add_argument('--datasets', nargs='+', choices=['source', 'f0921', 'oss2'], default=['source', 'f0921', 'oss2'])
    args = parser.parse_args()
    if args.stage == 'extract':
        extract(args.datasets)
    elif args.stage == 'predict':
        predict(args.datasets)
    else:
        globals()[args.stage]()


if __name__ == '__main__':
    main()
