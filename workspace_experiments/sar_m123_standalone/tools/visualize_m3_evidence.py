"""Render two real M3 forwards, their source references, and exact match evidence."""
import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
from PIL import Image
import torch

from sar_m123 import m3, tokens
from sar_m123.input import prepare
from sar_m123.pipeline import ShipClassifier, to_json


NAMES = ['散货船', '渔船', '普通货船', '成品油/化学品船', '集装箱船', '原油船', '拖船', '海工船']
GREEN, RED, INK, BLUE = '#138568', '#bd4054', '#263038', '#337cad'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def max_error(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def body_corners(geometry, placement):
    cx, cy, length, width, angle = geometry[:5]
    local = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]) * [length, width]
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    offset = [placement['crop_x'] - placement['pad_left'], placement['crop_y'] - placement['pad_top']]
    return local @ rotation.T + [cx, cy] + offset


@torch.inference_mode()
def compute_case(classifier, record, directory, trace, legacy_case, metadata):
    captured = []
    original_sinkhorn = m3.sinkhorn

    def capture(sim, qa, sa, steps=60, eps=.1):
        transport = original_sinkhorn(sim, qa, sa, steps, eps)
        mask = (qa > 0)[..., :, None] & (sa > 0)[..., None, :]
        captured.append(dict(K=sim.detach().clone(), T=(transport * mask).detach().clone(),
                             qa=qa.detach().clone(), sa=sa.detach().clone(), steps=steps, eps=eps))
        return transport

    features = classifier.extract(record, directory)
    with patch.object(m3, 'sinkhorn', capture):
        final, details = classifier.score(features, seed=17)
    match = details['matching']
    base = np.asarray(details['s12'], dtype=np.float32)
    dv, dh, weights = (match[k] for k in ('delta_vv', 'delta_vh', 'polar_weights'))
    model = classifier.m3._load(17)
    anchor = model['ck']['alpha_anchor']
    fixed_weights = np.array([anchor, 1-anchor], dtype=np.float32) * match['available']
    fixed = base + fixed_weights[0]*dv + fixed_weights[1]*dh
    rebuilt = base + weights[0]*dv + weights[1]*dh
    true, rival = int(legacy_case['true']), int(legacy_case['rival'])
    if int(metadata.loc[metadata.obs_id.eq(int(record['id'])), 'class_id'].iloc[0]) != true:
        raise AssertionError('Case truth does not match source metadata')

    # Capture is inside the actual forward; replay independently checks its aggregation.
    local_parts, matrices, replay_errors = [], [], []
    for region_id, item in enumerate(captured):
        k, t, qa, sa = (item[key] for key in ('K', 'T', 'qa', 'sa'))
        replay_t = original_sinkhorn(k, qa, sa, item['steps'], item['eps'])
        replay_t *= (qa > 0)[..., :, None] & (sa > 0)[..., None, :]
        replay_errors.append(max_error(t.cpu(), replay_t.cpu()))
        local_parts.append(torch.stack([(t*k).sum((-1, -2)), t.sum((-1, -2))], -1))
        matrices.append(dict(region_id=region_id, region_name=['body', 'side', 'axis_end'][region_id],
                             token_indices=tokens.REGIONS[region_id],
                             K=k[0].cpu().numpy(), T=t[0].cpu().numpy(),
                             query_mass=qa[0, 0, 0].cpu().numpy(),
                             reference_mass=sa[0].cpu().numpy(),
                             sinkhorn_steps=item['steps'], epsilon=item['eps']))
    local_replay = torch.stack(local_parts, -2).mean(2).cpu().numpy()[0]
    x = classifier.m3._tensor(m3.standardize(
        m3.local_features(local_replay[None], base[None]), model['ck']['readout_scaler']))
    replay_dv, replay_dh = model['vv'](x).cpu().numpy()[0], model['vh'](x).cpu().numpy()[0]
    replay_final = base + weights[0]*replay_dv + weights[1]*replay_dh
    refs_replay = classifier.m3.retrieve({k: v[None] for k, v in features['local'].items()})[0]
    np.testing.assert_array_equal(refs_replay, match['refs'])
    np.testing.assert_array_equal(match['reference_ids'], trace['trace']['matching']['reference_ids'])

    checks = dict(finalscore_reconstruction_max_abs=max_error(final, rebuilt),
                  transport_replay_max_abs=max(replay_errors),
                  local_aggregation_replay_max_abs=max_error(local_replay, match['local']),
                  readout_replay_max_abs=max(max_error(dv, replay_dv), max_error(dh, replay_dh)),
                  matching_to_final_replay_max_abs=max_error(final, replay_final),
                  saved_trace_final_max_abs=max_error(final, trace['class_scores']),
                  saved_trace_local_max_abs=max_error(match['local'], trace['trace']['matching']['local']),
                  refs_equal_fresh_retrieval_and_saved_trace=True)
    for key in ('finalscore_reconstruction_max_abs', 'transport_replay_max_abs',
                'local_aggregation_replay_max_abs', 'readout_replay_max_abs', 'matching_to_final_replay_max_abs'):
        assert checks[key] < 2e-6, (key, checks[key])
    assert checks['saved_trace_final_max_abs'] < 3e-4, checks
    assert checks['saved_trace_local_max_abs'] < 3e-4, checks
    assert int(final.argmax()) == trace['class_id']

    provenance = {(p['obs_id'], p['polarization']): p for p in legacy_case['provenance']}
    selected = []
    for cls in range(8):
        for rank, source_index in enumerate(match['reference_ids'][cls]):
            row = metadata.loc[int(source_index)]
            assert int(row.class_id) == cls and row.role == 'strategy_fit'
            selected.append(dict(class_id=cls, class_name=NAMES[cls], rank=rank+1,
                                 bank_row=int(match['refs'][cls, rank]), source_index=int(source_index),
                                 obs_id=int(row.obs_id), role=str(row.role),
                                 image_vv=str(row.image_vv), image_vh=str(row.image_vh)))

    views = []
    prepared = prepare(record, directory)
    for pol_id, pol in enumerate(('vv', 'vh')):
        path = (directory / record[pol]['image']).resolve()
        views.append(dict(kind='query', obs_id=int(record['id']), polarization=pol,
                          path=str(path), geometry=prepared[pol_id]['geometry'],
                          placement=prepared[pol_id]['placement']))
        for cls in (true, rival):
            for ref in (r for r in selected if r['class_id'] == cls):
                old = provenance[(ref['obs_id'], pol)]
                path = ref['image_'+pol]
                assert Path(path) == Path(old['path'])
                assert old['up_index'] == ref['source_index']
                views.append(dict(kind='true_ref' if cls == true else 'rival_ref',
                                  obs_id=ref['obs_id'], polarization=pol, path=path,
                                  geometry=old['geometry'], placement=old['placement'],
                                  source_index=ref['source_index'], bank_row=ref['bank_row'], rank=ref['rank']))
    for view in views:
        with Image.open(view['path']) as im:
            view['native_size_wh'] = list(im.size)
            pixels = np.asarray(im.convert('L'))
        if view['kind'] == 'query':
            legacy_image = provenance[(view['obs_id'], view['polarization'])]['path']
            with Image.open(legacy_image) as im:
                np.testing.assert_array_equal(pixels, np.asarray(im.convert('L')))
            view['legacy_pixels_equal'] = True
            view['legacy_path'] = legacy_image
        else:
            view['mapping_verified'] = True
    checks['source_reference_metadata_rows_verified'] = len(selected)
    checks['displayed_reference_image_mappings_verified'] = sum(v['kind'] != 'query' for v in views)
    checks['query_fixture_pixels_equal_legacy'] = True

    query_regions = m3.regional_descriptors(
        {k: v[None] for k, v in features['local'].items()}, classifier.m3.covariance_scaler)
    ref_regions = {k: v[match['refs']] for k, v in classifier.m3.bank_regions.items()}
    covariance_similarity = (query_regions['cov'][0][None, None] * ref_regions['cov']).sum(-1)
    valid = query_regions['valid'][0][None, None] & ref_regions['valid']
    raw_replay, _ = m3.gate_features(query_regions, classifier.m3.bank_regions, match['refs'][None],
                                    base[None], dv[None], dh[None])
    checks['gate_input_replay_max_abs'] = max_error(raw_replay[0], match['gate_raw'])
    assert checks['gate_input_replay_max_abs'] < 2e-6
    # The displayed descriptors are exactly the operands of the actual 144-value frequency block.
    freq = match['gate_raw'][:, 41:].reshape(8, 2, 3, 3, 8)
    cm, _ = m3.pair_stats(covariance_similarity[None], np.broadcast_to(valid[..., None], covariance_similarity.shape)[None])
    checks['displayed_covariance_to_gate_mean_max_abs'] = max_error(freq[..., 4], cm[0])
    assert checks['displayed_covariance_to_gate_mean_max_abs'] < 2e-6
    spectral = dict(wavelengths_px=[4, 8, 16], query_energy=query_regions['energy'][0],
                    reference_energy=ref_regions['energy'], covariance_similarity=covariance_similarity,
                    query_coverage=query_regions['coverage'][0], reference_coverage=ref_regions['coverage'],
                    valid=valid, query_covariance_descriptor=query_regions['cov'][0],
                    reference_covariance_descriptor=ref_regions['cov'], actual_gate_frequency_block=freq,
                    query_axes='polarization, region, wavelength',
                    reference_axes='class, reference_rank, polarization, region, wavelength',
                    energy_definition='Per-token mean over four directions of log1p(pooled native Gabor power/1e-4), then mass-weighted regional mean.',
                    covariance_definition='Dot product of standardized, region-pooled, unit-normalized 32-D joint descriptors at each wavelength.',
                    display='Both VV and VH, body region; query and all four displayed actual references.')

    pair = lambda v: float(np.float64(v[true])-np.float64(v[rival]))
    waterfall = {}
    for name, g, score in [('fixed_B', fixed_weights, fixed), ('selected', weights, final)]:
        terms = [pair(base), float(g[0])*pair(dv), float(g[1])*pair(dh)]
        waterfall[name] = dict(weights=g, terms=terms, cumulative=np.cumsum(terms),
                               margin=pair(score), prediction=int(score.argmax()),
                               true_score=float(score[true]), rival_score=float(score[rival]), scores=score,
                               margin_reconstruction_abs=abs(sum(terms)-pair(score)))

    return dict(obs_id=int(record['id']), seed=17, true_class=true, rival_class=rival,
                fixed_pair_names=[NAMES[true], NAMES[rival]], class_order=NAMES,
                s12=base, delta_vv=dv, delta_vh=dh, alpha_anchor=anchor,
                selected_alpha=match['alpha'], available=match['available'], finalscore=final,
                reference_ids=match['reference_ids'], refs=match['refs'], selected_source_refs=selected,
                local=match['local'], gate_raw=match['gate_raw'], views=views,
                matrices=matrices, spectral_conditions=spectral, waterfall=waterfall, checks=checks,
                displayed_matrix=dict(polarization='VH', region='body', reference_rank=1,
                                      classes=[true, rival], selection='fixed before observing matrix values'),
                matrix_axes='K/T: class, reference_rank, polarization, query_token, reference_token',
                K_definition='PairMetric-weighted cosine similarity, not exp(K/epsilon).',
                T_definition='Sinkhorn(exp(pad(K/0.1))) with unmatched row/column; real-real block after positive-mass mask; not row-normalized.',
                local_definition='For each class/polarization/region: mean over 2 refs of [sum(T*K), sum(T)].',
                scope='Only two preselected cases; no training, fitting, weights edits, or population estimate.')


def chip_axis(ax, view, title, marked=False):
    with Image.open(view['path']) as image:
        pixels = np.asarray(image.convert('L'))
    h, w = pixels.shape
    ax.imshow(pixels, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    if marked:
        corners = body_corners(view['geometry'], view['placement'])
        ax.add_patch(Polygon(corners, fill=False, edgecolor='#30dfcf', linewidth=1.1))
    ax.plot([4, 14], [h-5, h-5], color='white', lw=2)
    ax.text(9, h-7, '10 px', ha='center', va='bottom', color='white', fontsize=8)
    ax.set_title(title, fontsize=11, pad=7, color=INK)
    ax.set_xlim(-.5, w-.5)
    ax.set_ylim(h-.5, -.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#b6c4c9')


def heatmap(ax, matrix, title, kind, vmax):
    im = ax.imshow(matrix, cmap='coolwarm' if kind == 'K' else 'viridis',
                   vmin=-1 if kind == 'K' else 0, vmax=vmax, interpolation='nearest')
    ax.set_title(title, fontsize=11, pad=9)
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.tick_params(labelsize=9)
    ax.set_xlabel('参考 token j', fontsize=10)
    ax.set_ylabel('查询 token i', fontsize=10)
    for i in range(8):
        for j in range(8):
            value = matrix[i, j]
            color = 'white' if (abs(value) > .5 if kind == 'K' else value < .55*vmax) else INK
            ax.text(j, i, f'{value:.2f}' if kind == 'K' else f'{value:.3f}',
                    fontsize=7, ha='center', va='center', color=color)
    return im


def waterfall_axis(ax, item, title, ylim):
    terms, margin = item['terms'], item['margin']
    start = 0.
    for i, value in enumerate(terms):
        end = start+value
        ax.bar(i, abs(value), bottom=min(start, end), width=.64,
               color=GREEN if value >= 0 else RED, zorder=3)
        ax.text(i, max(start, end)+.012*(ylim[1]-ylim[0]), f'{value:+.5f}',
                ha='center', va='bottom', fontsize=10)
        ax.plot([i+.32, i+.68], [end, end], color='#8d989e', lw=.8)
        start = end
    ax.bar(3, abs(margin), bottom=min(0, margin), width=.64, color=BLUE, zorder=3)
    ax.text(3, max(0, margin)+.012*(ylim[1]-ylim[0]), f'{margin:+.5f}', ha='center', fontsize=10)
    ax.axhline(0, color=INK, lw=.9)
    ax.set_xticks(range(4), ['S12 分差', 'gVV × δVV\n分差增量', 'gVH × δVH\n分差增量', '最终分差'])
    ax.set_ylim(*ylim)
    ax.set_xlim(-.65, 3.6)
    ax.tick_params(labelsize=10)
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_ylabel('固定真实类分数 − 竞争类分数', fontsize=10)
    ax.grid(axis='y', color='#e4e8ea', zorder=0)
    ax.spines[['right', 'top']].set_visible(False)


def spectral_axis(ax, values, labels, title, kind, energy_max):
    vmax = energy_max if kind == 'energy' else 1
    im = ax.imshow(values, cmap='YlGnBu' if kind == 'energy' else 'coolwarm',
                   vmin=0 if kind == 'energy' else -1, vmax=vmax, aspect='auto', interpolation='nearest')
    ax.set_xticks(range(3), ['4 px', '8 px', '16 px'])
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(labelsize=9)
    ax.set_xlabel('Gabor 波长（原图像素）', fontsize=10)
    ax.set_title(title, fontsize=12, pad=10)
    for row in range(len(labels)):
        for col in range(3):
            val = values[row, col]
            white = val > energy_max*.6 if kind == 'energy' else abs(val) > .6
            ax.text(col, row, f'{val:.3f}', ha='center', va='center', fontsize=10,
                    color='white' if white else INK)
    bar = ax.figure.colorbar(im, ax=ax, fraction=.045, pad=.025)
    bar.ax.tick_params(labelsize=8)


def render(case, output):
    oid, true, rival = case['obs_id'], case['true_class'], case['rival_class']
    rescued = case['waterfall']['selected']['prediction'] == true
    fig = plt.figure(figsize=(20, 22.5), facecolor='white')
    upper = lambda y: (y*18.5+4)/22.5
    lower = lambda y: y*18.5/22.5
    fig.suptitle(f'M3 真实证据 | {oid} | seed 17 | 相对固定 B：'+('救回' if rescued else '误伤'),
                 x=.04, y=upper(.983), ha='left', fontsize=23, weight='bold', color=INK)
    fig.text(.04, upper(.953), f'固定类别对：{NAMES[true]}（真实） − {NAMES[rival]}（竞争）'
             '    |    原始 PNG → 冻结骨干 → 实际检索 / 匹配 → 分数', fontsize=12, color=INK)
    top = fig.add_gridspec(2, 5, left=.045, right=.96, bottom=upper(.578), top=upper(.923), hspace=.35, wspace=.13)
    for pol_id, pol in enumerate(('vv', 'vh')):
        views = [v for v in case['views'] if v['polarization'] == pol]
        for col, view in enumerate(views):
            if col == 0:
                title = f'查询 {oid} | {pol.upper()}\n真实：{NAMES[true]}'
            else:
                title = f'{"同类" if col < 3 else "竞争类"}参考 #{view["rank"]}：{view["obs_id"]} | {pol.upper()}\n'
                title += f'源索引 {view["source_index"]} · bank {view["bank_row"]}'
            chip_axis(fig.add_subplot(top[pol_id, col]), view, title,
                      marked=pol_id == 1 and col in (0, 1, 3))
    fig.text(.045, upper(.558), '原图均为 64 × 64 px，同尺度、固定灰度 0–255；网络输入为原尺度补零到 160。'
             '青框：下方矩阵对应的 VH 船体区域；每类真实检索 2 个参考。', fontsize=11)
    fig.text(.045, upper(.539), '局部匹配：VH / 船体 / 各类第 1 参考。8 个 token 按轴向 4 段 × 两侧编号；'
             'i、j = 2 × 轴向段号 + 侧号。图只展开此区域，JSON 保留全部真实矩阵。', fontsize=11)
    middle = fig.add_gridspec(1, 4, left=.055, right=.95, bottom=upper(.312), top=upper(.513), wspace=.4)
    body = case['matrices'][0]
    tmax = max(float(body['T'][cls, 0, 1].max()) for cls in (true, rival))
    for k, cls in enumerate((true, rival)):
        ref = next(r for r in case['selected_source_refs'] if r['class_id'] == cls and r['rank'] == 1)
        for j, kind in enumerate(('K', 'T')):
            ax = fig.add_subplot(middle[k*2+j])
            matrix = body[kind][cls, 0, 1]
            title = f'{"同类" if k == 0 else "竞争类"} {ref["obs_id"]} | {kind} '+('相似度' if kind == 'K' else '匹配质量')
            im = heatmap(ax, matrix, title, kind, 1 if kind == 'K' else tmax)
            bar = fig.colorbar(im, ax=ax, fraction=.046, pad=.03, shrink=.85)
            bar.ax.tick_params(labelsize=8)
    fig.text(.045, upper(.285), 'K = 学习度量下的加权余弦；T = 含未匹配槽的 Sinkhorn 实际质量，未逐行归一化。'
             '两类共用 K/T 色标。局部读出 = 两参考平均 [Σ(T×K), ΣT]。', fontsize=11)
    fig.text(.045, .386, '空间—频域条件的实际来源 | 船体区域 × 三种 Gabor 波长；'
             '左右分别查看 VV、VH，所有参考均与上方真实图像一一对应。', fontsize=12, weight='bold')
    spectral_grid = fig.add_gridspec(1, 4, left=.10, right=.95, bottom=.275, top=.36, wspace=.75)
    spectrum = case['spectral_conditions']
    refs = [r for cls in (true, rival) for r in case['selected_source_refs'] if r['class_id'] == cls]
    labels = [f'{"同类" if r["class_id"] == true else "竞争"} {r["obs_id"]}' for r in refs]
    energies = [np.stack([spectrum['query_energy'][pol, 0]] + [
        spectrum['reference_energy'][r['class_id'], r['rank']-1, pol, 0] for r in refs]) for pol in range(2)]
    energy_max = max(float(e.max()) for e in energies)
    for pol in range(2):
        spectral_axis(fig.add_subplot(spectral_grid[pol]), energies[pol], [f'查询 {oid}']+labels,
                      ('VV' if pol == 0 else 'VH')+' 三尺度能量描述', 'energy', energy_max)
        cov = np.stack([spectrum['covariance_similarity'][r['class_id'], r['rank']-1, pol, 0] for r in refs])
        spectral_axis(fig.add_subplot(spectral_grid[pol+2]), cov, labels,
                      ('VV' if pol == 0 else 'VH')+' 查询－参考共变相似度', 'covariance', energy_max)
    fig.text(.045, .247, '能量：逐方向 log1p(E / 1e−4) 后平均，token 按质量汇聚。'
             '共变：标准化、区域汇聚并单位化的 32 维联合描述点积。相同量在两极化共用色标。', fontsize=11)
    fig.text(.045, .230, '这些是实际门控输入的组成量，不是物理频谱或归因。'
             '源索引 → metadata 观测 ID → legacy 原图路径已逐一核对；查询原图与旧资源像素完全一致。', fontsize=11)
    bottom = fig.add_gridspec(1, 3, left=.06, right=.96, bottom=lower(.102), top=lower(.251),
                             width_ratios=[1, 1, .95], wspace=.32)
    points = [0.] + [x for value in case['waterfall'].values() for x in value['cumulative']]
    low, high = min(points), max(points)
    span = max(high-low, .1)
    limits = (low-.2*span, high+.3*span)
    for col, name in enumerate(('fixed_B', 'selected')):
        item = case['waterfall'][name]
        g = item['weights']
        title = ('固定 B' if col == 0 else 'selected 频域门控')+f' | gVV={g[0]:.4f}, gVH={g[1]:.4f}'
        waterfall_axis(fig.add_subplot(bottom[col]), item, title, limits)
    ax = fig.add_subplot(bottom[2])
    ax.axis('off')
    ax.set_title('固定 B 与 selected：同一类别对', fontsize=13, pad=12)
    b, s = case['waterfall']['fixed_B'], case['waterfall']['selected']
    rows = [['真实类分数', f'{b["true_score"]:+.5f}', f'{s["true_score"]:+.5f}'],
            ['竞争类分数', f'{b["rival_score"]:+.5f}', f'{s["rival_score"]:+.5f}'],
            ['固定对分差', f'{b["margin"]:+.5f}', f'{s["margin"]:+.5f}'],
            ['8 类最终预测', NAMES[b['prediction']], NAMES[s['prediction']]]]
    table = ax.table(cellText=rows, colLabels=['', '固定 B', 'selected'],
                     cellLoc='center', colWidths=[.32, .34, .34], bbox=[0, .25, 1, .75])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#dce2e5')
        if row == 0:
            cell.set_facecolor('#edf2f4')
    change = s['margin']-b['margin']
    ax.text(0, .12, f'权重变化带来的分差变化：{change:+.5f}', fontsize=11,
            color=GREEN if change > 0 else RED, transform=ax.transAxes)
    fig.text(.045, lower(.064), '瀑布严格使用同一类别对：ΔS = ΔS12 + gVV·ΔδVV + gVH·ΔδVH。'
             '分差 > 0 仅代表胜过所列竞争类；最终预测按全部 8 类 argmax。', fontsize=11)
    fig.text(.045, lower(.043), '声明：匹配热图不是物理归因；2 例不是整体比例；已有 8-bit 显示图不等于定标 RCS。'
             '未做遮挡实验，不能将局部亮点或纹理直接解释为误判原因。', fontsize=11, color=INK)
    fig.text(.045, lower(.022), '验证：finalscore 重构误差 '
             f'{case["checks"]["finalscore_reconstruction_max_abs"]:.2e}；'
             f'匹配聚合重放误差 {case["checks"]["local_aggregation_replay_max_abs"]:.2e}。'
             '全部精确值、原图路径、源索引与计算定义见同名 JSON。', fontsize=10, color='#5b666b')
    fig.savefig(output, dpi=170, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--legacy-root', type=Path,
                        default=ROOT.parent/'scene_aware_classification_20261018/model_closeout_20261023')
    parser.add_argument('--output', type=Path, default=ROOT/'evidence_gallery_20261025')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
                         'axes.unicode_minus': False, 'font.size': 11})
    requests = {r['id']: r for r in read_json(args.root/'validation_local/requests.json')}
    traces = {r['id']: next(p for p in r['predictions'] if p['seed'] == 17)
              for r in read_json(args.root/'outputs/predictions.json')}
    legacy_path = args.legacy_root/'evidence_preserving_redesign/anchored_polar_adjustment_20261025/analysis/cases.json'
    cases = {str(c['obs_id']): c for c in read_json(legacy_path) if c['seed'] == 17}
    metadata_path = args.legacy_root/'local_evidence_transport/paired_scale_readout/table.csv'
    metadata = pd.read_csv(metadata_path).set_index('table_index', drop=False)
    classifier = ShipClassifier(args.root/'assets', device='cpu')
    receipts = []
    for oid in ('13372', '26043'):
        case = compute_case(classifier, requests[oid], args.root/'validation_local', traces[oid], cases[oid], metadata)
        case['provenance'] = dict(requests=str(args.root/'validation_local/requests.json'),
                                  saved_trace=str(args.root/'outputs/predictions.json'),
                                  bank=str(args.root/'assets/m3/bank.npz'),
                                  reference_metadata=str(metadata_path), legacy_cases=str(legacy_path),
                                  font='C:/Windows/Fonts/msyh.ttc',
                                  original_reference_chips_read_only=True)
        case['legend_statements'] = ['匹配热图不是物理归因', '2例不是整体比例', '分数按固定class pair',
                                     '原始8-bit显示PNG；未对展示图额外拉伸；不是定标RCS']
        prefix = args.output/f'm3_evidence_{oid}_seed17'
        render(case, prefix.with_suffix('.png'))
        prefix.with_suffix('.json').write_text(json.dumps(to_json(case), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        receipt = dict(obs_id=int(oid), png=str(prefix.with_suffix('.png')), json=str(prefix.with_suffix('.json')),
                       fixed_B_margin=case['waterfall']['fixed_B']['margin'],
                       selected_margin=case['waterfall']['selected']['margin'], checks=case['checks'])
        receipts.append(receipt)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
    (args.output/'m3_validation.json').write_text(json.dumps(receipts, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
