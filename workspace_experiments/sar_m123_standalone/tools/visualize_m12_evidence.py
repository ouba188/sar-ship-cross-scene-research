"""Plot actual M1/M2 response readouts and exact frozen-head score accounting."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sar_m123.pipeline import ShipClassifier
from sar_m123.input import prepare
from sar_m123.baseline import extract as baseline_extract
from sar_m123.m12 import M12, attention_raw, centered_response
from sar_m123.regions import soft_regions

NAMES = ['散货船', '渔船', '普通货船', '成品油/化学品船', '集装箱船', '原油轮', '拖带船', '海工供应船']
OUT = ROOT/'evidence_gallery_20261025'


def setup():
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size': 11, 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.facecolor': 'white', 'figure.facecolor': 'white'})
    OUT.mkdir(exist_ok=True)


def image(ax, a, title):
    ax.imshow(a, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    ax.set_title(title, fontsize=12, pad=9)
    ax.set_xticks([0, 80, 159]); ax.set_yticks([0, 80, 159])
    ax.tick_params(labelsize=9)


def bars(ax, values, labels, title):
    colors = ['#475569' if i in (0, len(values)-1) else
              ('#16806a' if v >= 0 else '#c64f4f') for i, v in enumerate(values)]
    ax.barh(np.arange(len(values)), values, color=colors, height=.58)
    ax.axvline(0, color='#64748b', linewidth=.8)
    ax.set_yticks(np.arange(len(values)), labels)
    ax.invert_yaxis(); ax.set_title(title, fontsize=12)
    span = max(max(abs(np.array(values))), .04)
    ax.set_xlim(min(0, min(values))-span*.25, max(0, max(values))+span*.3)
    for i, value in enumerate(values):
        ax.text(value + (.02*span if value >= 0 else -.02*span), i,
                f'{value:+.4f}', va='center', ha='left' if value >= 0 else 'right', fontsize=11)
    ax.grid(axis='x', alpha=.15)


@torch.inference_mode()
def spatial_m1(model, items, maps, truth, competitor, summaries):
    views, errors = [], []
    for p, (item, feature) in enumerate(zip(items, maps)):
        if item is None:
            views.append(None)
            continue
        r_maps, a_plus, a_minus, packed = [], [], [], []
        W = torch.tensor(model.summary_reader.readers['layer3'][p],
                         dtype=torch.float64, device=model.backbone.device)
        tau = float(model.summary_reader.tau['layer3'][p])
        for k in range(4):
            response = torch.einsum('ck,bku->bcu', W, feature['layer3'][k:k+1].flatten(2).double())
            weights = np.rot90(item['footprint'], k).reshape(10, 16, 10, 16).mean((1, 3))
            w = torch.tensor(weights.reshape(1, 1, -1), device=W.device, dtype=torch.float64)
            norm = w/w.sum(-1, keepdim=True)
            mu, r = centered_response(response, norm)
            r -= (r*norm[:, :, None]).sum(-1, keepdim=True)
            logw = torch.where(norm > 0, norm.log(), -torch.inf)[:, :, None]
            plus, minus = (logw+r/tau).softmax(-1), (logw-r/tau).softmax(-1)
            qp, qm = (plus*r).sum(-1), (minus*r).sum(-1)
            packed.append(torch.stack((mu, (qp-qm)/2, (qp+qm)/2), -1)[0, 0].cpu().numpy())
            contrast = (r[0, 0, truth]-r[0, 0, competitor]).reshape(10, 10).cpu().numpy()
            # Responses on padding are computed by the backbone but not read by M1.
            r_maps.append(np.rot90(np.where(weights > 0, contrast, np.nan), -k))
            a_plus.append(np.rot90(plus[0, 0, truth].reshape(10, 10).cpu().numpy(), -k))
            a_minus.append(np.rot90(minus[0, 0, truth].reshape(10, 10).cpu().numpy(), -k))
        got = np.mean(packed, 0)
        expected = np.column_stack((summaries['mean3'][p], summaries['hb3'][p]))
        errors.append(float(np.max(abs(got-expected))))
        counted = np.isfinite(r_maps).sum(0)
        response = np.divide(np.nansum(r_maps, 0), counted,
                             out=np.full((10, 10), np.nan), where=counted > 0)
        views.append(dict(response=response, plus=np.mean(a_plus, 0), minus=np.mean(a_minus, 0)))
    assert max(errors) < 1e-9
    return views, max(errors)


@torch.inference_mode()
def m1_panel(model, oid, truth, items, maps, base, summaries):
    head = M12(ROOT/'assets', 29)
    s12, details = head(base, summaries)
    competitor = int(np.argmax(np.where(np.arange(8) == truth, -np.inf, base)))
    raw = attention_raw(base[None], summaries['mean3'][None], summaries['hb3'][None], summaries['valid'][None])
    mu, sd = head.scalers['attention']
    x = torch.tensor(((raw-mu)/sd).astype(np.float32))
    m = head.heads['attention']
    mean = m.mean(x[..., [0, 1, 2, 7, 8, 13, 14]]).squeeze(-1)[0].numpy()
    local = m.local(x[..., [0, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14]]).squeeze(-1)[0].numpy()
    dm, dl = mean[truth]-mean[competitor], local[truth]-local[competitor]
    margin = lambda z: float(z[truth]-z[competitor])
    error = abs(margin(details['s0'])-(margin(base)+float(dm)+float(dl)))
    assert error < 2e-6
    views, summary_error = spatial_m1(model, items, maps, truth, competitor, summaries)
    fig = plt.figure(figsize=(15, 11.3))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, .72], hspace=.43, wspace=.36)
    vmax = max(np.nanmax(abs(v['response'])) for v in views if v is not None)
    amax = max(max(v['plus'].max(), v['minus'].max()) for v in views if v is not None)
    for p, (item, view) in enumerate(zip(items, views)):
        axes = [fig.add_subplot(gs[p, c]) for c in range(4)]
        if item is None:
            for ax in axes:
                ax.text(.5, .5, '该极化无目标证据', ha='center'); ax.axis('off')
            continue
        image(axes[0], item['pixels'], ('VV' if p == 0 else 'VH')+' 原生160画布')
        for ax, key, title in zip(axes[1:], ('response', 'plus', 'minus'),
                                 ('类别相对响应差', '真实类高侧注意力', '真实类低侧注意力')):
            kwargs = dict(cmap='RdBu_r', vmin=-vmax, vmax=vmax) if key == 'response' else dict(cmap='magma', vmin=0, vmax=amax)
            im = ax.imshow(view[key], interpolation='nearest', **kwargs)
            ax.set_title(title, fontsize=12); ax.set_xticks([0, 5, 9]); ax.set_yticks([0, 5, 9])
            fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
    ax = fig.add_subplot(gs[2, :2])
    bars(ax, [margin(base), float(dm), float(dl), margin(details['s0'])],
         ['基础分差', '均值支路增量', '分布注意力增量', 'M1后分差'], '精确分数记账：前3项相加 = 最后一项')
    ax = fig.add_subplot(gs[2, 2:]); ax.axis('off')
    ax.text(0, .96, f'查询 {oid} | 配置29 | M1选中70步\n真实类：{NAMES[truth]}\n固定竞争类：{NAMES[competitor]}\n\n红响应：相对更支持真实类；蓝响应：相对更支持竞争类。\n高/低侧都由同一真实类响应计算，不是高频/低频。\n热图为四旋转逆对齐均值；不是像素因果贡献。', va='top', fontsize=11, linespacing=1.6)
    fig.suptitle(f'M1证据 | {oid}：平均响应之外，分布支路实际改变了多少分数？', fontsize=18, y=.99)
    fig.subplots_adjust(top=.94, bottom=.055, left=.08, right=.97)
    fig.text(.08, .015, '原chip不缩放进入模型；图中统一显示放大。热图保留10×10网格，补零不参与响应展示；VV/VH共用色标。', fontsize=10)
    fig.savefig(OUT/f'm1_{oid}_29.png', dpi=145); plt.close(fig)
    return dict(id=oid, module='M1', seed=29, truth=truth, competitor=competitor,
                base_margin=margin(base), mean_delta=float(dm), distribution_delta=float(dl),
                output_margin=margin(details['s0']), score_accounting_error=error,
                summary_reconstruction_error=summary_error)


def m2_panel(oid, truth, items, base, summaries):
    s12, details = M12(ROOT/'assets', 17)(base, summaries)
    s1 = details['s0']
    competitor = int(np.argmax(np.where(np.arange(8) == truth, -np.inf, s1)))
    margin = lambda z: float(z[truth]-z[competitor])
    delta = margin(details['delta_m2'])
    assert abs(margin(s12)-margin(s1)-delta) < 1e-8
    contrasts = summaries['summary1'][:, :, truth]-summaries['summary1'][:, :, competitor]
    vmax = max(abs(contrasts).max(), 1e-8)
    fig = plt.figure(figsize=(13.4, 10.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, .72], hspace=.4, wspace=.38)
    colors = np.array([[.12, .64, .47], [.94, .62, .13], [.21, .50, .86]])
    for p, item in enumerate(items):
        axes = [fig.add_subplot(gs[p, c]) for c in range(3)]
        if item is None:
            for ax in axes:
                ax.axis('off')
            continue
        pol = 'VV' if p == 0 else 'VH'
        image(axes[0], item['pixels'], pol+' 原始输入')
        image(axes[1], item['pixels'], pol+' 实际OBB池化区域')
        masks = soft_regions(item['geometry'])*item['footprint']
        alpha = masks.max(0)*.52
        rgba = np.concatenate((np.einsum('rhw,rc->hwc', masks, colors)/np.maximum(masks.sum(0)[..., None], 1e-15), alpha[..., None]), -1)
        axes[1].imshow(rgba, interpolation='nearest')
        im = axes[2].imshow(contrasts[p], cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto', interpolation='nearest')
        axes[2].set_title('区域摘要：真实类 − 竞争类', fontsize=12)
        axes[2].set_xticks(range(3), ['均值', '高低侧差', '高低侧中点'])
        axes[2].set_yticks(range(3), ['框内', '轴端', '侧向'])
        for i in range(3):
            for j in range(3):
                value = contrasts[p, i, j]
                axes[2].text(j, i, f'{value:+.2f}', ha='center', va='center',
                             color='white' if abs(value) > vmax*.58 else '#18232b')
        fig.colorbar(im, ax=axes[2], fraction=.045, pad=.03)
    ax = fig.add_subplot(gs[2, :2])
    bars(ax, [margin(s1), delta, margin(s12)], ['M1后分差', 'M2总增量', 'M2后分差'],
         f'固定类别对：{NAMES[truth]} − {NAMES[competitor]}')
    ax = fig.add_subplot(gs[2, 2]); ax.axis('off')
    ax.text(0, .98, '配置17：M2选中15步\n绿色：框内\n黄色：轴端（两端合并）\n蓝色：侧向（两侧合并）\n\n摘要数值不是最终区域贡献；\nM2还使用质量与上下文。\n这里没有把区域命名成物理机制。', va='top', fontsize=10, linespacing=1.5)
    fig.suptitle(f'M2证据 | {oid}：同一类别响应，保留它出现的区域身份', fontsize=18, y=.99)
    fig.subplots_adjust(top=.94, bottom=.055, left=.09, right=.96)
    fig.text(.09, .015, '区域由当前OBB计算，补零不计入。原图保留像素尺度；色块是实际池化覆盖，不是显著性或因果归因图。', fontsize=10)
    fig.savefig(OUT/f'm2_{oid}_17.png', dpi=150); plt.close(fig)
    return dict(id=oid, module='M2', seed=17, truth=truth, competitor=competitor,
                input_margin=margin(s1), delta=delta, output_margin=margin(s12),
                regional_summary_contrasts=contrasts.tolist(), quality=summaries['quality'].tolist(),
                score_accounting_error=abs(margin(s12)-margin(s1)-delta))


def main():
    setup()
    fixture = ROOT/'validation_local'
    rows = json.loads((fixture/'requests.json').read_text(encoding='utf8'))
    expected = np.load(fixture/'expected.npz')
    truths = {str(o): int(y) for o, y in zip(expected['obs_ids'], expected['truth'])}
    model = ShipClassifier(ROOT/'assets', 'cuda' if torch.cuda.is_available() else 'cpu')
    reports = []
    for row in rows:
        oid = row['id']
        if oid not in ('13372', '21151', '2418', '16596'):
            continue
        prepared = prepare(row, fixture)
        maps = model.backbone(prepared)
        baseline = baseline_extract(prepared, maps, model.backbone)
        target = [v if v is not None and v['valid_target'] else None for v in prepared]
        target_maps = [m if v is not None else None for v, m in zip(target, maps)]
        summaries = model.summary_reader.summarize(target, target_maps)
        from sar_m123.baseline import Baseline
        seed = 29 if oid in ('13372', '21151') else 17
        base, _ = Baseline(ROOT/'assets', seed)(baseline)
        if seed == 29:
            report = m1_panel(model, oid, truths[oid], target, target_maps, base, summaries)
        else:
            report = m2_panel(oid, truths[oid], target, base, summaries)
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    (OUT/'m12_evidence.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
