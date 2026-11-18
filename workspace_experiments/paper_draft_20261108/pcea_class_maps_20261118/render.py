"""Read-only routed-checkpoint class/region evidence visualization; no training."""
from pathlib import Path
import argparse
import csv
import html
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent
RUN = PAPER / 'pcea_routed_correction_20261118'
NAMES = ['Bulk', 'Fishing', 'General cargo', 'Chemical/product', 'Container', 'Crude', 'Tug', 'Offshore']
REGIONS = ['Body', 'Axial ends', 'Lateral']


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def capture_reader(module, args, output):
    shallow, _, masks, responses, valid, _ = args
    all_weights, all_scores, rebuilt = [], [], []
    for p in range(2):
        value = F.layer_norm(shallow[:, p].permute(0, 2, 3, 1), (512,)).permute(0, 3, 1, 2).flatten(2)
        mask, response, occupancy = (x.flatten(2) for x in (masks[:, p], responses[:, p], valid[:, p]))
        uniform = mask / mask.sum(-1, keepdim=True).clamp_min(1e-8)
        classes = response.shape[1]
        relative = (classes * response - response.sum(1, keepdim=True)) / (classes - 1)
        vw = occupancy / occupancy.sum(-1, keepdim=True).clamp_min(1e-8)
        centered = relative - (relative * vw).sum(-1, keepdim=True)
        score = 2 * torch.tanh(centered / ((centered.square() * vw).sum(-1, keepdim=True) + .01).sqrt())
        logits = score[:, None] + mask.clamp_min(1e-12).log()[:, :, None]
        weight = logits.masked_fill(mask[:, :, None] <= 0, -1e4).softmax(-1)
        weight = weight * (mask[:, :, None] > 0)
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-8)
        mean = torch.einsum('bru,bdu->brd', uniform, value)
        tokens = torch.einsum('brku,bdu->brkd', weight, value)
        rebuilt.append(torch.cat((mean[:, :, None], tokens), 2))
        all_weights.append(weight.reshape(-1, 3, 8, 20, 20))
        all_scores.append(score.reshape(-1, 8, 20, 20))
    raw = torch.stack(rebuilt, 1)
    error = float((raw - module.raw).abs().max())
    assert error < 1e-5, ('Token reconstruction', error)
    weights = torch.stack(all_weights, 1)
    sums = weights.sum((-2, -1))
    expected = (masks.sum((-2, -1)) > 0)[..., None].expand_as(sums).float()
    assert torch.allclose(sums, expected, atol=1e-6)
    assert torch.isfinite(weights).all()
    assert not bool((weights * (masks[:, :, :, None] <= 0)).any())
    CAPTURE.clear()
    CAPTURE.update(response=responses.detach().cpu().numpy(), score=torch.stack(all_scores, 1).cpu().numpy(),
                   weights=weights.cpu().numpy(), masks=masks.cpu().numpy(), valid=valid.cpu().numpy(),
                   token_reconstruction_error=error)


def image_axis(ax, pixels):
    ax.imshow(pixels, cmap='gray', vmin=0, vmax=1, interpolation='nearest', extent=(-.5, 159.5, 159.5, -.5))
    ax.set_xlim(-.5, 159.5)
    ax.set_ylim(159.5, -.5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def heat(ax, values, visible, limit, cmap='RdBu_r'):
    # Use actual origin-zero stride-8 cell boundaries, not an image resize.
    edges = np.r_[-.5, np.arange(4, 156, 8), 159.5]
    z = np.ma.masked_where(~visible, values)
    return ax.pcolormesh(edges, edges, z, cmap=cmap, vmin=-limit, vmax=limit,
                         shading='flat', alpha=.66, rasterized=False)


def savefig(fig, name):
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(ROOT / f'{name}.{ext}', dpi=180, facecolor='white')
    plt.close(fig)


def render_case(j, row, data, capture, on, off, references):
    sid, truth = row['id'], int(row['label'])
    rival = int(np.argmax(np.where(np.arange(8) == truth, -np.inf, off)))
    pixels = data['images'][j, :, 0].cpu().numpy()
    score, weights, masks = (capture[k][j] for k in ('score', 'weights', 'masks'))
    valid = capture['valid'][j, :, 0] > 0
    margin = lambda z: float(z[truth] - np.max(np.delete(z, truth)))
    result = dict(id=sid, index=int(INDEX[j]), truth=truth, rival=rival, port=row.get('port'),
                  seed=43, checkpoint=str(RUN / 'fits/routed_s43/best.pt'),
                  prediction_off=int(off.argmax()), prediction_on=int(on.argmax()),
                  margin_off=margin(off), margin_on=margin(on),
                  original_paths={p: row.get(p) for p in ('vv', 'vh')})
    title = f"ID {sid} | {row.get('port', '')} | true: {NAMES[truth]} | PCEA off/on: {NAMES[off.argmax()]} / {NAMES[on.argmax()]}"
    fig, axes = plt.subplots(2, 9, figsize=(18, 5), layout='constrained')
    for p in range(2):
        image_axis(axes[p, 0], pixels[p])
        axes[p, 0].set_title(('VV' if p == 0 else 'VH') + ' input')
        for k in range(8):
            ax = axes[p, k + 1]
            image_axis(ax, pixels[p])
            artist = heat(ax, score[p, k], valid[p], 2)
            label = NAMES[k] + (' [true]' if k == truth else ' [rival]' if k == rival else '')
            ax.set_title(label, fontsize=9)
    fig.suptitle(title + '\nCCREL class-competitive response; identical color scale, no per-panel stretching', fontsize=12)
    fig.colorbar(artist, ax=axes, shrink=.7, label='Bounded competitive response [-2, 2]')
    savefig(fig, f'{sid}_classes')
    fig, axes = plt.subplots(6, 4, figsize=(11.3, 16), layout='constrained')
    metrics = []
    for p in range(2):
        for r in range(3):
            n = p * 3 + r
            mask = masks[p, r]
            uniform = mask / max(float(mask.sum()), 1e-8)
            keep = mask > 0
            for col in range(4):
                image_axis(axes[n, col], pixels[p])
            axes[n, 0].pcolormesh(np.r_[-.5, np.arange(4, 156, 8), 159.5],
                                  np.r_[-.5, np.arange(4, 156, 8), 159.5],
                                  np.ma.masked_where(~keep, mask), cmap='Greens', vmin=0, vmax=1, alpha=.65)
            axes[n, 0].set_title(f"{('VV', 'VH')[p]} / {REGIONS[r]}\nRegion occupancy", fontsize=10)
            densities = []
            for col, k in ((1, truth), (2, rival)):
                density = np.log2(np.maximum(weights[p, r, k], 1e-12) / np.maximum(uniform, 1e-12))
                densities.append(density)
                artist = heat(axes[n, col], density, keep, 4)
                axes[n, col].set_title(NAMES[k] + '\nlog2(weight / uniform)', fontsize=10)
            difference = densities[0] - densities[1]
            heat(axes[n, 3], difference, keep, 4)
            axes[n, 3].set_title('True - rival\nlog2 density difference', fontsize=10)
            tv = .5 * np.abs(weights[p, r, truth] - weights[p, r, rival]).sum()
            metrics.append(dict(id=sid, polarization=('VV', 'VH')[p], region=REGIONS[r],
                                true_rival_total_variation=float(tv), region_mass=float(mask.sum())))
    fig.suptitle(title + f'\nFinal true-class margin: off {margin(off):+.6f}, on {margin(on):+.6f}\n'
                 'Actual regional readout weights, displayed relative to uniform pooling; not final-logit attribution', fontsize=11)
    fig.colorbar(artist, ax=axes, shrink=.4, label='log2 ratio / difference (display clipped at +/-4)')
    savefig(fig, f'{sid}_regions')
    np.savez_compressed(ROOT / f'{sid}_maps.npz', pixels=pixels, response=capture['response'][j],
                        score=score, weights=weights, masks=masks, valid=valid,
                        logits_on=on, logits_off=off, reference_ids=references)
    with (ROOT / f'{sid}_weights.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['polarization', 'region', 'class', 'grid_y', 'grid_x', 'weight', 'region_occupancy', 'competitive_response'])
        for p in range(2):
            for r in range(3):
                for k in range(8):
                    for y, x in np.argwhere(masks[p, r] > 0):
                        writer.writerow([('VV', 'VH')[p], REGIONS[r], NAMES[k], y, x,
                                         float(weights[p, r, k, y, x]), float(masks[p, r, y, x]), float(score[p, k, y, x])])
    return result, metrics


def main():
    global torch, np, F, plt, CAPTURE, INDEX
    parser = argparse.ArgumentParser()
    parser.add_argument('--ids', nargs='+', required=True, help='Existing validation IDs, in display order')
    args = parser.parse_args()
    m = load('class_maps_model', PAPER / 'pcea_class_local_20261117/model.py')
    torch, np, F = m.torch, m.np, m.F
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    print('Loading inputs', flush=True)
    inputs = m.Inputs()
    INDEX = np.array([next(int(i) for i in inputs.groups['validation'] if inputs.rows[i]['id'] == sid) for sid in args.ids])
    net = m.Model('generic')
    net.load_state_dict(torch.load(RUN / 'fits/routed_s43/best.pt', map_location='cuda', weights_only=True), strict=True)
    net.requires_grad_(False)
    net.accelerate()
    net.eval()
    CAPTURE = {}
    print('Replaying three existing cases', flush=True)
    with torch.no_grad():
        reference = net.reference(inputs.data(inputs.groups['references'], 'small'))
        data = inputs.data(INDEX, 'validation')
        rows = [inputs.rows[i] for i in INDEX]
        ordinary, _ = net(data, rows, reference, inputs.reference_rows)
        handle = net.ccrel.register_forward_hook(capture_reader)
        on, _ = net(data, rows, reference, inputs.reference_rows)
        capture = dict(CAPTURE)
        net.exchange.disabled = True
        off, _ = net(data, rows, reference, inputs.reference_rows)
        assert np.array_equal(capture['weights'], CAPTURE['weights'])
        assert np.array_equal(capture['response'], CAPTURE['response'])
        handle.remove()
    on, off, ordinary = (x.cpu().numpy() for x in (on, off, ordinary))
    hook_error = float(abs(on - ordinary).max())
    assert hook_error == 0
    errors = {}
    for mode, logits in (('on', on), ('off', off)):
        cache = np.load(RUN / f'fits/routed_s43/validation_{mode}.npz')
        positions = [int(np.flatnonzero(cache['ids'] == i)[0]) for i in INDEX]
        error = float(abs(logits - cache['score'][positions]).max())
        assert error < 2e-5, (mode, error)
        assert np.array_equal(logits.argmax(1), cache['score'][positions].argmax(1))
        errors[mode] = error
    reports, metrics = [], []
    for j, row in enumerate(rows):
        result, detail = render_case(j, row, data, capture, on[j], off[j], inputs.groups['references'])
        reports.append(result)
        metrics.extend(detail)
    report = dict(checks=dict(hook_logit_error=hook_error, cached_logit_errors=errors,
                  token_reconstruction_error=capture['token_reconstruction_error'],
                  upstream_maps_unchanged_on_off=True, weights_normalized=True), cases=reports, region_metrics=metrics)
    (ROOT / 'report_local.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    sections = []
    for case in reports:
        sid = case['id']
        sections.append(f'<section><h2>ID {sid}: {html.escape(NAMES[case["truth"]])}</h2>'
            f'<p>PCEA off/on margin: {case["margin_off"]:+.6f} / {case["margin_on"]:+.6f}</p>'
            f'<h3>Eight-class competitive responses</h3><a href="{sid}_classes.svg"><img src="{sid}_classes.png"></a>'
            f'<h3>Actual CCREL regional weights</h3><a href="{sid}_regions.svg"><img src="{sid}_regions.png"></a>'
            f'<p><a href="{sid}_weights.csv">Cell weights CSV</a> | <a href="{sid}_maps.npz">Original tensors</a></p></section>')
    (ROOT / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>PCEA class-specific evidence</title>'
        '<style>body{max-width:1600px;margin:24px auto;font-family:Arial,sans-serif;color:#202020}img{width:100%}'
        'section{border-top:1px solid #bbb;margin-top:28px}p{line-height:1.6}</style>'
        '<h1>类别响应与区域取证：最新PCEA真实检查点</h1>'
        '<p>固定seed43及16个源参考，不重训。每例分别显示VV/VH。第一组是八个类别的竞争响应；'
        '第二组是CCREL实际区域权重相对均值池化的密度。红色表示相对增强，蓝色表示相对降低，'
        '不是最终分类分数的正负贡献。点击图片打开文字可编辑的SVG；另有PDF、CSV和NPZ。</p>'
        '<p>取证图解释“如何读取”，PCEA开关分差解释“回写对本例有无作用”。两者不能互相替代。'
        '底图是实际送入模型的160像素输入；20×20网格按真实stride-8边界叠加，未进行平滑美化。</p>'
        + ''.join(sections), encoding='utf-8')
    print(json.dumps(report['checks']), flush=True)
    for result in reports:
        print(result['id'], result['margin_off'], result['margin_on'], flush=True)


if __name__ == '__main__':
    main()
