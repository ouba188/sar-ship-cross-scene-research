"""Read-only historical case replay and data visualization; no training/saving weights."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HISTORY = ROOT.parent / 'm3_explained_20261029'
sys.path.insert(0, str(HISTORY))
import trace_relations as tr
import numpy as np
import torch


@torch.no_grad()
def main():
    exp, joint = tr.exp, tr.j
    exp.DEVICE = 'cpu'
    exp.current.setup('cpu')
    torch.set_num_threads(2)
    rows, nf, refs, rid, arrays = exp.load()
    cache = joint.pair_cache()
    idx = next(i for i, row in enumerate(rows) if str(row['id']) == '24470')
    model = tr.review.restore('N_level_pattern', 43)
    model.requires_grad_(False)
    score, on = joint.forward(model, 'N_level_pattern', cache, arrays,
                             np.array([idx]), rid, rows, details=True)
    for op in model.m3[0]:
        op.frequency = False
    score_off, off = joint.forward(model, 'N_level_pattern', cache, arrays,
                                  np.array([idx]), rid, rows, details=True)
    original = exp.tensor(arrays['x'], [idx])
    rebuilt = tr.downstream(model, arrays, idx,
                            original + torch.stack([d['update'] for d in on], 1))
    removed = tr.downstream(model, arrays, idx, original)
    for op in model.m3[0]:
        op.gain.zero_()
    zero, _ = joint.forward(model, 'N_level_pattern', cache, arrays,
                           np.array([idx]), rid, rows, details=True)
    previous = np.load(HISTORY / 'relation_evidence/24470_traces.npz')['intervention_logits'][0]
    replay_error = float(np.max(np.abs(tr.arr(score)[0] - previous)))
    downstream_error = float((rebuilt - score).abs().max())
    removal_error = float((zero - removed).abs().max())
    assert replay_error < 1e-4, replay_error
    assert downstream_error < 1e-4, downstream_error
    assert removal_error < 1e-4, removal_error
    conditions = []
    for name, value in [('historical_full', score), ('frequency_off', score_off),
                        ('all_m3_writeback_off', removed)]:
        value = tr.arr(value)[0]
        conditions.append(dict(condition=name, prediction=int(value.argmax()),
                               predicted_class=tr.CLASSES[value.argmax()],
                               margin=float(tr.review.margin(value[None], np.array([3]))[0]),
                               logits=value.tolist()))
    reference_rows = [dict(id=str(rows[i]['id']), label=int(rows[i]['label']),
                           category=tr.CLASSES[rows[i]['label']],
                           port=rows[i].get('port'), pass_group=rows[i].get('pass_group'))
                      for i in rid]
    assert len(reference_rows) == 16
    payload = dict(query='24470', port=rows[idx].get('port'),
                   pass_group=rows[idx].get('pass_group'), candidate='N_level_pattern', seed=43,
                   base_checkpoint=str(exp.PACKAGE / 'weights/selected_s17.pt'),
                   candidate_checkpoint=str(joint.HERE / 'training/N_level_pattern_43/m3.pt'),
                   scope='Historical cached feature replay, no retraining or formal-model changes.',
                   replay_error=replay_error, downstream_error=downstream_error,
                   removal_error=removal_error, references=reference_rows, conditions=conditions)
    (ROOT / 'REPLAY_RECEIPT.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    # Show the complete native input grid, not just a display crop of the target.
    fig, axes = tr.plt.subplots(2, 4, figsize=(12, 6.5))
    names = ['24470', '24527', '7790', '31512']
    for col, name in enumerate(names):
        row = next(r for r in rows if str(r['id']) == name)
        for pol, view in enumerate(exp.data.prepare(row)):
            ax = axes[pol, col]
            ax.imshow(view['pixels'], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            masks = exp.tokens.slots(view['geometry'], view['footprint'])
            for region, color in zip(exp.tokens.REGIONS, (tr.GREEN, tr.BLUE, tr.ORANGE)):
                mask = masks[region].sum(0)
                if mask.max() > .05:
                    ax.contour(mask, levels=[.05], colors=[color], linewidths=.7)
            ax.set_title(f'{tr.POL[pol]} | {name}\n{tr.CLASSES[row["label"]]}', fontsize=10)
            ax.set_xticks([0, 80, 159]); ax.set_yticks([0, 80, 159])
    fig.suptitle('24470与实际参考：完整160×160原生画布，VV/VH分别查看', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(ROOT / 'native_inputs.png', dpi=160)
    tr.plt.close(fig)
    fig, axes = tr.plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={'width_ratios': [1.2, 1]})
    colors = [tr.BLUE, tr.GRAY, tr.ORANGE]
    titles = ['历史候选完整M3', '仅关闭M3频域比较', '关闭全部M3回写']
    x = np.arange(8)
    for i, (condition, color, title) in enumerate(zip(conditions, colors, titles)):
        axes[0].bar(x + (i-1)*.24, condition['logits'], .24, color=color, label=title)
    axes[0].set_xticks(x, tr.CLASSES, rotation=40, ha='right')
    axes[0].set_ylabel('分类logit，非概率'); axes[0].legend(fontsize=8)
    axes[1].barh(range(3), [r['margin'] for r in conditions], color=colors)
    axes[1].set_yticks(range(3), titles); axes[1].axvline(0, color='black', lw=.8)
    axes[1].set_xlabel('正确类分差：正数表示判对')
    for i, row in enumerate(conditions):
        axes[1].text(0.02, .92-i*.09, f'{titles[i]}：{row["margin"]:+.6f}',
                     transform=axes[1].transAxes, fontsize=9, color=colors[i])
    fig.suptitle('固定同一历史模型：区分频域贡献与全部M3回写贡献', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(ROOT / 'output_controls.png', dpi=180)
    fig.savefig(ROOT / 'output_controls.pdf')
    tr.plt.close(fig)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
