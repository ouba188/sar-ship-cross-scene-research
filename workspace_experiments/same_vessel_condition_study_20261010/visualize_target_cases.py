"""Display previously selected real chips with recorded decision accounting."""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

BASE = Path(__file__).resolve().parent
OUT = BASE/'target_response_adapter_balanced_20261010'
NAMES = ['散货船', '渔船', '杂货船', '成品油/化学品船', '集装箱船', '原油轮', '拖船', '海工补给船']
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def render(rows, path, title):
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 4.1*len(rows)),
                             gridspec_kw={'width_ratios': [1, 1, 1.5]}, squeeze=False)
    for i, row in enumerate(rows.itertuples()):
        old = row.fixed_rival if row.kind == 'rescue' else row.true_class
        new = row.true_class if row.kind == 'rescue' else row.fixed_rival
        angle = f'{row.incidence_deg:.1f}°' if pd.notna(row.incidence_deg) else '缺失'
        for j, pol in enumerate(('vv', 'vh')):
            with Image.open(getattr(row, 'image_relpath_'+pol)) as im:
                pixels = np.asarray(im.convert('L'))
            axes[i, j].imshow(pixels, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            axes[i, j].set_title(f'{row.port} | {pol.upper()} | {str(row.acq_time)[:10]}\n'
                                 f'入射角 {angle} | 原图 {pixels.shape[1]}×{pixels.shape[0]}', fontsize=10)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
        axes[i, 0].set_xlabel(f'真实：{NAMES[row.true_class]}\n{row.instance_id}', fontsize=9)
        axes[i, 1].set_xlabel(f'仅适配头：{NAMES[old]}\n内部适配后：{NAMES[new]}', fontsize=10)
        ax = axes[i, 2]
        vals = [row.source_prior_term, row.feature_with_old_head_term, row.support_head_refit_term]
        ax.barh([2, 1, 0], vals, color=['#147d92' if v >= 0 else '#be4d62' for v in vals])
        ax.set_yticks([2, 1, 0], ['原分类头分数变化', '旧残差头下特征变化', '支持头重新拟合'])
        ax.axvline(0, color='#444444', linewidth=.8)
        span = max(max(abs(v) for v in vals), .08)
        ax.set_xlim(min(0, min(vals))-.35*span, max(0, max(vals))+.35*span)
        for y, v in zip([2, 1, 0], vals):
            ax.text(v+(.025*span if v >= 0 else -.025*span), y, f'{v:+.3f}',
                    ha='left' if v >= 0 else 'right', va='center', fontsize=9)
        ax.set_title(f'真实类相对固定竞争类的分数间隔\n{row.old_margin:+.3f} → {row.new_margin:+.3f}', fontsize=10)
        ax.set_xlabel('正值帮助真实类；负值不利于真实类', fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(title+'\n原始8位裁片，不新增增强或方向模拟；图示为事后选例，不代表发生率', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95), h_pad=2, w_pad=2)
    fig.savefig(path, dpi=140, facecolor='white')
    plt.close(fig)


def main():
    cases = pd.read_csv(OUT/'case_score_accounting.csv')
    obs = pd.read_csv(BASE/'observations.csv')
    rows = cases.merge(obs[['instance_id', 'image_relpath_vv', 'image_relpath_vh',
                           'acq_time', 'incidence_deg']], on='instance_id', validate='many_to_one')
    assert len(rows) == len(cases) == 24
    rows.to_csv(OUT/'visualized_case_manifest.csv', index=False, encoding='utf-8-sig')
    for kind, name in [('rescue', '纠错'), ('harm', '误伤')]:
        part = rows[rows.kind.eq(kind)]
        chosen = part.groupby('port', sort=False).head(1)
        render(chosen, OUT/f'chips_{kind}_overview.png', f'目标内部适配：{name}案例（每港首例）')
        for port, group in part.groupby('port', sort=False):
            render(group, OUT/f'chips_{kind}_{port.replace(" ", "_")}.png', f'{port}：全部已选{name}案例')
    print(f'Rendered all {len(rows)} selected cases; 2 overview sheets and 6 full sheets.')


if __name__ == '__main__':
    main()
