"""Outcome-blind factor summaries and explicitly selected classification cases."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tifffile

HERE = Path(__file__).resolve().parent
FOLDER = HERE/'figures'


def main():
    summary = pd.read_csv(HERE/'image_response_summary.csv')
    choices = [('vv', 'body_neighbor_db', 'environment'), ('vh', 'body_neighbor_db', 'speed'),
               ('vv', 'azimuth_range_log_ratio', 'motion_geometry'), ('vh', 'azimuth_range_log_ratio', 'motion_geometry'),
               ('vv', 'ends_sides_balance', 'draught'), ('vh', 'ends_sides_balance', 'draught')]
    rows = [summary[summary.pol.eq(p) & summary.outcome.eq(o) & summary.arm.eq(a)].iloc[0] for p, o, a in choices]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, row in enumerate(rows):
        value = row.product_relative_mse_reduction_pct
        ax.errorbar(value, i, xerr=[[max(0, value-row.bootstrap_low_pct)], [max(0, row.bootstrap_high_pct-value)]],
                    fmt='o', color='#087e8b' if value > 0 else '#b2403b', capsize=3)
    ax.set_yticks(range(len(rows)), [f'{p.upper()} | {o}\n+ {a}' for p, o, a in choices], fontsize=9)
    ax.axvline(0, color='.45', lw=1)
    ax.set_xlabel('Relative reduction of acquisition-macro image-response MSE (%)')
    ax.set_title('Unseen acquisition and vessel | diagnostic regression, not classification gain', fontsize=10)
    ax.grid(axis='x', alpha=.2)
    fig.tight_layout()
    fig.savefig(FOLDER/'factor_prediction.png', dpi=160)
    plt.close(fig)
    edges = pd.read_csv(HERE/'classifier/changed_support_edges.csv')
    geom = pd.read_csv(HERE.parent/'oss1_joint_geometry_20261017/geometry.csv').set_index('global_row')
    meta = pd.read_csv(HERE/'metadata/oss1_factor_metadata.csv').set_index('global_row')
    selected = []
    # These are outcome-selected mechanism examples, not a representative sample.
    for arm, qid, draw in [('uint8_spatial', 982, 3), ('sigma0_spatial', 724, 4)]:
        e = edges[edges.origin.eq('cross') & edges.arm.eq(arm) & edges.query_global_row.eq(qid) & edges.draw.eq(draw)]
        dominant = e.reindex(e.total_expert_margin_change.abs().sort_values(ascending=False).index).head(3)
        ids = [qid]+dominant.support_global_row.astype(int).tolist()
        fig, axes = plt.subplots(2, len(ids), figsize=(12, 6))
        for j, index in enumerate(ids):
            pixels = tifffile.imread(geom.loc[index, 'calibrated_patch_path'])
            item = meta.loc[index]
            for i, (pol, channel) in enumerate([('VV', 1), ('VH', 0)]):
                axes[i, j].imshow(10*np.log10(pixels[:, :, channel].clip(1e-8)), cmap='gray', vmin=-35, vmax=5)
                extra = '' if j == 0 else f'\nmargin contribution {dominant.iloc[j-1].total_expert_margin_change:+.3f}'
                axes[i, j].set_title(f'{pol} {"query" if j == 0 else "support"} {index}\n{item.class_name}\n'
                    f'SOG {item.ais_sog_knots:.1f} kn, gap {item.ais_abs_timegap_s:.0f}s'+extra, fontsize=8)
                axes[i, j].tick_params(labelsize=6)
        fig.suptitle(f'{arm} | draw {draw} | outcome-selected case, signed expert-margin attribution\n'
                     'Support refitting included; not nearest-neighbor or causal attribution', fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, .91])
        fig.savefig(FOLDER/f'case_{arm}_{qid}.png', dpi=160)
        plt.close(fig)
        selected.append(dict(arm=arm, query=qid, draw=draw, support_rows=ids[1:],
                             expert_margin_change=e.total_expert_margin_change.sum()))
    pd.DataFrame(selected).to_csv(FOLDER/'case_selection.csv', index=False)
    data = pd.read_csv(HERE/'native_shape_measurements.csv')
    rows = []
    for pol, block in data.groupby('pol'):
        for measure in ['body_top5_mass', 'azimuth_range_log_ratio', 'outside_to_body_mean']:
            pivot = block.pivot(index='global_row', columns='representation', values=measure)
            for rep in ['sigma0', 'uint8_lowpass']:
                rows.append(dict(pol=pol, measure=measure, representation=rep,
                    n=int(pivot[['uint8', rep]].notna().all(axis=1).sum()),
                    spearman_vs_display=pivot['uint8'].corr(pivot[rep], method='spearman'),
                    median_absolute_change=(pivot[rep]-pivot.uint8).abs().median()))
    pd.DataFrame(rows).to_csv(HERE/'same_observation_processing.csv', index=False)


if __name__ == '__main__':
    main()
