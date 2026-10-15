"""Fixed-seed within-class/port visual comparison; not an angle-truth audit."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from extract_ship_axis_proxy import OUT, SOURCE


def main():
    observations = pd.read_csv(OUT/'observability/observations.csv')
    manifest = pd.read_csv(SOURCE/'manifest.csv', low_memory=False).set_index('instance_id')
    axes = pd.read_csv(OUT/'view_axes.csv', low_memory=False).set_index(['instance_id', 'polarization'])
    rng = np.random.default_rng(20261009)
    selected = []
    for port in ['Jebel Ali', 'Port Klang', 'Singapore']:
        for low, high in [(30, 45), (75, 90.0001)]:
            group = observations[observations.port.eq(port) & observations.class_id.eq(4)
                & observations.relative_acute_deg.ge(low) & observations.relative_acute_deg.lt(high)]
            selected.extend(rng.choice(group.index.to_numpy(), min(2, len(group)), replace=False))
    chosen = observations.loc[selected]
    chosen[['instance_id', 'port', 'relative_acute_deg', 'correct_fraction', 'gray_contrast']].to_csv(
        OUT/'observability/angle_case_selection.csv', index=False)
    fig, plots = plt.subplots(6, 4, figsize=(11, 15))
    for i, row in enumerate(chosen.itertuples()):
        for j, pol in enumerate(['vv', 'vh']):
            ax = plots.ravel()[2*i+j]
            with Image.open(manifest.loc[row.instance_id, 'image_relpath_'+pol]) as im:
                image = np.asarray(im.convert('L').resize((128, 128), Image.Resampling.BILINEAR))
            line = axes.loc[row.instance_id, pol]
            rad = np.deg2rad(line.angle_pixel_deg)
            ax.imshow(image, cmap='gray', vmin=0, vmax=255)
            ax.plot(line.cx+np.array([-22, 22])*np.cos(rad), line.cy+np.array([-22, 22])*np.sin(rad), c='#24dcb2', lw=1)
            ax.set_title(f'{row.port} {pol.upper()} | angle {row.relative_acute_deg:.0f} deg\nExample {i+1}, correctness {row.correct_fraction:.0%}', fontsize=8)
            ax.set_axis_off()
    fig.suptitle('Container-ship examples at two relative-angle intervals\nFixed random selection within port/interval, independent of correctness', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(OUT/'observability/within_port_angle_examples.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
