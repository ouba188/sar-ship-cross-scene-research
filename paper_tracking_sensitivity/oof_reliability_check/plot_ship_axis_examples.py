"""Label-independent per-port examples for checking the image axis proxy."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from extract_ship_axis_proxy import OUT, SOURCE


def main():
    pairs = pd.read_csv(OUT/'paired_axes.csv')
    views = pd.read_csv(OUT/'view_axes.csv', low_memory=False).set_index(['instance_id', 'polarization'])
    manifest = pd.read_csv(SOURCE/'manifest.csv', low_memory=False).set_index('instance_id')
    rng = np.random.default_rng(20261009)
    selected = [rng.choice(g.instance_id.to_numpy()) for _, g in pairs[pairs.accepted].groupby('port')]
    pd.DataFrame({'instance_id': selected}).to_csv(OUT/'visual_examples.csv', index=False)
    fig, axes = plt.subplots(5, 6, figsize=(15, 13))
    for i, instance in enumerate(selected):
        for j, pol in enumerate(['vv', 'vh']):
            ax = axes.ravel()[2*i+j]
            row = views.loc[instance, pol]
            with Image.open(manifest.loc[instance, 'image_relpath_'+pol]) as im:
                a = np.asarray(im.convert('L').resize((128, 128), Image.Resampling.BILINEAR))
            ax.imshow(a, cmap='gray', vmin=0, vmax=255)
            rad = np.deg2rad(row.angle_pixel_deg)
            ax.plot(row.cx+np.array([-23, 23])*np.cos(rad), row.cy+np.array([-23, 23])*np.sin(rad), c='#28d8b2', lw=1.2)
            ax.set_title(f'{row.port} {pol.upper()}\nExample {i+1:02d}, ratio={row.elongation:.1f}', fontsize=8)
            ax.set_axis_off()
    fig.suptitle('Fixed-seed accepted-axis samples, one instance per supported port\nBright-component proxies; no manual truth headings', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(OUT/'axis_examples.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    main()
