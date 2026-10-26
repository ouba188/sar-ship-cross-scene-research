"""Display original pixels and area-occupancy overlap at three stages."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from experiment import HERE, PILOT, old


def main():
    table, arrays = old.load()
    snapshots = {int(x['obs_id']): x for x in json.loads(old.SNAPSHOTS.read_text(encoding='utf-8'))}
    fig, axes = plt.subplots(4, 4, figsize=(12, 12), constrained_layout=True)
    for row, obs in enumerate((11062, 31089)):
        idx = int(np.flatnonzero(table.obs_id.eq(obs))[0])
        im, masks, valid, geom = old.native(table.iloc[idx], snapshots, arrays['available'][idx])
        for p, pol in enumerate(('VV','VH')):
            rr = row*2+p
            axes[rr, 0].imshow(im[p,0], cmap='gray', vmin=0, vmax=1)
            axes[rr, 0].set_title(f'obs {obs} / {pol} / native')
            for j, size in enumerate((40,20,10), 1):
                m = F.interpolate(torch.from_numpy(masks[p:p+1]),(size,size),mode='area')[0].numpy()
                # RGB identities: body red, end green, side blue; mixtures stay visible.
                rgba = np.concatenate((m.transpose(1,2,0), np.clip(m.sum(0)[...,None],0,1)*.8),axis=-1)
                axes[rr,j].imshow(im[p,0],cmap='gray',vmin=0,vmax=1)
                axes[rr,j].imshow(rgba,extent=(0,160,160,0),interpolation='nearest')
                axes[rr,j].set_title(f'layer{j}: {size}x{size}; body {m[0].sum():.2f} cells')
    for a in axes.flat:
        a.axis('off')
    fig.suptitle('Area occupancy: red = body, green = ends, blue = sides\nNot receptive-field segmentation; native input scale unchanged')
    fig.savefig(HERE/'resolution_examples.png',dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    main()
