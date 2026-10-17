"""Show actual local readout weights, not an illustrative attention map."""
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import HERE,MANIFEST
from local_reader import regions,ARMS


def main():
    m=pd.read_csv(MANIFEST);d=pd.read_csv(HERE/'detections.csv').set_index('global_row')
    g=pd.read_csv(HERE/'geometry.csv').set_index('global_row')
    selected=pd.read_csv(HERE/'visual_selection.csv').global_row.to_numpy()[[0,2,4,5]]
    fig,axes=plt.subplots(4,4,figsize=(13,13),layout='constrained')
    for i,row in enumerate(selected):
        with Image.open(m.loc[row,'chip_path_vv']) as im:
            native=np.array(im.convert('L'))
        axes[i,0].imshow(native,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
        axes[i,0].set_title(f'{row} | {m.loc[row,"class_name"]}',fontsize=9)
        for j,arm in enumerate(('image_ring','ship','joint'),1):
            body,weights,pos,cov=regions(d.loc[row],g.loc[row].radar_axis_rad,g.loc[row].incidence_rad,arm=arm)
            actual=weights.mean(0)
            axes[i,j].imshow(actual,cmap='magma',vmin=0,vmax=.003,interpolation='nearest')
            axes[i,j].contour(body,levels=[.5],colors=['cyan'],linewidths=.6)
            axes[i,j].set_title(f'{arm} | fixed sum=1\nmean coverage={cov.mean():.2f}',fontsize=9)
        for ax in axes[i]:
            ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Actual 12-token sampling weights; cyan=detected body; not learned attention')
    fig.savefig(HERE/'sampling_comparison.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    main()
