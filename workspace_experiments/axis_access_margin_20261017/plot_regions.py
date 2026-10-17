"""Display fixed generic regions on actual native VV/VH chips."""
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from universal_features import EDGES, ANGLES

HERE = Path(__file__).resolve().parent


def main():
    old = HERE.parent/'oss1_joint_geometry_20261017'
    geometry = pd.read_csv(old/'geometry.csv').set_index('global_row')
    detection = pd.read_csv(old/'detections.csv').set_index('global_row')
    fig,axes = plt.subplots(4,2,figsize=(7.6,12))
    colors = ['#55d6bc','#ffe178','#e9a5c5','#8abdf2']
    for row,index in enumerate([88,834,1044,786]):
        g,d = geometry.loc[index],detection.loc[index]
        for col,pol in enumerate(['vv','vh']):
            im = tifffile.imread(g['chip_path_'+pol])
            h,w = im.shape
            center = np.array([(w-1)/2,(h-1)/2])
            ax = axes[row,col]
            ax.imshow(im,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            for radius,color in zip(EDGES*min(h,w),colors):
                ax.add_patch(Circle(center,radius,fill=False,edgecolor=color,lw=.7))
            for angle in ANGLES:
                delta = np.array([np.cos(angle),np.sin(angle)])
                for sign in [-1,1]:
                    ends = center+sign*np.outer([EDGES[0],EDGES[-1]],min(h,w)*delta)
                    ax.plot(ends[:,0],ends[:,1],color='#c0c0c0',lw=.45,alpha=.7)
            ax.set_title(f'row {index} | {g.class_name} | {pol.upper()}\n'
                         f'{w}x{h}; accepted ship axis: {bool(d.axis_valid)}',fontsize=9)
            ax.set_xlim(-.5,w-.5);ax.set_ylim(h-.5,-.5)
            ax.tick_params(labelsize=7)
    fig.suptitle('Axis-independent local readout on native chips\n'
                 'Green: central image region; rings: three scales; lines: paired sector axes\n'
                 'Crop-centered regions, NOT hull segmentation or learned attention',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.93))
    (HERE/'figures').mkdir(exist_ok=True)
    fig.savefig(HERE/'figures/universal_native_regions.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    main()
