"""Plot measured full-model paths and non-cherry-picked support examples."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.dpi':180,'axes.titleweight':'bold'})
datasets=['self24ports','f0921','oss2']
names=['23 ports','F_0921','OpenSARShip 2.0']
m=pd.read_csv(OUT/'metrics_summary.csv',dtype={'mode':str})
m=m[m.layer=='full'].set_index(['dataset','mode'])
fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
x=np.arange(3);width=.19
for j,(mode,label,color) in enumerate([('100','Prior change only','#407cbd'),
    ('001','Extra relation only','#b46834'),('101','Prior + extra relation','#579963'),
    ('111','Complete dual model','#343434')]):
    y=np.array([m.loc[(ds,mode),'versus_single_ba'] for ds in datasets])
    bars=ax.bar(x+(j-1.5)*width,y,width,label=label,color=color)
    ax.bar_label(bars,labels=[f'{v:+.2f}' for v in y],padding=3,fontsize=9)
ax.axhline(0,color='#999999',linewidth=.7)
ax.set_xticks(x,names);ax.set_ylabel('BA gain over matched single route (pp)')
ax.set_ylim(-1.2,5.5);ax.legend(ncol=2,frameon=False,loc='upper left')
ax.set_title('Source prior and added support relations contribute jointly',pad=13)
fig.savefig(OUT/'FIGURE_PATHS.png');fig.savefig(OUT/'FIGURE_PATHS.pdf');plt.close(fig)

ps=pd.read_csv(OUT/'pixel_probe/summary.csv').set_index('condition')
conditions=['target_shuffle','outer_shuffle','target_flat','sham_flat','target_display','outer_display']
labels=['Target-neighborhood shuffle','Outer-region shuffle','Target-neighborhood flatten',
        'Equal-area corner flatten','Target-neighborhood display','Outer-region display']
values=ps.loc[conditions,'delta_ba'].to_numpy()
fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
bars=ax.barh(labels,values,color=['#407cbd','#999999','#407cbd','#999999','#579963','#b46834'])
ax.bar_label(bars,labels=[f'{v:+.2f}' for v in values],padding=4)
ax.invert_yaxis();ax.axvline(0,color='#444444',linewidth=.7);ax.set_xlim(-12,2)
ax.set_xlabel('Full-model BA change (pp); fixed original supports and coefficients')
ax.set_title('Actual fine-tuned R18: query-image evidence reaches the final output',pad=12)
fig.suptitle('OSS2 diagnostic sample: 128 images, 126 ships, 8 classes, 5 support draws',fontsize=10)
fig.savefig(OUT/'FIGURE_PIXEL_EVIDENCE.png');fig.savefig(OUT/'FIGURE_PIXEL_EVIDENCE.pdf');plt.close(fig)

cases=pd.read_csv(OUT/'cases.csv');cases=cases[cases.dataset=='oss2'].set_index('kind')
terms=pd.read_csv(OUT/'support_case_terms.csv');terms=terms[terms.dataset=='oss2']
frame=pd.read_csv('E:/Hermes/paper_tracking/sensitivity/opensarship2_transfer_20261007/target_manifest.csv',
                  usecols=['sample_id','chip_path_vh','class_id']).set_index('sample_id')
classes=['Bulk','Fishing','General cargo','Chemical tanker','Container','Crude tanker','Tug','Offshore']
fig,axes=plt.subplots(2,4,figsize=(11,6),layout='constrained')
for row,kind in enumerate(['rescue','harm']):
    case=cases.loc[kind]
    group=terms[terms.kind==kind]
    selected=pd.concat([group[group.extra_margin_term>0].nlargest(2,'extra_margin_term'),
                        group[group.extra_margin_term<0].nsmallest(1,'extra_margin_term')])
    ids=[case.sample_id]+selected.support_sample_id.tolist()
    titles=[f'{kind.upper()}: query, true {classes[int(case.true_class)]}\n'
            f'{classes[int(case.single_full_class)]} -> {classes[int(case.dual_full_class)]}']
    titles += [f'Support: {classes[int(a.support_class)]}\nAdded-relation margin {a.extra_margin_term:+.3f}'
               for a in selected.itertuples()]
    for col,(sid,title) in enumerate(zip(ids,titles)):
        with Image.open(frame.loc[sid,'chip_path_vh']) as im:
            pixels=np.asarray(im.convert('L'))
        axes[row,col].imshow(pixels,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
        axes[row,col].set_title(title,fontsize=10)
        axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
fig.suptitle('Median-margin success and failure cases, draw 0; signed terms are not deletion effects',fontsize=11)
fig.savefig(OUT/'FIGURE_CASES.png');fig.savefig(OUT/'FIGURE_CASES.pdf');plt.close(fig)
print('Three PNG/PDF figures written.')
