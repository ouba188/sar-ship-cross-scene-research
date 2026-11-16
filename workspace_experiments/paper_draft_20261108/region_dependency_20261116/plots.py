"""Editable summary plots from paired analysis tables."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
plt.rcParams['svg.fonttype']='none'
def read(name):
    with (ROOT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

data=read('polar_paired_summary.csv')
classes=('bulk','general_cargo','chemical_product_tanker','container')
names=('Bulk','General cargo','Chemical/product tanker','Container')
conditions=[f'{p}_mean50_keep{k}' for p in ('VV','VH') for k in (6,5,3)]
labels=[f'{p}: {r}' for p in ('VV','VH') for r in ('Body','Axial ends','Lateral')]
lookup={(r['condition'],r['group_value']):r for r in data if r['arm']=='full' and r['group']=='class'}
fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
for ax,field,title in zip(axes,('fixed_margin_delta_mean','delta_oa_mean'),('Change in fixed-rival margin','Change in class recall (pp)')):
    values=np.array([[float(lookup[c,k][field]) for k in classes] for c in conditions])
    vmax=abs(values).max()
    im=ax.imshow(values,cmap='RdBu_r',vmin=-vmax,vmax=vmax,aspect='auto')
    ax.set_xticks(range(4),[names[i]+'\nn='+str(int(float(lookup[conditions[0],k]['n_mean']))) for i,k in enumerate(classes)],rotation=22,ha='right')
    ax.set_yticks(range(6),labels);ax.set_title(title,fontsize=12)
    for i in range(6):
        for j in range(4):ax.text(j,i,f'{values[i,j]:+.2f}',ha='center',va='center',color='white' if abs(values[i,j])>.55*vmax else 'black',fontsize=10)
    fig.colorbar(im,ax=ax,shrink=.85)
fig.suptitle('Same queries with both polarizations available; 50% mean-preserving regional attenuation\nFull model; OSS2 validation; three fixed training seeds',fontsize=12)
for ext in ('svg','png'):fig.savefig(ROOT/('paired_polarization_effects.'+ext),dpi=180)
plt.close(fig)

data=read('path_effects_summary.csv')
arms=('m1_feedback_off','m2_write_off','m2_uniform','m3_off')
lookup={r['removed_path']:r for r in data if r['group']=='overall'}
v=np.array([[float(lookup[a][f]) for f in ('delta_oa_mean','delta_ba_mean')] for a in arms])
fig,ax=plt.subplots(figsize=(7.2,4.4),layout='constrained')
im=ax.imshow(v,cmap='RdBu_r',vmin=-abs(v).max(),vmax=abs(v).max(),aspect='auto')
ax.set_xticks((0,1),('OA difference (pp)','BA difference (pp)'))
ax.set_yticks(range(4),('M1 feedback','M2 internal write-back','M2 response-guided weights','M3 reference write-back'))
for i in range(4):
    for j in range(2):ax.text(j,i,f'{v[i,j]:+.3f}',ha='center',va='center',color='white' if abs(v[i,j])>1.3 else 'black')
ax.set_title('Full minus fixed-weight path removal\nOSS2 validation; mean of three seeds',fontsize=12)
fig.colorbar(im,ax=ax)
for ext in ('svg','png'):fig.savefig(ROOT/('path_net_effects.'+ext),dpi=180)
plt.close(fig)
print('Wrote paired_polarization_effects and path_net_effects, PNG + editable SVG')
