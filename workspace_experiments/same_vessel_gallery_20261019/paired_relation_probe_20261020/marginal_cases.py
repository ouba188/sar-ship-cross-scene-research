"""Positive-control case review, kept separate from primary-arm panels."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import importlib.util
from run import ROOT, load_data
spec=importlib.util.spec_from_file_location('paired_analysis',ROOT/'analyze.py')
analysis=importlib.util.module_from_spec(spec); spec.loader.exec_module(analysis)
read_scores=analysis.read_scores

f,_=load_data(); seed=1
rows,scores,base=read_scores(f,'marginal',seed)
y=f.iloc[rows].class_id.to_numpy(); good=scores.argmax(1)==y; old=base.argmax(1)==y
rival=np.argmax(np.where(np.arange(8)[None,:]==y[:,None],-np.inf,base),1)
before=base[np.arange(len(y)),y]-base[np.arange(len(y)),rival]
after=scores[np.arange(len(y)),y]-scores[np.arange(len(y)),rival]
records=[]
for label,mask in (('rescue',good&~old),('harm',~good&old)):
    ix=np.flatnonzero(mask); j=int(ix[np.argmax(np.abs((after-before)[ix]))]); r=f.iloc[rows[j]]
    fig,axs=plt.subplots(1,2,figsize=(9,4.5))
    fig.suptitle(f'marginal {label} | obs{int(r.obs_id)} | {r.class_name}\nseed1, margin {before[j]:+.3f} -> {after[j]:+.3f}')
    for p,pol in enumerate(('vv','vh')):
        image=np.array(Image.open(r['image_'+pol]).convert('L'))
        axs[p].imshow(image,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
        axs[p].set_title(f'{pol.upper()} native {image.shape[1]}x{image.shape[0]}')
    fig.tight_layout(rect=(0,0,1,.91)); fig.savefig(ROOT/f'marginal_{label}_{int(r.obs_id)}.png',dpi=125); plt.close(fig)
    records.append(dict(case=label,obs_id=int(r.obs_id),seed=seed,fold=int(r.fold),class_id=int(y[j]),
        base_prediction=int(base[j].argmax()),prediction=int(scores[j].argmax()),
        base_margin=float(before[j]),margin=float(after[j])))
pd.DataFrame(records).to_csv(ROOT/'marginal_cases.csv',index=False)
