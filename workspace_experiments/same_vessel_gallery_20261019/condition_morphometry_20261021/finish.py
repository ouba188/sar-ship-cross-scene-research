"""Summarize existing measurement caches; no image changes or classifier fitting."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import OUT

m=pd.read_csv(OUT/'measurements.csv')
p=pd.read_csv(OUT/'paired_changes.csv',dtype={'mmsi':str})
s=pd.read_csv(OUT/'change_summary.csv')
fits=pd.read_csv(OUT/'heldout_change_prediction.csv')
assert not m.duplicated(['obs_id','pol']).any()
assert len(m)==3640 and len(p)==1820
assert np.isfinite(m[['length_m','width_m','contrast','incidence']]).all().all()
assert (m.length_m>=m.width_m).all() and (m.width_m>0).all()
pred=pd.read_csv(OUT/'heldout_predictions.csv')
for r in fits.itertuples():
    f=pred[(pred.pol==r.pol)&(pred.metric==r.metric)]
    e0=np.abs(f[r.reference]-f.truth).mean();e1=np.abs(f[r.candidate]-f.truth).mean()
    np.testing.assert_allclose([e0,e1],[r.reference_mae,r.geometry_mae],rtol=1e-10)

rows=[]
for pol,f in p.groupby('pol'):
    for metric in ('target_mean','near_mean','near_sd','far_mean','contrast','target_saturated'):
        v=f['delta_'+metric].dropna().to_numpy()
        rows.append(dict(pol=pol,metric=metric,ships=len(v),median_abs_delta=float(np.median(abs(v))),median_delta=float(np.median(v))))
pd.DataFrame(rows).to_csv(OUT/'display_change_summary.csv',index=False)

fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
names=['length_m','width_m','r3_length_m','r3_width_m']
labels=['OBB length','OBB width','Response span: length','Response span: width']
for j,pol in enumerate(('vv','vh')):
    f=s[(s.pol==pol)&s.group.eq('all')].set_index('metric')
    v=f.loc[names,'median_absolute_ratio_change_pct'].to_numpy()
    axs[0].barh(np.arange(4)+(j-.5)*.34,v,height=.32,label=pol.upper(),color=('#147d92','#c16735')[j])
    for k,value in enumerate(v):axs[0].text(value+.4,k+(j-.5)*.34,f'{value:.1f}%',va='center',fontsize=9)
axs[0].set_yticks(range(4),labels);axs[0].invert_yaxis();axs[0].set_xlim(0,36)
axs[0].set_xlabel('Median max(A,B)/min(A,B) - 1 (%)');axs[0].legend()
axs[0].set_title('Within-vessel apparent size changes')
groups=[('vv','length_m','class_direction'),('vh','length_m','class_direction'),('vv','r3_length_m','class_direction'),('vh','r3_length_m','class_direction'),('vv','r3_width_m','class_incidence'),('vh','contrast','class_incidence')]
vals=[]
for pol,metric,arm in groups:
    row=fits[(fits.pol==pol)&(fits.metric==metric)&fits.reference.eq('class_metadata')&fits.candidate.eq(arm)].iloc[0]
    vals.append(row.mae_improvement_pct)
axs[1].barh(range(6),vals,color=['#147d92','#c16735']*3)
axs[1].set_yticks(range(6),['VV OBB length | direction','VH OBB length | direction','VV response length | direction','VH response length | direction','VV response width | incidence','VH contrast | incidence'])
axs[1].invert_yaxis();axs[1].set_xlabel('Held-out change prediction MAE reduction (%)')
axs[1].set_title('Added conditions vs class + timing/platform')
for i,v in enumerate(vals):axs[1].text(v+.1,i,f'{v:.2f}%',va='center',fontsize=9)
axs[1].set_xlim(0,max(vals)+2)
fig.suptitle('910 vessels; native PNG statistics, NOT calibrated backscatter or classification gains',fontsize=11)
fig.savefig(OUT/'summary_plot.png',dpi=150);plt.close(fig)
(OUT/'VERIFICATION.json').write_text(json.dumps(dict(passed=True,views=3640,paired_polarization_rows=1820,
    every_reported_regression_mae_recomputed=True,numerical_tests=3,original_images_modified=False),indent=2),encoding='utf-8')
print(pd.DataFrame(rows).to_string(index=False))
