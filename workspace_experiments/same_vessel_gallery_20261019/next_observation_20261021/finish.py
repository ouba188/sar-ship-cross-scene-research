"""Replay split/score checks and inspect fixed native-image cases."""
import json
import math
from pathlib import Path
from urllib.parse import unquote,urlparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from run import OUT,ROOT,GEOM,ARMS,fit_relation,predict,load


def main():
    f,_=load();p=pd.read_csv(OUT/'predictions.csv',dtype={'mmsi':str});s=pd.read_csv(OUT/'summary.csv')
    manifests=json.loads((OUT/'fit_manifest.json').read_text(encoding='utf-8'))
    lookup=f.set_index(['obs_id','pol'])
    assert np.isfinite(p[['pred_log','pred_m','abs_log_error','abs_m_error']]).all().all()
    assert not p.duplicated(['variant','pol','metric','mmsi','arm']).any()
    fits={r['fit_id']:r for r in manifests}
    for fit in manifests:
        src=lookup.loc[[(oid,fit['pol']) for oid in fit['source_obs_ids']]]
        assert src.split_fold.ne(fit['excluded_fold']).all()
        assert src.timestamp.lt(pd.Timestamp(fit['query_time'])).all()
        assert src.overpass.ne(fit['query_overpass']).all()
        assert src.groupby('mmsi').size().min()>=2
        if fit['fit_id'].startswith('omit_flagged_ship'):assert src.mmsi.ne('533150044').all()
    replay=[]
    for row in p.itertuples():
        q=lookup.loc[(row.obs_id,row.pol)]
        hist=lookup.loc[[(int(i),row.pol) for i in row.history_ids.split()]]
        assert hist.mmsi.eq(row.mmsi).all() and hist.timestamp.lt(q.timestamp).all()
        true_last=f[f.mmsi.eq(row.mmsi)&f.pol.eq(row.pol)].sort_values(['timestamp','obs_id']).iloc[-1].obs_id
        assert true_last==row.obs_id
        if row.arm=='history_mean':value=np.log(hist[row.metric]).mean()
        elif row.arm=='last_observation':value=np.log(hist[row.metric].iloc[-1])
        elif row.arm.startswith('own_'):
            fixed=[5.,1.,1.] if row.arm.endswith('fixedscale') else None
            beta=fit_relation(hist,row.metric,GEOM,1,fixed_scale=fixed)
            value=predict(hist,q,row.metric,GEOM,beta)
        else:
            arm='transfer_geometry' if row.arm.endswith('lastref') else row.arm
            columns=ARMS[arm]
            fit=fits[row.fit_id]
            beta=np.array([fit['coefficients'][arm][c] for c in columns])
            value=predict(hist,q,row.metric,columns,beta,last=row.arm.endswith('lastref'))
        replay.append(abs(value-row.pred_log))
    assert max(replay)<1e-10
    np.testing.assert_allclose(abs(p.pred_log-p.actual_log),p.abs_log_error,atol=1e-12)
    # Predictions must not read the held-out target statistic.
    example=f[f.mmsi.eq('249838000')&f.pol.eq('vv')].sort_values('timestamp')
    hist,q=example.iloc[:-1],example.iloc[-1].copy()
    beta=fit_relation(hist,'width_m',GEOM,1,fixed_scale=[5.,1.,1.])
    first=predict(hist,q,'width_m',GEOM,beta);q['width_m']=1e9
    assert first==predict(hist,q,'width_m',GEOM,beta)
    # A duplicated history contributes the same total ship weight and same estimate.
    double=pd.concat([hist,hist],ignore_index=True)
    np.testing.assert_allclose(beta,fit_relation(double,'width_m',GEOM,1,fixed_scale=[5.,1.,1.]),atol=1e-12)
    for key,g in p.groupby(['variant','pol','metric']):
        wide=g.pivot(index='mmsi',columns='arm',values='abs_log_error')
        for row in s[(s.variant.eq(key[0]))&s.pol.eq(key[1])&s.metric.eq(key[2])].itertuples():
            np.testing.assert_allclose((wide[row.baseline]-wide[row.arm]).mean(),row.mean_gain,atol=1e-12)
    checks=dict(manifests_checked=len(manifests),predictions_replayed=len(p),max_replay_error=max(replay),
        held_fold_excluded=True,source_strictly_earlier=True,query_pass_excluded=True,final_observation_fixed=True,
        target_statistic_not_used=True,ship_weight_duplicate_invariance=True,summary_replayed=True)
    (OUT/'VERIFICATION.json').write_text(json.dumps(checks,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(checks,indent=2),flush=True)
    cases=pd.read_csv(ROOT/'multitemporal_20261021/cases.csv',dtype={'mmsi':str})
    cp=p[p.variant.eq('all')&p.mmsi.isin(cases.mmsi)]
    cp.to_csv(OUT/'fixed_case_predictions.csv',index=False,lineterminator='\n')
    data=json.loads((ROOT/'gallery_data.js').read_text(encoding='utf-8').split('=',1)[1].strip().removesuffix(';'))['observations']
    fig,axes=plt.subplots(4,4,figsize=(17,17),layout='constrained')
    for r,ship in enumerate(cases.mmsi):
        seq=f[f.mmsi.eq(ship)&f.pol.eq('vv')].sort_values(['timestamp','obs_id'])
        ids=[int(seq.iloc[-2].obs_id),int(seq.iloc[-1].obs_id)]
        for phase,oid in enumerate(ids):
            for j,pol in enumerate(('vv','vh')):
                ax=axes[r,2*phase+j];o=data[str(oid)];v=o['images'][pol];pri=v['priors']
                path=unquote(urlparse(v['url']).path).lstrip('/')
                ax.imshow(np.asarray(Image.open(path).convert('L')),cmap='gray',vmin=0,vmax=255,interpolation='nearest',extent=(0,v['w'],v['h'],0))
                ax.add_patch(Polygon(pri['detections'][pri['target_index']]['corners'],fill=False,edgecolor='lime',lw=.8))
                axis=pri['ship_axis'];c=np.array(axis['center']);u=np.array([math.cos(axis['angle']),math.sin(axis['angle'])]);line=np.array([c-axis['length_px']*.5*u,c+axis['length_px']*.5*u]);ax.plot(line[:,0],line[:,1],color='yellow',lw=.8)
                a=pri['radar_angle'];ax.arrow(*pri['radar_center'],18*math.cos(a),18*math.sin(a),color='cyan',width=.4,head_width=3,length_includes_head=True)
                m=lookup.loc[(oid,pol)];text=f'{ship} | {"previous" if phase==0 else "held-out"} {pol.upper()} obs{oid}\n{o["time"][:10]} {o["port"]}\ninc {m.incidence:.1f} rel {m.phi_deg%180:.1f}; OBB L {m.length_m:.0f}m'
                if phase:
                    z=cp[cp.mmsi.eq(ship)&cp.pol.eq(pol)&cp.metric.eq('r3_length_m')].set_index('arm')
                    if len(z):text+=f'\nresponse L: true {z.actual_m.iloc[0]:.0f}, mean {z.loc["history_mean","pred_m"]:.0f}, transfer {z.loc["transfer_geometry","pred_m"]:.0f}m'
                ax.set_title(text,fontsize=8);ax.set_xlim(0,v['w']);ax.set_ylim(v['h'],0)
    fig.suptitle('Fixed cases: last historical image and held-out next image; predictor used ALL earlier eligible observations\nNative PNG pixels; green OBB / yellow undirected ship axis / cyan target-to-satellite',fontsize=12)
    fig.savefig(OUT/'fixed_cases_native.png',dpi=125);plt.close(fig)
    sel=s[s.variant.eq('all')&s.baseline.eq('history_mean')&s.arm.isin(['own_geometry_fixedscale','transfer_geometry','transfer_direction'])]
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    metrics=['length_m','width_m','r3_length_m','r3_width_m'];names=['OBB length','OBB width','Response length','Response width']
    for ax,pol in zip(axs,['vv','vh']):
        for offset,(arm,color) in enumerate([('own_geometry_fixedscale','#767676'),('transfer_direction','#17877e'),('transfer_geometry','#b24773')]):
            vals=sel[sel.pol.eq(pol)&sel.arm.eq(arm)].set_index('metric').loc[metrics,'relative_mae_reduction_pct']
            ax.barh(np.arange(4)+(offset-1)*.23,vals,height=.22,color=color,label=arm)
        ax.set_yticks(range(4),names);ax.axvline(0,color='black',lw=.8);ax.set_xlabel('Log-MAE reduction vs history mean (%)');ax.set_title(pol.upper());ax.legend(fontsize=7)
    fig.savefig(OUT/'prediction_gain.png',dpi=140);plt.close(fig)


if __name__=='__main__':main()
