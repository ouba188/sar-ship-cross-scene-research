"""Fixed-coordinate interpretation, not semantic discovery or model selection."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.metrics import roc_auc_score

from run_protected_drift_activation import OUT as PARENT, CANDIDATE, SOURCE


OUT = PARENT.parent/'latent_clue_cards_20261008'
REFERENCE = 'Antwerp-Bruges'
PAIR = (0, 4)
PROXY_NAMES = {
    'mean': '整体平均亮度', 'rim_mean': '外围平均亮度',
    'center_excess': '中心与外围亮度差', 'contrast': '亮度分位差',
    'bright_fraction': '相对亮区域占比', 'elongation': '亮散射分布细长比',
    'top_energy': '最亮10%像素能量占比', 'offset': '亮散射重心偏离',
}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def image_proxies(vv, vh):
    a = (np.asarray(vv, dtype=float)+np.asarray(vh, dtype=float))/510.
    side = a.shape[0]
    yy, xx = np.mgrid[:side, :side]/(side-1)-.5
    rim = (np.abs(xx)>.375) | (np.abs(yy)>.375)
    center = (np.abs(xx)<.25) & (np.abs(yy)<.25)
    p05, p50, p95, p99 = np.quantile(a, [.05,.5,.95,.99])
    mass = np.maximum(a-p50, 0)
    total = mass.sum()
    elongation = offset = np.nan
    if total > 1e-10:
        mx, my = (mass*xx).sum()/total, (mass*yy).sum()/total
        dx, dy = xx-mx, yy-my
        cov = np.array([[(mass*dx*dx).sum(), (mass*dx*dy).sum()],
                        [(mass*dx*dy).sum(), (mass*dy*dy).sum()]])/total
        eig = np.linalg.eigvalsh(cov)
        elongation = np.sqrt((eig[1]+1e-8)/(eig[0]+1e-8))
        offset = np.hypot(mx, my)
    top = np.sort(a.ravel())[-int(np.ceil(a.size*.1)):].sum()
    return dict(mean=a.mean(), rim_mean=a[rim].mean(),
        center_excess=a[center].mean()-a[rim].mean(), contrast=p95-p05,
        bright_fraction=np.mean(a>p50+.5*(p99-p50)), elongation=elongation,
        top_energy=top/max(a.sum(), 1e-10), offset=offset)


def load_proxies(frame):
    path = OUT/'image_proxies.csv'
    if path.exists():
        return pd.read_csv(path).set_index('manifest_row')
    rng = np.random.default_rng(20261008)
    ids = []
    for _, cell in frame.groupby(['port','class_id'], sort=True):
        ids.extend(rng.choice(cell.index.to_numpy(), min(32, len(cell)), replace=False))
    rows = []
    for step, idx in enumerate(sorted(ids)):
        row = frame.loc[idx]
        arrays = []
        for pol in ('vv','vh'):
            with Image.open(row['image_relpath_'+pol]) as im:
                arrays.append(np.asarray(im.convert('L').resize((128,128), Image.Resampling.BILINEAR)))
        rows.append(dict(manifest_row=int(idx), port=row.port, class_id=int(row.class_id),
                         **image_proxies(*arrays)))
        if (step+1)%500 == 0:
            print(f'PROXIES {step+1}/{len(ids)}', flush=True)
    answer = pd.DataFrame(rows).set_index('manifest_row')
    answer.to_csv(path, encoding='utf-8-sig')
    return answer


def directions(folder, reference_raw_drift):
    with (folder/'geometry.pkl').open('rb') as f:
        g = pickle.load(f)
    with np.load(CANDIDATE/'ports'/folder.name/'frozen.npz') as data:
        j = np.flatnonzero((data['pairs'][:,0]==PAIR[0]) & (data['pairs'][:,1]==PAIR[1]))[0]
        stable = data['conditional_penalty_basis'][:,j].astype(float)
    drift = g['qd'][:,0].copy()
    raw_drift = drift/g['feature_scale']
    cosine = raw_drift@reference_raw_drift/(np.linalg.norm(raw_drift)*np.linalg.norm(reference_raw_drift))
    sign = 1. if cosine>=0 else -1.
    return g, np.column_stack((stable, sign*drift)), sign, abs(cosine)


def scores(bank, g, basis, drift_sign):
    r = g['stable_rank']
    stable = np.asarray(bank[:,192:192+r])*g['stable_scale']+g['stable_center']
    return np.column_stack((stable@(g['qs'].T@basis[:,0]), drift_sign*bank[:,192+r]))


def balanced_class_stats(values, labels, ports, label):
    cells = [np.flatnonzero((labels==label)&(ports==p)) for p in np.unique(ports)]
    cells = [ids for ids in cells if len(ids)>=5]
    ids = np.concatenate(cells)
    weights = np.concatenate([np.full(len(r), 1./(len(cells)*len(r))) for r in cells])
    mean = np.sum(values[ids]*weights)
    variance = np.sum(weights*(values[ids]-mean)**2)
    between = np.var([values[r].mean() for r in cells])
    return ids, weights, mean, variance, between


def correlation_rows(values, proxy, fold, partition):
    records = []
    for (port, label), cell in proxy.groupby(['port','class_id'], sort=True):
        if len(cell)<12:
            continue
        for j, name in enumerate(('S1','D1')):
            x = values.loc[cell.index, name].to_numpy()
            for attribute in PROXY_NAMES:
                v = cell[attribute].to_numpy()
                use = np.isfinite(x)&np.isfinite(v)
                rho = np.nan
                if use.sum()>=12 and np.ptp(x[use])>1e-10 and np.ptp(v[use])>1e-10:
                    rho = float(spearmanr(x[use],v[use]).statistic)
                records.append(dict(fold=fold, partition=partition, port=port,
                    class_id=int(label), coordinate=name, attribute=attribute,
                    n=int(use.sum()), rho=rho))
    return records


def check_formulas():
    # Known images detect accidental use of class labels or port IDs as proxies.
    dark = np.zeros((128,128), dtype=np.uint8)
    p = image_proxies(dark, dark)
    assert p['mean']==0 and p['bright_fraction']==0 and np.isnan(p['elongation'])
    patch = dark.copy()
    patch[60:68,32:96] = 200
    p = image_proxies(patch, patch)
    assert p['elongation']>5 and p['offset']<1e-8 and p['center_excess']>0
    transposed = image_proxies(patch.T, patch.T)
    np.testing.assert_allclose(p['elongation'], transposed['elongation'])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(date='2026-10-08', pair=list(PAIR), fixed_before_readout=True,
        coordinates={'S1':'Frozen conditional-penalty direction for class0 vs class4',
                     'D1':'First protected drift SVD direction; sign aligned to reference raw covector'},
        reference=REFERENCE, image_sample='Seed20261008; max32 samples per class-port cell; no outcome selection',
        image_measurement='Mean of resized VV/VH intensities; proxy statistics, no true ship masks',
        source_summary='Equal class-port cells for image association, source-fit descriptive',
        heldout='Each target port uses its original source-fitted directions; query draw0; >=20 samples each class for pair tests',
        all_class_concepts=True, main_model_unchanged=True, fits=0, independent_confirmation=False,
        readout_removal='Fixed historical source base prior only, one unit direction to source feature mean; not full adapted model',
        no_causal_claim=True)
    protocol_path = OUT/'PROTOCOL.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding='utf-8'))!=protocol:
        raise ValueError('Use a new folder when changing the interpretation protocol')
    write_json(protocol_path, protocol)
    check_formulas()
    frame = pd.read_csv(SOURCE/'manifest.csv', low_memory=False, dtype={'mmsi':str})
    proxy = load_proxies(frame)
    with (PARENT/'ports'/REFERENCE/'geometry.pkl').open('rb') as f:
        ref = pickle.load(f)
    reference_raw_drift = ref['qd'][:,0]/ref['feature_scale']
    records, correlations, reliance = [], [], []
    validation_error = 0.
    for folder in sorted((PARENT/'ports').iterdir()):
        if not (folder/'geometry.pkl').exists():
            continue
        port = folder.name
        g, basis, sign, cosine = directions(folder, reference_raw_drift)
        sids, tids = np.load(folder/'source_rows.npy'), np.load(folder/'target_rows.npy')
        sb = np.load(folder/'source_bank.npy', mmap_mode='r')
        tb = np.load(folder/'target_bank.npy', mmap_mode='r')
        zs, zt = scores(sb,g,basis,sign), scores(tb,g,basis,sign)
        sy, ty = frame.class_id.to_numpy()[sids], frame.class_id.to_numpy()[tids]
        sp = frame.port.to_numpy()[sids]
        with np.load(CANDIDATE/'ports'/port/'frozen.npz') as data:
            ev = data['eval_mask'][0]
            j = np.flatnonzero((data['pairs'][:,0]==0)&(data['pairs'][:,1]==4))[0]
            error = float(np.max(np.abs(zt[:,0]-data['conditional_penalty_target_scores'][:,j])))
            validation_error = max(validation_error,error)
            if error>.002:
                raise ValueError(f'Coordinate reconstruction mismatch {port}: {error}')
        model = g['model']
        raw_gradient = (np.asarray(model['pca'].components_.T,dtype=float)/
                        np.asarray(model['scale'],dtype=float))@np.asarray(model['base'].coef_.T,dtype=float)
        covector = g['feature_scale'][:,None]*raw_gradient
        coefficients = basis.T@covector
        prior = np.asarray(tb[:,:128])@model['base'].coef_.T+model['base'].intercept_
        # Independently perturb cached PCA features to verify one-direction removal.
        delta_raw = zt[:3,0,None]*basis[:,0]*g['feature_scale']
        delta_xs = (delta_raw@model['pca'].components_.T)/model['scale']
        direct = (np.asarray(tb[:3,:128])-delta_xs)@model['base'].coef_.T+model['base'].intercept_
        np.testing.assert_allclose(direct,prior[:3]-zt[:3,0,None]*coefficients[0],atol=1e-9)
        for partition, ids, z in [('source',sids,zs),('target',tids[ev],zt[ev])]:
            sub = proxy.loc[proxy.index.intersection(ids)]
            values = pd.DataFrame(z,index=ids,columns=['S1','D1'])
            correlations.extend(correlation_rows(values,sub,port,partition))
            if port==REFERENCE:
                values.join(frame[['port','class_id']]).to_csv(OUT/f'reference_{partition}_coordinates.csv',index_label='manifest_row',encoding='utf-8-sig')
        for k,name in enumerate(('S1','D1')):
            a = balanced_class_stats(zs[:,k],sy,sp,0)
            b = balanced_class_stats(zs[:,k],sy,sp,4)
            orient = 1. if a[2]>=b[2] else -1.
            threshold = .5*(a[2]+b[2])
            sigma = np.sqrt(.5*(a[3]+b[3]))
            source_auc = roc_auc_score(np.r_[np.ones(len(a[0])),np.zeros(len(b[0]))],
                orient*np.r_[zs[a[0],k],zs[b[0],k]],sample_weight=np.r_[a[1],b[1]])
            source_ba = .5*(np.sum(a[1]*(orient*(zs[a[0],k]-threshold)>0))+
                              np.sum(b[1]*(orient*(zs[b[0],k]-threshold)<=0)))
            stats = dict(port=port, coordinate=name, source_auc=float(source_auc),
                source_ba=float(source_ba), source_gap=float(orient*(a[2]-b[2])/sigma),
                source_mean_drift_ratio=float(np.sqrt(.5*(a[4]+b[4]))/sigma),
                drift_reference_cosine=float(cosine), orientation=orient,
                source_prior_margin_coefficient=float(coefficients[k,0]-coefficients[k,4]))
            use = ev&np.isin(ty,PAIR)
            ya, yb = ty[use]==0, ty[use]==4
            if min(ya.sum(),yb.sum())<20:
                continue
            z = zt[use,k]
            auc = roc_auc_score(ya,orient*z)
            ba = .5*(np.mean(orient*(z[ya]-threshold)>0)+np.mean(orient*(z[yb]-threshold)<=0))
            shift = .5*(wasserstein_distance(zs[a[0],k],z[ya],u_weights=a[1])+
                         wasserstein_distance(zs[b[0],k],z[yb],u_weights=b[1]))/sigma
            records.append(dict(**stats,target_auc=float(auc), target_ba=float(ba),
                normalized_w1=float(shift),n0=int(ya.sum()),n4=int(yb.sum())))
            # Clamp to the source feature mean, not to a favorable class mean.
            removed = prior[use]-zt[use,k,None]*coefficients[k]
            margin = prior[use,0]-prior[use,4]
            after = removed[:,0]-removed[:,4]
            ba_before = .5*(np.mean(margin[ya]>0)+np.mean(margin[yb]<=0))
            ba_after = .5*(np.mean(after[ya]>0)+np.mean(after[yb]<=0))
            rms = np.sqrt(.5*(np.sum(a[1]*zs[a[0],k]**2)+np.sum(b[1]*zs[b[0],k]**2)))
            reliance.append(dict(port=port, coordinate=name, base_prior_pair_ba=ba_before,
                removed_pair_ba=ba_after, delta_ba=ba_after-ba_before,
                prediction_change=float(np.mean((margin>0)!=(after>0))),
                source_pair_margin_contribution_rms=float(rms*abs(coefficients[k,0]-coefficients[k,4]))))
        print(f'FOLD {port}: source={len(sids)} target={len(tids)}',flush=True)
        del sb,tb,g,model
    metrics = pd.DataFrame(records)
    corr = pd.DataFrame(correlations)
    usage = pd.DataFrame(reliance)
    metrics.to_csv(OUT/'held_port_pair_metrics.csv',index=False,encoding='utf-8-sig')
    corr.to_csv(OUT/'class_port_proxy_correlations.csv',index=False,encoding='utf-8-sig')
    usage.to_csv(OUT/'source_prior_direction_removal.csv',index=False,encoding='utf-8-sig')
    # Reference-source illustrations and each fold's held target are separate summaries.
    source_corr = corr[(corr.fold==REFERENCE)&(corr.partition=='source')]
    target_corr = corr[corr.partition=='target']
    summary_rows = []
    for name in ('S1','D1'):
        for attr in PROXY_NAMES:
            s = source_corr[(source_corr.coordinate==name)&(source_corr.attribute==attr)].rho.dropna()
            t = target_corr[(target_corr.coordinate==name)&(target_corr.attribute==attr)].rho.dropna()
            summary_rows.append(dict(coordinate=name,attribute=attr,source_mean=s.mean(),
                source_cells=len(s),target_mean=t.mean(),target_cells=len(t),
                target_positive_cells=int((t>0).sum()),target_negative_cells=int((t<0).sum())))
    cs = pd.DataFrame(summary_rows)
    cs.to_csv(OUT/'proxy_summary.csv',index=False,encoding='utf-8-sig')
    # Within-fold source-to-target association avoids treating rotating fold axes as identical.
    source_sameclass = corr[corr.partition=='source'].groupby(['fold','class_id','coordinate','attribute']).rho.mean().rename('source_rho')
    matched = target_corr.join(source_sameclass,on=['fold','class_id','coordinate','attribute'])
    matched['same_sign'] = matched.rho*matched.source_rho>0
    matched.to_csv(OUT/'source_target_association.csv',index=False,encoding='utf-8-sig')
    summary = dict(protocol=protocol, image_sample_rows=len(proxy), image_cells=len(proxy.groupby(['port','class_id'])),
        reconstruction_max_error=validation_error, pair_ports=int(metrics.port.nunique()),
        metrics=metrics.groupby('coordinate')[['source_auc','source_ba','target_auc','target_ba','normalized_w1']].mean().reset_index().to_dict('records'),
        removal=usage.groupby('coordinate')[['base_prior_pair_ba','removed_pair_ba','delta_ba','prediction_change','source_pair_margin_contribution_rms']].mean().reset_index().to_dict('records'),
        proxy_summary=summary_rows, drift_axis_cosine_min=float(metrics.drift_reference_cosine.min()),
        drift_axis_cosine_median=float(metrics.groupby('port').drift_reference_cosine.first().median()))
    write_json(OUT/'SUMMARY.json',summary)
    plot(metrics,cs,usage)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('protocol','proxy_summary')},ensure_ascii=False,indent=2),flush=True)


def plot(metrics,cs,usage):
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False,
                         'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes = plt.subplots(1,2,figsize=(13,5.8))
    for ax,name in zip(axes,('S1','D1')):
        rows = cs[cs.coordinate==name].set_index('attribute').loc[list(PROXY_NAMES)]
        y = np.arange(len(rows))
        ax.barh(y-.17,rows.source_mean,height=.3,color='#087F8C',label='固定划分的源港：均值')
        ax.barh(y+.17,rows.target_mean,height=.3,color='#BA4D62',label='逐港留出：均值')
        ax.axvline(0,color='#666666',lw=.8)
        ax.set_yticks(y,list(PROXY_NAMES.values()),fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(-.7,.7)
        ax.set_xlabel('类×港口内 Spearman 相关的等权均值')
        ax.set_title(f'{name}：'+('候选稳定判别线索' if name=='S1' else '第一漂移线索'))
        ax.legend(loc='lower right',fontsize=8)
    fig.suptitle('线索与图像属性的关联：相关不等于物理含义',fontsize=16)
    fig.text(.5,.015,'固定每类每港最多32张；代理来自VV/VH平均图像，不是真实船体测量。不同留港划分方向会变化；类别差异另存CSV。',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.94),w_pad=3)
    for ext in ('png','pdf'):
        fig.savefig(OUT/('clue_proxy_profiles.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(13,5.8))
    ports = sorted(metrics.port.unique())
    for name,color in [('S1','#087F8C'),('D1','#BA4D62')]:
        rows = metrics[metrics.coordinate==name].set_index('port').loc[ports]
        axes[0].plot(np.arange(len(ports)),rows.target_ba*100,'o-',color=color,label=name)
        ur = usage[usage.coordinate==name].set_index('port').loc[ports]
        axes[1].plot(np.arange(len(ports)),ur.delta_ba*100,'o-',color=color,label=name)
    for ax in axes:
        ax.set_xticks(np.arange(len(ports)),ports,rotation=45,ha='right',fontsize=8)
        ax.grid(axis='y',alpha=.2)
        ax.legend()
    axes[0].axhline(50,color='#777777',linestyle='--',lw=1)
    axes[0].set_ylabel('留出港二分类 BA (%)')
    axes[0].set_title('单条线索：固定源域阈值，不用目标标签拟合')
    axes[1].axhline(0,color='#777777',linestyle='--',lw=1)
    axes[1].set_ylabel('移除线索后的 BA 变化（百分点）')
    axes[1].set_title('仅基础源分类头：负值表示移除有损失')
    fig.suptitle('散货船 vs 集装箱船：迁移表现与读出依赖',fontsize=16)
    fig.text(.5,.015,'仅符合每类查询样本≥20的港口；draw0；不是八分类完整模型性能，也不是因果或独立新数据验证。',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.94),w_pad=3)
    for ext in ('png','pdf'):
        fig.savefig(OUT/('clue_transfer_and_usage.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    main()
