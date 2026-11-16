"""Paired prediction analysis; no training or GPU imports."""
import argparse
import csv
import json
import math
import os
from itertools import combinations
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='2'
import numpy as np

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent.parent
SEEDS=(17,29,43)
ARMS=('full','r50','m1_feedback_off','m2_write_off','m2_uniform','m3_off')
REGIONS=('body','ends','side')
CLASSES=('bulk','general_cargo','chemical_product_tanker','container')
PAIRS=((0,1),(0,2),(1,2))


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(name,rows):
    with (ROOT/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def rival(z,y):
    return np.where(np.eye(z.shape[1],dtype=bool)[y],-np.inf,z).argmax(1)


def margin(z,y,c):
    i=np.arange(len(y))
    return z[i,y]-z[i,c]


def game(v):
    phi=np.zeros((len(v),3));interaction=np.zeros((len(v),3))
    for i in range(3):
        for s in range(8):
            if s&(1<<i):continue
            n=s.bit_count();w=math.factorial(n)*math.factorial(2-n)/6
            phi[:,i]+=w*(v[:,s|(1<<i)]-v[:,s])
    for q,(i,j) in enumerate(PAIRS):
        k=next(k for k in range(3) if k not in (i,j))
        for s in (0,1<<k):
            interaction[:,q]+=.5*(v[:,s|(1<<i)|(1<<j)]-v[:,s|(1<<i)]-v[:,s|(1<<j)]+v[:,s])
    if not np.allclose(phi.sum(1),v[:,7]-v[:,0],atol=1e-6):
        raise AssertionError('Shapley efficiency failed')
    return phi,interaction


def self_test():
    v=np.array([[5+sum((2,-1,3)[i] for i in range(3) if s&(1<<i)) for s in range(8)]],float)
    p,j=game(v)
    assert np.allclose(p,[[2,-1,3]]) and np.allclose(j,0)
    pair=np.array([[6. if s&3==3 else 0. for s in range(8)]])
    p,j=game(pair)
    assert np.allclose(p,[[3,3,0]]) and np.allclose(j,[[6,0,0]])
    triple=np.array([[6. if s==7 else 0. for s in range(8)]])
    p,j=game(triple)
    assert np.allclose(p,[[2,2,2]]) and np.allclose(j,[[3,3,3]])
    print('PASS additive, pair interaction, triple interaction, efficiency')


def metrics(y,z,base):
    old=base.argmax(1);new=z.argmax(1);c=rival(base,y)
    delta=margin(z,y,c)-margin(base,y,c)
    strongest=margin(z,y,rival(z,y))-margin(base,y,c)
    present=np.unique(y)
    return dict(n=len(y),oa=100*float(np.mean(new==y)),
        ba=100*float(np.mean([np.mean(new[y==k]==k) for k in present])),
        delta_oa=100*float(np.mean((new==y).astype(float)-(old==y))),
        delta_ba=100*float(np.mean([np.mean(new[y==k]==k)-np.mean(old[y==k]==k) for k in present])),
        rescued=int(((old!=y)&(new==y)).sum()),harmed=int(((old==y)&(new!=y)).sum()),
        fixed_margin_delta=float(delta.mean()),abs_fixed_margin_delta=float(abs(delta).mean()),
        strongest_margin_delta=float(strongest.mean()),present_classes=len(present))


def summarize(rows,keys,fields):
    grouped={}
    for row in rows:grouped.setdefault(tuple(row[k] for k in keys),[]).append(row)
    out=[]
    for key,items in grouped.items():
        row=dict(zip(keys,key));row['seeds']=len(items)
        for field in fields:
            x=np.asarray([s[field] for s in items],float)
            row[field+'_mean']=float(x.mean());row[field+'_sd']=float(x.std(ddof=1)) if len(x)>1 else 0.
        out.append(row)
    return out


def cluster_interval(y,ships,delta,repeats=1000):
    # Average the three fixed seeds first; resample physical ships, not seed-events.
    names,inverse=np.unique(ships,return_inverse=True)
    count=np.zeros((len(names),4));total=np.zeros_like(count)
    d=delta.mean(0)
    for k in range(4):
        live=y==k
        count[:,k]=np.bincount(inverse[live],minlength=len(names))
        total[:,k]=np.bincount(inverse[live],weights=d[live],minlength=len(names))
    rng=np.random.default_rng(16016)
    draws=rng.multinomial(len(names),np.full(len(names),1/len(names)),size=repeats)
    n=draws@count;v=draws@total
    oa=v.sum(1)/n.sum(1)*100
    use=(n>0).all(1);ba=(v[use]/n[use]).mean(1)*100
    lo,hi=np.quantile(oa,[.025,.975]);bl,bh=np.quantile(ba,[.025,.975])
    return dict(ships=len(names),n=len(y),delta_oa=float(d.mean()*100),oa_low=float(lo),oa_high=float(hi),
        delta_ba=float(np.mean([d[y==k].mean() for k in range(4)])*100),ba_low=float(bl),ba_high=float(bh),bootstrap=repeats)


def main():
    self_test()
    rows=read(WORK/'m1_coverage_20261101/manifest.json')
    protocol=read(ROOT/'protocol.json');conditions={c['name']:c for c in protocol['conditions']}
    raw=np.load(ROOT/'predictions/full_s17/clean.npz');ids=raw['ids'];y=raw['y']
    ships=np.asarray([rows[i]['mmsi'] for i in ids]);passes=np.asarray([rows[i]['pass_group'] for i in ids])
    av=np.load(ROOT.parent/'recent_baselines_20261111/native_cache/available.npy')[ids]>0
    masks=np.load(ROOT/'native_regions.npy',mmap_mode='r');area=masks.sum((-2,-1));del masks
    data={};expected=0
    for arm in ARMS:
        names=list(conditions) if arm in ('full','r50') else ['clean']+[p+'_mean50_keep'+str(k) for p in ('VV','VH') for k in (3,5,6)]
        for seed in SEEDS:
            for name in names:
                path=ROOT/'predictions'/f'{arm}_s{seed}'/(name+'.npz')
                with np.load(path) as x:
                    assert np.array_equal(x['ids'],ids) and np.array_equal(x['y'],y),path
                    data[arm,seed,name]=x['score'].astype(float)
                expected+=1
    assert expected==234

    def eligible(name):
        c=conditions[name]
        if c['kind']=='clean':return np.ones(len(y),bool)
        p=c['p'];live=av[:,p].copy()
        if c['kind']=='coalition':
            changed=[i for i in range(3) if not c['keep']&(1<<i)]
            live&=(area[:,p,changed]>0).all(1)
        elif c['kind']=='permute':live&=area[:,p,c['r']]>1
        return live

    metrics_rows=[];attributions=[];interactions=[];path_rows=[];dids=[];intervals=[]
    groups=[('overall','all',np.ones(len(y),bool))]
    groups += [('class',CLASSES[k],y==k) for k in range(4)]
    groups += [('pass',p,passes==p) for p in np.unique(passes)]
    for (arm,seed,name),z in data.items():
        base=data[arm,seed,'clean'];live=eligible(name)
        for level,label,g in groups:
            use=live&g
            if not use.any():continue
            metrics_rows.append(dict(arm=arm,seed=seed,condition=name,group=level,group_value=label,
                **metrics(y[use],z[use],base[use])))
    save('metrics_seed.csv',metrics_rows)
    sm=summarize(metrics_rows,['arm','condition','group','group_value'],
        ['n','oa','ba','delta_oa','delta_ba','rescued','harmed','fixed_margin_delta','abs_fixed_margin_delta','strongest_margin_delta'])
    save('metrics_summary.csv',sm)

    paired=[]
    for arm in ('full','r50'):
        for seed in SEEDS:
            for vv in [name for name in conditions if name.startswith('VV')]:
                vh='VH'+vv[2:];common=eligible(vv)&eligible(vh)
                for name in (vv,vh):
                    for level,label,g in groups:
                        use=common&g
                        if use.any():paired.append(dict(arm=arm,seed=seed,condition=name,group=level,group_value=label,
                            **metrics(y[use],data[arm,seed,name][use],data[arm,seed,'clean'][use])))
    save('polar_paired_seed.csv',paired)
    save('polar_paired_summary.csv',summarize(paired,['arm','condition','group','group_value'],
        ['n','delta_oa','delta_ba','rescued','harmed','fixed_margin_delta','abs_fixed_margin_delta']))

    sample_phi=[]
    for arm in ('full','r50'):
        for seed in SEEDS:
            base=data[arm,seed,'clean'];c=rival(base,y)
            for p,pol in enumerate(('VV','VH')):
                v=np.stack([margin(data[arm,seed,'clean' if k==7 else f'{pol}_mean50_keep{k}'],y,c) for k in range(8)],1)
                phi,j=game(v);denom=abs(phi).sum(1);live=av[:,p]&(area[:,p]>0).all(1)
                ratio=np.divide(phi,denom[:,None],out=np.zeros_like(phi),where=denom[:,None]>1e-12)
                for k in np.flatnonzero(live):
                    sample_phi.append(dict(arm=arm,seed=seed,index=int(ids[k]),polarization=pol,
                        phi_body=phi[k,0],phi_ends=phi[k,1],phi_side=phi[k,2],
                        interaction_body_ends=j[k,0],interaction_body_side=j[k,1],interaction_ends_side=j[k,2]))
                for level,label,g in groups:
                    use=live&g
                    if not use.any():continue
                    for i,region in enumerate(REGIONS):
                        attributions.append(dict(arm=arm,seed=seed,polarization=pol,region=region,group=level,group_value=label,n=int(use.sum()),
                            phi=float(phi[use,i].mean()),abs_phi=float(abs(phi[use,i]).mean()),positive_fraction=float((phi[use,i]>0).mean()),
                            signed_ratio=float(ratio[use,i].mean()),abs_ratio=float(abs(ratio[use,i]).mean()),
                            area_mean=float(area[use,p,i].mean()),zero_total=int((denom[use]<=1e-12).sum())))
                    for q,(i,k) in enumerate(PAIRS):
                        interactions.append(dict(arm=arm,seed=seed,polarization=pol,pair=REGIONS[i]+'_'+REGIONS[k],group=level,group_value=label,n=int(use.sum()),
                            interaction=float(j[use,q].mean()),abs_interaction=float(abs(j[use,q]).mean()),positive_fraction=float((j[use,q]>0).mean())))
    save('shapley_samples.csv',sample_phi);save('shapley_seed.csv',attributions);save('interaction_seed.csv',interactions)
    save('shapley_summary.csv',summarize(attributions,['arm','polarization','region','group','group_value'],['n','phi','abs_phi','signed_ratio','abs_ratio','positive_fraction','area_mean','zero_total']))
    save('interaction_summary.csv',summarize(interactions,['arm','polarization','pair','group','group_value'],['n','interaction','abs_interaction','positive_fraction']))

    for arm in ARMS[2:]:
        deltas=[]
        for seed in SEEDS:
            full=data['full',seed,'clean'];off=data[arm,seed,'clean'];c=rival(full,y)
            for level,label,g in groups:
                path_rows.append(dict(removed_path=arm,seed=seed,group=level,group_value=label,
                    **metrics(y[g],full[g],off[g])))
            deltas.append((full.argmax(1)==y).astype(float)-(off.argmax(1)==y))
            for pol in ('VV','VH'):
                for k in (3,5,6):
                    name=f'{pol}_mean50_keep{k}';live=eligible(name)
                    d=(margin(data['full',seed,name],y,c)-margin(full,y,c))-(margin(data[arm,seed,name],y,c)-margin(off,y,c))
                    for level,label,g in groups:
                        use=live&g
                        if use.any():dids.append(dict(removed_path=arm,seed=seed,condition=name,group=level,group_value=label,n=int(use.sum()),
                            did=float(d[use].mean()),abs_did=float(abs(d[use]).mean())))
        intervals.append(dict(contrast='full_minus_'+arm,**cluster_interval(y,ships,np.array(deltas))))
    save('path_effects_seed.csv',path_rows)
    save('path_effects_summary.csv',summarize(path_rows,['removed_path','group','group_value'],['n','delta_oa','delta_ba','rescued','harmed']))
    save('path_did_seed.csv',dids)
    save('path_did_summary.csv',summarize(dids,['removed_path','condition','group','group_value'],['n','did','abs_did']))
    for arm in ('full','r50'):
        for name,cond in conditions.items():
            if name=='clean' or (cond['kind']=='coalition' and cond['keep'] not in (3,5,6)):continue
            live=eligible(name)
            d=np.array([(data[arm,s,name].argmax(1)==y).astype(float)-(data[arm,s,'clean'].argmax(1)==y) for s in SEEDS])[:,live]
            intervals.append(dict(contrast=arm+'_'+name+'_minus_clean',**cluster_interval(y[live],ships[live],d)))
    save('cluster_intervals.csv',intervals)
    replay={}
    for arm in ('full','r50'):
        for seed in SEEDS:replay[f'{arm}_s{seed}']=read(ROOT/'predictions'/f'{arm}_s{seed}'/'replay.json')
    receipt=dict(prediction_sets=expected,queries=len(y),ships=len(np.unique(ships)),passes=len(np.unique(passes)),
        available=av.sum(0).tolist(),class_counts=np.bincount(y,minlength=4).tolist(),
        replay=replay,bootstrap='MMSI clusters, mean over three fixed seeds first; conditional on these seeds and four observed passes',
        self_tests='additive/pair/triple games passed; efficiency checked on every actual sample',
        analysis='complete; fixed-weight validation diagnosis, not new held-out test or retrained ablation')
    (ROOT/'analysis_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    for kind in ('full','r50'):
        path=ROOT/f'status_{kind}.json'
        status=read(path);status['state']='predictions_and_analysis_complete'
        path.write_text(json.dumps(status,indent=2),encoding='utf-8')
    clean=[r for r in sm if r['condition']=='clean' and r['group']=='overall']
    lines=['# 固定权重诊断：自动汇总','',
        'OSS2验证集228条、223艘船、4个过境；三个既有种子；234组预测。有效VV212、VH222。',
        '下表为各路径原图指标；不是重新训练的模块删除消融。','',
        '|路径|OA均值±标准差|BA均值±标准差|','|---|---:|---:|']
    for r in clean:lines.append(f"|{r['arm']}|{r['oa_mean']:.3f}±{r['oa_sd']:.3f}|{r['ba_mean']:.3f}±{r['ba_sd']:.3f}|")
    lines+=['','## 路径净贡献','正数代表Full优于关闭该路径。区间按船重采样，先平均三个固定种子，不是跨训练总体置信区间。','',
        '|对照|ΔOA|95%区间|ΔBA|95%区间|','|---|---:|---|---:|---|']
    for r in intervals[:4]:lines.append(f"|{r['contrast']}|{r['delta_oa']:+.3f}|[{r['oa_low']:+.3f},{r['oa_high']:+.3f}]|{r['delta_ba']:+.3f}|[{r['ba_low']:+.3f},{r['ba_high']:+.3f}]|")
    lines+=['','## 文件说明',
        '- metrics：同一有效子集上，操作后减原图；逐类BA等于该类召回率，逐过境BA仅对当地出现的类别平均。',
        '- shapley：以区域结构50%衰减为基准，原图固定竞争类；不是准确率贡献百分比。',
        '- interaction：三区域二元Shapley交互；正负是相对于本次输入基准的判别相互作用。',
        '- path_did：统一使用Full原图竞争类，计算输入操作效应Full减路径关闭；与原图路径净收益分开。',
        '- cluster_intervals：条件性船级区间，不能代表新港口或新过境不确定性。',
        '- 普通R50与Full训练历史不同，只作系统诊断；不得跨分类头直接比较原始logit幅度。',
        '- R50历史TF32缓存与本轮FP32在seed17有1条原图预测不同；本轮所有差分使用新算的FP32原图，不混用历史缓存。']
    (ROOT/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    self_test() if args.self_test else main()
