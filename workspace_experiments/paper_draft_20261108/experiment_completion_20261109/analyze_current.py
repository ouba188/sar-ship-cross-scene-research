"""Complete paired reporting from existing predictions; no fitting or selection."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[1]
M12=WORK/'m1_guided_m2_20261106'
ESS=ROOT.parent/'essential_experiments_20261109'
SEEDS=(17,29,43)
NAMES=('Bulk carrier','General cargo','Chemical/product','Container')
ZH=('散货船','普通货船','化学品/成品油轮','集装箱船')


def save_csv(name,rows):
    with (ROOT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def metric(y,z):
    pred=z.argmax(-1)
    cm=np.zeros((4,4),int)
    np.add.at(cm,(y,pred),1)
    recall=np.diag(cm)/np.maximum(cm.sum(1),1)
    precision=np.diag(cm)/np.maximum(cm.sum(0),1)
    f1=2*recall*precision/np.maximum(recall+precision,1e-12)
    p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
    return dict(oa=100*float((pred==y).mean()),ba=100*float(recall[cm.sum(1)>0].mean()),
                macro_f1=100*float(f1[cm.sum(1)>0].mean()),
                nll=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-20)).mean()))


def main():
    ROOT.mkdir(exist_ok=True)
    rows=json.loads((WORK/'m1_coverage_20261101/manifest.json').read_text(encoding='utf-8'))
    summaries=[];classes=[];passes=[];intervals=[];confusions=[]
    for split in ('validation','evaluation'):
        scores={}
        for seed in SEEDS:
            paths={'full':M12/f'class_s{seed}/{split}.npz',
                   'mean_readout':M12/f'mean_s{seed}/{split}.npz',
                   'content_readout':M12/f'content_s{seed}/{split}.npz',
                   'no_m3_refit':ESS/f'no_m3_refit/class_s{seed}/{split}.npz'}
            for arm,path in paths.items():
                d=np.load(path)
                if not scores:
                    ids=d['ids'];y=np.array([rows[i]['label'] for i in ids])
                    ships=np.array([str(rows[i]['mmsi']) for i in ids])
                    # Use platform + absolute orbit, merging adjacent datatake/product slices.
                    orbit=np.array(['_'.join(rows[i]['pass_group'].split('_')[:2]) for i in ids])
                assert np.array_equal(ids,d['ids']),path
                scores[seed,arm]=d['score']
                summaries.append(dict(split=split,seed=seed,arm=arm,n=len(y),**metric(y,d['score'])))
        for arm in paths:
            stack=np.stack([scores[s,arm] for s in SEEDS])
            pred=stack.argmax(-1);good=pred==y[None]
            for c in range(4):
                mask=y==c
                vals=100*good[:,mask].mean(1)
                classes.append(dict(split=split,arm=arm,label=c,class_name=ZH[c],
                                    observations=int(mask.sum()),ships=len(set(ships[mask])),
                                    recall_mean=float(vals.mean()),recall_std=float(vals.std(ddof=1))))
                for k in range(4):
                    confusions.append(dict(split=split,arm=arm,true=c,predicted=k,
                                           count_mean=float((pred[:,mask]==k).sum(1).mean()),
                                           row_percent=float(100*(pred[:,mask]==k).mean())))
            for value in np.unique(orbit):
                mask=orbit==value
                stats=[metric(y[mask],scores[s,arm][mask]) for s in SEEDS]
                passes.append(dict(split=split,arm=arm,pass_group=value,n=int(mask.sum()),
                                   ships=len(set(ships[mask])),present_classes=','.join(map(str,np.unique(y[mask]))),
                                   oa=float(np.mean([r['oa'] for r in stats])),
                                   ba_present_classes=float(np.mean([r['ba'] for r in stats]))))
        # Paired, class-stratified ship-cluster bootstrap, conditional on the three stored fits.
        # Sample a ship with all its observations; the three training seeds are not extra ships.
        units=[]
        for c in range(4):
            members=[np.flatnonzero(ships==v) for v in sorted(set(ships[y==c]))]
            assert all(np.all(y[u]==c) for u in members),'Mixed-class MMSI requires a different resampling unit'
            units.append(members)
        for control in ('mean_readout','content_readout','no_m3_refit'):
            full=np.stack([scores[s,'full'].argmax(1)==y for s in SEEDS]).mean(0)
            base=np.stack([scores[s,control].argmax(1)==y for s in SEEDS]).mean(0)
            delta=full-base
            actual_oa=100*float(delta.mean())
            actual_ba=100*float(np.mean([delta[y==c].mean() for c in range(4)]))
            rng=np.random.default_rng(20261109);samples=[]
            for _ in range(3000):
                totals=[];counts=[]
                for members in units:
                    selected=rng.integers(0,len(members),len(members))
                    tot=sum(float(delta[members[i]].sum()) for i in selected)
                    count=sum(len(members[i]) for i in selected)
                    totals.append(tot);counts.append(count)
                samples.append([100*sum(totals)/sum(counts),100*np.mean(np.array(totals)/counts)])
            ci=np.quantile(samples,[.025,.975],axis=0)
            intervals.append(dict(split=split,control=control,oa_delta=actual_oa,oa_low=float(ci[0,0]),oa_high=float(ci[1,0]),
                                  ba_delta=actual_ba,ba_low=float(ci[0,1]),ba_high=float(ci[1,1]),
                                  units=len(set(ships)),replicates=3000,
                                  scope='paired class-stratified MMSI bootstrap; fixed three fits and observed passes'))
    for name,table in [('summary.csv',summaries),('per_class.csv',classes),('per_pass.csv',passes),
                       ('paired_intervals.csv',intervals),('confusions.csv',confusions)]:
        save_csv(name,table)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(12.8,4.1),layout='constrained')
    for ax,arm,title in zip(axes,('mean_readout','no_m3_refit','full'),('Mean readout','No RCEW / refit','Current full')):
        part=[r for r in confusions if r['split']=='evaluation' and r['arm']==arm]
        cm=np.array([r['row_percent'] for r in part]).reshape(4,4)
        im=ax.imshow(cm,vmin=0,vmax=100,cmap='Blues')
        for i in range(4):
            for j in range(4):ax.text(j,i,f'{cm[i,j]:.1f}',ha='center',va='center',color='white' if cm[i,j]>55 else '#222')
        ax.set_xticks(range(4),['Bulk','Cargo','Tanker','Container'],rotation=35,ha='right')
        ax.set_yticks(range(4),NAMES);ax.set_title(title);ax.set_xlabel('Predicted class')
    fig.colorbar(im,ax=axes,shrink=.8,label='Row-normalized %, mean of three seeds')
    for ext in ('png','pdf','svg'):fig.savefig(ROOT/f'confusion_evaluation.{ext}',dpi=180)
    plt.close(fig)
    lines=['# 当前模型补充定量分析','',
           '本轮复用预测，不新增训练、不改变模型、不选择检查点。所有数值为当前同一OSS2协议；不是多数据集主表。',
           '补充逐类召回率、宏F1、逐过境结果、混淆矩阵及配对船级bootstrap。','',
           '## 评价集逐类召回率','',
           '| 类别 | 样本 | 均值读出 | 移除RCEW配套训练 | 完整组合 |','|---|---:|---:|---:|---:|']
    for c in range(4):
        select={r['arm']:r for r in classes if r['split']=='evaluation' and r['label']==c}
        lines.append(f'| {ZH[c]} | {select["full"]["observations"]} | {select["mean_readout"]["recall_mean"]:.3f} | {select["no_m3_refit"]["recall_mean"]:.3f} | {select["full"]["recall_mean"]:.3f} |')
    lines+=['','## 配对区间','',
            '按类别分层抽样MMSI，保留同船全部观测，3000次bootstrap。每条观测先对三个固定训练重复取平均贡献；不把种子作为新样本。区间只描述当前评价船群的不确定性，不包含新过境、新划分和完整骨干独立重训的全部变异。','',
            '| 对照 | OA差及95%区间 | BA差及95%区间 |','|---|---|---|']
    for r in intervals:
        if r['split']=='evaluation':
            lines.append(f'| {r["control"]} | {r["oa_delta"]:+.3f} [{r["oa_low"]:+.3f}, {r["oa_high"]:+.3f}] | {r["ba_delta"]:+.3f} [{r["ba_low"]:+.3f}, {r["ba_high"]:+.3f}] |')
    lines+=['','逐过境BA只对该过境实际存在的类别平均，不能当作每景四类完全齐全。混淆矩阵为三次逐模型结果平均，不是logit集成模型。',
            '这些补充说明现有收益的类别与场景分布，不能替代新的公平方法对比和从共同起点训练的完整模块消融。']
    (ROOT/'CURRENT_RESULTS_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
