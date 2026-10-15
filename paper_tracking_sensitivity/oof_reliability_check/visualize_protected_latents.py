"""Show extracted latent coordinates and raw examples without claiming semantics."""

import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name] = '2'

import json
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from run_protected_drift_activation import OUT, CANDIDATE, SOURCE


DEST = OUT/'latent_interpretation'
PORT = 'Antwerp-Bruges'
CLASS_NAMES = {0:'散货船',4:'集装箱船'}


def save(fig, name):
    fig.savefig(DEST/(name+'.png'),dpi=300,bbox_inches='tight',facecolor='white')
    fig.savefig(DEST/(name+'.pdf'),bbox_inches='tight',facecolor='white')
    plt.close(fig)


def gallery(chosen, scores, title, name):
    fig, axes = plt.subplots(2,4,figsize=(12,6.9))
    for col, (_,row) in enumerate(chosen.iterrows()):
        for ir, pol in enumerate(('vv','vh')):
            path = row['image_relpath_'+pol]
            with Image.open(path) as im:
                arr = np.asarray(im)
            axes[ir,col].imshow(arr,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            axes[ir,col].set_xticks([])
            axes[ir,col].set_yticks([])
            axes[ir,col].grid(False)
            if ir == 0:
                axes[ir,col].set_title(f'{CLASS_NAMES[int(row.class_id)]} | {row.port}\n'
                    f'{row.example_group}，坐标 {scores.loc[row.name]:+.2f}',fontsize=10)
            if col == 0:
                axes[ir,col].set_ylabel(pol.upper(),fontsize=12)
    fig.suptitle(title,fontsize=16,y=.99)
    fig.subplots_adjust(top=.84,bottom=.10,hspace=.12,wspace=.08)
    fig.text(.5,.025,'原始VV/VH裁片，固定0–255显示；按预定坐标/类内分位点选例，非物理因素归因或注意力图。',
             ha='center',fontsize=10,color='#555555')
    save(fig,name)


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':11,
        'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,
        'axes.grid':True,'grid.alpha':.18,'axes.titleweight':'bold','pdf.fonttype':42})
    folder = OUT/'ports'/PORT
    with (folder/'geometry.pkl').open('rb') as f:
        g = pickle.load(f)
    source_bank = np.load(folder/'source_bank.npy',mmap_mode='r')
    target_bank = np.load(folder/'target_bank.npy',mmap_mode='r')
    ids, target_ids = np.load(folder/'source_rows.npy'), np.load(folder/'target_rows.npy')
    cols = ['instance_id','port','class_name','class_id','mmsi','image_relpath_vv','image_relpath_vh']
    frame = pd.read_csv(SOURCE/'manifest.csv',usecols=cols,keep_default_na=False)
    sf = frame.iloc[ids].copy()
    r = g['stable_rank']
    stable_norm = np.asarray(source_bank[:,192:192+r])
    stable_raw = stable_norm*g['stable_scale']+g['stable_center']
    drift = np.asarray(source_bank[:,192+r:])
    with np.load(CANDIDATE/'ports'/PORT/'frozen.npz') as saved:
        pairs = saved['pairs']
        j = np.flatnonzero((pairs[:,0] == 0) & (pairs[:,1] == 4))[0]
        direction = saved['conditional_penalty_basis'][:,j]
    stable_score = stable_raw@(g['qs'].T@direction)
    drift_score = drift[:,0]
    use = sf.class_id.isin((0,4)).to_numpy()
    shown = {}
    for key,value in (('stable',stable_score),('drift',drift_score)):
        shown[key] = (value-value[use].mean())/value[use].std()
        sf[key] = shown[key]
    cells = sf[use].groupby(['port','class_id']).agg(n=('instance_id','size'),
        stable=('stable','mean'),drift=('drift','mean')).reset_index()
    counts = cells.pivot(index='port',columns='class_id',values='n')
    common_ports = counts.index[(counts[0].fillna(0)>=5)&(counts[4].fillna(0)>=5)].tolist()
    cells.to_csv(DEST/'source_class_port_coordinates.csv',index=False,encoding='utf-8-sig')
    geometry = pd.read_csv(OUT/'geometry_summary.csv')
    overlap = geometry.nuisance_energy_in_stable_fraction
    captured = geometry.selected_energy_fraction
    energy = np.array([overlap.mean(),((1-overlap)*captured).mean(),((1-overlap)*(1-captured)).mean()])*100
    fig,axes = plt.subplots(3,1,figsize=(14,10.5),gridspec_kw={'height_ratios':[1,1,.48]})
    colors = {0:'#007C91',4:'#B04458'}
    for ax,key,title in zip(axes[:2],('stable','drift'),
        ('A  候选稳定信息：散货船–集装箱船的源拟合判别方向',
         'B  可分离漂移信息：同类跨港均值变化的第一主方向 D1')):
        for label in (0,4):
            rows = cells[(cells.class_id == label)&cells.port.isin(common_ports)].set_index('port').loc[common_ports]
            ax.scatter(np.arange(len(common_ports))+(-.09 if label == 0 else .09),rows[key],
                c=colors[label],marker='o' if label == 0 else '^',s=48,label=CLASS_NAMES[label],zorder=3)
            ax.axhline(rows[key].mean(),color=colors[label],lw=1,linestyle='--',alpha=.65)
        ax.set_title(title,loc='left',fontsize=13,pad=12)
        ax.set_ylabel('投影坐标（展示标准化）')
        ax.set_xticks(np.arange(len(common_ports)),common_ports,rotation=28,ha='right',fontsize=9)
        ax.legend(loc='upper right',ncol=2,frameon=True,framealpha=.85,fontsize=10)
    labels = ['候选空间内仍存在的漂移','选出的可分离漂移','其他未处理漂移']
    barcolors = ['#A878AD','#007C91','#B5BDC5']
    start = 0.
    for value,color,label in zip(energy,barcolors,labels):
        axes[2].barh(0,value,left=start,color=color,height=.5)
        axes[2].text(start+value/2,0,f'{label}\n{value:.2f}%',ha='center',va='center',
                     color='white' if color != '#B5BDC5' else '#243238',fontsize=10)
        start += value
    axes[2].set_title('C  源域同类跨港均值漂移能量的分解（23个划分等权平均）',loc='left',fontsize=13,pad=12)
    axes[2].set_xlim(0,100)
    axes[2].set_ylim(-.5,.5)
    axes[2].set_yticks([])
    axes[2].set_xlabel('占源域同类跨港均值漂移能量的比例 (%)；不是信息量或准确率贡献')
    fig.suptitle('提取出的稳定候选与漂移隐信息：真实数据展示',fontsize=18,y=.99)
    fig.subplots_adjust(top=.92,bottom=.08,hspace=.74,left=.08,right=.98)
    fig.text(.5,.015,'A/B固定使用Antwerp-Bruges留港划分的源数据，点为类–港均值；仅展示两类均≥5条的港口。各轴分别标准化，非距离比较或泛化验证。',
             ha='center',fontsize=9,color='#555555')
    save(fig,'latent_information_overview')

    used = set()
    examples = []
    for label in (0,4):
        subset = sf[(sf.port == 'Jebel Ali')&(sf.class_id == label)]
        for quantile in (.35,.65):
            target = subset.stable.quantile(quantile)
            candidates = subset.iloc[np.argsort(np.abs(subset.stable.to_numpy()-target))]
            candidates = candidates[~candidates.mmsi.isin(used)]
            chosen = candidates.iloc[0].copy()
            used.add(chosen.mmsi)
            chosen['example_group'] = f'类内{int(quantile*100)}%分位附近'
            examples.append(chosen)
    stable_examples = pd.DataFrame(examples)
    gallery(stable_examples,sf.stable,'同港不同船型：稳定判别方向的真实裁片示例','stable_coordinate_examples')
    bulk = cells[(cells.class_id == 0)&(cells.n>=5)].sort_values('drift')
    selected_ports = list(bulk.port.iloc[:2])+list(bulk.port.iloc[-2:])
    examples, used = [], set()
    for k,port in enumerate(selected_ports):
        subset = sf[(sf.port == port)&(sf.class_id == 0)]
        candidates = subset.iloc[np.argsort(np.abs(subset.drift.to_numpy()-subset.drift.mean()))]
        candidates = candidates[~candidates.mmsi.isin(used)]
        chosen = candidates.iloc[0].copy()
        used.add(chosen.mmsi)
        chosen['example_group'] = '低坐标港均值附近' if k < 2 else '高坐标港均值附近'
        examples.append(chosen)
    drift_examples = pd.DataFrame(examples)
    gallery(drift_examples,sf.drift,'同为散货船、不同源港：漂移方向 D1 的真实裁片示例','drift_coordinate_examples')
    pd.concat((stable_examples.assign(selection='stable'),drift_examples.assign(selection='drift'))).to_csv(
        DEST/'example_selection.csv',index_label='manifest_row',encoding='utf-8-sig')
    np.savez_compressed(DEST/'latent_vectors_AntwerpBruges.npz', qs=g['qs'],qd=g['qd'],
        source_rows=ids,target_rows=target_ids,feature_center=g['feature_center'],feature_scale=g['feature_scale'],
        stable_center=g['stable_center'],stable_scale=g['stable_scale'],
        source_stable_coordinates=stable_raw,source_stable_normalized=stable_norm,source_drift_coordinates=drift,
        target_stable_coordinates=target_bank[:,192:192+r]*g['stable_scale']+g['stable_center'],
        target_drift_coordinates=target_bank[:,192+r:],illustrated_pair=np.array([0,4]),illustrated_pair_direction=direction)
    record = dict(date='2026-10-08',outer_port=PORT,source_rows=len(ids),target_rows=len(target_ids),
        stable_rank=r,drift_rank=g['drift_rank'],pair=[0,4],source_only_illustration=True,
        displayed_ports=common_ports,energy_percent=energy.tolist(),
        stable_examples='Fixed Jebel Ali classes0/4, nearest withinclass35/65percent quantiles, distinct MMSI',
        drift_examples='Class0 two lowest/highest D1 source-port means with>=5rows; nearest mean samples, distinct MMSI',
        image_display='Unmodified VV/VH PNG pixels; fixed0..255 display',
        caveat='No class or physical semantic interpretation follows from a single coordinate or representative images',
        fits=0,new_query_selection=0,main_model_unchanged=True)
    (DEST/'VISUALIZATION_PROTOCOL.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
