"""Plot the whole observed path from saved numeric evidence, with no new inference."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'m3_explained_20261029'))
import trace_relations as tr
import numpy as np
from matplotlib.patches import Rectangle


def draw():
    r=json.loads((ROOT/'REPLAY_RECEIPT.json').read_text(encoding='utf-8'))
    n=np.load(ROOT/'intermediate.npz')
    s=r['selected']; qi,ri=s['query_slot'],s['reference_slot']
    refs=r['references']; k=next(i for i,v in enumerate(refs) if v['id']=='31512')
    records,_,_,_=tr.exp.records()
    query=next(v for v in records if str(v['id'])=='12457')
    view=tr.exp.data.prepare(query)[1]
    fig=tr.plt.figure(figsize=(25,5.2),layout='constrained')
    grid=fig.add_gridspec(1,6,width_ratios=[1,1,1,1.15,1,1.15])
    axes=[fig.add_subplot(grid[0,i]) for i in range(6)]
    a=axes[0]; a.imshow(view['pixels'],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
    masks=tr.exp.tokens.slots(view['geometry'],view['footprint'])
    for slots,color in zip(tr.exp.tokens.REGIONS,(tr.GREEN,tr.BLUE,tr.ORANGE)):
        a.contour(masks[slots].sum(0),levels=[.05],colors=[color],linewidths=.6)
    a.contour(masks[qi],levels=[.05],colors=['#B2188B'],linewidths=1.2)
    a.set_xticks([]);a.set_yticks([]);a.set_title('① 查询与槽位\n12457 / VH / 散货船',fontsize=12)
    a.set_xlabel('实际输入160画布\n框内8、侧向8、轴端4',fontsize=10)
    a=axes[1];x=np.arange(2);e=np.array(s['similarities']);w=np.array(s['weights'])
    a.bar(x-.22,e,.22,label='相似度E',color=tr.BLUE)
    a.bar(x,w,.22,label='权重α',color=tr.ORANGE)
    a.bar(x+.22,e*w,.22,label='αE',color=tr.GREEN)
    a.set_xticks(x,['深层','中层']);a.set_ylim(0,1.05);a.legend(fontsize=8)
    a.set_title('② 两层空间比较\nS = ΣαE = '+f'{s["similarity"]:.4f}',fontsize=12)
    a.set_xlabel(f'查询槽位{qi} → 31512槽位{ri}\n无频域比较',fontsize=10)
    a=axes[2];t=n['VH_transport_0'][0,k];a.imshow(t,cmap='Blues',vmin=0,vmax=t.max())
    a.add_patch(Rectangle((ri-.5,qi-.5),1,1,fill=False,ec='#B2188B',lw=1.8))
    a.set_xticks(range(8));a.set_yticks(range(8));a.set_xlabel('参考框内槽位');a.set_ylabel('查询框内槽位')
    a.set_title('③ 区域内软对应\n选中T = '+f'{s["transport"]:.5f}',fontsize=12)
    a=axes[3];weights=n['VH_reference_weights'][0,:,qi]
    a.barh(range(16),weights,color=[tr.GREEN if j==k else tr.BLUE for j in range(16)])
    a.set_yticks(range(16),[v['id'] for v in refs],fontsize=8)
    a.set_title('④ 聚合16个参考\n31512权重β = '+f'{s["reference_weight"]:.4f}',fontsize=12)
    a.set_xlabel('T对齐32维V，再按β混合',fontsize=10)
    a=axes[4];delta=np.linalg.norm(n['update'][0,1],axis=0)
    im=a.imshow(delta,cmap='magma',vmin=0,vmax=delta.max());fig.colorbar(im,ax=a,fraction=.035)
    a.set_title('⑤ 实际VH回写\nFout = Fq + ΔF',fontsize=12)
    a.set_xlabel('10×10网格，通道L2范数\n更新强度不是有益程度',fontsize=10)
    a=axes[5];values=[v['margin'] for v in r['conditions'][:2]]
    a.barh([0,1],values,color=[tr.GRAY,tr.BLUE]);a.axvline(0,color='black',lw=.7)
    a.set_yticks([0,1],['关闭M3','空间M3']);a.set_xlim(-1.1,.48)
    for j,v in enumerate(values):a.text(v+.02,j,f'{v:+.5f}',va='center',fontsize=10)
    a.set_title('⑥ 真实分类翻转\n化学品/成品油轮 → 散货',fontsize=12)
    a.set_xlabel('正确类分差；正值才判对',fontsize=10)
    fig.suptitle('空间M3真实救回：12457 / Jebel Ali / S1A_058688 / 历史N_level_pattern seed17',fontsize=18)
    for ext in ('png','pdf'):
        fig.savefig(ROOT/f'09_overview.{ext}',dpi=170,bbox_inches='tight',facecolor='white')
    tr.plt.close(fig)


if __name__=='__main__':draw()
