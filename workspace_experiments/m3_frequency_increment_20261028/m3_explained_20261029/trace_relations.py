"""Two fixed historical cases: branch evidence, image links, and value-path removal."""
from pathlib import Path
import sys
import json
import html
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from draw import plt, ROOT, BLUE, GREEN, GRAY, ORANGE, RED

BASE = ROOT.parent / 'joint_m3_readout'
sys.path.insert(0, str(BASE))
import review

j, exp = review.joint, review.exp
OUT = ROOT / 'relation_evidence'
REGIONS = ('框内', '侧向', '轴端')
POL = ('VV', 'VH')
BRANCH = ('深层', '细层', '4px', '8px', '16px')
CLASSES = ('散货', '渔船', '普通货船', '化学品/成品油轮', '集装箱', '原油轮', '拖船', '海工')


def export(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=150, bbox_inches='tight')
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    return name+'.png'


def arr(x):
    return x.detach().cpu().numpy()


def decompose(model, p, d, arrays, idx, rid):
    """Exact linear decomposition at fixed T and W; no reallocation after removal."""
    op = model.m3[0][p]
    source = exp.tensor(arrays['x'], rid)[:, p]
    masks = exp.tensor(arrays['slots'], rid)[:, p]
    valid = exp.tensor(arrays['valid'], rid)[:, p]
    mass = exp.tensor(arrays['mass'], rid)[:, p]
    masks = torch.minimum(masks.clamp(0,1), j.grid_occupancy(valid.to(source),source.shape[-2:]).clamp(0,1))
    z = F.layer_norm(source.permute(0,2,3,1),(source.shape[1],)).permute(0,3,1,2)
    values = torch.einsum('kshw,kdhw->ksd',masks,op.project(z))/mass.clamp_min(1e-8)[...,None]
    qm = exp.tensor(arrays['mass'],[idx])[:,p]
    parts=[]
    for r, slots in enumerate(exp.tokens.REGIONS):
        t=d['transport'][r]
        moved=torch.einsum('bkij,kjd->bkid',t,values[:,slots])
        offered=qm[:,None,slots]/qm[:,None,slots].sum(-1,keepdim=True).clamp_min(1e-20)
        moved=moved/offered[...,None].clamp_min(1e-20)
        parts.append(d['reference_weights'][:,:,slots,None]*moved)
    slot_values=torch.cat(parts,2)[0]  # reference x slot x 32
    masks=exp.tensor(arrays['slots'],[idx])[:,p]
    valid=exp.tensor(arrays['valid'],[idx])[:,p]
    masks=torch.minimum(masks.clamp(0,1),j.grid_occupancy(valid,source.shape[-2:]).clamp(0,1))
    write=masks/masks.sum(1,keepdim=True).clamp_min(1)

    def update(refs=None, slots=None):
        v=slot_values.clone()
        if refs is not None:
            keep=torch.zeros(len(rid),dtype=torch.bool);keep[refs]=True;v[~keep]=0
        if slots is not None:
            keep=torch.zeros(20,dtype=torch.bool);keep[slots]=True;v[:,~keep]=0
        mixed=torch.einsum('bshw,sd->bdhw',write,v.sum(0))
        return op.gain*op.output(mixed)

    error=float((update()-d['update']).abs().max())
    assert error<2e-5, ('writeback decomposition mismatch',error)
    return update,error


def downstream(model, arrays, idx, updated):
    n=len(updated)
    ids=np.repeat(idx,n)
    valid,slots,energy=[exp.tensor(arrays[k],ids) for k in ('valid','slots','energy')]
    available=exp.tensor(arrays['available'],ids)
    deep=model.backbone.model.layer4(updated.flatten(0,1))
    glob=exp.regional_mean(deep,valid.flatten(0,1),valid.flatten(0,1))[0][:,0].reshape(n,2,-1)
    low=exp.tensor(arrays['low'],ids)
    pooled=[]
    for p in range(2):
        reg,en=model.region_integrals(slots[:,p],energy[:,p])
        extra=model.deep_pool[p](updated[:,p],reg,valid[:,p],en)
        pooled.append((low[:,p]+model.deep_output[p](extra['pooled'])).flatten(1))
    values=torch.cat((glob,torch.stack(pooled,1)),-1)*available[...,None]
    return model.head(torch.cat((values.flatten(1),available),-1))


def branch_panels(case,p,refs,on,off,rows,rid,records):
    fig,axes=plt.subplots(6,6,figsize=(17,15))
    for block,k in enumerate(refs):
        d,do=on[p],off[p]
        # Display the body matrix; all regions are exported separately as raw tensors.
        e=arr(d['branch_similarity'][0][0,k]); a=arr(d['branch_weights'][0][0,k])
        t=arr(d['transport'][0][0,k]); live=t>0
        s=arr(d['similarity'][0][0,k]); so=arr(do['similarity'][0][0,k])
        for rr,data in enumerate((e,a,e*a)):
            for b in range(5):
                ax=axes[block*3+rr,b]
                im=ax.imshow(np.ma.masked_where(~live,data[...,b]),cmap='viridis' if rr==1 else 'RdBu_r',
                             vmin=0 if rr==1 else -1,vmax=1,interpolation='nearest')
                if rr==0: ax.set_title(BRANCH[b])
                if b==0: ax.set_ylabel(f'参考{rows[rid[k]]["id"]}\n'+('E 相似度','α 权重','αE 加权项')[rr])
                ax.set_xticks([0,7]);ax.set_yticks([0,7])
            ax=axes[block*3+rr,5]
            v=(so,s,s-so)[rr]
            lim=1 if rr<2 else max(float(abs(v[live]).max()) if live.any() else 0,.001)
            im=ax.imshow(np.ma.masked_where(~live,v),cmap='RdBu_r',vmin=-lim,vmax=lim,interpolation='nearest')
            ax.set_title(('频域关 S','频域开 S','变化 ΔS')[rr]);fig.colorbar(im,ax=ax,fraction=.045)
            ax.set_xticks([0,7]);ax.set_yticks([0,7])
        for r in range(3):
            er=arr(d['branch_similarity'][r][0,k]); ar=arr(d['branch_weights'][r][0,k])
            tr=arr(d['transport'][r][0,k]);den=max(tr.sum(),1e-20)
            for b in range(5):
                records.append(dict(query=case,pol=POL[p],reference=rows[rid[k]]['id'],region=REGIONS[r],branch=BRANCH[b],
                    matched_similarity=float((er[...,b]*tr).sum()/den),matched_weight=float((ar[...,b]*tr).sum()/den),
                    matched_term=float((ar[...,b]*er[...,b]*tr).sum()/den)))
    fig.suptitle(f'{case} / {POL[p]}：同类与异类参考的五路比较（候选 N）\n横轴参考框内槽位，纵轴查询框内槽位；白色无有效匹配。E、αE共用[-1,1]；α使用[0,1]。',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95))
    return export(fig,f'{case}_{POL[p]}_branches')


def image_links(case,p,idx,refs,on,off,rows,rid,edge_rows):
    q=exp.data.prepare(rows[idx])[p]
    qm=exp.tokens.slots(q['geometry'],q['footprint'])
    fig,axes=plt.subplots(2,2,figsize=(15,11))
    yy,xx=np.mgrid[:160,:160]
    def center(mask):
        den=mask.sum();return (float((mask*xx).sum()/den),float((mask*yy).sum()/den)) if den>0 else None
    for row,k in enumerate(refs):
        ref=exp.data.prepare(rows[rid[k]])[p]
        rm=exp.tokens.slots(ref['geometry'],ref['footprint'])
        for col,d in enumerate((off[p],on[p])):
            ax=axes[row,col]
            # Both chips share the native 160px display canvas; no model input resizing.
            canvas=np.zeros((160,344));canvas[:,:160]=q['pixels'];canvas[:,184:]=ref['pixels']
            ax.imshow(canvas,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            for m,color in zip(exp.tokens.REGIONS,(GREEN,BLUE,ORANGE)):
                for masks,offset in ((qm,0),(rm,184)):
                    reg=masks[m].sum(0)
                    if reg.max()>.05:ax.contour(xx+offset,yy,reg,levels=[.05],colors=[color],linewidths=.65)
            for r,slots in enumerate(exp.tokens.REGIONS):
                t=arr(d['transport'][r][0,k]); order=np.argsort(-t.ravel())
                chosen=[]
                for flat in order:
                    a,b=np.unravel_index(flat,t.shape)
                    if t[a,b]<=0:break
                    if a in [u for u,v in chosen]:continue
                    qa,rb=center(qm[slots[a]]),center(rm[slots[b]])
                    if qa is None or rb is None:continue
                    chosen.append((a,b))
                    color=(GREEN,BLUE,ORANGE)[r]
                    strength=float(t[a,b]/max(t.sum(),1e-20))
                    ax.plot([qa[0],rb[0]+184],[qa[1],rb[1]],color=color,lw=.8+5*strength,alpha=.8)
                    ax.scatter([qa[0],rb[0]+184],[qa[1],rb[1]],s=18,facecolors='none',edgecolors=color)
                    edge_rows.append(dict(query=case,pol=POL[p],reference=rows[rid[k]]['id'],frequency=('off','on')[col],region=REGIONS[r],
                        query_slot=int(slots[a]),reference_slot=int(slots[b]),transport=float(t[a,b]),regional_fraction=strength,
                        qx=qa[0],qy=qa[1],rx=rb[0],ry=rb[1]))
                    if len(chosen)==2:break
            ax.set_title(f'{"频域关" if col==0 else "频域开"} | 查询{case} → 参考{rows[rid[k]]["id"]} ({CLASSES[rows[rid[k]]["label"]]})',fontsize=11)
            ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle(f'{case} / {POL[p]}：局部匹配回到原始chip\n绿色框内 / 蓝色侧向 / 橙色轴端。每区域显示2条最强且查询槽位不同的连接，非全部匹配。',fontsize=14)
    fig.text(.05,.025,'线端点为区域槽位的原图加权中心，不是散射点定位；线宽按该参考、该区域内部匹配量占比显示。',fontsize=10)
    fig.tight_layout(rect=(0,.05,1,.94))
    return export(fig,f'{case}_{POL[p]}_links')


def weights_panel(case,on,off,rows,rid):
    fig,axes=plt.subplots(2,3,figsize=(17,9))
    maximum=max(float(d['reference_weights'].max()) for d in on+off)
    difference=max(float((on[p]['reference_weights']-off[p]['reference_weights']).abs().max()) for p in range(2))
    for p in range(2):
        a,b=arr(on[p]['reference_weights'][0]),arr(off[p]['reference_weights'][0])
        for c,data in enumerate((b,a,a-b)):
            ax=axes[p,c];limit=maximum if c<2 else max(difference,1e-6)
            im=ax.imshow(data,cmap='viridis' if c<2 else 'RdBu_r',vmin=0 if c<2 else -limit,vmax=limit,aspect='auto')
            ax.set_title(f'{POL[p]} / '+('频域关','频域开','开 − 关')[c]);fig.colorbar(im,ax=ax,fraction=.035)
            ax.set_yticks(range(len(rid)),[str(rows[i]['id']) for i in rid],fontsize=8)
            ax.set_xticks([0,7,8,15,16,19]);ax.axvline(7.5,color='white',lw=.8);ax.axvline(15.5,color='white',lw=.8)
            ax.set_xlabel('查询槽位：框内0–7 / 侧向8–15 / 轴端16–19')
    fig.suptitle(f'{case}：真正用于回写的参考权重 W（列为查询槽位，不再先取平均）',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.96))
    return export(fig,f'{case}_weights')


@torch.no_grad()
def main():
    OUT.mkdir(exist_ok=True)
    exp.DEVICE='cpu';exp.current.setup('cpu');torch.set_num_threads(4)
    rows,nf,refs,rid,arrays=exp.load();cache=j.pair_cache()
    branch_rows=[];edge_rows=[];interventions=[];cases=[]
    for case,seed in (('24470',43),('12457',17)):
        idx=next(i for i,r in enumerate(rows) if str(r['id'])==case)
        model=review.restore('N_level_pattern',seed);model.requires_grad_(False)
        score,on=j.forward(model,'N_level_pattern',cache,arrays,np.array([idx]),rid,rows,details=True)
        for op in model.m3[0]:op.frequency=False
        score_off,off=j.forward(model,'N_level_pattern',cache,arrays,np.array([idx]),rid,rows,details=True)
        for op in model.m3[0]:op.frequency=True
        saved=dict(np.load(BASE/'training'/f'N_level_pattern_{seed}'/'final_development.npz'))
        pos=int(np.flatnonzero(saved['ids']==idx)[0]);replay=float(abs(arr(score)[0]-saved['score'][pos]).max())
        assert replay<1e-4,replay
        source=exp.tensor(arrays['x'],[idx]);base=source+torch.stack([d['update'] for d in on],1)
        variants=[base,source+torch.stack([d['update'] for d in off],1)]
        names=['全部开启','全部频域关闭'];component_records=[];pictures=[];selection=[]
        ry=np.array([rows[i]['label'] for i in rid]);label=rows[idx]['label']
        raw={}
        for p in range(2):
            live=arr(on[p]['reference_weights'][0].sum(0))>0
            mean=arr(on[p]['reference_weights'][0])[:,live].mean(1)
            same=np.flatnonzero((ry==label)&(mean>0));other=np.flatnonzero((ry!=label)&(mean>0))
            chosen=[int(same[np.argmax(mean[same])]),int(other[np.argmax(mean[other])])]
            selection.append([str(rows[rid[k]]['id']) for k in chosen])
            update,error=decompose(model,p,on[p],arrays,idx,rid)
            for name,kw in [(f'{POL[p]} 去{REGIONS[r]}',dict(slots=s)) for r,s in enumerate(exp.tokens.REGIONS)]+[
                (f'{POL[p]} 去参考{rows[rid[k]]["id"]}',dict(refs=[k])) for k in chosen]:
                contribution=update(**kw);changed=base.clone();changed[:,p]-=contribution
                variants.append(changed);names.append(name)
                component_records.append(dict(name=name,norm=float(contribution.norm()),decomposition_error=error))
            pictures.append(branch_panels(case,p,chosen,on,off,rows,rid,branch_rows))
            pictures.append(image_links(case,p,idx,chosen,on,off,rows,rid,edge_rows))
            for state,ds in (('on',on),('off',off)):
                for key in ('transport','similarity','branch_similarity','branch_weights'):
                    for r in range(3):raw[f'{POL[p]}_{state}_{key}_{r}']=arr(ds[p][key][r])
                raw[f'{POL[p]}_{state}_reference_weights']=arr(ds[p]['reference_weights'])
        results=[]
        for start in range(0,len(variants),3):
            results.append(downstream(model,arrays,idx,torch.cat(variants[start:start+3])))
        results=torch.cat(results)
        parity=float((results[0]-score[0]).abs().max());assert parity<1e-4,parity
        off_parity=float((results[1]-score_off[0]).abs().max());assert off_parity<1e-4,off_parity
        margins=review.margin(arr(results),np.full(len(results),label))
        for k,name in enumerate(names):
            interventions.append(dict(query=case,seed=seed,condition=name,truth=label,prediction=int(results[k].argmax()),
                margin=float(margins[k]),margin_change=float(margins[k]-margins[0]),
                replay_error=replay,downstream_parity=parity))
        raw['intervention_logits']=arr(results);np.savez_compressed(OUT/f'{case}_traces.npz',**raw)
        pictures.append(weights_panel(case,on,off,rows,rid))
        fig,axes=plt.subplots(1,2,figsize=(15,7))
        axes[0].barh(range(len(names)),margins,color=[BLUE if m>=0 else RED for m in margins])
        axes[0].set_yticks(range(len(names)),names);axes[0].invert_yaxis();axes[0].axvline(0,color=GRAY,lw=.8)
        axes[0].set_title('关闭后正确类分差');axes[0].set_xlabel('正确类 − 最强错误类')
        delta=margins-margins[0]
        axes[1].barh(range(len(names)),delta,color=[GREEN if m>=0 else RED for m in delta])
        axes[1].set_yticks(range(len(names)),names);axes[1].invert_yaxis();axes[1].axvline(0,color=GRAY,lw=.8)
        axes[1].set_title('关闭后分差变化：正值表示移除后更好')
        fig.suptitle(f'{case} / seed {seed}：固定匹配和参考权重，只移除指定回写项\n“全部频域关闭”会重新计算匹配；其余局部关闭不重新分配权重。',fontsize=13)
        fig.tight_layout(rect=(0,0,1,.92));pictures.append(export(fig,f'{case}_local_removal'))
        cases.append(dict(query=case,seed=seed,label=CLASSES[label],port=rows[idx]['port'],orbit=j.probe.orbit(rows[idx]),
            margin_on=float(margins[0]),margin_off=float(margins[1]),references=selection,components=component_records,
            image_paths={p:rows[idx][p]['image'] for p in ('vv','vh')},pictures=pictures))
        print('CASE_DONE',case,seed,'margin',margins.tolist(),flush=True)
        del model
    pd.DataFrame(branch_rows).to_csv(OUT/'branch_terms.csv',index=False)
    pd.DataFrame(edge_rows).to_csv(OUT/'displayed_links.csv',index=False)
    pd.DataFrame(interventions).to_csv(OUT/'local_removal.csv',index=False)
    (OUT/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
    blocks=[]
    for case in cases:
        blocks.append(f'<section><h2>{case["query"]} · {html.escape(case["label"])} · seed {case["seed"]}</h2>'+
            ''.join(f'<figure><img src="{p}" loading="lazy"><figcaption>{p}</figcaption></figure>' for p in case['pictures'])+'</section>')
    (OUT/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>M3关系证据</title><style>body{font:16px system-ui;color:#202b31;max-width:1500px;margin:32px auto;padding:0 18px}img{width:100%;height:auto}figure{margin:28px 0}section{border-top:1px solid #ccc;margin-top:32px}figcaption{color:#667}p{line-height:1.7}</style>'
        '<h1>M3：比较依据 → 图像对应 → 回写依赖</h1><p>固定历史候选 N，不重训，不修改正式主模型。24470检验微小频域救回；12457用于区分联合训练收益与频域贡献。'
        '标签只用于事后诊断同类/混淆参考。可视化连接是区域槽位中心，不是物理散射点。局部关闭固定匹配和权重，不重分配。</p>'+
        ''.join(blocks)+'</html>',encoding='utf-8')


if __name__=='__main__':main()
