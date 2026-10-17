"""Trace actual query/support changes and render unmodified SAR inputs."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_slots as run
from slot_reader import SlotReader, centered_features
from body_context_gate import _coefficients

HERE=Path(__file__).resolve().parent
PIXELS=Path("D:/SAR_completion_cache_20261010/source_inputs128.npy")


def show(ax,pixels,row,geo,centers,label,pol=0):
    im=pixels[int(row),pol]
    ax.imshow(im,cmap="gray",vmin=0,vmax=255,interpolation="nearest")
    h,w=im.shape
    cx,cy=(centers+.5)/56*np.array([w,h])
    for k,color,name in ((0,"cyan","ship axis"),(1,"yellow","radar proxy")):
        if geo[k+3]>0:
            a=geo[k]
            length=.24*min(h,w)
            dx,dy=length*np.cos(a),length*np.sin(a)
            ax.plot([cx-dx,cx+dx],[cy-dy,cy+dy],color=color,linewidth=.9,linestyle="--" if k==1 else "-")
    ax.set_title(label,fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    events=pd.read_csv(HERE/"paired_events.csv",dtype={"mmsi":str})
    candidates=events.loc[events.arm.eq("geometry") & events.seed.eq(0) & events.origin.eq("cross") & events.enabled].sort_values(["port","global_row","fold","draw"])
    picked=[]
    for kind, mask in (("rescue",candidates.rescue),("harm",candidates.harm),("unchanged",~candidates.rescue & ~candidates.harm)):
        f=candidates.loc[mask]
        if len(f):
            picked.append((kind,f.iloc[0]))
    if not picked:
        (HERE/"CASE_ANALYSIS_ZH.md").write_text("# 案例\n\n无可启用的跨采集查询，不生成虚构案例。\n",encoding="utf-8")
        return
    pixels=np.load(PIXELS,mmap_mode="r")
    fig,axes=plt.subplots(len(picked),4,figsize=(14,4.5*len(picked)),squeeze=False)
    records=[]; support_records=[]; slot_records=[]
    cache={}
    for ri,(kind,row) in enumerate(picked):
        if row.port not in cache:
            data=run.load(row.port,True)
            run.prepare_baseline(data,0)
            saved=torch.load(HERE/"trained"/row.port.replace(" ","_")/"geometry_seed0.pt",weights_only=False)
            model=SlotReader("geometry").to(run.DEVICE)
            model.load_state_dict(saved["state"]); model.eval()
            raw=run.raw_descriptors(model,data)
            with np.load(HERE/"slots_cache"/(row.port.replace(" ","_")+".npz")) as z:
                pos=np.searchsorted(z["rows"],data["meta"].global_row)
                centers=z["centers"][pos]
            cache[row.port]=(data,model,raw,centers)
        data,model,raw,centers=cache[row.port]
        d=data["tensors"]
        item=next(x for x in data["cache"]["target"] if x["t"]["fold"]==row.fold and x["t"]["origin"]==row.origin and x["t"]["draw"]==row.draw)
        qloc=int(np.where(data["meta"].iloc[item["q"].cpu().numpy()].sample_id.to_numpy()==row.sample_id)[0][0])
        qid=int(item["q"][qloc]); sids=item["s"].cpu().numpy()
        prob,newqs,psi=run.prediction(model,data,item,raw_all=raw)
        ns=len(sids); newss=item["ss"]+.75*(psi[:ns]@psi[:ns].T)
        cb=_coefficients(item["ss"],item["prior_s"],item["sy"])
        cn=_coefficients(newss,item["prior_s"],item["sy"])
        truth=int(row.truth)
        oldprob=item["old_prob"][qloc].cpu().numpy()
        rivals=oldprob.copy(); rivals[truth]=-np.inf; rival=int(rivals.argmax())
        direct=((newqs[qloc]-item["qs"][qloc])*(cb[:,truth]-cb[:,rival]))
        resolving=newqs[qloc]*((cn[:,truth]-cn[:,rival])-(cb[:,truth]-cb[:,rival]))
        exact=(newqs[qloc]@(cn[:,truth]-cn[:,rival])-item["qs"][qloc]@(cb[:,truth]-cb[:,rival]))
        torch.testing.assert_close(direct.sum()+resolving.sum(),exact,atol=1e-10,rtol=1e-10)
        total=(direct+resolving).cpu().numpy(); sy=item["sy"].cpu().numpy()
        cs=np.flatnonzero(sy==truth); rs=np.flatnonzero(sy==rival)
        right=int(cs[np.argmax(total[cs])]) if len(cs) else int(np.argmax(total))
        wrong=int(rs[np.argmin(total[rs])]) if len(rs) else int(np.argmin(total))
        geo=d["geometry"].cpu().numpy()
        globalrows=data["meta"].global_row.to_numpy()
        show(axes[ri,0],pixels,globalrows[qid],geo[qid],centers[qid],f"{kind}: query VV, class {truth}\n{row.port} | MMSI {row.mmsi}")
        show(axes[ri,1],pixels,globalrows[qid],geo[qid],centers[qid],f"query VH | B {int(row.pred_B)} -> C {int(row.pred)}\nprob margin {oldprob[truth]-oldprob[rival]:+.3f} -> {float(prob[qloc,truth]-prob[qloc,rival]):+.3f}",1)
        for col,si,title in ((2,right,"true-class support" if len(cs) else "no true-class support"),(3,wrong,"rival-class support" if len(rs) else "no rival-class support")):
            idx=sids[si]
            show(axes[ri,col],pixels,globalrows[idx],geo[idx],centers[idx],f"{title}: class {int(sy[si])}\nexpert margin contribution {total[si]:+.3f}")
        rec=dict(kind=kind,port=row.port,mmsi=row.mmsi,sample_id=row.sample_id,global_row=int(row.global_row),fold=str(row.fold),draw=int(row.draw),truth=truth,rival=rival,
                 pred_B=int(row.pred_B),pred_C=int(row.pred),prob_margin_B=float(oldprob[truth]-oldprob[rival]),prob_margin_C=float(prob[qloc,truth]-prob[qloc,rival]),
                 direct_kernel_effect=float(direct.sum()),coefficient_resolve_effect=float(resolving.sum()),expert_margin_change=float(exact),valid_supports=int(row.valid_supports))
        records.append(rec)
        for si,idx in enumerate(sids):
            support_records.append(dict(case=kind,query_id=row.sample_id,support_id=data["meta"].iloc[idx].sample_id,support_mmsi=data["meta"].iloc[idx].mmsi,support_class=int(sy[si]),
                support_valid=bool(d["valid"][idx]),kernel_delta=float(newqs[qloc,si]-item["qs"][qloc,si]),direct=float(direct[si]),resolve=float(resolving[si]),total=float(total[si])))
            slots=(.75*psi[ns+qloc].reshape(3,4,16)*psi[si].reshape(3,4,16)).sum(-1).cpu().numpy()
            for ring in range(3):
                for slot in range(4):
                    slot_records.append(dict(case=kind,support_id=data["meta"].iloc[idx].sample_id,ring=ring,slot=slot,delta_kernel=float(slots[ring,slot])))
    fig.suptitle("Actual SAR inputs | cyan: unsigned ship proxy; yellow: unsigned radar proxy",fontsize=12)
    fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.92,hspace=.30,wspace=.06)
    fig.savefig(HERE/"query_support_cases.png",dpi=160); plt.close(fig)
    pd.DataFrame(records).to_csv(HERE/"case_accounting.csv",index=False)
    pd.DataFrame(support_records).to_csv(HERE/"case_support_contributions.csv",index=False)
    pd.DataFrame(slot_records).to_csv(HERE/"case_slot_kernel_contributions.csv",index=False)
    lines=["# 查询—支持案例与决策路径", "", "固定seed0，跨采集、可启用查询；救回/误伤/不变各按港口、原始行、折、抽样排序取第一个。这是事后解释实例，不是发生率估计。", "",
           "图像是模型实际拉伸uint8 VV/VH输入，不是原始辐射定标数据。青线/黄线分别为无向船轴与雷达轴代理，不命名船首/尾迹。", "", "![实际SAR查询与支持](query_support_cases.png)", "",
           "完整32支持均参与求解，图中只展示真实类中贡献最大的支持与竞争类中贡献最小的支持；不存在该类时会明确标注替代。按支持贡献选图不能证明其外观对应同一物理机制。", ""]
    for r in records:
        lines += [f"## {r['kind']}：{r['mmsi']}", "", f"{r['port']}，真实类{r['truth']}，固定竞争类{r['rival']}。最终预测B={r['pred_B']}，C={r['pred_C']}。真实类对竞争类概率差{r['prob_margin_B']:+.6f}→{r['prob_margin_C']:+.6f}。",
                  f"专家分差变化{r['expert_margin_change']:+.6f}，分解为保持旧支持系数的直接核效应{r['direct_kernel_effect']:+.6f}，以及重求解系数效应{r['coefficient_resolve_effect']:+.6f}。这是精确代数路径，不是物理因果分解。", ""]
    lines += ["## 解释边界", "", "正确类关系改善、完整分类获益和船体/海面物理机制是不同层级。分区贡献可定位计算依赖，但不自动区分框外目标关联响应和环境纹理；需要结合实际图像观察，不把角度依赖直接命名为物理机制。所有支持、分区贡献及精确和式核对保存在CSV。"]
    (HERE/"CASE_ANALYSIS_ZH.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(pd.DataFrame(records).to_string(index=False))


if __name__=="__main__":
    main()
