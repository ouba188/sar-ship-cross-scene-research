"""Paired source evaluation and native-chip evidence for the utility pilot."""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"
import sys
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image

HERE = Path(__file__).resolve().parent
SCENE = HERE.parents[1]
OBB = SCENE / "eight_class_evidence_20261023/obb_dynamic_readout"
sys.path.insert(0, str(OBB))
import common as geometry
from run_router import STRATEGIES, center_scores, utilities


def paired_tables(pred):
    rows, by_class = [], []
    for seed, group in pred[pred.role.eq("evaluation")].groupby("seed"):
        new = group[group.arm.eq("utility")].set_index("instance_id")
        for arm in ("base", "body", "full", "equal", "fixed", "ce_only"):
            other = group[group.arm.eq(arm)].set_index("instance_id").loc[new.index]
            np.testing.assert_array_equal(new.class_id, other.class_id)
            y = new.class_id.to_numpy()
            a = new.prediction.to_numpy() == y
            b = other.prediction.to_numpy() == y
            delta = a.astype(float) - b.astype(float)
            rows.append(dict(seed=int(seed), comparison=arm, n=len(y),
                             rescue=int(np.sum(a & ~b)), harm=int(np.sum(~a & b)),
                             flips=int(np.sum(new.prediction.to_numpy()!=other.prediction.to_numpy())),
                             delta_OA=float(delta.mean()*100),
                             delta_BA=float(np.mean([delta[y==c].mean() for c in np.unique(y)])*100)))
            for c in np.unique(y):
                take = y == c
                by_class.append(dict(seed=int(seed), comparison=arm, class_id=int(c),
                                     n=int(take.sum()), rescue=int(np.sum(a[take]&~b[take])),
                                     harm=int(np.sum(~a[take]&b[take])),
                                     delta_recall=float(delta[take].mean()*100)))
    return pd.DataFrame(rows), pd.DataFrame(by_class)


def cases_to_view(pred):
    group = pred[pred.role.eq("evaluation") & pred.seed.eq(17)]
    new = group[group.arm.eq("utility")].set_index("instance_id")
    chosen = []
    for arm in ("ce_only", "fixed"):
        other = group[group.arm.eq(arm)].set_index("instance_id").loc[new.index]
        a, b = new.prediction.eq(new.class_id), other.prediction.eq(new.class_id)
        outcomes = [("rescued", a & ~b), ("harmed", ~a & b)]
        if arm == "ce_only":
            outcomes += [("CC", a & b), ("WW", ~a & ~b)]
        for outcome, mask in outcomes:
            pool = new[mask].sort_values(["obs_id", "row_index"])
            if pool.empty:
                chosen.append(dict(comparison=arm, outcome=outcome, group_n=0, available=False))
                continue
            row = pool.iloc[0]
            chosen.append(dict(comparison=arm, outcome=outcome, group_n=len(pool), available=True,
                               instance_id=row.name, obs_id=int(row.obs_id), row_index=int(row.row_index)))
    return pd.DataFrame(chosen)


def render_case(record, pred, raw, upstream_ids, table, valid, snapshots, out):
    i = int(record.row_index)
    row = table.iloc[i]
    loc = int(np.flatnonzero(upstream_ids == i)[0])
    g = pred[pred.role.eq("evaluation") & pred.seed.eq(17) & pred.instance_id.eq(row.instance_id)].set_index("arm")
    arms = ("base", "body", "full", "fixed", "ce_only", "utility")
    scores = {a: g.loc[a, [f"s{c}" for c in range(8)]].to_numpy(float) for a in arms}
    y = int(row.class_id)
    rival = int(np.argmax(np.where(np.arange(8)==y, -np.inf, scores["utility"])))
    colors = {"base":"#555555", "body":"#bb7825", "full":"#147b72",
              "fixed":"#7a579a", "ce_only":"#3778ad", "utility":"#c54342"}
    fig, ax = plt.subplots(2, 4, figsize=(21, 10), layout="constrained")
    for p, pol in enumerate(("vv", "vh")):
        with Image.open(row[f"image_{pol}"]) as im:
            img = np.asarray(im.convert("L"))
        canvas, footprint, box, place = geometry.native(table, snapshots, i, p)
        ax[p, 0].imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax[p, 0].set_title(f"{pol.upper()} original {img.shape[1]}x{img.shape[0]}, valid={bool(valid[i,p])}")
        if box is not None:
            u = np.array([np.cos(box[4]), np.sin(box[4])])*box[2]/2
            v = np.array([-np.sin(box[4]), np.cos(box[4])])*box[3]/2
            c = box[:2] - [place["pad_left"]-place["crop_x"], place["pad_top"]-place["crop_y"]]
            ax[p, 0].add_patch(Polygon(np.array([c-u-v,c+u-v,c+u+v,c-u+v]), fill=False, edgecolor="#35cf76"))
        ax[p, 1].imshow(canvas, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        regions = geometry.soft_regions(box)*footprint
        for r, color in enumerate(("#35cf76", "#e9ac2c", "#32bfd3")):
            rgba = np.zeros((160,160,4)); rgba[...,:3] = matplotlib.colors.to_rgb(color)
            rgba[...,3] = .5*regions[r]
            ax[p,1].imshow(rgba, interpolation="nearest")
        ax[p,1].set_title("Native160: body / ends / sides; RF35")
        labels = [f"{r}:{s}" for r in ("body","end","side") for s in ("mu","h","b")]
        values = [raw[loc,y,1+p*18+k*2]-raw[loc,rival,1+p*18+k*2] for k in range(9)]
        ax[p,2].bar(np.arange(9),values,color=colors["full"])
        ax[p,2].set_xticks(np.arange(9),labels,rotation=75,fontsize=7)
        ax[p,2].axhline(0,color="black",lw=.6)
        ax[p,2].set_title(f"Actual regional input: class{y} minus class{rival}\nNot a contribution or physical explanation",fontsize=9)
    for arm in arms:
        ax[0,3].plot(np.arange(8),scores[arm],label=arm,color=colors[arm],marker="o",ms=3)
    ax[0,3].legend(fontsize=8);ax[0,3].set_xticks(range(8));ax[0,3].set_title(f"Class scores; true={y}; fixed rival={rival}")
    weight = np.array([[g.loc["utility",f"w_{k}_{c}"] for k in STRATEGIES] for c in range(8)],float)
    ax[1,3].imshow(weight,aspect="auto",vmin=0,vmax=1,cmap="Greens")
    ax[1,3].set_xticks(range(3),STRATEGIES);ax[1,3].set_yticks(range(8))
    ax[1,3].set_title("Utility coordinator weights by candidate class")
    for c in range(8):
        for k in range(3):ax[1,3].text(k,c,f"{weight[c,k]:.2f}",ha="center",va="center",fontsize=8)
    selection_note = "numerical diagnostic case" if str(record.outcome).startswith("coverage-change") else "smallest obs_id in group"
    fig.suptitle(f"SOURCE EVALUATION seed17 | obs{row.obs_id} | {row.class_name} | {row.port}\n"
                 f"{record.outcome} relative to {record.comparison}; {selection_note}; no target training",fontsize=12)
    fig.savefig(out/f"case_{int(row.obs_id)}.png",dpi=110);plt.close(fig)
    z = center_scores(np.stack([scores[a] for a in STRATEGIES]))
    delta = z-z[0]
    terms = weight[y]*delta[:,y]-weight[rival]*delta[:,rival]
    reconstructed = z[0,y]-z[0,rival]+terms.sum()
    actual = scores["utility"][y]-scores["utility"][rival]
    assert abs(reconstructed-actual)<2e-5
    details = dict(instance_id=row.instance_id,obs_id=int(row.obs_id),class_id=y,rival=rival,
                   port=row.port,pass_group=row.pass_group,image_vv=row.image_vv,image_vh=row.image_vh,
                   decomposition_error=float(abs(reconstructed-actual)))
    for arm in arms:
        details[f"prediction_{arm}"] = int(scores[arm].argmax())
        details[f"margin_{arm}"] = float(scores[arm][y]-scores[arm][rival])
    for k,name in enumerate(STRATEGIES):details[f"margin_contribution_{name}"] = float(terms[k])
    return details


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,default=HERE/'router_results')
    parser.add_argument('--output',type=Path,default=HERE/'analysis')
    args=parser.parse_args()
    results=args.results
    assert (results/"complete.json").exists()
    pred=pd.read_csv(results/"predictions.csv",dtype={"mmsi":str})
    out=args.output;out.mkdir(exist_ok=True)
    pairs,classes=paired_tables(pred)
    pairs.to_csv(out/"paired.csv",index=False);classes.to_csv(out/"class_effects.csv",index=False)
    metrics=pd.read_csv(results/"metrics.csv")
    means=metrics[metrics.role.eq("evaluation")].groupby("arm")[["OA","BA","Brier","utility_mse","utility_proxy_regret"]].agg(["mean","std"])
    means.to_csv(out/"evaluation_summary.csv")
    proxy_rows=[]
    for seed in (17,29,43):
        upstream=pd.read_csv(HERE/"upstream/table.csv")
        with np.load(HERE/"upstream"/f"strategies_seed{seed}.npz") as z:
            score=z["logits"].astype(float)
        tr=upstream.role.eq("utility_train").to_numpy()
        truth=utilities(score,upstream.class_id.to_numpy(int))[...,1:]
        constant=truth[tr].mean(axis=(0,1),keepdims=True)
        for role in ("utility_train","selection","evaluation"):
            ix=upstream.role.eq(role).to_numpy()
            zero=float(np.mean(truth[ix]**2))
            const=float(np.mean((truth[ix]-constant)**2))
            for arm in ("ce_only","utility"):
                fitted=metrics[(metrics.seed==seed)&(metrics.role==role)&(metrics.arm==arm)].iloc[0]
                proxy_rows.append(dict(seed=seed,role=role,arm=arm,
                    mse=float(fitted.utility_mse),zero_mse=zero,train_constant_mse=const,
                    mse_over_zero=float(fitted.utility_mse)/max(zero,1e-12),
                    mse_over_constant=float(fitted.utility_mse)/max(const,1e-12)))
    pd.DataFrame(proxy_rows).to_csv(out/"utility_learning.csv",index=False)
    cases=cases_to_view(pred);cases.to_csv(out/"case_selection.csv",index=False)
    table,valid,snapshots=geometry.load_data()
    with np.load(HERE/"upstream/strategies_seed17.npz") as z:
        raw=z["regional_raw"].copy();ids=z["indices"].copy()
    records=[]
    for _,record in cases[cases.available.eq(True)].drop_duplicates("instance_id").iterrows():
        records.append(render_case(record,pred,raw,ids,table,valid,snapshots,out))
    pd.DataFrame(records).to_csv(out/"cases.csv",index=False)
    print(means.to_string())
    print(pairs.to_string(index=False))
    print("Saved",len(records),"native VV/VH case panels")


if __name__=="__main__":main()
