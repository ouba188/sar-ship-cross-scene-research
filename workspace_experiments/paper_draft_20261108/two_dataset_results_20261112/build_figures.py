"""Recompute publication figures from saved test predictions; no inference."""
from pathlib import Path
import csv
import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent
WORK = PAPER.parent
OUT = PAPER / "figures/results_20261112"
SEEDS = (17, 29, 43)
STORE = Path("C:/Users/25791/AppData/Local/sar_ports24_fullscale_20261110")
SPECS = {
    "oss2": dict(title="OpenSARShip 2.0", data=WORK / "m1_coverage_20261101", n=378,
        names=["Bulk", "General", "Chem./prod.", "Container"],
        zh=["散货船", "普通货船", "化学品／成品油轮", "集装箱船"],
        folders={"Plain": PAPER / "staged_ablation_20261109/plain_s{seed}_lr1",
                 "Full-Staged": PAPER / "staged_ablation_20261109/full_s{seed}_lr1",
                 "Full-Tuned": PAPER / "full_finetune_20261111/selected_s{seed}",
                 "R50-Tuned": PAPER / "r50_tuned_20261111/selected_s{seed}"}),
    "ports24": dict(title="Self-built 24-port", data=STORE, n=3075,
        names=["Bulk", "Fishing", "General", "Chem./prod.", "Container", "Crude", "Tug", "Offshore"],
        zh=["散货船", "渔船", "普通货船", "化学品／成品油轮", "集装箱船", "原油轮", "拖轮／拖带船", "海工补给船"],
        folders={"Plain": STORE / "plain_s{seed}_lr1",
                 "Full-Staged": STORE / "full_s{seed}_lr1",
                 "Full-Tuned": PAPER / "ports24_sync_20261112/selected_s{seed}"}),
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_csv(name, rows):
    with (ROOT / name).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, name):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def heatmap(ax, matrix, names, title, difference=False, vmax=None):
    c = len(names)
    ticks = ["BC", "FV", "GC", "PT", "CS", "CT", "TT", "OS"] if c == 8 else names
    limit = vmax if vmax is not None else max(1., np.max(np.abs(matrix)))
    im = ax.imshow(matrix, cmap="RdBu_r" if difference else "Blues",
                   vmin=-limit if difference else 0, vmax=limit if difference else 100)
    ax.set_xticks(range(c), ticks, rotation=40 if c == 4 else 0, ha="right" if c == 4 else "center", rotation_mode="anchor")
    ax.set_yticks(range(c), ticks)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title, fontsize=10, pad=10)
    ax.set_xticks(np.arange(c + 1)-.5, minor=True)
    ax.set_yticks(np.arange(c + 1)-.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=0)
    for i in range(c):
        for j in range(c):
            val = matrix[i, j]
            text = "0.0" if abs(val) < .05 else f"{val:+.1f}" if difference else f"{val:.1f}"
            bright = abs(val) > .64 * limit if difference else val > 60
            ax.text(j, i, text, ha="center", va="center", fontsize=7.2 if c == 8 else 9,
                    color="white" if bright else "#202020")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "svg.fonttype": "none", "pdf.fonttype": 42, "axes.spines.top": False,
        "axes.spines.right": False, "axes.unicode_minus": True})
    results, sources, cells, classes = {}, [], [], []
    for key, spec in SPECS.items():
        rows = read_json(spec["data"] / "manifest.json")
        ids = np.array(read_json(spec["data"] / "groups.json")["evaluation"], dtype=int)
        y = np.array([rows[i]["label"] for i in ids])
        c = len(spec["names"])
        assert len(ids) == spec["n"] and len(set(ids)) == len(ids)
        support = np.bincount(y, minlength=c)
        assert np.all(support > 0)
        methods = {}
        for method, template in spec["folders"].items():
            cms = []
            for seed in SEEDS:
                path = Path(str(template).format(seed=seed)) / "evaluation.npz"
                with np.load(path, allow_pickle=False) as saved:
                    actual, score = saved["ids"], saved["score"]
                    assert len(set(actual)) == len(ids) and set(actual) == set(ids), path
                    assert score.shape == (len(ids), c) and np.isfinite(score).all(), path
                    index = {int(v): j for j, v in enumerate(actual)}
                    score = score[[index[int(i)] for i in ids]]
                    label_key = "label" if "label" in saved else "y" if "y" in saved else None
                    if label_key:
                        assert np.array_equal(saved[label_key], [rows[int(i)]["label"] for i in actual]), path
                cm = np.bincount(c*y+score.argmax(1), minlength=c*c).reshape(c, c)
                cms.append(cm)
                sources.append(dict(dataset=key, method=method, seed=seed, path=str(path)))
                for i in range(c):
                    for j in range(c):
                        cells.append(dict(dataset=key, method=method, seed=seed,
                            true_class=spec["names"][i], predicted_class=spec["names"][j],
                            count=int(cm[i,j]), support=int(support[i]), percent=100*cm[i,j]/support[i]))
            cms = np.array(cms)
            normalized = cms / support[None, :, None] * 100
            recall = np.diagonal(normalized, axis1=1, axis2=2)
            f1 = 200*np.diagonal(cms, axis1=1, axis2=2)/(support[None,:]+cms.sum(1))
            oa = 100*np.trace(cms, axis1=1, axis2=2)/len(ids)
            ba = recall.mean(1)
            methods[method] = dict(cm=normalized.mean(0).tolist(), recall=recall.mean(0).tolist(),
                recall_sd=recall.std(0,ddof=1).tolist(), f1=f1.mean(0).tolist(),
                oa=float(oa.mean()), oa_sd=float(oa.std(ddof=1)),
                ba=float(ba.mean()), ba_sd=float(ba.std(ddof=1)),
                macro_f1=float(f1.mean(1).mean()), macro_f1_sd=float(f1.mean(1).std(ddof=1)))
            for i in range(c):
                classes.append(dict(dataset=key, method=method, class_name=spec["names"][i],
                    class_zh=spec["zh"][i], support=int(support[i]),
                    recall=recall[:,i].mean(), recall_sd=recall[:,i].std(ddof=1), f1=f1[:,i].mean()))
        results[key] = dict(n=len(ids), ships=len({rows[i]["mmsi"] for i in ids}),
            names=spec["names"], support=support.tolist(), methods=methods)

    # A seed-averaged matrix describes repeated training, not a three-model ensemble.
    delta_cells = []
    for key, spec in SPECS.items():
        methods = results[key]["methods"]
        full = np.array(methods["Full-Tuned"]["cm"])
        baselines = ["Plain", "Full-Staged"] + (["R50-Tuned"] if key == "oss2" else [])
        for baseline in baselines:
            old = np.array(methods[baseline]["cm"])
            delta = full-old
            assert np.allclose(delta.sum(1),0)
            limit = 22.0 if baseline == "Plain" else float(max(1, np.ceil(np.max(np.abs(delta)))))
            for i, name in enumerate(spec["names"]):
                for j, predicted in enumerate(spec["names"]):
                    delta_cells.append(dict(dataset=key, baseline=baseline, true_class=name,
                        predicted_class=predicted, delta_percentage_points=delta[i,j]))
            fig, axes = plt.subplots(1, 2, figsize=(8.3,4.2), layout="constrained")
            im = heatmap(axes[0], full, spec["names"], f"{spec['title']}: Full-Tuned")
            fig.colorbar(im, ax=axes[0], shrink=.70, label="Row-normalized (%)")
            im = heatmap(axes[1], delta, spec["names"], f"Full-Tuned minus {baseline}", True, limit)
            fig.colorbar(im, ax=axes[1], shrink=.70, label="Difference (pp)")
            save(fig, f"{key}_full_delta_{baseline.lower()}")
            for label, matrix, diff in ((baseline,old,False),("Full-Tuned",full,False),("delta",delta,True)):
                if label == "Full-Tuned" and baseline != "Plain":
                    continue
                fig, ax = plt.subplots(figsize=(5.2,4.5), layout="constrained")
                im = heatmap(ax,matrix,spec["names"], f"{spec['title']}: {label}",diff,limit)
                fig.colorbar(im,ax=ax,shrink=.8,label="Difference (pp)" if diff else "Row-normalized (%)")
                name = f"{key}_delta_{baseline.lower()}" if diff else f"{key}_{label.lower()}_cm"
                save(fig,name)

    fig, axes = plt.subplots(1,2,figsize=(8.6,3.8),layout="constrained",gridspec_kw={"width_ratios":[1,1.8]})
    for ax,(key,spec) in zip(axes,SPECS.items()):
        methods = results[key]["methods"]
        x = np.arange(len(spec["names"]))
        for offset, method, color in ((-1,"Plain","#91979e"),(0,"Full-Staged","#4a90ad"),(1,"Full-Tuned","#c86b42")):
            v=methods[method]
            ax.bar(x+offset*.25,v["recall"],.23,label=method,color=color,
                yerr=v["recall_sd"],capsize=2,error_kw={"elinewidth":.7})
        ax.set_xticks(x,spec["names"],rotation=35,ha="right")
        ax.set_ylim(0,100)
        ax.set_ylabel("Recall (%)")
        ax.set_title(spec["title"])
        ax.set_axisbelow(True)
        ax.grid(axis="y",alpha=.2)
    axes[0].legend(fontsize=7,loc="upper left",frameon=False)
    save(fig,"per_class_recall")
    write_csv("confusion_by_seed.csv",cells)
    write_csv("confusion_difference.csv",delta_cells)
    write_csv("per_class.csv",classes)
    (ROOT/"summary.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
    (ROOT/"sources_local.json").write_text(json.dumps(sources,indent=2),encoding="utf-8")
    print(json.dumps({k:{m:{n:v[n] for n in ("oa","ba","macro_f1")} for m,v in r["methods"].items()} for k,r in results.items()},indent=2))


if __name__ == "__main__":
    main()
