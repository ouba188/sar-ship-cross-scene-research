"""Exact support-level margin accounting for frozen B64 case changes."""

import json
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "2"

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent / "body_context_expansion"
OUT = ROOT / "explainability"


def coefficients(kernel, prior, labels):
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    root = np.sqrt(len(labels) / (len(counts) * counts[inverse]))
    target = 2 * np.eye(8)[labels] - 1
    system = root[:, None] * kernel * root[None, :] + 8 * np.eye(len(labels))
    return root[:, None] * np.linalg.solve(system, root[:, None] * (target - prior))


def softmax(score):
    value = np.exp(score - score.max(axis=1, keepdims=True))
    return value / value.sum(axis=1, keepdims=True)


def reconstruct(a, q, s, arm):
    z, b, r = a["expert"], a["base"], a["relation"]
    kss, kqs = z[s] @ z[s].T, z[q] @ z[s].T
    if arm != "original":
        local = a["ship"] if arm == "ship" else np.concatenate(
            [np.sqrt(.5) * a["ship"], np.sqrt(.5) * a["ring"]], axis=1)
        kss += .25 * (16 * local[s] @ local[s].T - z[s, 128:256] @ z[s, 128:256].T)
        kqs += .25 * (16 * local[q] @ local[s].T - z[q, 128:256] @ z[s, 128:256].T)
    kss = .75 * kss + .25 * ((r[s] * a["weights"]) @ r[s].T)
    kqs = .75 * kqs + .25 * ((r[q] * a["weights"]) @ r[s].T)
    alpha = coefficients(kss, a["prior_e"][s], a["labels"][s])
    score = a["prior_e"][q] + kqs @ alpha
    base_alpha = coefficients(b[s] @ b[s].T, a["prior_b"][s], a["labels"][s])
    base_score = a["prior_b"][q] + (b[q] @ b[s].T) @ base_alpha
    return dict(kernel=kqs, alpha=alpha, score=score, prob=.5 * (softmax(score) + softmax(base_score)))


def main():
    OUT.mkdir(exist_ok=True)
    frozen = json.loads((ROOT / "frozen.json").read_text(encoding="utf-8"))
    cases, support_rows, checks = [], [], []
    with threadpool_limits(limits=2):
        for pool in frozen["pools"]:
            if not pool["selected"]:
                continue
            slug = pool["port"].replace(" ", "_")
            folder = ROOT / slug
            with np.load(folder / "readout.npz") as cache:
                a = {key: np.asarray(cache[key], dtype=np.int64 if key == "labels" else np.float64)
                     for key in cache.files if key in ("base", "expert", "prior_b", "prior_e", "relation", "ship", "ring", "weights", "labels")}
            meta = pd.read_csv(folder / "metadata.csv", dtype={"mmsi": str})
            tasks = json.loads((folder / "tasks.json").read_text(encoding="utf-8"))
            tasks = [t for t in tasks if t["budget"] == 64]
            if not tasks:
                continue
            archive = pd.read_csv(folder / "predictions_B64.csv", dtype={"mmsi": str})
            ship_archive = pd.read_csv(OUT / f"{slug}_ship_B64_predictions.csv", dtype={"mmsi": str})
            for task_id, task in enumerate(tasks):
                if task["origin"] != "cross":
                    continue
                q, s = np.asarray(task["q"]), np.asarray(task["s"])
                result = {arm: reconstruct(a, q, s, arm) for arm in ("original", "ship", "additive")}
                for arm, output in result.items():
                    records = ship_archive if arm == "ship" else archive
                    records = records.loc[records.task.eq(task_id) & records.arm.eq(arm)].set_index("sample_id")
                    records = records.loc[meta.iloc[q].sample_id]
                    saved = records[[f"p{i}" for i in range(8)]].to_numpy()
                    error = np.max(np.abs(output["prob"] - saved))
                    np.testing.assert_allclose(output["prob"], saved, atol=1e-10, rtol=1e-10)
                    checks.append(dict(port=pool["port"], task=task_id, arm=arm, probability_error=float(error)))
                for before, after in (("original", "ship"), ("original", "additive"), ("ship", "additive")):
                    old, new = result[before], result[after]
                    oldpred, newpred = old["prob"].argmax(1), new["prob"].argmax(1)
                    truth = a["labels"][q]
                    flip = (oldpred == truth) != (newpred == truth)
                    for qi in np.flatnonzero(flip):
                        cls = int(truth[qi])
                        rescued = newpred[qi] == cls
                        rival = int(oldpred[qi] if rescued else newpred[qi])
                        a0 = old["alpha"][:, cls] - old["alpha"][:, rival]
                        a1 = new["alpha"][:, cls] - new["alpha"][:, rival]
                        k0, k1 = old["kernel"][qi], new["kernel"][qi]
                        direct = (k1 - k0) * (a0 + a1) / 2
                        refit = (k0 + k1) * (a1 - a0) / 2
                        contribution = direct + refit
                        score0 = old["score"][qi, cls] - old["score"][qi, rival]
                        score1 = new["score"][qi, cls] - new["score"][qi, rival]
                        np.testing.assert_allclose(contribution.sum(), score1-score0, atol=1e-10, rtol=1e-10)
                        case_id = f"{slug}_{task_id}_{qi}_{after}_vs_{before}"
                        item = dict(case_id=case_id, port=pool["port"], fold=task["fold"], draw=task["draw"],
                            sample_id=meta.iloc[q[qi]].sample_id, mmsi=meta.iloc[q[qi]].mmsi,
                            before=before, after=after, event="rescue" if rescued else "harm", truth=cls, rival=rival,
                            predicted_before=int(oldpred[qi]), predicted_after=int(newpred[qi]),
                            true_prob_before=old["prob"][qi, cls], true_prob_after=new["prob"][qi, cls],
                            fused_margin_before=old["prob"][qi, cls]-old["prob"][qi, rival],
                            fused_margin_after=new["prob"][qi, cls]-new["prob"][qi, rival],
                            expert_margin_before=score0, expert_margin_after=score1,
                            direct_kernel_effect=direct.sum(), coefficient_refit_effect=refit.sum(),
                            total_effect=contribution.sum(), accounting_error=float(abs(contribution.sum()-(score1-score0))))
                        cases.append(item)
                        for si, row in enumerate(s):
                            support_rows.append(dict(case_id=case_id, support_id=meta.iloc[row].sample_id,
                                support_mmsi=meta.iloc[row].mmsi, support_class=int(a["labels"][row]),
                                similarity_before=k0[si], similarity_after=k1[si],
                                coefficient_margin_before=a0[si], coefficient_margin_after=a1[si],
                                direct_kernel_effect=direct[si], coefficient_refit_effect=refit[si], total_effect=contribution[si]))
    cases = pd.DataFrame(cases)
    support_rows = pd.DataFrame(support_rows)
    cases.to_csv(OUT / "case_events.csv", index=False)
    support_rows.to_csv(OUT / "case_support_contributions.csv", index=False)
    pd.DataFrame(checks).to_csv(OUT / "case_replay_checks.csv", index=False)
    selected = cases.sort_values(["port", "fold", "draw", "sample_id"]).groupby(["before", "after", "event"], sort=True).head(1)
    selected.to_csv(OUT / "case_examples.csv", index=False)
    rows = ["# 改判案例的精确支持贡献", "", "固定B64跨采集10任务、五次支持抽样；全部救回/误伤均保存，示例按港口、采集、抽样、样本ID顺序取首个，未按最大收益挑选。",
        "", "## 怎样解释", "",
        "固定真实类t和错分竞争类r，用专家分差的变化进行记账。源先验与基础分支不变。对第i个支持案例，令a_i为它对t-r的校正系数，k_i为查询-支持核值，则：",
        "", "`k1*a1 - k0*a0 = (k1-k0)*(a0+a1)/2 + (k0+k1)/2*(a1-a0)`", "",
        "第一项记为核关系变化，第二项记为系数重求解变化。二者逐支持求和精确还原专家分差变化；它是对称的代数分解，不是两种独立物理因果。最终概率经过softmax与基础分支融合，不能把该分数贡献直接叫概率贡献。",
        "", f"已核对{len(checks)}组NumPy独立重构概率与保存预测；最大误差{max(r['probability_error'] for r in checks):.3g}。{len(cases)}个改判事件的分差分解最大误差{cases.accounting_error.max():.3g}。同一船在不同抽样/比较中可重复出现。"]
    names = {"original": "原模型", "ship": "仅框内补充", "additive": "框内+环带"}
    for c in selected.to_dict("records"):
        top = support_rows.loc[support_rows.case_id.eq(c["case_id"])].copy()
        positive_sum = top.loc[top.total_effect.gt(0), "total_effect"].sum()
        negative_sum = top.loc[top.total_effect.lt(0), "total_effect"].sum()
        top = pd.concat([top.loc[top.total_effect.gt(0)].nlargest(3, "total_effect"),
                         top.loc[top.total_effect.lt(0)].nsmallest(3, "total_effect")])
        rows.extend(["", f"## {names[c['before']]} → {names[c['after']]}：{'救回' if c['event']=='rescue' else '误伤'}", "",
            f"港口{c['port']}，样本`{c['sample_id']}`，MMSI `{c['mmsi']}`，抽样{c['draw']}。真实类{c['truth']}，预测{c['predicted_before']}→{c['predicted_after']}。",
            f"真实类最终概率{100*c['true_prob_before']:.3f}%→{100*c['true_prob_after']:.3f}%；相对竞争类{c['rival']}的融合概率差{100*c['fused_margin_before']:.3f}→{100*c['fused_margin_after']:.3f}个百分点。",
            f"专家分差变化{c['total_effect']:+.6f} = 核关系项{c['direct_kernel_effect']:+.6f} + 系数重求解项{c['coefficient_refit_effect']:+.6f}。",
            f"全部支持贡献中，正项合计{positive_sum:+.6f}，负项合计{negative_sum:+.6f}。",
            "", "正向和负向贡献各最大的三个支持案例（正值推向真实类，负值推向竞争类）：", "",
            "| 支持ID | 类别 | 核值前→后 | 核关系项 | 系数项 | 合计 |", "| --- | --- | --- | --- | --- | --- |"])
        for x in top.to_dict("records"):
            rows.append(f"| {x['support_id']} | {x['support_class']} | {x['similarity_before']:.4f}→{x['similarity_after']:.4f} | {x['direct_kernel_effect']:+.5f} | {x['coefficient_refit_effect']:+.5f} | {x['total_effect']:+.5f} |")
        rows.append("只列正负方向各三个，不能用这六项代替全部64个支持贡献总和。")
    ship_cases = cases.loc[cases.before.eq("original") & cases.after.eq("ship")]
    opposed = int((ship_cases.direct_kernel_effect * ship_cases.coefficient_refit_effect < 0).sum())
    rows.extend(["", "## 全部框内补充改判的共同现象", "",
        f"原模型到仅框内补充的{len(ship_cases)}个改判事件中，有{opposed}个的核关系项与系数重求解项方向相反。这个代数现象表明最终变化经常包含抵消，不能只看相似度或最近邻排序来预测分类结果。它不是两种可独立干预因素的因果比例，也不能外推到未改判样本。"])
    rows.extend(["", "## 解释边界", "",
        "表中相似度是包含原特征与类别关系的完整专家核，不是纯船体或纯背景相似度。少数支持贡献可相互抵消，标签相同不保证贡献为正。例子解释这些已发生的改判，不证明所有未见船都会遵循同一机制，也不从抽象通道给出未经验证的散射物理名称。",
        "", "全部事件和64支持贡献保留在case_events.csv、case_support_contributions.csv；总体有效性须回到FINDINGS_ZH.md中的全量配对统计，而不是单个成功案例。"])
    (OUT / "CASE_FINDINGS_ZH.md").write_text("\n".join(rows)+"\n", encoding="utf-8")
    print(cases.groupby(["before", "after", "event"]).size().to_string())
    print(selected[["before", "after", "event", "port", "sample_id", "total_effect", "direct_kernel_effect", "coefficient_refit_effect"]].to_string(index=False))
    print("max replay", max(r["probability_error"] for r in checks), "accounting", cases.accounting_error.max())


if __name__ == "__main__":
    main()
