"""Summarize the frozen experiment without selecting methods on target labels."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import recall_score
from body_context_gate import Gate

ROOT = Path(__file__).resolve().parent / "body_context"


def markdown(frame):
    display = frame.copy()
    for name in display.select_dtypes(include="number").columns:
        display[name] = display[name].map(lambda x: f"{x:.4f}")
    return "| " + " | ".join(display.columns) + " |\n| " + " | ".join(
        ["---"] * len(display.columns)) + " |\n" + "\n".join(
            "| " + " | ".join(map(str, row)) + " |" for row in display.itertuples(index=False, name=None))


def read_many(pattern):
    files = sorted(ROOT.glob(pattern))
    return pd.concat([pd.read_csv(f).assign(pool=f.parent.name) for f in files], ignore_index=True)


def main():
    target = read_many("*/target_metrics.csv")
    preds = read_many("*/target_predictions.csv")
    source = pd.concat([read_many("*/baselines_selection.csv"),
                        read_many("*/static_seed[012]_selection.csv"),
                        read_many("*/conditional_seed[012]_selection.csv")], ignore_index=True)
    keys = ["pool", "fold", "task", "arm", "seed", "shuffle"]
    # This check detects metric/row misalignment; predictions are the independent input.
    check = preds.groupby(keys).apply(lambda x: pd.Series(dict(
        oa=100 * (x.truth == x.pred).mean(),
        ba=100 * recall_score(x.truth, x.pred, labels=np.unique(x.truth),
                             average="macro", zero_division=0))), include_groups=False).reset_index()
    merged = target.merge(check, on=keys, validate="one_to_one", suffixes=("", "_check"))
    np.testing.assert_allclose(merged[["oa", "ba"]], merged[["oa_check", "ba_check"]], atol=1e-10)
    real = target.loc[target.shuffle.eq(-1)]
    per_fold = real.groupby(["fold", "origin", "arm"])[["oa", "ba", "macro_nll", "macro_brier", "macro_map_r"]].mean()
    summary = per_fold.groupby(["origin", "arm"]).mean().reset_index()
    summary.to_csv(ROOT / "target_summary.csv", index=False)
    per_fold.to_csv(ROOT / "target_per_fold.csv")
    src = source.loc[source.shuffle.eq(-1)].groupby(["pool", "origin", "arm"])[["oa", "ba"]].mean()
    src.to_csv(ROOT / "source_per_pool.csv")
    src.groupby(["origin", "arm"]).mean().to_csv(ROOT / "source_summary.csv")

    checkpoints, diagnostics, coverage = [], [], []
    for folder in sorted(ROOT.iterdir()):
        if not folder.is_dir():
            continue
        episodes = json.loads((folder / "episodes.json").read_text(encoding="utf-8"))
        meta = pd.read_csv(folder / "metadata.csv", dtype={"mmsi": str})
        for split in ("fit", "selection"):
            tasks = episodes[split]
            q = sorted({i for t in tasks for i in t["q"]})
            coverage.append(dict(pool=folder.name, split=split, tasks=len(tasks), qchips=len(q),
                                 qships=meta.iloc[q].mmsi.nunique(), classes=meta.iloc[q].class_id.nunique(),
                                 acquisitions=meta.iloc[q].acquisition.nunique(), ports=meta.iloc[q].port.nunique()))
        for mode in ("static", "conditional"):
            for seed in range(3):
                item = torch.load(folder / f"{mode}_seed{seed}.pt", weights_only=False)
                checkpoints.append(dict(pool=folder.name, mode=mode, seed=seed,
                                        step=item["best_step"], source_ba=item["selection_ba"], seconds=item["seconds"]))
                tail_path = folder / f"{mode}_seed{seed}_last.pt"
                if tail_path.exists():
                    tail = torch.load(tail_path, weights_only=False)
                    diag = dict(tail["diagnostics"])
                    with np.load(folder / "readout.npz") as cache:
                        source_positions = pd.read_csv(folder / "source_split.csv").position.to_numpy(int)
                        ring = torch.tensor(cache["ring"][source_positions], dtype=torch.float64)
                    gate = Gate(mode).double()
                    gate.load_state_dict(tail["state"])
                    with torch.no_grad():
                        logits = gate.c[None, :].expand(len(ring), -1)
                        if mode == "conditional":
                            logits = ring @ gate.W.T + gate.c
                        adjustment = .25 * torch.tanh(logits)
                        diag.update(adjustment_abs_mean=float(adjustment.abs().mean()),
                                    adjustment_abs_max=float(adjustment.abs().max()),
                                    adjustment_across_sample_std=float(adjustment.std(0).mean()))
                    diagnostics.append(dict(pool=folder.name, mode=mode, seed=seed, **diag))
    checkpoints = pd.DataFrame(checkpoints)
    checkpoints.to_csv(ROOT / "checkpoints.csv", index=False)
    diagnostics = pd.DataFrame(diagnostics)
    diagnostics.to_csv(ROOT / "training_diagnostics.csv", index=False)
    pd.DataFrame(coverage).to_csv(ROOT / "source_task_coverage.csv", index=False)

    pairs = ["pool", "fold", "origin", "draw", "sample_id"]
    baseline = preds.loc[preds.arm.eq("original") & preds.shuffle.eq(-1), pairs + ["pred", "truth"]]
    changed = preds.loc[preds.shuffle.eq(-1)].merge(baseline, on=pairs, validate="many_to_one", suffixes=("", "_base"))
    assert (changed.truth == changed.truth_base).all()
    changed["rescued"] = (changed.pred != changed.pred_base) & (changed.pred == changed.truth)
    changed["harmed"] = (changed.pred != changed.pred_base) & (changed.pred_base == changed.truth)
    flips = changed.groupby(["fold", "origin", "arm", "seed", "truth"])[["rescued", "harmed"]].sum()
    flips.to_csv(ROOT / "target_class_flips.csv")
    changed.loc[changed.rescued | changed.harmed].to_csv(ROOT / "target_changed_cases.csv", index=False)
    # Tail vs additive describes actual learned perturbations before checkpoint selection.
    tail_rows = []
    for f in sorted(ROOT.glob("*/*seed[12]_last_predictions.csv")):
        tail = pd.read_csv(f)
        base = pd.read_csv(f.parent / "baselines_selection_predictions.csv")
        base = base.loc[base.arm.eq("additive")]
        joined = tail.merge(base, on=["task", "sample_id"], suffixes=("", "_add"), validate="one_to_one")
        prob = joined[[f"p{c}" for c in range(8)]].to_numpy()
        original_prob = joined[[f"p{c}_add" for c in range(8)]].to_numpy()
        row_l1 = np.abs(prob-original_prob).sum(1)
        tail_rows.append(dict(pool=f.parent.name, mode=tail.arm.iloc[0], seed=int(tail.seed.iloc[0]),
            mean_probability_l1=row_l1.mean(), max_probability_l1=row_l1.max(),
            decision_changes=int(np.sum(joined.pred != joined.pred_add)),
            mean_margin_change=float((joined.margin-joined.margin_add).mean())))
    tail_df = pd.DataFrame(tail_rows)
    tail_df.to_csv(ROOT / "tail_effects.csv", index=False)
    cross = per_fold.xs("cross", level="origin").reset_index()
    delta = cross.pivot(index="fold", columns="arm", values="ba")
    for name in ("ship", "ring", "additive", "static", "conditional"):
        delta[name] = delta[name] - delta.original
    delta = delta.drop(columns="original").reset_index()
    delta.to_csv(ROOT / "cross_ba_deltas.csv", index=False)
    seed_summary = real.loc[real.seed.ge(0)].groupby(["seed", "fold", "origin", "arm"])[["oa", "ba"]].mean()
    seed_summary = seed_summary.groupby(["seed", "origin", "arm"]).mean()
    seed_summary.to_csv(ROOT / "target_by_training_seed.csv")
    report = ["# 船体/环带实验结果", "", "2026-10-15。已完成源端训练与冻结目标评价，未替换主模型。",
        "", "## 当前判断", "",
        "保留环带独立补充这个低成本候选，不把本轮条件加权升级为新模块。共同底座上，环带跨采集OA/BA为52.433/46.405%，相对原模型52.204/46.037%提高0.229/0.368个百分点；三个任务BA正向、三个负向，不要求全面获胜。",
        "",
        "船体+环带等权组合跨采集BA下降0.316个百分点，同采集BA提高0.521个百分点，说明存在用途边界，并非组合一律无效。固定/条件加权均没有在目标任务上额外改变OA/BA；15次条件训练14次选0步，仅Singapore源池种子0选600步，源端多救回一个查询，另两个种子未复现。这里Singapore指对应留出模型的源验证池，不是用Singapore目标标签选择。",
        "", "## 协议", "",
        "五个source-train-only共同底座，六个已使用的目标采集，B16、同/跨采集各五次支持抽样。所有对照共享底座、PCA、类别关系、先验及50%融合。目标标签未选择模块参数/检查点。此处基线不是历史train+val基线，不能混减。",
        "", "## 完整模型结果", "", markdown(summary),
        "", "跨采集逐折BA相对原模型变化，单位百分点：", "", markdown(delta),
        "", "环带相对原模型在跨采集五次支持抽样中累计27次救回、14次误伤。这些是重复任务中的改判次数，不是41艘独立船。类别3净救回6次、类别4净救回5次，其他类有小幅正负变化；见逐类与逐船明细，不将多数类收益一票否决。",
        "", "本轮MAP@R使用完整第二路支持核，只对支持中存在对应类别的查询计算，并保存覆盖数量；不与此前末层GAP/浅层混合检索的绝对值直接混比。目标三个训练种子的OA/BA完全相同，标准差0；这是多数检查点退回加性方案、唯一活动门控未改变top-1的结果，不代表训练出的动态机制稳定有效。",
        "", "## 实际源任务覆盖", "", markdown(pd.DataFrame(coverage)),
        "", "任务数包含同/跨采集与重复支持抽样，不能当作独立采集数。原验证池八类齐全不代表实际可组成的查询任务仍有八类；上表按真正参与的查询统计。各模型池也不是彼此独立数据集。",
        "", "## 学到了什么", "", markdown(checkpoints),
        "", "训练末态的固定任务NLL和参数变化（种子1/2保留末态；种子0仅存预定选择结果）：", "",
        markdown(diagnostics.groupby("mode")[["initial_fit_nll", "last_fit_nll", "adjustment_abs_mean", "adjustment_across_sample_std"]].mean().reset_index()),
        "", "训练末态相对直接融合的实际概率扰动：", "",
        markdown(tail_df.groupby("mode")[["mean_probability_l1", "max_probability_l1", "decision_changes"]].mean().reset_index()),
        "", "门控只改变局部特征中四分之一的归一化船体项。经过原D18替换与类别核混合后，其可变点积在专家核中的系数为0.75*0.25*16*0.25=0.75。它还受到支持ridge求解和双路概率融合影响；不能把这个系数直接当作最终准确率上限。小模块确实可能学动，但对最终概率的影响很小。当前证据不能区分扩大作用范围后有用，还是这组条件信息与坐标重加权本来不匹配。",
        "", "## 解释边界与实现记录", "",
        "零初始化与直接融合等价、支持缺类权重、独立显式ridge回归、有限差分梯度共七项测试通过；真实任务NumPy/Torch概率也核对通过。目标表OA/BA由保存的逐查询预测独立重算通过。",
        "",
        "准备阶段发现全局行号/压缩行号混用，两个受影响缓存已在训练前重建；旧错误缓存未用于训练评价。源端按BA相同时选最早检查点，选择0意味着该训练并未获得可选择的BA提升，不代表没有梯度。",
        "",
        "本轮没有增加骨干前向或重训骨干。三个训练种子是支持任务次序的独立随机重复，零初始化本身相同；五次支持抽样不是五组独立船。环带未验证为纯水域，不能命名为纯因果校正。所选零门控的置乱不变是数学结果，不是新的机制证据。",
        "", f"实际完成30次小模块训练，记录的训练/选择/诊断总时长{checkpoints.seconds.sum():.1f}秒；另有五次共同底座重建。尚未单独测量从原图提取环带到最终预测的新增端到端延迟，不能声称已经证明零成本。",
        "",
        "明细见target_class_flips.csv、target_changed_cases.csv、tail_effects.csv及各源池selection/target预测。未自动调大门控幅度或更换优化目标追求正结果。"]
    (ROOT / "FINDINGS_ZH.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print("CHECKPOINTS\n", checkpoints.groupby(["mode", "step"]).size().to_string())
    print("DELTA\n", delta.to_string(index=False))


if __name__ == "__main__":
    main()
