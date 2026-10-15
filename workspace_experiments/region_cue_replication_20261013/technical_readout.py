"""Quantify archived-cache versus same-pipeline original decisions, without new fits."""

from replication_context import *


def main():
    rows, coverage = [], []
    for name, fold in all_tasks():
        folder = OUT / "predictions" / name / f"fold_{fold}"
        receipt = json.loads((folder / "COMPLETE.json").read_text(encoding="utf-8"))
        data = rc.run.archive(folder / "cohort.npz")
        frame = pd.read_csv(folder / "cohort.csv", keep_default_na=False)
        control = pd.read_csv(folder / "regions.csv")
        original = np.load(folder / "baseline.npy", mmap_mode="r")
        archive = np.load(folder / "archived_baseline.npy", mmap_mode="r")
        labels = frame.class_id.to_numpy(int)
        for draw, mask in enumerate(data["eval_masks"]):
            mask = mask & control.region_valid.to_numpy(bool)
            y = labels[mask]
            w = rc.weights(y)
            for arm, index in (("R", 0), ("T", 1)):
                p = original[draw, index][mask].argmax(-1)
                q = archive[draw, index][mask].argmax(-1)
                rows.append(dict(dataset=receipt["dataset"], task=name, fold=fold, draw=draw,
                    arm=arm, n=len(y), flipped=int((p != q).sum()),
                    delta_oa=100*float(np.mean(p == y)-np.mean(q == y)),
                    delta_ba=float(w[p == y].sum()-w[q == y].sum())))
        coverage.append(dict(task=name, fold=fold, rows=len(frame),
            invalid_support_counts=receipt["invalid_support_counts"],
            valid_queries=receipt["valid_queries"],
            legacy_max_probability_error=max(r["max_probability_error"] for r in receipt["legacy_replays"])))
    dst = OUT / "summary_all"
    dst.mkdir(exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(dst / "technical_original_contexts.csv", index=False)
    task = table.groupby(["dataset", "task", "arm"])[["delta_oa", "delta_ba"]].mean().reset_index()
    summary = task.groupby(["dataset", "arm"])[["delta_oa", "delta_ba"]].mean().reset_index()
    summary.to_csv(dst / "technical_original_dataset_means.csv", index=False)
    write(dst / "technical_coverage.json", coverage)
    print(summary.to_string(index=False))
    print("MAX_CONTEXT_ABS_BA", table.delta_ba.abs().max())
    print("LEGACY_MAX_PROBABILITY_ERROR", max(r["legacy_max_probability_error"] for r in coverage))


if __name__ == "__main__":
    main()
