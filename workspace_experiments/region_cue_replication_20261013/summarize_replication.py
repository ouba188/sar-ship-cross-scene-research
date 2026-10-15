"""Average within task before across ports; never pool images as ports."""

from replication_context import *
import argparse


def equal_task_summary(frame, keys, values):
    task = frame.groupby(["dataset", "task", *keys], dropna=False)[values].mean().reset_index()
    summary = task.groupby(["dataset", *keys], dropna=False)[values].mean().reset_index()
    return task, summary


def summarize(scope):
    tasks = [(n, f) for n, f in all_tasks() if scope == "all" or
             ((n in ("oss1", "oss2")) == (scope == "public"))]
    dst = OUT / ("summary_"+scope)
    dst.mkdir(parents=True, exist_ok=True)
    tables = {}
    for filename, keys in (("path_effects", ["scope", "condition", "path"]),
                           ("utility", ["scope", "condition", "mode"]),
                           ("classes", ["scope", "condition", "path", "class_id"])):
        frames = []
        for name, fold in tasks:
            folder = OUT / "predictions" / name / f"fold_{fold}"
            if not (folder / "READOUT_COMPLETE.json").exists():
                raise RuntimeError("Incomplete readout: " + str(folder))
            frames.append(pd.read_csv(folder / (filename+".csv")))
        frame = pd.concat(frames, ignore_index=True)
        values = [c for c in frame if c not in ["dataset", "task", "fold", "draw", *keys]]
        task, summary = equal_task_summary(frame, keys, values)
        frame.to_csv(dst / (filename+".csv"), index=False)
        task.to_csv(dst / (filename+"_task_means.csv"), index=False)
        summary.to_csv(dst / (filename+"_dataset_means.csv"), index=False)
        tables[filename] = frame
    f = tables["path_effects"]
    contrasts = []
    for (dataset, task, source_fold, draw, subgroup, path), a in f.groupby(["dataset", "task", "fold", "draw", "scope", "path"]):
        a = a.set_index("condition")
        for first, second in (("target_shuffle", "outer_shuffle"), ("outer_display", "target_display"),
                              ("target_flat", "sham_flat"), ("wide_target_flat", "target_flat")):
            contrasts.append(dict(dataset=dataset, task=task, fold=source_fold, draw=draw, scope=subgroup,
                path=path, contrast=first+"-"+second,
                delta_ba=float(a.loc[first, "delta_ba"]-a.loc[second, "delta_ba"]),
                old_rescue_lost=float(a.loc[first, "old_rescue_lost"]-a.loc[second, "old_rescue_lost"])))
    c = pd.DataFrame(contrasts)
    c.to_csv(dst / "paired_contrasts.csv", index=False)
    keys = ["scope", "path", "contrast"]
    task, summary = equal_task_summary(c, keys, ["delta_ba", "old_rescue_lost"])
    task.to_csv(dst / "contrast_task_means.csv", index=False)
    summary.to_csv(dst / "contrast_dataset_means.csv", index=False)
    counts = task.groupby(["dataset", *keys]).agg(tasks=("delta_ba", "size"),
        negative_tasks=("delta_ba", lambda a: int((a < -1e-10).sum())),
        positive_tasks=("delta_ba", lambda a: int((a > 1e-10).sum())),
        minimum=("delta_ba", "min"), maximum=("delta_ba", "max")).reset_index()
    counts.to_csv(dst / "contrast_task_directions.csv", index=False)
    context_counts = c.groupby(["dataset", *keys]).agg(contexts=("delta_ba", "size"),
        negative_contexts=("delta_ba", lambda a: int((a < -1e-10).sum())),
        positive_contexts=("delta_ba", lambda a: int((a > 1e-10).sum()))).reset_index()
    context_counts.to_csv(dst / "contrast_context_directions.csv", index=False)
    utility = tables["utility"].copy()
    utility["net_change"] = utility.net-utility.original_net
    task, summary = equal_task_summary(utility, ["scope", "condition", "mode"], ["net", "net_change", "r_ba", "t_ba"])
    task.to_csv(dst / "net_task_means.csv", index=False)
    summary.to_csv(dst / "net_dataset_means.csv", index=False)
    print(pd.read_csv(dst / "path_effects_dataset_means.csv").query(
        "scope == 'full' and path in ['T_Aonly','T_Bonly'] and condition in ['target_shuffle','outer_shuffle','target_display','outer_display']"
        )[["dataset", "condition", "path", "delta_ba"]].to_string(index=False), flush=True)
    print(summary.query("scope == 'full' and mode == 'joint'").to_string(index=False), flush=True)
    write(dst / "COMPLETE.json", dict(task_fold_units=len(tasks), tasks=len(set(n for n, f in tasks)),
        port_summary="Equal port means after averaging its fixed five supports",
        public_summary="Each dataset mean across five source folds x five supports",
        independent_training_seed_inference=False, rows={k: len(v) for k, v in tables.items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scope", choices=("public", "ports", "all"))
    summarize(parser.parse_args().scope)
