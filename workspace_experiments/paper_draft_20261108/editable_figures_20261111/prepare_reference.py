"""Prepare historical reference-bank figure data; never train or infer.

Only saved metrics, query IDs/labels, manifests and source text are read.
Outputs are restricted to reference_data next to this script.
"""

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
ROOT = WORKSPACE / "scene_aware_classification_20261018/model_closeout_20261023"
STUDY = ROOT / "m3_reference_coverage_20261030"
OUT = HERE / "reference_data"
REQUESTED = [1, 2, 4, 8, 16]
STUDY_ID = "m3_reference_coverage_20261030_anchor_corrected"
OLD = ROOT / "local_evidence_transport/conditional_relation_operator/reference_resampling"
CONFIGS = {
    "private": {
        "dataset_label": "Private historical development",
        "manifest_root": WORKSPACE / "sar_m123_current_20261027/manifests",
        "cohort": "development",
        "n_queries": 160,
        "n_model_classes": 8,
        "model_id": "spatial_m12_full_20261029_spatial_s17",
        "checkpoint": ROOT / "spatial_m12_full_20261029/training/spatial_s17/model.pt",
    },
    "oss2": {
        "dataset_label": "OpenSARShip-2 historical validation",
        "manifest_root": ROOT / "final_oss2_replication_20261029/manifests",
        "cohort": "validation",
        "n_queries": 228,
        "n_model_classes": 4,
        "model_id": "final_oss2_replication_20261029_full_s17_adapt",
        "checkpoint": ROOT / "final_oss2_replication_20261029/training/full_s17/adapt/model.pt",
    },
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_location(path, marker):
    for line, text in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if marker in text:
            return {"path": path.as_posix(), "line": line, "excerpt": text.strip()}
    raise ValueError(f"Missing source evidence: {path}: {marker}")


def main():
    long_rows, members, groups = [], [], defaultdict(list)
    provenance = {
        "study_id": STUDY_ID,
        "prepared_on": "2026-11-11",
        "figure_scope": "Historical candidate diagnostic, NOT current RCEW sensitivity",
        "requested_references_per_class": REQUESTED,
        "metric_units": "percent",
        "metric_definitions": {
            "OA": "100 * correct queries / all queries",
            "BA": "100 * mean class recall over labels present in the query cohort",
        },
        "replicate_definition": "One saved model-order-seed and reference-bank context per condition",
        "sd_definition": "Sample SD (ddof=1); null/empty for n<2, never replaced with zero",
        "aggregation": "Separate dataset, model, cohort, bank strategy and references per class",
        "execution": {
            "new_training": False,
            "new_inference": False,
            "checkpoint_loaded": False,
            "score_arrays_read": False,
            "cached_npz_fields_read": ["ids", "y"],
            "output_directory": OUT.as_posix(),
        },
        "historical_intervention": {
            "training_during_bank_replacement": False,
            "model_parameters": "Previously trained endpoints, frozen during the bank probe",
            "model_order_seeds_observed": [17],
            "random_bank_seed": 1030,
            "random_bank_sampling": "NumPy default_rng(1030), reset per select() call; select a random eligible observation within each class, excluding already selected MMSIs in that class. Not uniform over ships.",
            "coverage_bank_sampling": "Deterministic greedy cosine facility coverage; inverse-MMSI-count weights; distinct ships within each class; protected anchor_last descriptors",
            "original_bank": "Inherited manifest; no new draw. Original construction seed is not asserted here.",
            "query_pair_exclusion": "Same observation ID or same nonempty MMSI is masked by pair_mask",
            "bank_size_definition": "Stored source reference observations per class, not per-polarization valid count and not target labels",
        },
        "exclusions": [
            {"source": (STUDY / "private").as_posix(), "reason": "Legacy uncorrected deep-descriptor statistics; use anchor_corrected only"},
            {"source": (STUDY / "oss2").as_posix(), "reason": "Legacy uncorrected deep-descriptor statistics; use anchor_corrected only"},
            {"source": (STUDY / "FINDINGS_ZH.md").as_posix(), "reason": "Full-cohort three-order center/control training changes model parameters; it is not the frozen bank-size probe"},
            {"source": (OLD / "FINDINGS_ZH.md").as_posix(), "reason": "2026-10-24 uses a different conditional-relation candidate, fixed 8/class (64 total), banks original plus 101/307/509/711/907; includes a separate episodic-training study. No additional size point for this curve."},
        ],
        "datasets": {},
        "code_evidence": {},
    }
    evidence = {
        "bank_seed": (STUDY / "probe.py", "rng=np.random.default_rng(1030)"),
        "protected_anchor": (STUDY / "probe.py", "return net.anchor_last"),
        "bank_sizes": (STUDY / "probe.py", "for n in (k,2*k)"),
        "frozen_parameters": (STUDY / "probe.py", "net.requires_grad_(False)"),
        "private_queries": (STUDY / "probe.py", "queries=cache.rows['development']"),
        "public_queries": (STUDY / "probe.py", "queries=groups['validation']"),
        "private_manifests": (ROOT / "spatial_m12_refit_20261029/run.py", "rows = {k: read_manifest"),
        "public_manifest_root": (ROOT / "m3_reference_delta_20261030/public.py", "PUB=ROOT/"),
        "metric_definition": (ROOT / "m3_reference_delta_20261030/run.py", "def metrics(score, y)"),
        "identity_mask": (WORKSPACE / "sar_m123_standalone/sar_m123/current/data.py", "def pair_mask"),
    }
    provenance["code_evidence"] = {k: source_location(*v) for k, v in evidence.items()}

    for dataset, cfg in CONFIGS.items():
        result_path = STUDY / "anchor_corrected" / dataset / "results_s17.json"
        results = read_json(result_path)
        manifest_root = cfg["manifest_root"]
        fit = read_json(manifest_root / "source_fit.json")
        queries = read_json(manifest_root / f"{cfg['cohort']}.json")
        original = read_json(manifest_root / "references.json")
        lookup = {str(r["id"]): r for r in fit}
        query_ids = [str(r["id"]) for r in queries]
        query_labels = [r["label"] for r in queries]
        query_counts = Counter(query_labels)
        class_labels = sorted({r["label"] for r in fit})
        require(len(queries) == cfg["n_queries"], f"{dataset}: changed query count")
        require(len(class_labels) == cfg["n_model_classes"], f"{dataset}: changed model taxonomy")
        endpoint_path = STUDY / "analysis/anchor_corrected" / dataset / "endpoints.csv"
        with endpoint_path.open(encoding="utf-8-sig", newline="") as stream:
            endpoint_rows = {r["arm"]: (line, r) for line, r in enumerate(csv.DictReader(stream), 2) if r["arm"].startswith("bank_") and r["seed"] == "17"}
        require(set(endpoint_rows) == {f"bank_{name}" for name in results}, f"{dataset}: mismatched CSV/JSON arms")
        details = {
            "model_id": cfg["model_id"],
            "checkpoint_provenance_only": cfg["checkpoint"].as_posix(),
            "cohort": cfg["cohort"],
            "n_queries": len(queries),
            "query_class_counts": dict(sorted(query_counts.items())),
            "n_classes_in_BA": len(query_counts),
            "n_model_classes": len(class_labels),
            "query_ids_in_order": query_ids,
            "query_manifest": (manifest_root / f"{cfg['cohort']}.json").as_posix(),
            "reference_manifest": (manifest_root / "references.json").as_posix(),
            "source_fit_manifest": (manifest_root / "source_fit.json").as_posix(),
            "source_fit_count": len(fit),
            "source_metrics": result_path.as_posix(),
            "corroborating_csv": endpoint_path.as_posix(),
            "banks": {},
        }
        for bank_name, bank in results.items():
            strategy = bank_name.split("_")[0]
            refs = [lookup[rid] for rid in bank["reference_ids"]]
            counts = Counter(r["label"] for r in refs)
            require(set(counts) == set(class_labels) and len(set(counts.values())) == 1,
                    f"{dataset}/{bank_name}: not equal references per class")
            per_class = next(iter(counts.values()))
            require(len(refs) == bank["reference_count"], f"{bank_name}: incorrect reference total")
            require(len({r["id"] for r in refs}) == len(refs), f"{bank_name}: repeated reference ID")
            if strategy == "original":
                require(bank["reference_ids"] == [str(r["id"]) for r in original], f"{dataset}: original manifest changed")
            else:
                require(int(bank_name.rsplit("_", 1)[1]) == per_class, f"{bank_name}: misleading size name")
                for c in class_labels:
                    require(len({str(r["mmsi"]) for r in refs if r["label"] == c}) == per_class,
                            f"{bank_name}: repeated within-class reference ship")
            prediction_path = result_path.parent / f"{bank_name}_s17.npz"
            with np.load(prediction_path, allow_pickle=False) as stored:
                require(stored["ids"].astype(str).tolist() == query_ids, f"{bank_name}: query IDs/order mismatch")
                require(stored["y"].tolist() == query_labels, f"{bank_name}: query labels mismatch")
            line, endpoint = endpoint_rows[f"bank_{bank_name}"]
            require(int(endpoint["n"]) == len(queries), f"{bank_name}: CSV query count mismatch")
            for metric in ("OA", "BA"):
                require(math.isclose(bank["metrics"][metric.lower()], float(endpoint[metric]), abs_tol=1e-10, rel_tol=0),
                        f"{bank_name}: JSON and CSV {metric} mismatch")
            eligible = [sum(str(q["id"]) != str(r["id"]) and not (
                q.get("mmsi") and r.get("mmsi") and str(q["mmsi"]) == str(r["mmsi"])) for r in refs) for q in queries]
            details["banks"][bank_name] = {
                "reference_ids": bank["reference_ids"],
                "actual_per_class_counts": dict(sorted(counts.items())),
                "reference_total": len(refs),
                "distinct_ship_count": len({str(r["mmsi"]) for r in refs}),
                "eligible_reference_count_range": [min(eligible), max(eligible)],
                "n_queries_with_identity_exclusion": sum(v < len(refs) for v in eligible),
                "queries_and_labels_match_manifest_exactly": True,
                "cached_prediction_file": prediction_path.as_posix(),
                "metrics_csv_line": line,
                "bank_draw_seed": 1030 if strategy == "random" else None,
                "draw_identity": "random_seed1030" if strategy == "random" else ("source_greedy_deterministic" if strategy == "coverage" else "inherited_original_manifest"),
            }
            context = {
                "study_id": STUDY_ID,
                "dataset": dataset,
                "dataset_label": cfg["dataset_label"],
                "model_id": cfg["model_id"],
                "cohort": cfg["cohort"],
                "strategy": strategy,
                "references_per_class": per_class,
                "reference_total": len(refs),
                "n_queries": len(queries),
                "n_model_classes": len(class_labels),
                "n_classes_in_ba": len(query_counts),
            }
            unit_id = f"{dataset}/{bank_name}/model_order17"
            for metric in ("OA", "BA"):
                long_rows.append({**context, "model_order_seed": 17,
                    "bank_draw_seed": 1030 if strategy == "random" else "",
                    "draw_identity": details["banks"][bank_name]["draw_identity"],
                    "unit_id": unit_id, "metric": metric, "value_pct": bank["metrics"][metric.lower()],
                    "training_during_bank_probe": False,
                    "source_json": result_path.as_posix(), "source_json_pointer": f"/{bank_name}/metrics/{metric.lower()}",
                    "source_csv": endpoint_path.as_posix(), "source_csv_line": line})
            groups[(dataset, cfg["model_id"], cfg["cohort"], strategy, per_class)].append((context, bank["metrics"]))
            for rank, r in enumerate(refs, 1):
                members.append({"dataset": dataset, "bank": bank_name, "model_order_seed": 17,
                    "references_per_class": per_class, "bank_position": rank, "reference_id": str(r["id"]),
                    "class_label": r["label"], "mmsi": r["mmsi"], "pass_group": r.get("pass_group", ""),
                    "vv_supplied": r.get("vv") is not None, "vh_supplied": r.get("vh") is not None})
        observed = sorted({k[-1] for k in groups if k[0] == dataset})
        details["observed_references_per_class"] = observed
        details["missing_requested_points"] = sorted(set(REQUESTED) - set(observed))
        details["coverage_by_strategy"] = {
            strategy: {"observed": sorted({k[-1] for k in groups if k[0] == dataset and k[3] == strategy}),
                       "missing": sorted(set(REQUESTED) - {k[-1] for k in groups if k[0] == dataset and k[3] == strategy})}
            for strategy in ("original", "random", "coverage")
        }
        provenance["datasets"][dataset] = details

    plot_rows = []
    for key, values in sorted(groups.items()):
        context = values[0][0]
        oa, ba = ([r[1][m] for r in values] for m in ("oa", "ba"))
        plot_rows.append({**context, "oa_mean": mean(oa), "oa_sd": stdev(oa) if len(oa) > 1 else "",
            "ba_mean": mean(ba), "ba_sd": stdev(ba) if len(ba) > 1 else "", "n": len(values),
            "sd_status": "estimable" if len(values) > 1 else "undefined_n1",
            "series_id": "/".join(str(x) for x in key[:-1]),
            "model_order_seeds": "17", "bank_draw_seed": 1030 if key[3] == "random" else "",
            "figure_scope": "historical_candidate_diagnostic"})
    provenance["verification"] = {
        "reference_members_and_class_counts_checked": True,
        "all_cached_query_ids_and_labels_match_manifests": True,
        "metrics_json_and_csv_agree": True,
        "distinct_condition_count": len(plot_rows),
        "long_metric_row_count": len(long_rows),
        "missing_points_have_no_numeric_rows": True,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv("reference_long.csv", long_rows)
    write_csv("reference_plot_ready.csv", plot_rows)
    write_csv("reference_bank_membership.csv", members)
    (OUT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_readme(plot_rows, provenance)
    print(json.dumps({"output": OUT.as_posix(), "conditions": len(plot_rows), "long_rows": len(long_rows),
        "missing": {k: v["missing_requested_points"] for k, v in provenance["datasets"].items()},
        "training": False, "inference": False}, ensure_ascii=False, indent=2))


def write_readme(rows, provenance):
    lines = [
        "# Historical Reference-Bank Size Diagnostic",
        "",
        "This is the 2026-10-30 corrected-anchor candidate diagnostic, NOT a complete sensitivity study of the current RCEW model.",
        "Only saved observations are exported. No training, inference, score-array reading, interpolation, imputation or mock data is performed.",
        "",
        "## Files",
        "",
        "- reference_long.csv: one row per observed condition and metric; exact JSON pointer and CSV line provenance.",
        "- reference_plot_ready.csv: one row per condition, OA/BA mean and sample SD, n, model/cohort/strategy identity.",
        "- reference_bank_membership.csv: actual bank member IDs, class labels, ships and supplied polarizations. Supplied views are not a guarantee of valid regional evidence.",
        "- provenance.json: actual class counts, exact query IDs, model paths (not loaded), sampling details, source-code locations and exclusions.",
        "",
        "## Observations",
        "",
        "| Dataset | Strategy | References/class | Total | OA (%) | BA (%) | n | SD |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['strategy']} | {r['references_per_class']} | {r['reference_total']} | {r['oa_mean']:.9f} | {r['ba_mean']:.9f} | {r['n']} | not estimable |")
    lines.extend([
        "",
        "## Interpretation And Plotting",
        "",
        "Use separate panels for private and OSS2. Keep original, random and coverage separate; series_id defines a connectable series. Only random and coverage have two size points per dataset. Original is a single marker at its actual size, not an extra repeat or an observed horizontal baseline.",
        "Private: 160 fixed historical development queries; 8-class model and source bank, but only 7 classes present in the query cohort. BA averages those 7 present classes. Sizes 2 and 4 mean 16 and 32 stored references.",
        "OSS2: 228 fixed source-validation queries, four present classes. Sizes 8 and 16 mean 32 and 64 stored references. These are not the 378 final evaluation queries.",
        "All points use one already-trained model-order endpoint (17). Random-bank seed 1030 is a sampling seed, not another model-training seed. Coverage is deterministic given the source descriptors/order. The original bank is inherited from its manifest; its construction seed is not inferred from the model seed.",
        "The historical probe freezes network parameters while replacing source-fit references. Later center/control training uses seeds 17/29/43 and different parameter states; those results are excluded. This preparation itself performs no training or inference.",
        "Each condition has n=1. Mean equals the saved observation. Sample SD is undefined and is an empty CSV field, not 0. Do not draw SD error bars or claim independent replication. Query count is not replicate n; strategies are not replicates.",
        "Random selection resets default_rng(1030) per call and samples eligible observations without within-class MMSI reuse. Different budgets are not guaranteed nested banks. Coverage minimizes inverse-MMSI-weighted nearest-descriptor distance using the corrected protected anchor path. It uses source-fit data only.",
        "Sizes count stored observations per class before query ID/MMSI masking and polarization/region validity. Exact eligible-count ranges and per-bank members are recorded; no hidden change of the x-axis to effective polarized evidence counts is made.",
        "The private random doubled bank has lower OA/BA than the original; both coverage sizes have OA 45.0. All OSS2 OA/BA points coincide. Preserve overlapping values as separate series in the data; do not jitter numeric values or select the best bank. Probability metrics can change even when OA/BA agree.",
        "",
        "## Missing Requested Points",
        "",
        "Requested grid: 1, 2, 4, 8, 16 per class. Availability is scoped to the audited corrected historical probe, not a claim about every experiment in the workspace.",
    ])
    for dataset, d in provenance["datasets"].items():
        lines.append(f"- {dataset}: observed {d['observed_references_per_class']}; missing {d['missing_requested_points']}.")
    lines.extend([
        "- The current RCEW model has no matching size-sweep observations in this export; none of these historical points may be relabeled as current RCEW.",
        "- Missing budgets have no rows in the performance CSVs. Do not interpolate, add placeholder points or combine private 2/4 with OSS2 8/16 into a curve.",
        "",
        "## Sources And Exclusions",
        "",
        f"Primary study: {STUDY.as_posix()}",
        "Use anchor_corrected/{private,oss2}/results_s17.json, corroborated by analysis/anchor_corrected/{private,oss2}/endpoints.csv and saved NPZ ids/y. FINDINGS_ZH.md and the final correction section of PROTOCOL_ZH.md take precedence over the earlier uncorrected narrative.",
        "The preserved legacy probe used live layer3 for statistics instead of protected anchor_last. Its results are not mixed into the corrected experiment.",
        f"Also inspected: {(OLD / 'FINDINGS_ZH.md').as_posix()}",
        "The 2026-10-24 reference-resampling study has eight references/class at all six banks, a different conditional-relation candidate, different query cohorts, and a separate episodic-training phase. It cannot fill any size point of the 2026-10-30 model. Original and random banks there also differ in representativeness and pass coverage.",
        "No new chip-level causal claims are made by this numeric preparation. Historical case analysis remains in the source study's CASE_NOTES_ZH.md; plotting these saved points is not a new experimental round.",
        "",
        "## Reproduce",
        "",
        "Run with an environment that already provides NumPy:",
        "",
        "```powershell",
        f'python -B "{(HERE / "prepare_reference.py").as_posix()}"',
        "```",
        "",
        "The script verifies actual reference class counts, query ID/label equality and JSON/CSV metric agreement before writing. A mismatch stops output instead of silently changing study identity. It imports no project model/training module and writes only this reference_data directory.",
    ])
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
