"""Check fold sharing/purges and independently verify reported cue-order direction."""

from crossfit import *
import pickle
from scipy.stats import mannwhitneyu


def main():
    markers = ["PREDICTIONS_FROZEN.json", "READOUT_COMPLETE.json", "DIAGNOSTICS_COMPLETE.json",
        "audit/COMPLETE.json", "image_diagnostics/COMPLETE.json", "image_diagnostics/CLASS_SIGNAL_COMPLETE.json",
        "image_diagnostics/ORIGIN_CONTROLS_COMPLETE.json"]
    for marker in markers:
        json.loads((OUT / marker).read_text(encoding="utf-8"))
    target_count = 0
    for fold in range(5):
        directory = OUT / f"fold_{fold}/fits"
        rows = permitted_rows(fold)
        np.testing.assert_array_equal(np.load(directory / "source_rows.npy"), rows)
        with (directory / "geometry.pkl").open("rb") as handle:
            geometry = pickle.load(handle)
        np.testing.assert_array_equal(geometry["source_rows"], rows)
        for name in fold_ports(fold) + list(PUBLIC_TASKS):
            path = OUT / f"fold_{fold}/targets/shared_frozen" / name
            result = json.loads((path / "COMPLETE.json").read_text(encoding="utf-8"))
            if Path(result["mapper_path"]) != directory / "mapper.npz" or Path(result["shared_geometry_path"]) != directory / "geometry.pkl":
                raise ValueError("Target used a different fold fit")
            if result["source_n"] != len(rows):
                raise ValueError("Target used different source cohort")
            target_count += 1
    meta = pd.read_csv(OUT / "audit/metadata.csv", usecols=["dataset", "sample_id", "class_id", "contrast_proxy"])
    oss1 = pd.read_csv(OUT / "image_diagnostics/oss1.csv", usecols=["sample_id", "contrast_proxy"])
    table = pd.read_csv(OUT / "image_diagnostics/class_pair_ordering.csv")
    errors = []
    for dataset in ["self24ports", *PUBLIC_TASKS]:
        part = meta[meta.dataset == dataset]
        if dataset == "oss1":
            part = part.drop(columns="contrast_proxy").merge(oss1, on="sample_id", validate="one_to_one")
        a = part.loc[part.class_id == 1, "contrast_proxy"].to_numpy()
        b = part.loc[part.class_id == 4, "contrast_proxy"].to_numpy()
        expected = float(mannwhitneyu(b, a).statistic/(len(a)*len(b)))
        observed = table[(table.dataset == dataset)&(table.task == "all")&(table.descriptor == "contrast_proxy")
            &(table.class_a == 1)&(table.class_b == 4)].auc_b_higher.item()
        errors.append(abs(expected-observed))
        np.testing.assert_allclose(expected, observed, atol=1e-12, rtol=0)
    write(OUT / "COMPLETE.json", dict(status="complete", targets=target_count, held_ports=len(PORTS), source_folds=5,
        source_rows_match_mapper_and_geometry=True, shared_model_within_each_fold=True,
        held_port_and_known_vessel_exclusions=True, cue_auc_independent_rank_max_error=max(errors),
        report="FINDINGS_ZH.md", prior_outputs_unchanged=True, no_new_encoder_training=True,
        upstream_pretraining_isolation=False, causal_crop_intervention_run=False))
    print(json.dumps(dict(targets=target_count, auc_max_error=max(errors), status="complete")), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
