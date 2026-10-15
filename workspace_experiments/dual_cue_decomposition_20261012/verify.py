"""Independent endpoint-probability and metric checks for the fixed diagnostic."""

from run import OUT, ROOT, SOURCE, DATASETS, folders, info_for, KINDS, ARMS, READOUTS
from run import np, pd, softmax, write_json
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss
import warnings


def main():
    metrics = pd.read_csv(OUT / "metrics.csv")
    endpoint_error = 0.
    metric_error = 0.
    sum_error = 0.
    checked_datasets = set()
    for folder in folders():
        info, suffix = info_for(folder)
        dst = OUT / "targets" / suffix
        probs = np.load(dst / "probabilities.npy", mmap_mode="r")
        with np.load(dst / "predictions.npz") as z:
            saved = {k: z[k] for k in z.files}
        old = ROOT / "dual_evidence_20261010/targets" / suffix
        scores = np.load(old / "scores.npy", mmap_mode="r")
        with np.load(old / "predictions.npz") as z:
            modes = list(z["modes"])
        indices = np.unique(np.r_[np.linspace(0, len(saved["rows"])-1, 64, dtype=int),
                                   saved["support"].ravel()])
        for arm, old_arm in (("R", "RR"), ("T", "RD")):
            for hi, head in enumerate(READOUTS):
                for draw in range(5):
                    reference = scores[draw, modes.index(old_arm + "_" + head), indices]
                    if head == "expert":
                        reference = softmax(reference, axis=1)
                    actual = probs[0, draw, ARMS.index(arm), hi, indices]
                    endpoint_error = max(endpoint_error, float(np.max(np.abs(actual-reference))))
                    np.testing.assert_allclose(actual, reference, atol=2e-7, rtol=1e-6)
        if info["dataset"] in checked_datasets:
            continue
        checked_datasets.add(info["dataset"])
        manifest = (SOURCE / "manifest.csv" if info["dataset"] == "self24ports"
                    else ROOT / DATASETS[info["dataset"]][0] / "target_manifest.csv")
        truth = pd.read_csv(manifest, usecols=["class_id"]).class_id.to_numpy(int)[saved["rows"]]
        ev = saved["eval_masks"][0]
        y = truth[ev]
        _, inv, counts = np.unique(y, return_inverse=True, return_counts=True)
        weights = 1. / counts[inv]
        for ki, kind in enumerate(KINDS):
            for ai, arm in enumerate(ARMS):
                for hi, head in enumerate(READOUTS):
                    p = np.asarray(probs[ki, 0, ai, hi, ev], float)
                    sum_error = max(sum_error, float(np.max(np.abs(p.sum(1)-1))))
                    np.testing.assert_allclose(p.sum(1), 1., atol=1e-7, rtol=0.)
                    row = metrics[(metrics.dataset == info["dataset"]) & (metrics.task == info["task"])
                        & (metrics.kind == kind) & (metrics.arm == arm) & (metrics["head"] == head)
                        & (metrics.draw == 0)].iloc[0]
                    # Saved float32 probabilities pass their precision bound above.
                    # Predictions of source classes absent from a query remain errors.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="The y_prob values do not sum to one.*")
                        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
                        calculated = dict(ba=100*balanced_accuracy_score(y, saved["predictions"][ki, 0, ai, hi, ev]),
                            brier_micro=brier_score_loss(y, p, labels=np.arange(8), scale_by_half=False),
                            brier_macro=brier_score_loss(y, p, labels=np.arange(8), scale_by_half=False, sample_weight=weights),
                            nll_micro=log_loss(y, p, labels=np.arange(8)),
                            nll_macro=log_loss(y, p, labels=np.arange(8), sample_weight=weights))
                    for key, value in calculated.items():
                        metric_error = max(metric_error, abs(value-row[key]))
                        np.testing.assert_allclose(value, row[key], atol=1e-10, rtol=1e-10)
    write_json(OUT / "VERIFIED.json", dict(endpoint_probability_max_error=endpoint_error,
        endpoints_checked="All25 tasks,5 draws, both heads, both endpoints; 64 evenly spaced rows plus support union",
        independent_sklearn_metrics_max_error=metric_error,
        float32_probability_sum_max_error=sum_error,
        independent_metric_checks="First fixed task per dataset, draw0, both mappers, all arms and readouts",
        full_population_endpoint_class_predictions_checked_during_prediction=True))
    print("VERIFIED", endpoint_error, metric_error, flush=True)


if __name__ == "__main__":
    main()
