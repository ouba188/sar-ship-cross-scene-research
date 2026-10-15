"""Re-solve unchanged supports under transformed features and fixed source fits."""

from context import *
import pickle
from threadpoolctl import threadpool_limits


def data_for(name, condition):
    old = crossfit.target_arrays(name)
    base = "oss2" if name == "oss2_grdh" else name
    path = OUT / "features" / base / condition
    if not (path / "COMPLETE.json").exists():
        raise RuntimeError(f"Unfinished input features: {base}/{condition}")
    ids = np.load(path / "ids.npy")
    positions = pd.Index(ids).get_indexer(old["ids"])
    if (positions < 0).any():
        raise ValueError("Target subset ID missing")
    np.testing.assert_array_equal(ids[positions], old["ids"])
    for key in ("raw", "c4", "extra"):
        old[key] = np.load(path / (key+".npy"), mmap_mode="r")[positions]
    return old


def main():
    freeze()
    if not (OUT / "FEATURES_COMPLETE.json").exists():
        raise RuntimeError("Complete all input features before inference")
    for fold in FOLDS:
        evaluate = crossfit.configure_evaluation(fold)
        fits = PREVIOUS / f"fold_{fold}/fits"
        with (fits / "geometry.pkl").open("rb") as f:
            bundle = pickle.load(f)
        np.testing.assert_array_equal(bundle["source_rows"], np.load(fits / "source_rows.npy"))
        for condition in CONDITIONS:
            evaluate.OUT = OUT / "predictions" / condition / f"fold_{fold}"
            evaluate.fit_dir = lambda regime, name: fits
            evaluate.mapper_path = lambda regime, name: fits / "mapper.npz"
            evaluate.target_arrays = lambda regime, name, grdh_only=False: data_for(name, condition)
            for name in tasks(fold):
                evaluate.predict("shared_frozen", name, bundle)
    write(OUT / "PREDICTIONS_FROZEN.json", dict(contexts=36, conditions=CONDITIONS, folds=FOLDS,
        support_query_ids_unchanged=True, no_source_refit=True, all_predictions_saved_before_outcomes=True))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
