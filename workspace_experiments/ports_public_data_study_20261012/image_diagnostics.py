"""Matched display-image proxies and within-class risk associations.

This is a post-outcome exploratory diagnostic, not a new selected predictor.
"""

from crossfit import *
from scipy.stats import spearmanr
import c4_condition_descriptors as descriptor

DST = OUT / "image_diagnostics"
FIELDS = list(descriptor.FEATURES) + ["native_width_px_vv"]
CONTRASTS = (("P", "R"), ("E", "R"), ("PE", "P"), ("T", "R"))


def main():
    DST.mkdir(exist_ok=True)
    if not (DST / "oss1.csv").exists():
        descriptor.MANIFESTS["oss1"] = public.original("oss1") / "target_manifest.csv"
        receipt = descriptor.run_dataset("oss1", DST)
        write(DST / "OSS1_EXTRACTION.json", dict(**receipt, definitions=descriptor.FEATURES,
            recipe="Exact c4_condition_descriptors.py.describe/read_pair helper, existing source/OSS2 recipe",
            physical_interpretation=False, label_or_outcome_selection=False))
    metadata = pd.read_csv(OUT / "audit/metadata.csv", low_memory=False)
    extra = pd.read_csv(DST / "oss1.csv")
    one = metadata[metadata.dataset == "oss1"].drop(columns=FIELDS).merge(extra[["sample_id"]+FIELDS],
        on="sample_id", how="left", validate="one_to_one")
    metadata = pd.concat([metadata[metadata.dataset != "oss1"], one], ignore_index=True)
    numbers, correlations, median_strata = [], [], []
    for (dataset, c), part in metadata.groupby(["dataset", "class_id"]):
        for field in FIELDS:
            x = pd.to_numeric(part[field], errors="coerce")
            numbers.append(dict(dataset=dataset, class_id=int(c), descriptor=field, n=len(x), valid=int(x.notna().sum()),
                mean=float(x.mean()), median=float(x.median()), q25=float(x.quantile(.25)), q75=float(x.quantile(.75))))
    for dataset, part in metadata.groupby("dataset"):
        for field in FIELDS:
            x = pd.to_numeric(part[field], errors="coerce")
            numbers.append(dict(dataset=dataset, class_id=-1, descriptor=field, n=len(x), valid=int(x.notna().sum()),
                mean=float(x.mean()), median=float(x.median()), q25=float(x.quantile(.25)), q75=float(x.quantile(.75))))
    for fold in range(5):
        sr = permitted_rows(fold)
        source = metadata[metadata.dataset == "self24ports"].set_index("manifest_row").loc[sr]
        for name in fold_ports(fold) + list(PUBLIC_TASKS):
            dst = OUT / f"fold_{fold}/targets/shared_frozen" / name
            saved = archive(dst / "predictions.npz")
            dataset = name if name in PUBLIC_TASKS else "self24ports"
            frame = metadata[metadata.dataset == dataset].set_index("sample_id").loc[saved["ids"]]
            y = frame.class_id.to_numpy(int)
            p = np.load(dst / "probabilities.npy", mmap_mode="r")
            valid = saved["eval_masks"].sum(0)
            gain = {}
            for a, b in CONTRASTS:
                result = np.zeros(len(y))
                for draw, mask in enumerate(saved["eval_masks"]):
                    pa = np.asarray(p[1, draw, public.ARMS.index(a), 0], float)
                    pb = np.asarray(p[1, draw, public.ARMS.index(b), 0], float)
                    loss = np.sum((pb-np.eye(8)[y])**2-(pa-np.eye(8)[y])**2, axis=1)
                    result[mask] += loss[mask]
                gain[a+"-"+b] = result/np.maximum(valid, 1)
            for field in FIELDS:
                values = pd.to_numeric(frame[field], errors="coerce").to_numpy()
                threshold = float(pd.to_numeric(source[field], errors="coerce").median())
                for c in np.unique(y):
                    mask = (y == c) & (valid > 0) & np.isfinite(values)
                    for contrast, benefit in gain.items():
                        meta = dict(fold=fold, dataset=dataset, task=name, class_id=int(c), descriptor=field,
                            contrast=contrast)
                        if mask.sum() >= 30 and len(np.unique(values[mask])) > 1:
                            correlations.append(dict(**meta, n=int(mask.sum()),
                                spearman=float(spearmanr(values[mask], benefit[mask]).statistic)))
                        for side, ix in (("source_low", mask & (values <= threshold)),
                                         ("source_high", mask & (values > threshold))):
                            if ix.any():
                                median_strata.append(dict(**meta, side=side, threshold=threshold,
                                    n=int(ix.sum()), risk_gain=float(benefit[ix].mean())))
            print("IMAGE ASSOCIATIONS", fold, name, flush=True)
    pd.DataFrame(numbers).to_csv(DST / "summary.csv", index=False)
    pd.DataFrame(correlations).to_csv(DST / "within_class_correlations.csv", index=False)
    pd.DataFrame(median_strata).to_csv(DST / "source_median_strata.csv", index=False)
    write(DST / "COMPLETE.json", dict(oss1_rows=len(extra), kind="mlp", head="expert", targets=38,
        diagnostic="Within-task/class Spearman >=30 query rows; query-draw averaged benefit",
        thresholds="Each fold's source median, no target-optimized threshold",
        limits="Display-intensity/bright-footprint proxies, not calibrated backscatter or vessel segmentation. Within-class associations remain confounded; no causal or significance claims."))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
