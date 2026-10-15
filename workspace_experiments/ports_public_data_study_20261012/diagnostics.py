"""Source-defined shift strata and within-class associations, without refits."""

from crossfit import *
from analyze import target_frame
from dual_reconstruction_core import reconstruct
from scipy.stats import spearmanr

CONTRASTS = (("P", "R"), ("E", "R"), ("PE", "P"), ("T", "R"))


def input_shift(x, mapper):
    z = ((x-mapper["x_mean"])/mapper["x_std"]).astype(float)
    return np.mean(z*z, axis=1)


def main():
    if not (OUT / "PREDICTIONS_FROZEN.json").exists():
        raise RuntimeError("Inference must finish before diagnostic readout")
    strata, correlations, summaries, thresholds = [], [], [], []
    for fold in range(5):
        evaluate = configure_evaluation(fold)
        mapper = archive(OUT / f"fold_{fold}/fits/mapper.npz")
        rows, raw, c4, _ = source_arrays(fold)
        source_z = np.concatenate([input_shift(np.c_[raw[ix], c4[ix]].astype(np.float32), mapper)
            for ix in np.array_split(rows, max(1, (len(rows)+511)//512))])
        cuts = np.quantile(source_z, [.25, .5, .75, .99])
        thresholds.append(dict(fold=fold, source_n=len(rows), q25=cuts[0], q50=cuts[1], q75=cuts[2], q99=cuts[3]))
        for name in fold_ports(fold) + list(PUBLIC_TASKS):
            data = target_arrays(name)
            dst = OUT / f"fold_{fold}/targets/shared_frozen" / name
            saved = archive(dst / "predictions.npz")
            np.testing.assert_array_equal(data["ids"], saved["ids"])
            y = target_frame(name, saved["ids"]).class_id.to_numpy(int)
            probs = np.load(dst / "probabilities.npy", mmap_mode="r")
            n = len(y)
            z = np.empty(n)
            errors = np.empty((2, n))
            for lo in range(0, n, 512):
                s = slice(lo, min(lo+512, n))
                x = np.c_[data["raw"][s], data["c4"][s]].astype(np.float32)
                z[s] = input_shift(x, mapper)
                for ki, kind in enumerate(public.KINDS):
                    yp = reconstruct(x, mapper, kind)
                    errors[ki, s] = np.mean(((yp.astype(float)-data["extra"][s])/mapper["y_std"])**2, axis=1)
            buckets = np.searchsorted(cuts, z)
            meta = dict(fold=fold, task=name, dataset=name if name in PUBLIC_TASKS else "self24ports")
            for ki, kind in enumerate(public.KINDS):
                summaries.append(dict(**meta, kind=kind, n=n, input_z_mean=float(z.mean()),
                    input_z_median=float(np.median(z)), fraction_above_source99=float(np.mean(z>cuts[-1])),
                    reconstruction_mse=float(errors[ki].mean())))
                sums = np.zeros((len(CONTRASTS), n))
                counts = np.sum(saved["eval_masks"], axis=0)
                for draw, mask in enumerate(saved["eval_masks"]):
                    sc = np.bincount(y[data["support"][draw]], minlength=8)
                    loss = {}
                    for arm in ("R", "P", "E", "PE", "T"):
                        p = np.asarray(probs[ki, draw, public.ARMS.index(arm), 0], float)
                        loss[arm] = np.sum((p-np.eye(8)[y])**2, axis=1)
                    for ci, (a, b) in enumerate(CONTRASTS):
                        gain = loss[b]-loss[a]
                        sums[ci, mask] += gain[mask]
                        for c in np.unique(y[mask]):
                            for bucket in range(5):
                                ix = mask & (y == c) & (buckets == bucket)
                                if ix.any():
                                    strata.append(dict(**meta, kind=kind, draw=draw, class_id=int(c),
                                        support_n=int(sc[c]), bucket=bucket, contrast=a+"-"+b,
                                        n=int(ix.sum()), gain=float(gain[ix].mean()),
                                        input_z=float(z[ix].mean()), reconstruction_mse=float(errors[ki, ix].mean())))
                average = sums/np.maximum(counts, 1)
                for ci, (a, b) in enumerate(CONTRASTS):
                    for c in np.unique(y):
                        ix = (y == c) & (counts > 0)
                        if ix.sum() < 30:
                            continue
                        for descriptor, values in (("input_z", z), ("reconstruction_mse", errors[ki])):
                            rho = float(spearmanr(values[ix], average[ci, ix]).statistic)
                            correlations.append(dict(**meta, kind=kind, class_id=int(c), contrast=a+"-"+b,
                                descriptor=descriptor, n=int(ix.sum()), spearman=rho))
            np.savez_compressed(dst / "data_diagnostics.npz", ids=data["ids"], input_z=z,
                source_shift_bucket=buckets, reconstruction_mse=errors, source_thresholds=cuts)
            print("DIAGNOSTICS", fold, name, flush=True)
    pd.DataFrame(thresholds).to_csv(OUT / "source_shift_thresholds.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "shift_summary.csv", index=False)
    pd.DataFrame(strata).to_csv(OUT / "shift_class_effects.csv", index=False)
    pd.DataFrame(correlations).to_csv(OUT / "within_class_correlations.csv", index=False)
    write(OUT / "DIAGNOSTICS_COMPLETE.json", dict(source_defined_quantiles=[.25, .5, .75, .99],
        correlations="Per-task, per-class >=30 rows; query-draw-averaged Brier gain; descriptive, no independent-sample p-values",
        input_shift="Mean squared source-standardized R50original+C4 coordinate; not a physical sensor-distance measure",
        no_refits=True, no_target_threshold_selection=True))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
