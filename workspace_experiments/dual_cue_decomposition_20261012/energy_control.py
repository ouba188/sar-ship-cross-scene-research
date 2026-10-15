"""Bounded F0921 diagnosis: separate SSL kernel shape from diagonal energy."""

from run import (OUT, ROOT, KINDS, ARMS, BATCH, READOUTS, Inputs, EvidenceContext,
                 load_task, reconstruct, ssl_map, teacher_map, coefficients,
                 base_cross, np, pd, softmax, ba, write_json, threadpool_limits)
import time

DST = OUT / "f0921_energy_control"


def main():
    write_json(DST / "PROTOCOL.json", dict(
        posthoc_reason="F0921 PE improves less than T and has much larger diagonal energy with negative P/E cross terms",
        task="f0921", kinds=KINDS, support_budget=64, draws=5,
        intervention="KR + gamma*(PP'+EE'); gamma=sum_support||T||^2 / sum_support(||P||^2+||E||^2)",
        fixed="Same R50 prior, base, source maps,64supports,ridge8,softmax and equal fusion",
        gamma_selection="One analytic trace match per support draw; no parameter search, no query rows or labels",
        boundary="Uses true R18 support features for matching and residuals for queries; not a single-backbone replacement",
        claim="Matching the SSL trace does not match its spectrum or establish physical causality",
        query_metrics="Only after BOTH mapper outputs are saved; exploratory follow-up, not independent confirmation"))
    folder = ROOT / "learned_frequency_residual_20261009/tasks/f0921"
    data = load_task(folder)
    inputs = Inputs(folder)
    ctx = EvidenceContext(folder, data)
    with np.load(ROOT / "dual_reconstruction_20261010/targets/f0921/f0921/mapper.npz") as z:
        mapper = {k: z[k] for k in z.files}
    olddst = OUT / "targets/f0921/f0921"
    old = np.load(olddst / "probabilities.npy", mmap_mode="r")
    with np.load(olddst / "coefficients.npz") as z:
        oldcoef = {k: z[k] for k in z.files}
    ref = np.unique(data["support"])
    locs = [np.searchsorted(ref, s) for s in data["support"]]
    fr = ctx.features(ref)
    kr = ctx.cross(fr, fr)["R"]
    tr = teacher_map(fr)
    kb = base_cross(data, ref, ref, inputs.info["eta"])
    n = len(data["rows"])
    result = np.lib.format.open_memmap(DST / "probabilities.npy", mode="w+", dtype=np.float32,
                                       shape=(2, 5, 2, n, 8))
    predictions = np.empty((2, 5, 2, n), np.int8)
    factors = []
    max_replay_error = 0.
    started = time.perf_counter()
    for ki, kind in enumerate(KINDS):
        pr = ssl_map(reconstruct(inputs.target_x(ref), mapper, kind), ctx.model, ctx.stats)
        er = tr-pr
        kpe = pr@pr.T + er@er.T
        states = []
        for draw, loc in enumerate(locs):
            gamma = float(np.sum(tr[loc]**2) / (np.sum(pr[loc]**2)+np.sum(er[loc]**2)))
            labels = data["support_labels"][draw]
            scaled = kr[np.ix_(loc, loc)] + gamma*kpe[np.ix_(loc, loc)]
            np.testing.assert_allclose(np.trace(scaled-kr[np.ix_(loc, loc)]), np.sum(tr[loc]**2), rtol=1e-12)
            alpha = coefficients(scaled, fr["priors"]["r50"][loc], labels, 8.)
            ab = coefficients(kb[np.ix_(loc, loc)], data["prior"][data["support"][draw]], labels, 8.)
            states.append((gamma, alpha, ab))
            factors.append(dict(kind=kind, draw=draw, gamma=gamma))
        for lo in range(0, n, BATCH):
            s = slice(lo, min(lo+BATCH, n))
            f = ctx.features(s)
            k = ctx.cross(f, fr)["R"]
            p = ssl_map(reconstruct(inputs.target_x(s), mapper, kind), ctx.model, ctx.stats)
            e = teacher_map(f)-p
            delta = p@pr.T + e@er.T
            b = base_cross(data, s, ref, inputs.info["eta"])
            for draw, loc in enumerate(locs):
                gamma, alpha, ab = states[draw]
                replay = softmax(f["priors"]["r50"] + (k[:, loc]+delta[:, loc])@oldcoef[f"{kind}_{draw}_PE"], axis=1)
                err = float(np.max(np.abs(replay-old[ki, draw, ARMS.index("PE"), 0, s])))
                max_replay_error = max(max_replay_error, err)
                np.testing.assert_allclose(replay, old[ki, draw, ARMS.index("PE"), 0, s], atol=2e-7, rtol=1e-6)
                expert = softmax(f["priors"]["r50"] + (k[:, loc]+gamma*delta[:, loc])@alpha, axis=1)
                base = softmax(data["prior"][s]+b[:, loc]@ab, axis=1)
                for hi, value in enumerate((expert, .5*(base+expert))):
                    if not np.isfinite(value).all():
                        raise ValueError("Nonfinite scale-control probability")
                    result[ki, draw, hi, s] = value
                    predictions[ki, draw, hi, s] = value.argmax(1)
            if lo % (BATCH*80) == 0:
                print("ENERGY CONTROL", kind, s.stop, n, flush=True)
    result.flush()
    np.savez_compressed(DST / "predictions.npz", predictions=predictions, kinds=np.array(KINDS),
        rows=data["rows"], ids=data["ids"], support=data["support"], eval_masks=data["eval_masks"])
    write_json(DST / "PREDICTIONS_FROZEN.json", dict(gamma=factors, seconds=time.perf_counter()-started,
        max_PE_probability_replay_error=max_replay_error, no_query_labels_used_for_gamma=True))
    frame = pd.read_csv(ROOT / "f0921_transfer_20261007/target_manifest.csv", usecols=["sample_id", "class_id"])
    np.testing.assert_array_equal(frame.sample_id.to_numpy(str)[data["rows"]], data["ids"])
    truth = frame.class_id.to_numpy(int)[data["rows"]]
    records = []
    for ki, kind in enumerate(KINDS):
        for draw, ev in enumerate(data["eval_masks"]):
            y = truth[ev]
            for hi, head in enumerate(READOUTS):
                p = np.asarray(result[ki, draw, hi, ev], float)
                loss = np.sum((p-np.eye(8)[y])**2, axis=1)
                pred = predictions[ki, draw, hi, ev]
                records.append(dict(kind=kind, draw=draw, head=head, arm="PE_scaled", oa=100*np.mean(pred == y),
                    ba=ba(y, pred), brier_micro=float(loss.mean()),
                    brier_macro=float(np.mean([loss[y == c].mean() for c in np.unique(y)]))))
    table = pd.DataFrame(records)
    original = pd.read_csv(OUT / "metrics.csv")
    original = original[(original.dataset == "f0921") & original.arm.isin(["P", "PE", "T"])]
    table = pd.concat([table, original[table.columns]], ignore_index=True)
    table.to_csv(DST / "metrics.csv", index=False)
    summary = table.groupby(["kind", "head", "arm"])[["oa", "ba", "brier_micro", "brier_macro"]].mean().reset_index()
    summary.to_csv(DST / "summary.csv", index=False)
    write_json(DST / "COMPLETE.json", dict(tasks=1, kinds=2, draws=5, no_search=True,
        support_trace_matched=True, max_PE_probability_replay_error=max_replay_error,
        seconds=time.perf_counter()-started))
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
