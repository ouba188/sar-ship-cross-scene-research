"""Frozen confidence-versus-support-relation controls on cached ring probes."""

import itertools
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import probe_ring_predictability as old

OUT = old.ROOT / "ring_relation_verification"
CONF = old.FEATURES[:5]
PAIRS = list(itertools.combinations(CONF, 2))[:8]
VARIANTS = ("confidence5", "confidence_poly18", "confidence_relation12", "full18")


def features(frame, variant):
    if variant == "full18":
        return frame[old.FEATURES].copy()
    if variant == "confidence_relation12":
        return frame[old.FEATURES[:12]].copy()
    x = frame[CONF].copy()
    if variant == "confidence_poly18":
        for c in CONF:
            x[c + "_square"] = frame[c] ** 2
        for a, b in PAIRS:
            x[a + "_times_" + b] = frame[a] * frame[b]
    return x


def diagnostics(frame, score, chosen):
    event = frame.gain.ne(0).to_numpy()
    labels = frame.gain.eq(1).to_numpy()
    w = old.risk_weights(frame)
    two = len(np.unique(labels[event])) == 2
    return dict(rows=len(frame), ships=int(frame.mmsi.nunique()),
        scenes=int(frame[["port", "acquisition"]].drop_duplicates().shape[0]),
        events=int(event.sum()), event_ships=int(frame.loc[event, "mmsi"].nunique()),
        rescue=int(frame.gain.eq(1).sum()), harm=int(frame.gain.eq(-1).sum()),
        selected_rescue=int((frame.gain.eq(1).to_numpy() & chosen).sum()),
        selected_harm=int((frame.gain.eq(-1).to_numpy() & chosen).sum()),
        auc=float(roc_auc_score(labels[event], score[event])) if two else np.nan,
        weighted_auc=float(roc_auc_score(labels[event], score[event], sample_weight=w[event])) if two else np.nan,
        delta_ba=float(100 * np.sum(w * frame.gain.to_numpy() * chosen)))


def main():
    start = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    sources = {}
    matched = pd.read_csv(old.ROOT / "ring_matched/source_scored.csv", dtype={"mmsi": str})
    sources["matched16"] = matched.loc[matched.organization.eq("single") & matched.budget.eq(16)].copy()
    sources["expanded"] = pd.read_csv(old.ROOT / "ring_relaxed/query_coverage/source_scored.csv", dtype={"mmsi": str})
    models, choices, source_rows, coefs, counts = {}, {}, [], [], []
    replay_error = 0.
    with threadpool_limits(limits=2):
        for protocol, data in sources.items():
            for (pool, budget), group in data.groupby(["pool", "budget"], sort=True):
                fit = group.loc[group.split.eq("fit") & group.origin.eq("cross")].copy()
                selection = group.loc[group.split.eq("selection") & group.origin.eq("cross")].copy()
                events = fit.gain.ne(0).to_numpy()
                if fit.empty or fit.loc[events, "gain"].nunique() != 2:
                    counts.append(dict(protocol=protocol, pool=pool, budget=budget, trained=False,
                                       reason="source_fit_missing_two_outcomes"))
                    continue
                w = old.risk_weights(fit)
                event_w = w[events] / w[events].mean()
                y = fit.loc[events, "gain"].eq(1).to_numpy(int)
                for variant in VARIANTS:
                    x = features(fit, variant)
                    scaler = StandardScaler().fit(x, sample_weight=w)
                    learner = LogisticRegression(C=.1, solver="lbfgs", max_iter=1000, random_state=20261015)
                    learner.fit(scaler.transform(x)[events], y, sample_weight=event_w)
                    model = make_pipeline(scaler, learner)
                    key = (protocol, pool, int(budget), variant)
                    models[key] = model
                    selected = None
                    for split, frame in (("fit", fit), ("selection", selection)):
                        if frame.empty:
                            continue
                        score = model.predict_proba(features(frame, variant))[:, 1]
                        chosen = (score > .5) & frame.disagree.to_numpy()
                        d = diagnostics(frame, score, chosen)
                        source_rows.append(dict(protocol=protocol, pool=pool, budget=budget,
                                                variant=variant, split=split, **d))
                        if split == "selection":
                            selected = bool(d["delta_ba"] > 0.)
                        if variant == "full18":
                            error = float(np.abs(score - frame.score_logistic.to_numpy()).max())
                            replay_error = max(replay_error, error)
                            np.testing.assert_allclose(score, frame.score_logistic, atol=1e-10)
                    choices[key] = selected
                    counts.append(dict(protocol=protocol, pool=pool, budget=budget, variant=variant,
                        trained=True, parameters=x.shape[1] + 1, selection_available=selected is not None,
                        enabled=selected, fit_rows=len(fit), event_rows=int(events.sum()),
                        event_ships=int(fit.loc[events, "mmsi"].nunique()), selection_rows=len(selection)))
                    coefs.extend(dict(protocol=protocol, pool=pool, budget=budget, variant=variant,
                        feature=c, coefficient=float(v)) for c, v in zip(x.columns, learner.coef_[0]))
                print("SOURCE", protocol, pool, budget, "events", events.sum(), flush=True)
        joblib.dump(models, OUT / "models.joblib")
        old.save_json(OUT / "SOURCE_FROZEN.json", [dict(protocol=k[0], pool=k[1], budget=k[2],
            variant=k[3], enabled=v, threshold=.5) for k, v in choices.items()])
        pd.DataFrame(source_rows).to_csv(OUT / "source_diagnostics.csv", index=False)
        pd.DataFrame(counts).to_csv(OUT / "model_coverage.csv", index=False)
        pd.DataFrame(coefs).to_csv(OUT / "coefficients.csv", index=False)

        target = pd.read_csv(old.OUT / "target_scored.csv", dtype={"mmsi": str})
        archive_matched = pd.read_csv(old.ROOT / "ring_matched/target_decisions.csv", dtype={"mmsi": str})
        archive_expanded = pd.read_csv(old.ROOT / "ring_relaxed/query_coverage/target_decisions.csv", dtype={"mmsi": str})
        units, decisions, target_diagnostics = [], [], []
        for protocol, pool, budget in sorted({k[:3] for k in models}):
            frame = target.loc[target.pool.eq(pool) & target.budget.eq(budget)].copy()
            for arm in ("body", "mix", "half"):
                chosen = np.full(len(frame), arm == "mix")
                m = old.metrics(frame, chosen, arm)
                m["protocol"] = protocol
                units.append(m)
            for variant in VARIANTS:
                key = (protocol, pool, budget, variant)
                score = models[key].predict_proba(features(frame, variant))[:, 1]
                chosen = (score > .5) & frame.disagree.to_numpy()
                for name, selection in ((variant, chosen), (variant + "_enabled", chosen if choices[key] else np.zeros(len(frame), bool))):
                    if name.endswith("_enabled") and choices[key] is None:
                        continue
                    for m in (old.metrics(frame, selection, name), old.random_expected(frame, selection, name)):
                        m["protocol"] = protocol
                        units.append(m)
                d = frame[old.UNIT + ["sample_id", "mmsi", "truth", "body_pred", "mix_pred", "gain", "disagree"]].copy()
                d["protocol"], d["variant"], d["score"], d["chosen"] = protocol, variant, score, chosen
                d["source_enabled"] = choices[key]
                for origin, indices in frame.groupby("origin").indices.items():
                    g = frame.iloc[indices]
                    weights = old.risk_weights(g)
                    d.loc[g.index, "ba_contribution"] = 100 * weights * g.gain.to_numpy() * chosen[indices]
                    target_diagnostics.append(dict(protocol=protocol, pool=pool, budget=budget,
                        origin=origin, variant=variant, **diagnostics(g, score[indices], chosen[indices])))
                decisions.append(d)
                if variant == "full18":
                    archived = archive_matched if protocol == "matched16" else archive_expanded
                    recipe = "single_budget" if protocol == "matched16" else "relaxed"
                    ref = archived.loc[archived.recipe.eq(recipe) & archived.pool.eq(pool) & archived.budget.eq(budget)]
                    keys = ["pool", "budget", "port", "acquisition", "origin", "draw", "sample_id"]
                    replay = d.merge(ref[keys + ["score_logistic"]], on=keys, validate="one_to_one")
                    assert len(replay) == len(d)
                    error = float(np.abs(replay.score - replay.score_logistic).max())
                    replay_error = max(replay_error, error)
                    np.testing.assert_allclose(replay.score, replay.score_logistic, atol=1e-10)
        units = pd.concat(units, ignore_index=True)
        units.to_csv(OUT / "target_metrics.csv", index=False)
        pd.concat(decisions, ignore_index=True).to_csv(OUT / "target_decisions.csv", index=False)
        pd.DataFrame(target_diagnostics).to_csv(OUT / "target_diagnostics.csv", index=False)
        tables, draws = [], []
        for protocol, g in units.groupby("protocol"):
            a = old.aggregate(g)
            a["protocol"] = protocol
            tables.append(a)
            a = old.aggregate(g, ["draw"])
            a["protocol"] = protocol
            draws.append(a)
        pd.concat(tables, ignore_index=True).to_csv(OUT / "target_per_port.csv", index=False)
        pd.concat(draws, ignore_index=True).to_csv(OUT / "target_per_draw.csv", index=False)
    old.save_json(OUT / "DONE.json", dict(seconds=time.perf_counter()-start, models=len(models),
        cells=len(models)//len(VARIANTS), full18_replay_max_error=replay_error,
        frozen_source_before_target=True, variants=list(VARIANTS), no_target_threshold_selection=True))
    print("DONE", len(models), "models; replay", replay_error, flush=True)


if __name__ == "__main__":
    main()
