"""Frozen T/K replacements; no fitting, checkpoint selection, or model writes."""
import os
import sys
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
sys.dont_write_bytecode = True
from pathlib import Path
import importlib.util
import json
import time
import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
HERE = OUT.parent
spec = importlib.util.spec_from_file_location("frequency_train", HERE / "train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)
ARMS = ("oldT_oldK", "newT_newK", "oldT_newK", "newT_oldK")
ROLES = ("selection", "evaluation")
DESCRIPTORS = ("content_dot", "delta_K", "abs_delta_K", "transport_L1",
               "old_real_mass", "new_real_mass")


def rival(score, y):
    return np.where(np.arange(8)[None] == y[:, None], -np.inf, score).argmax(1)


def margin(score, y):
    row = np.arange(len(y))
    return score[row, y] - score[row, rival(score, y)]


@torch.inference_mode()
def reconstruct(reader, gate):
    scores = {arm: [] for arm in ARMS}
    locals_out = {arm: [] for arm in ARMS}
    descriptors = []
    checks = dict(pair_score_formula_error=0., mass_channel_error=0.,
                  masked_mass_max=0., invalid_pair_count=0)
    for first in range(0, len(reader.ids), train.BATCH):
        idx = torch.arange(first, min(first + train.BATCH, len(reader.ids)), device="cuda")
        refs = reader.refs[idx]
        q, s = reader.x[idx], reader.x[refs]
        qe, se = reader.e[idx], reader.e[refs]
        a, b = reader.mass[idx], reader.mass[refs]
        groups = {arm: [] for arm in ARMS}
        description = []
        for region in train.REGIONS:
            product = q[:, :, region][:, None, None, :, :, None, :] * s[..., region, :][..., None, :, :]
            eq = qe[:, :, region][:, None, None, :, :, None, :]
            es = se[..., region, :][..., None, :, :]
            old_k = product.sum(-1)
            dk = .25 * (product * gate(eq, es)).sum(-1)
            new_k = old_k + dk
            qa, sa = a[:, None, None, :, region], b[..., region]
            old_t = train.sinkhorn(old_k, qa, sa)[0]
            new_t = train.sinkhorn(new_k, qa, sa)[0]
            keep = (qa > 0)[..., :, None] & (sa > 0)[..., None, :]
            checks["invalid_pair_count"] += int((~keep).sum())
            pairs = ((old_t, old_k), (new_t, new_k), (old_t, new_k), (new_t, old_k))
            region_values = {}
            for arm, (t, k) in zip(ARMS, pairs):
                value = train.pair_score(t, k, qa, sa)
                # Direct masked accounting detects an accidental null normalization
                # or a mismatch between the similarity and mass channels.
                masked = t * keep
                direct = torch.stack([(masked * k).sum((-1, -2)), masked.sum((-1, -2))], -1)
                checks["pair_score_formula_error"] = max(checks["pair_score_formula_error"], float((value-direct).abs().max()))
                checks["masked_mass_max"] = max(checks["masked_mass_max"], float((masked * ~keep).abs().max()))
                region_values[arm] = value
                groups[arm].append(value)
            for lhs, rhs in (("oldT_oldK", "oldT_newK"), ("newT_newK", "newT_oldK")):
                checks["mass_channel_error"] = max(checks["mass_channel_error"], float((region_values[lhs][..., 1]-region_values[rhs][..., 1]).abs().max()))
            count = keep.sum((2, 4, 5))
            means = [(v * keep).sum((2, 4, 5)) / count.clamp_min(1) for v in (old_k, dk, dk.abs())]
            means = [v.masked_fill(count == 0, float("nan")) for v in means]
            desc = means + [((new_t-old_t).abs()*keep).sum((-1, -2)).mean(2),
                            region_values["oldT_oldK"][..., 1].mean(2),
                            region_values["newT_newK"][..., 1].mean(2)]
            description.append(torch.stack(desc, -1))
        descriptors.append(torch.stack(description, -2).cpu().numpy())
        for arm in ARMS:
            local = torch.stack(groups[arm], -2).mean(2)
            score = reader.anchor[idx] + reader.delta(local, idx) - reader.old_delta[idx]
            scores[arm].append(score.cpu().numpy())
            locals_out[arm].append(local.cpu().numpy())
    return ({k: np.concatenate(v) for k, v in scores.items()},
            {k: np.concatenate(v) for k, v in locals_out.items()},
            np.concatenate(descriptors), checks)


def main():
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    original_cases = json.loads((HERE / "analysis/cases.json").read_text(encoding="utf-8"))
    assert [int(c["obs_id"]) for c in original_cases] == [220, 12716, 25486, 25876]
    metric_rows, paired_rows, margin_rows, class_rows = [], [], [], []
    prediction_rows, case_rows, case_local_rows, kernel_rows, checks = [], [], [], [], []
    receipts = []
    for seed in train.SEEDS:
        reader = train.Reader("frequency", seed)
        ck = torch.load(HERE / "models" / f"frequency_{seed}.pt", map_location="cpu", weights_only=False)
        scaler_error = max(float(np.max(np.abs(np.asarray(reader.scaler[k], dtype=float)-np.asarray(ck["scaler"][k], dtype=float)))) for k in reader.scaler)
        assert scaler_error == 0., scaler_error
        gate = train.Gate().cuda().eval().requires_grad_(False)
        gate.load_state_dict(ck["state"])
        receipts.append(ck["receipt"])
        scores, locals_out, desc, check = reconstruct(reader, gate)
        saved = np.load(HERE / "results" / f"frequency_{seed}.npz")
        np.testing.assert_array_equal(saved["ids"], reader.ids)
        check.update(seed=seed, n=len(reader.ids), scaler_error=scaler_error,
                     old_head_replay=reader.oldhead_replay,
                     new_score_replay=float(np.max(np.abs(scores["newT_newK"]-saved["score"]))),
                     new_local_replay=float(np.max(np.abs(locals_out["newT_newK"]-saved["local"]))),
                     old_score_replay=float(np.max(np.abs(scores["oldT_oldK"]-reader.strong))),
                     old_local_replay=float(np.max(np.abs(locals_out["oldT_oldK"]-reader.oldlocal.cpu().numpy()))),
                     new_prediction_mismatch=int(np.sum(scores["newT_newK"].argmax(1)!=saved["score"].argmax(1))),
                     old_prediction_mismatch=int(np.sum(scores["oldT_oldK"].argmax(1)!=reader.strong.argmax(1))))
        assert max(check[k] for k in ("new_score_replay", "new_local_replay", "old_score_replay", "old_local_replay")) < 3e-5, check
        assert check["old_prediction_mismatch"] == check["new_prediction_mismatch"] == 0, check
        assert check["pair_score_formula_error"] == check["mass_channel_error"] == check["masked_mass_max"] == 0., check
        checks.append(check)
        # Use the exact cached anchor as baseline after validating oldT/oldK.
        scores["strong"] = reader.strong
        f = reader.table
        y = f.class_id.to_numpy(int)
        fixed_wrong = rival(reader.strong, y)
        base_margin = margin(reader.strong, y)
        base_correct = reader.strong.argmax(1) == y
        for role in ROLES:
            take = np.flatnonzero(f.role.eq(role))
            yy = y[take]
            for arm in ("strong", *ARMS[1:]):
                score = scores[arm]
                m = train.metrics(yy, score[take])
                metric_rows.append(dict(seed=seed, role=role, arm=arm, **m))
                pred = score.argmax(1)
                row = np.arange(len(y))
                true_score = score[row, y]
                wrong_score = score[row, rival(score, y)]
                fixed_score = score[row, fixed_wrong]
                for subset, mask in (("all", np.ones(len(take), bool)), ("old_correct", base_correct[take]), ("old_wrong", ~base_correct[take])):
                    ix = take[mask]
                    margin_rows.append(dict(seed=seed, role=role, arm=arm, subset=subset, n=len(ix),
                        true_logit=float(true_score[ix].mean()), best_wrong_logit=float(wrong_score[ix].mean()),
                        margin=float((true_score-wrong_score)[ix].mean()),
                        fixed_old_rival_margin=float((true_score-fixed_score)[ix].mean()),
                        margin_gain=float(((true_score-wrong_score)-base_margin)[ix].mean())))
                for c in range(8):
                    ix = take[yy == c]
                    class_rows.append(dict(seed=seed, role=role, arm=arm, class_id=c,
                        class_name=f.loc[f.class_id.eq(c), "class_name"].iloc[0], n=len(ix),
                        recall=float((pred[ix]==c).mean()), margin=float(margin(score[ix], y[ix]).mean())))
                for j in take:
                    prediction_rows.append(dict(seed=seed, role=role, arm=arm, obs_id=int(f.iloc[j].obs_id),
                        class_id=int(y[j]), prediction=int(pred[j]), true_score=float(true_score[j]),
                        best_wrong=int(rival(score[j:j+1], y[j:j+1])[0]), best_wrong_score=float(wrong_score[j]),
                        margin=float(true_score[j]-wrong_score[j]), fixed_old_rival=int(fixed_wrong[j]),
                        fixed_old_rival_margin=float(true_score[j]-fixed_score[j])))
            for arm, comparator in (("newT_newK", "strong"), ("oldT_newK", "strong"), ("newT_oldK", "strong"),
                                    ("newT_newK", "oldT_newK"), ("newT_newK", "newT_oldK")):
                aa, bb = scores[arm][take], scores[comparator][take]
                am, bm = train.metrics(yy, aa), train.metrics(yy, bb)
                ac, bc = aa.argmax(1)==yy, bb.argmax(1)==yy
                paired_rows.append(dict(seed=seed, role=role, arm=arm, comparator=comparator,
                    rescued=int((ac & ~bc).sum()), harmed=int((bc & ~ac).sum()),
                    prediction_flips=int((aa.argmax(1)!=bb.argmax(1)).sum()),
                    OA_pp=100*(am["OA"]-bm["OA"]), BA_pp=100*(am["BA"]-bm["BA"]),
                    NLL_delta=am["NLL"]-bm["NLL"], balanced_NLL_delta=am["balanced_NLL"]-bm["balanced_NLL"],
                    margin_gain=float((margin(aa, yy)-margin(bb, yy)).mean())))
            for c in range(8):
                ix = take[yy == c]
                true_desc = desc[ix, c]
                wrong_desc = desc[ix, fixed_wrong[ix]]
                all_wrong_desc = np.nanmean(np.delete(desc[ix], c, axis=1), axis=1)
                for relation, values in (("true", true_desc), ("old_best_wrong", wrong_desc), ("all_wrong_mean", all_wrong_desc)):
                    for p, pol in enumerate(("VV", "VH")):
                        for r, region in enumerate(("body", "side", "end")):
                            v = values[:, p, r]
                            kernel_rows.append(dict(seed=seed, role=role, true_class=c, relation=relation,
                                pol=pol, region=region, n=len(ix), valid_kernel_queries=int(np.isfinite(v[:, 0]).sum()),
                                **{name: float(np.nanmean(v[:, k])) for k, name in enumerate(DESCRIPTORS)}))
        for original in original_cases:
            j = int(original["local_index"])
            assert int(f.iloc[j].obs_id) == original["obs_id"]
            c, wr = int(y[j]), int(fixed_wrong[j])
            for arm in ("strong", *ARMS[1:]):
                s = scores[arm][j]
                winner = int(s.argmax())
                wrong = int(rival(s[None], y[j:j+1])[0])
                case_rows.append(dict(seed=seed, original_case=original["case"], obs_id=int(original["obs_id"]),
                    arm=arm, true_class=c, prediction=winner, best_wrong=wrong,
                    true_score=float(s[c]), best_wrong_score=float(s[wrong]), margin=float(s[c]-s[wrong]),
                    fixed_old_rival=wr, fixed_old_rival_margin=float(s[c]-s[wr]),
                    scores=s.tolist()))
            if seed == int(original["seed"]):
                for arm in ARMS:
                    for relation, cls in (("true", c), ("old_best_wrong", wr)):
                        for p, pol in enumerate(("VV", "VH")):
                            for r, region in enumerate(("body", "side", "end")):
                                value = locals_out[arm][j, cls, p, r]
                                case_local_rows.append(dict(seed=seed, obs_id=int(original["obs_id"]), arm=arm,
                                    relation=relation, class_id=cls, pol=pol, region=region,
                                    affinity=float(value[0]), real_mass=float(value[1])))
        print("FINISHED", seed, "replay", check, flush=True)
        del reader, gate
        torch.cuda.empty_cache()
    for name, rows in (("metrics", metric_rows), ("paired", paired_rows), ("margins", margin_rows),
                       ("classes", class_rows), ("kernel_descriptive", kernel_rows), ("case_local", case_local_rows)):
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(OUT / "predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame([{k:v for k,v in c.items() if k != "scores"} for c in case_rows]).to_csv(OUT / "cases.csv", index=False)
    (OUT / "cases.json").write_text(json.dumps(dict(predictions=case_rows, original_cases=original_cases), indent=2), encoding="utf-8")
    role_counts = f.groupby("role").agg(n=("obs_id", "size"), ships=("mmsi", "nunique"), passes=("pass_group", "nunique")).reset_index().to_dict("records")
    receipt = dict(seconds=time.perf_counter()-start, executable=sys.executable, fits=0, checkpoint_selection=0,
        checkpoints=receipts, roles=role_counts, checks=checks,
        formula="strong + old_head(new_local, fixed_context) - old_head(old_local, fixed_context)",
        replacements="old/new T crossed with old/new K; pair_score returns masked affinity AND unnormalized real match mass",
        descriptive="content_dot/delta_K: valid token pairs pooled over two refs within each pol/region; then equal-query mean. No transport weighting. Empty cells excluded. Transport/mass include empty zero cells.",
        evaluation="Historical development evaluation; descriptive replacements, no training or selection on either reported role.")
    (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(pd.DataFrame(metric_rows).groupby(["role", "arm"])[["OA", "BA", "NLL"]].mean().to_string(), flush=True)
    print("COMPLETE", round(receipt["seconds"], 3), flush=True)


if __name__ == "__main__":
    main()
