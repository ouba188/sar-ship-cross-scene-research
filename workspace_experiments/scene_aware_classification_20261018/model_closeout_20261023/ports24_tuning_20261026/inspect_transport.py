"""Trace actual soft correspondence on the reviewed cases; no new fitting."""
import json
import numpy as np
import pandas as pd
import torch
from tune import HERE, PREV, SEEDS, ta, op, PairMetric, load, write


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    data, ext = load(PREV/'bundle/features.npz'), load(PREV/'bundle/external.npz')
    features = {k: np.concatenate((data[k], ext[k])) for k in ta.FEATURES}
    refs = np.concatenate((data['refs'], ext['refs']))
    obs = np.r_[data['obs_id'], ext['obs_id']]
    cases = json.loads((HERE/'analysis/cases.json').read_text(encoding='utf-8'))
    original = op.sinkhorn
    rows, replay = [], []
    for seed in SEEDS:
        subset = [c for c in cases if c['seed'] == seed]
        ids = np.array([c['index'] for c in subset])
        initial = PREV/'initial_assets/warm'/f'seed{seed}'
        metric = PairMetric().cuda().eval()
        metric.load_state_dict(op.load_checkpoint(initial/'both_metric.pt')['state'])
        scaler = op.load_checkpoint(initial/'metric_scaler.pt')
        for eps in (.1, .4):
            captured = []
            def trace(sim, qa, sa, steps=60, eps=.1):
                t = original(sim, qa, sa, steps=steps, eps=eps)
                t = t*((qa > 0)[..., :, None] & (sa > 0)[..., None, :])
                total = t.sum((-1, -2))
                prob = t/total[..., None, None].clamp_min(1e-20)
                h = -(prob*prob.clamp_min(1e-20).log()).sum((-1, -2))
                peak = prob.flatten(-2).max(-1).values
                captured.append((total.cpu().numpy(), h.cpu().numpy(), peak.cpu().numpy()))
                return t
            op.sinkhorn = trace
            try:
                matcher = op.Matcher(features, refs, scaler, 'cuda', temperature=eps)
                local = matcher.predict(ids, metric, batch=len(ids))
            finally:
                op.sinkhorn = original
            assert len(captured) == 3
            if eps == .1:
                old = load(PREV/'results'/f'seed{seed}/both.npz')['local'][ids]
                np.testing.assert_allclose(local, old, atol=2e-5, rtol=1e-5)
                replay.append(dict(seed=seed, max_local_error=float(abs(local-old).max())))
            for region, (mass, entropy, peak) in enumerate(captured):
                for i, case in enumerate(subset):
                    for cls in (case['true'], case['rival']):
                        for ref in range(2):
                            for pol in range(2):
                                if mass[i, cls, ref, pol] > 0:
                                    rows.append(dict(obs_id=int(obs[ids[i]]), seed=seed,
                                        candidate_class=cls, relationship='true' if cls == case['true'] else 'rival',
                                        reference_obs=int(obs[refs[ids[i], cls, ref]]),
                                        polarization=('VV', 'VH')[pol], region=region, temperature=eps,
                                        matched_mass=float(mass[i, cls, ref, pol]),
                                        conditional_entropy=float(entropy[i, cls, ref, pol]),
                                        largest_pair_share=float(peak[i, cls, ref, pol])))
            del matcher
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE/'case_transport.csv', index=False)
    summary = frame.groupby('temperature')[['matched_mass', 'conditional_entropy', 'largest_pair_share']].mean()
    summary.to_csv(HERE/'case_transport_summary.csv')
    write(HERE/'transport_replay.json', dict(replay=replay, reviewed_cases=len(cases), new_training=False,
        interpretation='Case-level matched-plan diagnostics, not full-cohort proportions or isolated causal effect of temperature.'))
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    main()
