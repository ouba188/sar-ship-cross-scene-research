"""Check that cached expanded matches equal the selected standalone operator."""
import numpy as np
import torch
import evaluate as run
from sar_m123 import m3, tokens


def main():
    torch.set_num_threads(2)
    runtime = m3.M3(run.PACKAGE/'assets')
    tok = np.load(run.EXT/'tokens.npz')
    mean = (tok['layer1']-runtime.projection['mean'])@runtime.projection['components'].T
    mean[tok['mass'] <= 0] = 0
    # Include a missing-VH query to detect availability/empty-region mismatches.
    rows = np.array([0, 1, int(np.flatnonzero(~tok['valid'][:, 1])[0])])
    feat = dict(x=tokens.unit(mean[rows]), mass=tok['mass'][rows],
                joint=np.load(run.COV/'joint_external.npz')['joint'][rows])
    checks = []
    for seed in run.SEEDS:
        z = np.load(run.HERE/'results'/f'seed{seed}.npz')
        with torch.inference_mode():
            local = runtime.match(feat, z['refs'][rows], runtime._load(seed))
        expected = np.load(run.COV/'results'/f'expanded_nearest2_{seed}.npz')['both_local'][rows]
        error = float(abs(local-expected).max())
        assert error < 3e-5, (seed, error)
        checks.append(dict(seed=seed, rows=rows.tolist(), max_local_error=error))
    run.write_json(run.HERE/'matching_replay_checks.json', checks)
    print(checks)


if __name__ == '__main__':
    main()
