"""Complete a paired 2x2 polarization intervention using existing single-arm caches."""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'region_dependency_20261116'
spec = importlib.util.spec_from_file_location('regional_diagnosis', OLD / 'run.py')
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
np, torch = base.np, base.torch
original_perturb = base.perturb


def joint_perturb(images, masks, valid, ids, condition):
    if condition['kind'] != 'joint':
        return original_perturb(images, masks, valid, ids, condition)
    keep = 7 ^ (1 << condition['r'])
    out = original_perturb(images, masks, valid, ids,
                           dict(kind='coalition', p=0, keep=keep))
    return original_perturb(out, masks, valid, ids,
                            dict(kind='coalition', p=1, keep=keep))


def check_operation(inputs):
    images = torch.from_numpy(np.array(inputs.arrays['pixels'][inputs.ids])).float().unsqueeze(2) / 255
    valid = torch.from_numpy(np.array(inputs.arrays['footprint'][inputs.ids])).float().unsqueeze(2)
    masks = np.asarray(inputs.masks)
    errors = []
    for r in range(3):
        both = joint_perturb(images, masks, valid, inputs.ids, dict(kind='joint', r=r))
        vv = original_perturb(images, masks, valid, inputs.ids, dict(kind='coalition', p=0, keep=7 ^ (1 << r)))
        vh = original_perturb(images, masks, valid, inputs.ids, dict(kind='coalition', p=1, keep=7 ^ (1 << r)))
        assert torch.equal(both[:, 0], vv[:, 0]) and torch.equal(both[:, 1], vh[:, 1])
        for p in range(2):
            m = torch.from_numpy(masks[:, p, r].copy())
            delta = both[:, p, 0] - images[:, p, 0]
            assert torch.equal(both[:, p, 0][~m], images[:, p, 0][~m])
            err = (delta * m).sum((-2, -1)).abs() / m.sum((-2, -1)).clamp_min(1)
            assert err.max() < 1e-6
            errors.append(float(err.max()))
        assert torch.isfinite(both).all()
    return dict(max_regional_mean_error=max(errors), other_pixels_unchanged=True,
                each_joint_channel_equals_single_operation=True)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    inputs = base.Inputs()
    inputs.prepare_masks()
    receipt = dict(checks=check_operation(inputs), seeds=[], training=False,
                   original_cache=str(OLD), formal_model_changed=False)
    base.write(ROOT / 'receipt.json', receipt)
    print(json.dumps(receipt['checks']), flush=True)
    if args.check_only:
        return
    base.perturb = joint_perturb
    try:
        for seed in base.SEEDS:
            folder = ROOT / 'predictions' / f'full_s{seed}'
            folder.mkdir(parents=True, exist_ok=True)
            net, _ = base.model(seed, 'full')
            reference = net.reference(inputs.data(inputs.rids, 'small'))
            clean = base.predict(net, inputs, reference, 'full', 'full', dict(kind='clean'), base.MICRO)
            cached = np.load(OLD / 'predictions' / f'full_s{seed}' / 'clean.npz')
            assert np.array_equal(cached['ids'], inputs.ids)
            error = float(abs(clean - cached['score'][:len(clean)]).max())
            assert error < 1e-4 and np.array_equal(clean.argmax(1), cached['score'][:len(clean)].argmax(1))
            entry = dict(seed=seed, clean_replay_error=error, regions=[])
            for r, region in enumerate(base.REGIONS):
                path = folder / f'both_{region}_mean50.npz'
                started = time.perf_counter()
                if not path.exists():
                    z = base.predict(net, inputs, reference, 'full', 'full', dict(kind='joint', r=r))
                    np.savez_compressed(path, ids=inputs.ids, y=inputs.y, score=z)
                entry['regions'].append(dict(region=region, seconds=time.perf_counter() - started))
                print(json.dumps(dict(seed=seed, region=region, seconds=entry['regions'][-1]['seconds'])), flush=True)
            receipt['seeds'].append(entry)
            base.write(ROOT / 'receipt.json', receipt)
            del net, reference
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        base.perturb = original_perturb


if __name__ == '__main__':
    main()
