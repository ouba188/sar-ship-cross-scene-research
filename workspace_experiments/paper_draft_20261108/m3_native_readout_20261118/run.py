"""Bounded matched continuation for the native-resolution M3 readout repair."""
from pathlib import Path
import argparse
import gc
import json
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
spec_path = ROOT/'model.py'
import importlib.util
spec = importlib.util.spec_from_file_location('native_readout_model', spec_path)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
torch, np = m.torch, m.np
recipe = m.load('native_readout_recipe', ROOT.parent/'pcea_class_local_20261117/run.py')
compact = m.load('native_readout_compact', ROOT.parent/'size_adaptive_slots_20261118/run.py')
BASE = ROOT.parent/'m3_coarse_supplement_20261118/fits/continue_s17'
ARMS = ('fine_anchor', 'fine_live')
RATES = dict(backbone=1e-6, modules=1e-6, reader=1e-5, exchange=1e-5)
write, read = recipe.write, recipe.read


def runtime():
    recipe.ROOT = ROOT
    recipe.m.Inputs = m.Inputs
    inputs, fit, inner = recipe.runtime()
    recipe.m = m
    recipe.RATES, recipe.STEPS, recipe.INTERVAL = RATES, 120, 30
    config = read(BASE/'config.json')
    assert all(config['rates'][k] == v for k, v in RATES.items())
    assert config['steps'] == 120 and config['initialization'] == str(m.START)
    order = np.random.default_rng(17+20261117).permutation(fit)[:120*24]
    assert np.array_equal(order, np.load(BASE/'training_ids_local.npy'))
    assert np.array_equal(inner, np.load(BASE/'inner.npz')['ids'])
    scope = read(ROOT/'scope.json')
    scope.update(all_arms_start=str(m.START), arms=list(ARMS), steps=120, seed=17,
                 reused_baseline=str(BASE), rates=RATES, parameters_added=0,
                 test3075_read=False)
    write(ROOT/'scope.json', scope)
    return inputs, fit, inner


def evaluation_inputs():
    compact.m, compact.torch, compact.np = m.old, torch, np
    compact.ROOT = ROOT/'evaluation_inputs'
    compact.ROOT.mkdir(parents=True, exist_ok=True)
    inputs = compact.Inputs()
    native = m.NativeReads(inputs.rows)
    original = inputs.data
    inputs.data = lambda ids: dict(original(ids), read40=native.batch(ids))
    return inputs


def scores(net, inputs, ids, reference):
    values = []
    for first in range(0, len(ids), 12):
        part = ids[first:first+12]
        z, _ = net(inputs.data(part), [inputs.rows[i] for i in part], reference,
                   [inputs.rows[i] for i in inputs.refs])
        values.append(z.detach().cpu().numpy())
    return np.concatenate(values)


def smoke(inputs, fit):
    ids = fit[:4]
    data = inputs.data(ids, 'small')
    net = m.Model('legacy')
    old = m.old.Model('generic')
    old.load_state_dict(torch.load(m.START, map_location='cuda', weights_only=True), strict=True)
    baseline = m.full.predict(old, inputs, ids, 'small')
    replay = m.full.predict(net, inputs, ids, 'small')
    error = float(abs(baseline-replay).max())
    assert error < 1e-6, error
    del old
    refdata = inputs.data(inputs.groups['references'], 'small')
    with torch.no_grad():
        a0, c0 = net.extract(data, True)
        reference = net.reference(refdata)
        off0 = net.finish(a0, c0, [inputs.rows[i] for i in ids], reference,
                          inputs.reference_rows, False)[0]
    del net
    gc.collect()
    torch.cuda.empty_cache()
    records = {}
    for arm in ARMS:
        net = m.Model(arm)
        with torch.no_grad():
            a, c = net.extract(data, True)
            reference = net.reference(refdata)
            off = net.finish(a, c, [inputs.rows[i] for i in ids], reference,
                             inputs.reference_rows, False)[0]
        off_error = float((off-off0).abs().max())
        assert off_error < 1e-5, off_error
        for key in ('mass', 'write', 'slots', 'available'):
            assert torch.equal(a[key], a0[key]), key
        delta = float((a['desc']-a0['desc']).abs().max())
        assert delta > 1e-6
        net.train()
        loss, z = m.full.loss_for(net, inputs, ids, 1)
        loss.backward()
        groups = recipe.groups_for(net)
        norms = {g['name']: sum(float(p.grad.square().sum()) for p in g['params']
                               if p.grad is not None)**.5 for g in groups}
        assert all(np.isfinite(v) and v > 0 for v in norms.values()), norms
        net.zero_grad(set_to_none=True)
        direct, _ = net.extract(data, True)
        direct['desc'][..., 256:].square().mean().backward()
        m1_norm = sum(float(p.grad.square().sum()) for p in net.net.m1.parameters()
                      if p.grad is not None)**.5
        assert (m1_norm > 0) == (arm == 'fine_live'), m1_norm
        records[arm] = dict(m3_off_max_error=off_error, descriptor_change=delta,
                            loss=float(loss.detach()), gradients=norms,
                            direct_m1_descriptor_gradient=m1_norm)
        del net, loss, z, groups, direct
        gc.collect()
        torch.cuda.empty_cache()
    # A high-resolution perturbation invisible to the exact existing downsampler.
    w = torch.tensor(m.geometry._downsample_weights(), dtype=torch.float64, device='cuda')
    generator = torch.Generator(device='cuda').manual_seed(17)
    candidate = torch.randn(40, 40, generator=generator, device='cuda', dtype=torch.float64)
    null = torch.eye(40, dtype=w.dtype, device=w.device)-w.T@torch.linalg.solve(w@w.T, w)
    detail = (null@candidate)[None, None]
    f2 = torch.zeros(1, 1, 20, 20, device='cuda', dtype=torch.float64)
    coarse = m.raw.pool_multiscale(detail, f2, data['read'][:1, 0].double(), 'coarse_fixed')
    fine = m.raw.pool_multiscale(detail, f2, data['read40'][:1, 0].double(), 'fine_fixed')
    coarse_max, fine_max = float(coarse.abs().max()), float(fine.abs().max())
    assert coarse_max < 1e-10 and fine_max > 1e-5, (coarse_max, fine_max)
    # All geometric transforms retain the same native slot areas.
    area_error = 0.
    cell_area = torch.tensor(m.geometry._cell_area(40), device='cuda')
    for code in range(8):
        other = inputs.data(ids[:1], 'small', code)['read40']
        area_error = max(area_error, float(((other*cell_area).sum((-2, -1))-
                                  (data['read40'][:1]*cell_area).sum((-2, -1))).abs().max()))
    assert area_error < 1e-3, area_error
    write(ROOT/'smoke.json', dict(passed=True, replay_error=error, arms=records,
          invisible_coarse_detail=coarse_max, visible_native_detail=fine_max,
          augmentation_native_area_error=area_error, unchanged_write_and_mass=True))
    print('SMOKE PASSED', json.dumps(read(ROOT/'smoke.json')), flush=True)


@torch.no_grad()
def evaluate():
    assert all((ROOT/'fits'/f'{a}_s17/receipt.json').exists() for a in ARMS)
    inputs = evaluation_inputs()
    records = []
    for arm in ('legacy',)+ARMS:
        dest = BASE if arm == 'legacy' else ROOT/'fits'/f'{arm}_s17'
        net = m.Model(arm)
        net.load_state_dict(torch.load(dest/'best.pt', map_location='cuda', weights_only=True), strict=True)
        net.requires_grad_(False).eval()
        net.accelerate()
        reference = net.reference(inputs.data(inputs.refs))
        for split, ids in inputs.splits.items():
            output = ROOT/'evaluation'/f'{arm}_{split}_local.npz'
            if output.exists():
                saved = np.load(output)
                assert np.array_equal(saved['ids'], ids)
                score = saved['score']
            else:
                score = scores(net, inputs, ids, reference)
            if arm == 'legacy':
                cached = np.load(BASE/f'{split}_on_local.npz')
                assert np.array_equal(cached['ids'], ids)
                assert np.max(abs(cached['score']-score)) < 2e-5
            output.parent.mkdir(exist_ok=True)
            np.savez_compressed(output, ids=ids, label=inputs.y[ids], score=score)
            metrics = m.full.metrics(inputs.y[ids], score)
            records.append(dict(arm=arm, split=split,
                selected_step=read(dest/'receipt.json')['best']['step'], **metrics))
            write(ROOT/'results.json', records)
            print('EVAL', arm, split, metrics['oa'], metrics['ba'], flush=True)
        if arm != 'legacy':
            # Same selected weights; change only the M3 evidence reader on both sides.
            net.readout_arm = 'legacy'
            reference = net.reference(inputs.data(inputs.refs))
            for split, ids in inputs.splits.items():
                score = scores(net, inputs, ids, reference)
                np.savez_compressed(ROOT/'evaluation'/f'{arm}_legacyread_{split}_local.npz',
                                    ids=ids, label=inputs.y[ids], score=score)
                metrics = m.full.metrics(inputs.y[ids], score)
                records.append(dict(arm=arm+'_legacyread', split=split,
                    selected_step=read(dest/'receipt.json')['best']['step'], **metrics))
                write(ROOT/'results.json', records)
                print('EVAL', arm, 'legacyread', split, metrics['oa'], metrics['ba'], flush=True)
        del net
        gc.collect()
        torch.cuda.empty_cache()
    write(ROOT/'status.json', dict(phase='complete', new_fits=2, reused_fits=1,
                                   evaluations=10, test3075_read=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('smoke', 'train', 'evaluate'), default='smoke')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.phase == 'evaluate':
        evaluate()
    else:
        inputs, fit, inner = runtime()
        if args.phase == 'smoke':
            smoke(inputs, fit)
        else:
            assert read(ROOT/'smoke.json')['passed']
            for arm in ARMS:
                recipe.fit_one(inputs, fit, inner, arm, 17)
            del inputs
            gc.collect()
            evaluate()
