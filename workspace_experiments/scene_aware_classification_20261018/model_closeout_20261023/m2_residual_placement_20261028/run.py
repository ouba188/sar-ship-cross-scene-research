"""Equal-width M2 at residual output, post-add/pre-ReLU, and stage output."""
import importlib.util
import json
from pathlib import Path
import argparse
import gc

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent/'m2_block_placement_20261028'
spec = importlib.util.spec_from_file_location('previous_position_runner', PREVIOUS/'run.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
np, torch = previous.np, previous.torch
legacy, RELEASE = previous.legacy, previous.RELEASE
Cache, construct = previous.Cache, previous.construct
setup, read_manifest, pair_mask = previous.setup, previous.read_manifest, previous.pair_mask
ARMS = dict(baseline=(), stage='stage', preadd='preadd', postadd='postadd')
SEEDS = (17, 29, 43)


def write(name, value):
    (ROOT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def check(state, records, caches, bank):
    q = caches['source'].get(np.arange(4))
    legal = pair_mask(records['source'][:4], records['refs'], 'cuda')
    base = construct(state, (), 17)
    with torch.no_grad():
        expected = base(q, bank, legal)
    del base
    report = dict(initial_identity={}, gradient={}, native_block_hook_parity={},
                  cached_vs_live_prefix={}, reused_checkpoints={}, extra_parameters={})
    for arm, position in ARMS.items():
        m = construct(state, position, 17)
        report['extra_parameters'][arm] = sum(p.numel() for p in m.early.parameters())
        with torch.no_grad():
            error = float((m(q, bank, legal)-expected).abs().max())
        assert error < 1e-4, (arm, 'identity', error)
        report['initial_identity'][arm] = error
        if arm in ('preadd', 'postadd'):
            opt = torch.optim.AdamW(m.early.parameters(), lr=3e-4)
            labels = torch.tensor([r['label'] for r in records['source'][:4]], device='cuda')
            for step in range(4):
                opt.zero_grad(set_to_none=True)
                torch.nn.functional.cross_entropy(m(q, bank, legal), labels).backward()
                for key, module in m.early.items():
                    output = sum(float(t[-1].weight.grad.abs().sum()) for t in module.transforms)
                    inner = sum(float(t[0].weight.grad.abs().sum()) for t in module.transforms)
                    assert output > 0, (arm, key, step, 'output')
                    if step == 3:
                        assert inner > 0, (arm, key, 'inner')
                        report['gradient'][f'{arm}_{key}'] = dict(output=output, inner=inner)
                opt.step()
            with torch.no_grad():
                direct = m.stage2(q, {})
                block = m.base.backbone.model.layer2[-1]
                if arm == 'preadd':
                    hook = block.bn3.register_forward_hook(lambda module, args, value: m.feedback(value, q, {}))
                else:
                    # ResNet reuses this ReLU three times; intercept only the final addition.
                    calls = [0]
                    def before_relu(module, args):
                        calls[0] += 1
                        return (m.feedback(args[0], q, {}),) if calls[0] == 3 else None
                    hook = block.relu.register_forward_pre_hook(before_relu)
                try:
                    native = block(q['prefix2'].flatten(0, 1)).reshape_as(direct)
                finally:
                    hook.remove()
                if arm == 'postadd':
                    assert calls[0] == 3
                error = float((direct-native).abs().max())
                assert error < 1e-5, (arm, 'native-order', error)
                report['native_block_hook_parity'][arm] = error
                live = {k:v for k,v in q.items() if k != 'prefix2'}
                error = float((m(q, bank, legal)-m(live, bank, legal)).abs().max())
                assert error < 1e-4, (arm, 'cache', error)
                report['cached_vs_live_prefix'][arm] = error
            del opt
        del m
        gc.collect(); torch.cuda.empty_cache()
    assert len({report['extra_parameters'][a] for a in ('stage','preadd','postadd')}) == 1
    original = previous.PREVIOUS
    for arm, oldarm in [('baseline','baseline'), ('stage','layer2')]:
        for seed in SEEDS:
            checkpoint = torch.load(original/'checkpoints'/f'{oldarm}_{seed}.pt', weights_only=True, map_location='cpu')
            m = construct(state, ARMS[arm], seed)
            assert {n for n,p in m.named_parameters() if p.requires_grad} == set(checkpoint['state'])
            m.load_state_dict(checkpoint['state'], strict=False)
            score = legacy.predict(m, caches['dev'], records['dev'][:16], bank, records['refs'], 16)
            saved = np.load(original/f'completed_{oldarm}_{seed}.npz')['score'][:16]
            error = float(abs(saved-score).max())
            assert error < 1e-4, (arm, seed, 'replay', error)
            report['reused_checkpoints'][f'{arm}_{seed}'] = error
            del m
            gc.collect(); torch.cuda.empty_cache()
    write('diagnostics.json', report)
    print('CHECKS PASSED', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('prepare','train'), default='train')
    args = parser.parse_args(); setup('cuda')
    records = {k:read_manifest(RELEASE/'manifests'/f'{v}.json') for k,v in
               [('source','source_adapt'),('dev','development'),('refs','references')]}
    selected = json.loads((RELEASE/'manifests/selection_archived.json').read_text(encoding='utf-8'))
    lookup = {str(r['id']):r for r in selected}
    selection = [lookup[str(r['id'])] for r in records['source']]
    state = torch.load(RELEASE/'weights/teacher.pt', weights_only=True, map_location='cpu')['state']
    for name, rows in records.items():
        ids = json.loads((previous.PREVIOUS/'cache'/name/'complete.json').read_text())['ids']
        assert ids == [str(r['id']) for r in rows]
    caches = {name:Cache(name) for name in records}
    bank = caches['refs'].get(np.arange(len(records['refs'])))
    if args.mode == 'prepare':
        check(state, records, caches, bank)
        return
    if not (ROOT/'diagnostics.json').exists():
        raise RuntimeError('Run prepare first.')
    legacy.ROOT, legacy.ARMS, legacy.construct = ROOT, ARMS, construct
    old = np.load(PREVIOUS/'predictions.npz')
    assert old['ids'].tolist() == [str(r['id']) for r in records['dev']]
    scores = np.zeros((4,3,len(records['dev']),8), np.float32)
    bypasses = scores.copy(); runs = []
    for ai, arm in enumerate(ARMS):
        for si, seed in enumerate(SEEDS):
            if arm in ('baseline','stage'):
                old_ai = old['arms'].tolist().index(arm)
                score, bypass = old['scores'][old_ai,si], old['bypasses'][old_ai,si]
                receipt = next(r.copy() for r in json.loads((PREVIOUS/'runs.json').read_text()) if r['arm']==arm and r['seed']==seed)
                receipt['reused_from'] = str(PREVIOUS/'runs.json')
            elif (ROOT/f'completed_{arm}_{seed}.npz').exists():
                saved = np.load(ROOT/f'completed_{arm}_{seed}.npz')
                score, bypass = saved['score'], saved['bypass']
                receipt = json.loads((ROOT/f'run_{arm}_{seed}.json').read_text())
            else:
                score, bypass, receipt = legacy.train_one(state, arm, seed, caches['source'], caches['dev'],
                    records['source'], records['dev'], bank, records['refs'], selection, 16)
                receipt.update(model_class='Layer2BlockM2', position=ARMS[arm],
                               new_m2_per_polarization=1, adapter_channels=512, grid=[20,20])
                np.savez(ROOT/f'completed_{arm}_{seed}.npz', score=score, bypass=bypass)
                write(f'run_{arm}_{seed}.json', receipt)
            scores[ai,si] = score; bypasses[ai,si] = bypass; runs.append(receipt)
            write('runs.json', runs)
    np.savez(ROOT/'predictions.npz', ids=old['ids'], labels=old['labels'], initial=old['initial'],
             scores=scores, bypasses=bypasses, arms=np.array(list(ARMS)), seeds=np.array(SEEDS))
    write('complete.json', dict(new_fits=6, reused_fits=6, endpoint='fixed_epoch6', tuning=False,
                               matched_width=512, default_model_changed=False))
    print('COMPLETE: six new equal-width fits, six reused controls', flush=True)


if __name__ == '__main__':
    main()
