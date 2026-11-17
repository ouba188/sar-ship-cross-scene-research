"""Reuse the matched four-epoch runner; cache label-free same-input teacher scores."""
from pathlib import Path
import argparse
import importlib.util
import sys
import gc

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('resolution_runner_previous', ROOT.parent/'pcea_conditional_modulation_20261117/run.py')
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)
runner.ROOT = ROOT
runner.ARMS = ('local', 'competition', 'both')
runner.CONFIGS = {k: runner.CONFIGS[k] for k in ('base', 'slow')}
original_fit = runner.fit


def teacher(model, inputs, seed):
    if getattr(inputs, 'teacher_seed', None) == seed:
        return
    torch, np, full = model.torch, model.np, model.full
    path = ROOT/f'teacher_s{seed}.npz'
    if not path.exists():
        inputs.teacher = None
        net = full.LiveFull(seed, 'full')
        net.load_state_dict(torch.load(full.ROOT/f'selected_s{seed}/best.pt', map_location='cpu', weights_only=True))
        net.requires_grad_(False).eval()
        net.accelerate()
        arrays = {}
        with torch.inference_mode():
            ref = net.reference(inputs.data(inputs.groups['references'], 'small'))
            for split in ('small', 'validation'):
                ids = inputs.groups[split]
                allcodes = []
                for code in range(8 if split == 'small' else 1):
                    scores = []
                    for first in range(0, len(ids), 12):
                        part = ids[first:first+12]
                        z, _ = net(inputs.data(part, split, code), [inputs.rows[i] for i in part], ref, inputs.reference_rows)
                        scores.append(z.cpu().numpy())
                    allcodes.append(np.concatenate(scores))
                    print('TEACHER', seed, split, code, flush=True)
                arrays[split] = np.stack(allcodes, 1)
                arrays[split+'_ids'] = ids
        runner.replay(model, arrays['validation_ids'], arrays['validation'][:, 0], full.ROOT/f'selected_s{seed}/validation.npz')
        np.savez_compressed(path, **arrays)
        del net, ref
        gc.collect()
        torch.cuda.empty_cache()
    with np.load(path, allow_pickle=False) as cached:
        inputs.teacher = {}
        for split in ('small', 'validation'):
            assert np.array_equal(cached[split+'_ids'], inputs.groups[split])
            inputs.teacher[split] = torch.as_tensor(cached[split], device='cuda')
    inputs.teacher_seed = seed


def fit(model, inputs, arm, seed, config):
    teacher(model, inputs, seed)
    return original_fit(model, inputs, arm, seed, config)


runner.fit = fit


def runtime():
    model = runner.load('resolution_model', ROOT/'model.py')
    class Inputs(model.full.Inputs):
        teacher = None
        def data(self, ids, split, code=0):
            data = super().data(ids, split, code)
            if self.teacher is not None and split in self.teacher:
                lookup = self.resident[split][0]
                ix = model.torch.as_tensor([lookup[int(i)] for i in ids], device='cuda')
                data['teacher_score'] = self.teacher[split][ix, code]
            return data
    inputs = Inputs()
    for split in ('small', 'validation'):
        inputs.prepare_geometry(split)
    return model, inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('tune', 'confirm', 'smoke'), default='tune')
    parser.add_argument('--arms', nargs='+', choices=runner.ARMS, default=list(runner.ARMS))
    args = parser.parse_args()
    model, inputs = runtime()
    if args.phase == 'smoke':
        teacher(model, inputs, 17)
        for arm in args.arms:
            print(runner.test_gradients(model, inputs, arm), flush=True)
    elif args.phase == 'tune':
        runner.tune(model, inputs, args.arms)
    else:
        runner.confirm(model, inputs, args.arms)


if __name__ == '__main__':
    main()
