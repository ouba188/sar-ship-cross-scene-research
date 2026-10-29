"""Keep the class-separated M3 operator fixed; compare two source losses."""
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
CLASS_ROOT = HERE.parent / 'm3_class_separated_20261030'
sys.path.insert(0, str(CLASS_ROOT))
spec = importlib.util.spec_from_file_location('margin_class_study', CLASS_ROOT/'experiment.py')
study = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = study
spec.loader.exec_module(study)
p = study.previous
torch, np = study.torch, study.np
from losses import objective


def restore(path):
    blob = torch.load(path, map_location='cpu', weights_only=True)
    net = study.attach(p.original(blob['dataset'], blob['seed']), 'aligned', blob['reference_labels'])
    net.load_state_dict(blob['state'], strict=True)
    return net.eval().requires_grad_(False)


def main(dataset, seeds):
    p.setup()
    cache, (_, source, queries, refs) = p.data(dataset)
    bank = cache.take(refs)
    lookup = {str(r['id']): i for i, r in enumerate(source)}
    sy = np.array([r['label'] for r in source])
    y = np.array([r['label'] for r in queries])
    ref_labels = [r['label'] for r in refs]
    for seed in seeds:
        groups = p.json.loads((study.PREVIOUS/dataset/'exposures'/f'schedule_s{seed}.json').read_text())
        schedule = [np.array([lookup[str(k)] for k in g]) for g in groups]
        old = p.original(dataset, seed)
        baseline = cache.predict(old, queries, refs, bank)
        small_baseline = cache.predict(old, queries[:4], refs, bank)
        teacher = cache.predict(old, source, refs, bank)
        with np.load(CLASS_ROOT/dataset/'predictions'/f'aligned_s{seed}.npz') as saved:
            assert np.array_equal(saved['baseline'], baseline)
        del old
        for arm in ('fd_lm', 'margin'):
            out = HERE/dataset/'training'; out.mkdir(parents=True, exist_ok=True)
            receipt = out/f'{arm}_s{seed}.json'
            if receipt.exists():
                raise FileExistsError(receipt)
            net = study.attach(p.original(dataset, seed), 'aligned', ref_labels)
            initial_error = float(abs(cache.predict(net, queries[:4], refs, bank)-small_baseline).max())
            assert initial_error < 1e-6, initial_error
            calibration, distance_metric = study.calibrate(net, cache, source, refs, bank)
            selected = study.configure(net)
            names = {n for n, _ in selected}
            params = [v for _, v in selected]
            fixed = {n:v.detach().cpu().clone() for n,v in net.state_dict().items() if n not in names}
            opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=.01)
            curve = []; gradient = {}; tick = p.time.perf_counter()
            for step, ix in enumerate(schedule):
                group = [source[i] for i in ix]
                target = torch.tensor(teacher[ix], device='cuda')
                labels = torch.tensor(sy[ix], device='cuda')
                opt.zero_grad(set_to_none=True)
                score = cache.score(net, group, refs, bank)
                ce, guard = objective(score, target, labels, arm)
                loss = ce + guard
                assert torch.isfinite(loss)
                if step == 15:
                    gc = torch.cat([g.flatten() for g in torch.autograd.grad(ce, params, retain_graph=True)])
                    gp = torch.cat([g.flatten() for g in torch.autograd.grad(guard, params, retain_graph=True)])
                    gradient = dict(step=step+1, ce_norm=float(gc.norm()), guard_norm=float(gp.norm()),
                        cosine=float(torch.dot(gc, gp)/(gc.norm()*gp.norm()).clamp_min(1e-20)))
                loss.backward()
                if step == 0:
                    assert any(v.grad is not None and v.grad.norm()>0 for v in params)
                opt.step()
                curve.append(dict(step=step+1, ce=float(ce.detach()), guard=float(guard.detach())))
                if (step+1)%16 == 0:
                    print('SOURCE_GUARD', dataset, arm, seed, curve[-1], flush=True)
            assert all(torch.equal(net.state_dict()[n].detach().cpu(),v) for n,v in fixed.items())
            net.requires_grad_(False)
            score = cache.predict(net, queries, refs, bank)
            source_score = cache.predict(net, source, refs, bank)
            pred = HERE/dataset/'predictions'; pred.mkdir(parents=True, exist_ok=True)
            np.savez(pred/f'{arm}_s{seed}.npz', score=score, baseline=baseline, y=y,
                     ids=np.array([str(r['id']) for r in queries]))
            np.savez(pred/f'{arm}_source_s{seed}.npz', score=source_score, baseline=teacher, y=sy,
                     ids=np.array([str(r['id']) for r in source]))
            torch.save(dict(dataset=dataset, seed=seed, arm=arm, reference_labels=ref_labels,
                state={n:v.detach().cpu() for n,v in net.state_dict().items()}), out/f'{arm}_s{seed}.pt')
            for m in net.m3[0]: m.extra_control = 'off'
            off_error = float(abs(cache.predict(net, queries[:4], refs, bank)-small_baseline).max())
            assert off_error < 1e-4
            report = dict(dataset=dataset, seed=seed, arm=arm, operator='class_separated_aligned',
                source_count=len(source), query_count=len(queries), reference_count=len(refs),
                steps=len(schedule), batch_size=16, lr=1e-4, weight_decay=.01,
                trainable_parameters=sum(v.numel() for v in params), trainable_names=sorted(names),
                calibration=calibration, source_distance=distance_metric, gradient=gradient,
                initial_error=initial_error, off_error=off_error, frozen_unchanged=True,
                baseline=p.metrics(baseline,y), after=p.metrics(score,y),
                source_before=p.metrics(teacher,sy), source_after=p.metrics(source_score,sy),
                curve=curve, seconds=p.time.perf_counter()-tick, evaluation_used=False)
            del net, opt, params, selected, fixed, loss, ce, guard, score
            net = restore(out/f'{arm}_s{seed}.pt')
            with np.load(pred/f'{arm}_s{seed}.npz') as saved:
                expected = saved['score'][:4]
            report['restore_error'] = float(abs(cache.predict(net,queries[:4],refs,bank)-expected).max())
            q = p.batch([p.prepare_record(r) for r in queries[:4]], 'cuda')
            r = p.batch([p.prepare_record(v) for v in refs], 'cuda')
            with torch.no_grad():
                r['prefix'] = net.prefix(r['images'], stop=2)
                native = p.forward_batch(net,q,r,p.pair_mask(queries[:4],refs,'cuda')).cpu().numpy()
            report['native_error'] = float(abs(native-expected).max())
            assert max(report['restore_error'],report['native_error']) < 1e-4
            p.write(receipt,report)
            print('COMPLETE_SOURCE_GUARD',dataset,arm,seed,report['after'],report['source_after'],gradient,flush=True)
            del net,q,r


if __name__ == '__main__':
    parser = p.argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['private','oss2'], required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[17])
    args = parser.parse_args()
    main(args.dataset, args.seeds)
