"""Center M3 comparisons within the original adaptation budget, not after it."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'m3_reference_coverage_20261030'))
from center import *
from sar_m123.current.model import LIVE, training_policy
from sar_m123.current.training import adaptation_loss

HERE = Path(__file__).resolve().parent


def restore(path):
    blob = torch.load(path, map_location='cpu', weights_only=True)
    net = (SpatialM12ResNet50(None, 'pre3x3', 'local', 8)
           if blob['dataset'] == 'private' else api.AblationModel('full', 4))
    if blob['arm'] == 'center_joint':
        for p, old in enumerate(net.m3[0]):
            m = CenteredMatch(dynamic=old.dynamic, frequency=old.frequency, temperature=old.temperature)
            for i in range(5):
                m.register_buffer(f'center_{i}', torch.zeros_like(blob['state'][f'm3.0.{p}.center_{i}']))
            m.centered = True
            net.m3[0][p] = m
    net.load_state_dict(blob['state'], strict=True)
    return net.cuda().eval().requires_grad_(False)


def main(dataset, seeds):
    setup()
    if dataset == 'private':
        cache = PrivateCache()
        add_adapt(cache.rows, cache.arrays, cache.indices)
        cache.lookup.update({str(r['id']): int(i) for r, i in zip(cache.rows['adapt'], cache.indices['adapt'])})
        fit, train, queries, refs = [cache.rows[k] for k in ('fit', 'adapt', 'development', 'references')]
    else:
        rows = {k: read_manifest(PUB/'manifests'/f'{k}.json')
                for k in ('source_fit', 'source_adapt', 'validation', 'references')}
        cache = Cache(rows)
        fit, train, queries, refs = [rows[k] for k in ('source_fit', 'source_adapt', 'validation', 'references')]
    bank = cache.take(refs)
    weights = class_weights(train, 'adapt')
    labels = np.array([r['label'] for r in queries])
    for seed in seeds:
        if dataset == 'private':
            fitpath = ROOT/'spatial_m12_refit_20261029/training'/f'spatial_s{seed}/model.pt'
            originalpath = ROOT/'spatial_m12_full_20261029/training'/f'spatial_s{seed}/model.pt'
            teacherpath = ROOT/'spatial_m12_full_20261029/selection'/f'spatial_s{seed}.json'
            loader = load_model
        else:
            fitpath = PUB/'training'/f'full_s{seed}/fit/model.pt'
            originalpath = PUB/'training'/f'full_s{seed}/adapt/model.pt'
            teacherpath = PUB/'teachers'/f'full_s{seed}.json'
            loader = api.load
        teachers = json.loads(teacherpath.read_text(encoding='utf-8'))
        assert [(str(t['id']), t['label']) for t in teachers] == [(str(r['id']), r['label']) for r in train]
        net = loader(originalpath, 'cuda')
        baseline = cache.predict(net, queries, refs, bank)
        del net
        for arm in ('raw_joint', 'center_joint'):
            out = HERE/dataset/'training'
            out.mkdir(parents=True, exist_ok=True)
            if (out/f'{arm}_s{seed}.json').exists():
                raise FileExistsError(out/f'{arm}_s{seed}.json')
            net = loader(fitpath, 'cuda')
            before = cache.predict(net, queries, refs, bank)
            teacher_score = cache.predict(net, train[:4], refs, bank)
            teacher_error = float(abs(teacher_score-np.array([t['original'] for t in teachers[:4]])).max())
            assert teacher_error < 1e-4, teacher_error
            calibration = None
            if arm == 'center_joint':
                calibration = attach_center(net, cache, fit)
                off = cache.predict(net, queries[:4], refs, bank)
                assert float(abs(off-before[:4]).max()) < 1e-4
                for m in net.m3[0]:
                    m.centered = True
            training_policy(net, 'adapt')
            for n, p in net.named_parameters():
                if n.startswith('m3.') and ('.project.' in n or '.output.' in n or n in {'m3.0.0.gain', 'm3.0.1.gain'}):
                    p.requires_grad_(True)
            selected = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
            names = {n for n, p in selected}
            fixed = {n: v.detach().cpu().clone() for n, v in net.state_dict().items() if n not in names}
            optimizer = torch.optim.AdamW([p for n, p in selected], lr=3e-5, weight_decay=.01)
            rng = np.random.default_rng(seed)
            steps, curve = 0, []
            started = time.perf_counter()
            for epoch in range(1, 7):
                terms = []
                for ix in np.array_split(rng.permutation(len(train)), range(16, len(train), 16)):
                    q = [train[i] for i in ix]
                    y = torch.tensor([r['label'] for r in q], device='cuda')
                    w = torch.tensor(weights[ix], device='cuda')
                    teacher = torch.tensor([teachers[i]['original'] for i in ix], device='cuda')
                    useful = torch.tensor([teachers[i]['useful'] for i in ix], device='cuda')
                    optimizer.zero_grad(set_to_none=True)
                    z = cache.score(net, q, refs, bank)
                    loss, t = adaptation_loss(z, y, teacher, w, useful, .3683066666126251)
                    assert torch.isfinite(loss)
                    loss.backward()
                    if steps == 0:
                        gradients = {n: float(p.grad.norm()) for n, p in selected if p.grad is not None}
                    optimizer.step()
                    steps += 1
                    terms.append({k: float(v.detach()) for k, v in t.items()})
                curve.append(dict(epoch=epoch, **{k: float(np.mean([t[k] for t in terms])) for k in terms[0]}))
                print('JOINT_CENTER', dataset, arm, seed, steps, curve[-1], flush=True)
            assert steps == (54 if dataset == 'private' else 96)
            assert all(torch.equal(net.state_dict()[n].detach().cpu(), v) for n, v in fixed.items())
            net.requires_grad_(False)
            score = cache.predict(net, queries, refs, bank)
            pred = HERE/dataset/'predictions'
            pred.mkdir(parents=True, exist_ok=True)
            np.savez(pred/f'{arm}_s{seed}.npz', score=score, baseline=baseline, y=labels,
                     ids=np.array([str(r['id']) for r in queries]))
            torch.save(dict(dataset=dataset, arm=arm, state={n:v.detach().cpu() for n,v in net.state_dict().items()}),
                       out/f'{arm}_s{seed}.pt')
            report = dict(dataset=dataset, arm=arm, seed=seed, source_count=len(train), query_count=len(queries),
                steps=steps, epochs=6, lr=3e-5, trainable_names=sorted(names), gradients=gradients,
                frozen_unchanged=True, calibration=calibration, teacher_error=teacher_error,
                before=metrics(before, labels), baseline=metrics(baseline, labels), after=metrics(score, labels),
                curve=curve, seconds=time.perf_counter()-started, evaluation_used=False)
            del optimizer, selected, fixed, net, z, loss, t
            net = restore(out/f'{arm}_s{seed}.pt')
            q = queries[:4]
            replay = cache.predict(net, q, refs, bank)
            report['restore_error'] = float(abs(replay-score[:4]).max())
            values = batch([prepare_record(r) for r in q], 'cuda')
            native_bank = batch([prepare_record(r) for r in refs], 'cuda')
            with torch.no_grad():
                native_bank['prefix'] = net.prefix(native_bank['images'], stop=2)
                native = forward_batch(net, values, native_bank, pair_mask(q, refs, 'cuda')).cpu().numpy()
            report['native_error'] = float(abs(native-score[:4]).max())
            assert max(report['restore_error'], report['native_error']) < 1e-4, report
            write(out/f'{arm}_s{seed}.json', report)
            print('COMPLETE', dataset, arm, seed, report['after'], flush=True)
            del net, values, native_bank


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['private', 'oss2'], required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[17])
    args = parser.parse_args()
    main(args.dataset, args.seeds)
