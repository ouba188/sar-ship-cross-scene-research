"""Fixed-budget M3 source replay; the rest of the complete model stays fixed."""
from diagnose import *
from collections import defaultdict


def pools(rows):
    result = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        result[r['label']][str(r['mmsi'])].append(i)
    return {c: list(v.values()) for c, v in result.items()}


def draw(pool, labels, rng):
    return [int(rng.choice(pool[c][int(rng.integers(len(pool[c])))])) for c in labels]


def main(dataset, seeds):
    setup()
    cache, (fit, adapt, queries, refs) = data(dataset)
    bank = cache.take(refs)
    labels = np.array([r['label'] for r in queries])
    ap, fp = pools(adapt), pools(fit)
    assert set(ap) == set(fp)
    classes = sorted(ap)
    nsteps = 6*((len(adapt)+15)//16)
    for seed in seeds:
        net = original(dataset, seed)
        before = cache.predict(net, queries, refs, bank)
        at = cache.predict(net, adapt, refs, bank)
        ft = cache.predict(net, fit, refs, bank)
        del net
        # Same labels and first-half observations in both arms; only second-half provenance differs.
        rng, replay_rng = np.random.default_rng(seed), np.random.default_rng(seed+10000)
        schedule = []
        for _ in range(nsteps):
            c = rng.choice(classes, 16).tolist()
            ai = draw(ap, c, rng)
            fi = draw(fp, c[8:], replay_rng)
            schedule.append((ai, fi))
        for arm in ('adapt_only', 'source_mix'):
            out = HERE/dataset/'training'
            out.mkdir(parents=True, exist_ok=True)
            if (out/f'{arm}_s{seed}.json').exists():
                raise FileExistsError(out/f'{arm}_s{seed}.json')
            net = original(dataset, seed)
            selected = parameters(net)
            initial_parameters = torch.cat([p.detach().flatten().cpu() for n,p in selected])
            names = {n for n,p in selected}
            fixed = {n:v.detach().cpu().clone() for n,v in net.state_dict().items() if n not in names}
            opt = torch.optim.AdamW([p for n,p in selected], lr=1e-4, weight_decay=.01)
            curve, exposures = [], []
            start = time.perf_counter()
            for step, (ai, fi) in enumerate(schedule):
                if arm == 'adapt_only':
                    q, t = [adapt[i] for i in ai], at[ai]
                    provenance = ['source_adapt']*16
                else:
                    q = [adapt[i] for i in ai[:8]]+[fit[i] for i in fi]
                    t = np.concatenate([at[ai[:8]], ft[fi]])
                    provenance = ['source_adapt']*8+['source_fit']*8
                y = torch.tensor([r['label'] for r in q], device='cuda')
                teacher = torch.as_tensor(t, device='cuda')
                opt.zero_grad(set_to_none=True)
                z = cache.score(net, q, refs, bank)
                ce = F.cross_entropy(z, y)
                kl = F.kl_div(z.log_softmax(1), teacher.log_softmax(1), log_target=True, reduction='none').sum(1)
                keep = (teacher.argmax(1).eq(y)*kl).mean()
                loss = ce+.1*keep
                assert torch.isfinite(loss)
                loss.backward()
                opt.step()
                curve.append(dict(step=step+1, ce=float(ce.detach()), keep=float(keep.detach())))
                exposures.extend(dict(step=step+1, id=str(r['id']), label=r['label'], split=s)
                                 for r,s in zip(q,provenance))
                if (step+1)%((len(adapt)+15)//16) == 0:
                    print('REPLAY', dataset, seed, arm, curve[-1], flush=True)
            assert all(torch.equal(net.state_dict()[n].detach().cpu(), v) for n,v in fixed.items())
            net.requires_grad_(False)
            score = cache.predict(net, queries, refs, bank)
            pred = HERE/dataset/'predictions'
            pred.mkdir(parents=True, exist_ok=True)
            np.savez(pred/f'{arm}_s{seed}.npz', score=score, baseline=before, y=labels,
                     ids=np.array([str(r['id']) for r in queries]))
            torch.save(dict(dataset=dataset, arm=arm, state={n:v.detach().cpu() for n,v in net.state_dict().items()}),
                       out/f'{arm}_s{seed}.pt')
            report = dict(dataset=dataset, arm=arm, seed=seed, source_adapt=len(adapt), source_fit=len(fit),
                query_count=len(queries), reference_count=len(refs), reference_ids=[str(r['id']) for r in refs],
                steps=nsteps, batch_size=16, sample_presentations=16*nsteps, lr=1e-4,
                trainable_names=sorted(names), frozen_unchanged=True, baseline=metrics(before, labels),
                after=metrics(score, labels), curve=curve, seconds=time.perf_counter()-start,
                source_unique_ids=len({e['id'] for e in exposures}), evaluation_used=False)
            delta = torch.cat([p.detach().flatten().cpu() for n,p in selected])-initial_parameters
            report['parameter_delta_norm'] = float(delta.norm())
            gradpath = HERE/dataset/'diagnostics'/f'gradients_s{seed}.pt'
            if gradpath.exists():
                grads = torch.load(gradpath, weights_only=True)
                report['initial_gradient_dot_actual_delta'] = {k:float(g@delta) for k,g in grads.items()}
                report['negative_gradient_cosine_actual_delta'] = {k:cosine(-g,delta) for k,g in grads.items()}
            write(HERE/dataset/'exposures'/f'{arm}_s{seed}.json', exposures)
            del net, opt, selected, fixed, loss, z
            restored = original(dataset, seed)
            blob = torch.load(out/f'{arm}_s{seed}.pt', map_location='cpu', weights_only=True)
            restored.load_state_dict(blob['state'], strict=True)
            q = queries[:4]
            replay = cache.predict(restored, q, refs, bank)
            report['restore_error'] = float(abs(replay-score[:4]).max())
            values = batch([prepare_record(r) for r in q], 'cuda')
            native_bank = batch([prepare_record(r) for r in refs], 'cuda')
            with torch.no_grad():
                native_bank['prefix'] = restored.prefix(native_bank['images'], stop=2)
                native = forward_batch(restored, values, native_bank, pair_mask(q, refs, 'cuda')).cpu().numpy()
            report['native_error'] = float(abs(native-score[:4]).max())
            assert max(report['restore_error'], report['native_error']) < 1e-4
            write(out/f'{arm}_s{seed}.json', report)
            print('COMPLETE', dataset, arm, seed, report['baseline'], report['after'], flush=True)
            del restored, blob, values, native_bank


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['private','oss2'], required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[17])
    a = p.parse_args()
    main(a.dataset, a.seeds)
