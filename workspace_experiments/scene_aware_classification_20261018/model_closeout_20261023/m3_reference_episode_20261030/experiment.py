"""Change source reference organization, not architecture or query schedule."""
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'm3_pairwise_repair_20261030'
spec = importlib.util.spec_from_file_location('episode_previous', PREVIOUS / 'experiment.py')
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
torch, np, F = previous.torch, previous.np, previous.F
json, time = previous.json, previous.time


def draw_bank(rows, per_class, rng):
    pools = defaultdict(lambda: defaultdict(list))
    for row in rows:
        pools[int(row['label'])][str(row['mmsi'])].append(row)
    selected = []
    for c in sorted(pools):
        ships = sorted(pools[c])
        if len(ships) < per_class:
            raise ValueError((c, len(ships), per_class))
        for ship in rng.choice(ships, size=per_class, replace=False):
            candidates = pools[c][ship]
            selected.append(candidates[int(rng.integers(len(candidates)))])
    return selected


def banks_for_steps(rows, original, steps, seed):
    classes = {r['label'] for r in original}
    count = len(original) // len(classes)
    assert all(sum(r['label'] == c for r in original) == count for c in classes)
    assert classes == {r['label'] for r in rows}
    rng = np.random.default_rng(seed + 103000)
    fixed = draw_bank(rows, count, rng)
    changing = [fixed]
    for _ in range(steps // 2 - 1):
        changing.append(draw_bank(rows, count, rng))
    return fixed, changing


@torch.no_grad()
def source_probe(net, cache, rows, refs, banks):
    y = np.array([r['label'] for r in rows])
    original = cache.predict(net, rows, refs, cache.take(refs))
    result = []
    for group in banks:
        z = cache.predict(net, rows, group, cache.take(group))
        delta = z - original
        centered = delta - delta.mean(1, keepdims=True)
        common = centered.mean(0, keepdims=True)
        total = float((centered ** 2).sum())
        result.append(dict(metrics=previous.metrics(z, y),
                           changed=int((z.argmax(1) != original.argmax(1)).sum()),
                           max_logit_delta=float(abs(delta).max()),
                           common_fraction=float(len(rows)*(common**2).sum()/total) if total else None))
    return dict(original=previous.metrics(original, y), banks=result)


def main(dataset, seeds):
    previous.setup()
    cache, (fit, source, queries, refs) = previous.data(dataset)
    for key in ('id', 'mmsi', 'pass_group'):
        assert not ({str(r[key]) for r in fit} & {str(r[key]) for r in source})
    bank = cache.take(refs)
    lookup = {str(r['id']): i for i, r in enumerate(source)}
    y = np.array([r['label'] for r in queries])
    sy = np.array([r['label'] for r in source])
    for seed in seeds:
        schedule_path = PREVIOUS / dataset / 'exposures' / f'schedule_s{seed}.json'
        schedule = [np.array([lookup[str(k)] for k in ids])
                    for ids in json.loads(schedule_path.read_text())]
        fixed_bank, episodic_banks = banks_for_steps(fit, refs, len(schedule), seed)
        previous.write(HERE/dataset/'exposures'/f'reference_s{seed}.json',
                       dict(fixed=[str(r['id']) for r in fixed_bank],
                            changing=[[str(r['id']) for r in b] for b in episodic_banks]))
        net = previous.original(dataset, seed)
        baseline = cache.predict(net, queries, refs, bank)
        teacher = cache.predict(net, source, refs, bank)
        with np.load(PREVIOUS/dataset/'predictions'/f'ce_s{seed}.npz') as old:
            assert np.array_equal(baseline, old['baseline'])
            assert np.array_equal(old['ids'], [str(r['id']) for r in queries])
        probe = source_probe(net, cache, source, refs, episodic_banks[:4])
        previous.write(HERE/dataset/f'source_probe_s{seed}.json', probe)
        print('SOURCE_BANK_PROBE', dataset, seed, probe, flush=True)
        del net
        for arm in ('fixed_alt', 'episodic'):
            out = HERE/dataset/'training'
            out.mkdir(parents=True, exist_ok=True)
            path = out/f'{arm}_s{seed}.json'
            if path.exists():
                raise FileExistsError(path)
            net = previous.original(dataset, seed)
            selected = previous.trainable(net)
            names = {n for n,p in selected}
            frozen = {n:v.detach().cpu().clone() for n,v in net.state_dict().items() if n not in names}
            opt = torch.optim.AdamW([p for n,p in selected], lr=1e-4, weight_decay=.01)
            curve = []; start = time.perf_counter()
            exposed = set()
            for step, ix in enumerate(schedule):
                group = [source[i] for i in ix]
                chosen = refs if step % 2 == 0 else (fixed_bank if arm == 'fixed_alt' else episodic_banks[step//2])
                exposed.update(str(r['id']) for r in chosen)
                values = bank if step % 2 == 0 else cache.take(chosen)
                labels = torch.tensor(sy[ix], device='cuda')
                target = torch.tensor(teacher[ix], device='cuda')
                opt.zero_grad(set_to_none=True)
                z = cache.score(net, group, chosen, values)
                ce = F.cross_entropy(z, labels)
                kl = F.kl_div(z.log_softmax(1), target.log_softmax(1), log_target=True, reduction='none').sum(1)
                keep = (kl*target.argmax(1).eq(labels)).mean()
                loss = ce + .1*keep
                assert torch.isfinite(loss)
                loss.backward()
                if step == 0:
                    gradients = {n:float(p.grad.norm()) for n,p in selected if p.grad is not None}
                    assert any(v > 0 for v in gradients.values())
                opt.step()
                curve.append(dict(step=step+1, ce=float(ce.detach()), keep=float(keep.detach()), original_bank=step%2==0))
                if (step+1) % 16 == 0:
                    print('EPISODE', dataset, arm, seed, curve[-1], flush=True)
            assert all(torch.equal(net.state_dict()[n].detach().cpu(), v) for n,v in frozen.items())
            net.requires_grad_(False)
            score = cache.predict(net, queries, refs, bank)
            source_score = cache.predict(net, source, refs, bank)
            pred = HERE/dataset/'predictions'; pred.mkdir(parents=True, exist_ok=True)
            np.savez(pred/f'{arm}_s{seed}.npz', score=score, baseline=baseline, y=y, ids=np.array([str(r['id']) for r in queries]))
            np.savez(pred/f'{arm}_source_s{seed}.npz', score=source_score, baseline=teacher, y=sy, ids=np.array([str(r['id']) for r in source]))
            torch.save(dict(dataset=dataset, seed=seed, arm=arm, state={n:v.detach().cpu() for n,v in net.state_dict().items()}), out/f'{arm}_s{seed}.pt')
            report = dict(dataset=dataset, arm=arm, seed=seed, steps=len(schedule), lr=1e-4, batch_size=16,
                source_count=len(source), query_count=len(queries), reference_count=len(refs),
                reference_ids=[str(r['id']) for r in refs], unique_training_references=len(exposed),
                trainable_names=sorted(names), trainable_parameters=sum(p.numel() for n,p in selected),
                baseline=previous.metrics(baseline,y), after=previous.metrics(score,y),
                source_before=previous.metrics(teacher,sy), source_after=previous.metrics(source_score,sy),
                gradients=gradients, curve=curve, frozen_unchanged=True, evaluation_used=False,
                seconds=time.perf_counter()-start, schedule=str(schedule_path))
            del net,opt,selected,frozen,z,loss,ce,keep,values
            net = previous.restore(out/f'{arm}_s{seed}.pt')
            small = queries[:4]
            report['restore_error'] = float(abs(cache.predict(net,small,refs,bank)-score[:4]).max())
            qdata = previous.batch([previous.prepare_record(r) for r in small], 'cuda')
            rdata = previous.batch([previous.prepare_record(r) for r in refs], 'cuda')
            with torch.no_grad():
                rdata['prefix'] = net.prefix(rdata['images'], stop=2)
                native = previous.forward_batch(net,qdata,rdata,previous.pair_mask(small,refs,'cuda')).cpu().numpy()
            report['native_error'] = float(abs(native-score[:4]).max())
            assert max(report['restore_error'], report['native_error']) < 1e-4
            previous.write(path,report)
            print('COMPLETE',dataset,arm,seed,report['after'],flush=True)
            del net,qdata,rdata


if __name__ == '__main__':
    p = previous.argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['private','oss2'],required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[17])
    a = p.parse_args(); main(a.dataset,a.seeds)
