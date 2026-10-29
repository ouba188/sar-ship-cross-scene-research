"""Paired class-preserving M3 writers under the existing source schedule."""
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'm3_pairwise_repair_20261030'
spec = importlib.util.spec_from_file_location('class_separated_previous', PREVIOUS / 'experiment.py')
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
torch, np, F = previous.torch, previous.np, previous.F
from class_match import attach, configure


@torch.no_grad()
def calibrate(net, cache, source, refs, bank):
    totals = [dict(square=0., count=0) for _ in net.m3[0]]
    scores = []
    for start in range(0, len(source), 16):
        group = source[start:start+16]
        cache.score(net, group, refs, bank)
        numerator, denominator = [], []
        for p, m in enumerate(net.m3[0]):
            d = m.diagnostic
            totals[p]['square'] += float(d['selected_square'])
            totals[p]['count'] += d['selected_numel']
            live = d['class_valid']
            numerator.append((d['class_evidence'] * live).sum(-1))
            denominator.append(live.sum(-1))
        count = torch.stack(denominator).sum(0)
        score = torch.stack(numerator).sum(0) / count.clamp_min(1)
        scores.append(score.masked_fill(count == 0, -1e4).cpu().numpy())
    for p, m in enumerate(net.m3[0]):
        scale = (totals[p]['square']/totals[p]['count'])**.5
        assert np.isfinite(scale) and scale > 1e-8
        m.extra_scale.fill_(scale)
        totals[p]['scale'] = scale
    score = np.concatenate(scores)
    y = np.array([r['label'] for r in source])
    return totals, previous.metrics(score, y)


def restore(path):
    blob = torch.load(path, map_location='cpu', weights_only=True)
    net = attach(previous.original(blob['dataset'], blob['seed']), blob['arm'], blob['reference_labels'])
    net.load_state_dict(blob['state'], strict=True)
    return net.eval().requires_grad_(False)


def main(dataset, seeds):
    previous.setup()
    cache, (_, source, queries, refs) = previous.data(dataset)
    bank = cache.take(refs)
    labels_ref = [int(r['label']) for r in refs]
    y = np.array([r['label'] for r in queries])
    sy = np.array([r['label'] for r in source])
    lookup = {str(r['id']): i for i, r in enumerate(source)}
    for seed in seeds:
        schedule_path = PREVIOUS/dataset/'exposures'/f'schedule_s{seed}.json'
        schedule = [np.array([lookup[str(k)] for k in group]) for group in previous.json.loads(schedule_path.read_text())]
        net = previous.original(dataset, seed)
        baseline = cache.predict(net, queries, refs, bank)
        small_baseline = cache.predict(net, queries[:4], refs, bank)
        teacher = cache.predict(net, source, refs, bank)
        with np.load(PREVIOUS/dataset/'predictions'/f'ce_s{seed}.npz') as old:
            assert np.array_equal(baseline, old['baseline'])
            assert np.array_equal(old['ids'], [str(r['id']) for r in queries])
        del net
        for arm in ('aligned', 'pooled'):
            out = HERE/dataset/'training'; out.mkdir(parents=True, exist_ok=True)
            receipt = out/f'{arm}_s{seed}.json'
            if receipt.exists(): raise FileExistsError(receipt)
            net = attach(previous.original(dataset, seed), arm, labels_ref)
            initial_error = float(abs(cache.predict(net, queries[:4], refs, bank)-small_baseline).max())
            assert initial_error < 1e-6
            calibration, source_distance = calibrate(net, cache, source, refs, bank)
            selected = configure(net)
            names = {n for n, p in selected}
            fixed = {n:v.detach().cpu().clone() for n,v in net.state_dict().items() if n not in names}
            opt = torch.optim.AdamW([p for n,p in selected], lr=1e-4, weight_decay=.01)
            curve=[]; start=previous.time.perf_counter()
            for step, ix in enumerate(schedule):
                group = [source[i] for i in ix]
                target = torch.tensor(teacher[ix], device='cuda')
                labels = torch.tensor(sy[ix], device='cuda')
                opt.zero_grad(set_to_none=True)
                score = cache.score(net, group, refs, bank)
                ce = F.cross_entropy(score, labels)
                kl = F.kl_div(score.log_softmax(1), target.log_softmax(1), log_target=True, reduction='none').sum(1)
                keep = (kl*target.argmax(1).eq(labels)).mean()
                loss=ce+.1*keep
                assert torch.isfinite(loss)
                loss.backward()
                if step == 0:
                    gradients = {n:float(p.grad.norm()) for n,p in selected}
                    assert any(v>0 for v in gradients.values())
                opt.step()
                curve.append(dict(step=step+1, ce=float(ce.detach()), keep=float(keep.detach())))
                if (step+1)%16 == 0: print('CLASS_MATCH',dataset,arm,seed,curve[-1],flush=True)
            assert all(torch.equal(net.state_dict()[n].detach().cpu(),v) for n,v in fixed.items())
            net.requires_grad_(False)
            score=cache.predict(net,queries,refs,bank)
            source_score=cache.predict(net,source,refs,bank)
            pred=HERE/dataset/'predictions'; pred.mkdir(parents=True,exist_ok=True)
            np.savez(pred/f'{arm}_s{seed}.npz',score=score,baseline=baseline,y=y,ids=np.array([str(r['id']) for r in queries]))
            np.savez(pred/f'{arm}_source_s{seed}.npz',score=source_score,baseline=teacher,y=sy,ids=np.array([str(r['id']) for r in source]))
            torch.save(dict(dataset=dataset,seed=seed,arm=arm,reference_labels=labels_ref,
                state={n:v.detach().cpu() for n,v in net.state_dict().items()}),out/f'{arm}_s{seed}.pt')
            for m in net.m3[0]: m.extra_control='off'
            off=cache.predict(net,queries,refs,bank)
            off_error=float(abs(off-baseline).max())
            assert off_error<1e-4
            report=dict(dataset=dataset,seed=seed,arm=arm,source_count=len(source),query_count=len(queries),
                reference_count=len(refs),reference_ids=[str(r['id']) for r in refs],reference_labels=labels_ref,
                steps=len(schedule),batch_size=16,lr=1e-4,calibration=calibration,source_distance=source_distance,
                trainable_names=sorted(names),trainable_parameters=sum(p.numel() for n,p in selected),
                initial_error=initial_error,off_error=off_error,frozen_unchanged=True,
                baseline=previous.metrics(baseline,y),after=previous.metrics(score,y),
                source_before=previous.metrics(teacher,sy),source_after=previous.metrics(source_score,sy),
                gradients=gradients,curve=curve,seconds=previous.time.perf_counter()-start,evaluation_used=False)
            del net,opt,selected,fixed,loss,ce,keep
            net=restore(out/f'{arm}_s{seed}.pt')
            report['restore_error']=float(abs(cache.predict(net,queries[:4],refs,bank)-score[:4]).max())
            q=previous.batch([previous.prepare_record(r) for r in queries[:4]],'cuda')
            r=previous.batch([previous.prepare_record(v) for v in refs],'cuda')
            with torch.no_grad():
                r['prefix']=net.prefix(r['images'],stop=2)
                native=previous.forward_batch(net,q,r,previous.pair_mask(queries[:4],refs,'cuda')).cpu().numpy()
            report['native_error']=float(abs(native-score[:4]).max())
            assert max(report['restore_error'],report['native_error'])<1e-4
            previous.write(receipt,report)
            print('COMPLETE_CLASS_MATCH',dataset,arm,seed,report['after'],source_distance,flush=True)
            del net,q,r


if __name__=='__main__':
    p=previous.argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['private','oss2'],required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[17])
    a=p.parse_args(); main(a.dataset,a.seeds)
