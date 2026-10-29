"""Same legal source schedule as the preceding CE control; no final test."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent/'m3_pairwise_repair_20261030'
sys.path.insert(0, str(PREVIOUS))
import importlib.util
spec = importlib.util.spec_from_file_location('pairwise_training', PREVIOUS/'experiment.py')
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
from pairwise_training import (torch, np, F, setup, data, original, write, metrics,
    time, argparse, json, batch, prepare_record, pair_mask, forward_batch)
sys.path.insert(0, str(HERE))
from interaction import attach, configure


@torch.no_grad()
def calibrate(net, cache, rows, refs, bank):
    totals = [dict(square=0., count=0) for _ in net.m3[0]]
    for start in range(0, len(rows), 16):
        cache.score(net, rows[start:start+16], refs, bank)
        for p, m in enumerate(net.m3[0]):
            totals[p]['square'] += float(m.diagnostic['selected_square'])
            totals[p]['count'] += m.diagnostic['selected_numel']
    for p, m in enumerate(net.m3[0]):
        scale = (totals[p]['square']/totals[p]['count'])**.5
        assert np.isfinite(scale) and scale > 1e-8
        m.extra_scale.fill_(scale)
        totals[p]['scale'] = scale
    return totals


def restore(path):
    blob = torch.load(path, map_location='cpu', weights_only=True)
    net = attach(original(blob['dataset'], blob['seed']), blob['arm'])
    net.load_state_dict(blob['state'], strict=True)
    return net.eval().requires_grad_(False)


def main(dataset, seeds):
    setup()
    cache, (_, source, queries, refs) = data(dataset)
    bank = cache.take(refs)
    y = np.array([r['label'] for r in queries])
    sy = np.array([r['label'] for r in source])
    lookup = {str(r['id']):i for i,r in enumerate(source)}
    for seed in seeds:
        schedule_path = PREVIOUS/dataset/'exposures'/f'schedule_s{seed}.json'
        schedule = [np.array([lookup[str(k)] for k in group]) for group in json.loads(schedule_path.read_text())]
        net = original(dataset, seed)
        baseline = cache.predict(net, queries, refs, bank)
        baseline_small = cache.predict(net, queries[:4], refs, bank)
        teacher = cache.predict(net, source, refs, bank)
        with np.load(PREVIOUS/dataset/'predictions'/f'ce_s{seed}.npz') as old:
            assert np.array_equal(old['ids'], [str(r['id']) for r in queries])
            assert np.array_equal(baseline, old['baseline'])
        del net
        for arm in ('query_only', 'ref_only', 'interaction'):
            out = HERE/dataset/'training'
            out.mkdir(parents=True, exist_ok=True)
            if (out/f'{arm}_s{seed}.json').exists():
                raise FileExistsError(out/f'{arm}_s{seed}.json')
            net = attach(original(dataset, seed), arm)
            initial = cache.predict(net, queries[:4], refs, bank)
            initial_error = float(abs(initial-baseline_small).max())
            assert initial_error < 1e-6
            calibration = calibrate(net, cache, source, refs, bank)
            selected = configure(net)
            names = {n for n,p in selected}
            fixed = {n:v.detach().cpu().clone() for n,v in net.state_dict().items() if n not in names}
            opt = torch.optim.AdamW([p for n,p in selected], lr=1e-4, weight_decay=.01)
            curve = []; start = time.perf_counter()
            for step, ix in enumerate(schedule):
                q = [source[i] for i in ix]
                labels = torch.tensor(sy[ix], device='cuda')
                target = torch.tensor(teacher[ix], device='cuda')
                opt.zero_grad(set_to_none=True)
                z = cache.score(net, q, refs, bank)
                ce = F.cross_entropy(z, labels)
                kl = F.kl_div(z.log_softmax(1), target.log_softmax(1), log_target=True, reduction='none').sum(1)
                keep = (kl*target.argmax(1).eq(labels)).mean()
                loss = ce+.1*keep
                assert torch.isfinite(loss)
                loss.backward()
                if step == 0:
                    gradients = {n:float(p.grad.norm()) for n,p in selected}
                    assert any(v>0 for v in gradients.values())
                opt.step()
                curve.append(dict(step=step+1, ce=float(ce.detach()), keep=float(keep.detach())))
                if (step+1) % ((len(source)+15)//16) == 0:
                    print('INTERACTION',dataset,arm,seed,curve[-1],flush=True)
            assert all(torch.equal(net.state_dict()[n].detach().cpu(), v) for n,v in fixed.items())
            net.requires_grad_(False)
            scores = cache.predict(net, queries, refs, bank)
            source_scores = cache.predict(net, source, refs, bank)
            pred = HERE/dataset/'predictions'; pred.mkdir(parents=True,exist_ok=True)
            np.savez(pred/f'{arm}_s{seed}.npz', score=scores, baseline=baseline, y=y, ids=np.array([str(r['id']) for r in queries]))
            torch.save(dict(dataset=dataset, seed=seed, arm=arm,
                state={n:v.detach().cpu() for n,v in net.state_dict().items()}), out/f'{arm}_s{seed}.pt')
            controls = {}
            for control in ('off', 'uniform'):
                for m in net.m3[0]:m.extra_control=control
                control_scores = cache.predict(net, queries, refs, bank)
                if control == 'off':assert abs(control_scores-baseline).max()<1e-4
                np.savez(pred/f'{arm}_{control}_s{seed}.npz', score=control_scores, baseline=baseline, y=y, ids=np.array([str(r['id']) for r in queries]))
                controls[control] = dict(metrics=metrics(control_scores,y),
                    max_logit_difference=float(abs(control_scores-scores).max()),
                    changed=int((control_scores.argmax(1)!=scores.argmax(1)).sum()))
            report = dict(dataset=dataset,seed=seed,arm=arm,source_count=len(source),query_count=len(queries),
                steps=len(schedule),batch_size=16,lr=1e-4,reference_count=len(refs),reference_ids=[str(r['id']) for r in refs],
                schedule=str(schedule_path),calibration=calibration,trainable_names=sorted(names),
                trainable_parameters=sum(p.numel() for n,p in selected),frozen_unchanged=True,
                initial_error=initial_error,baseline_batch_error=float(abs(baseline_small-baseline[:4]).max()),
                baseline=metrics(baseline,y),after=metrics(scores,y),
                source_before=metrics(teacher,sy),source_after=metrics(source_scores,sy),gradients=gradients,
                controls=controls,curve=curve,seconds=time.perf_counter()-start,evaluation_used=False)
            del net,opt,selected,fixed,z,loss,ce,keep
            net = restore(out/f'{arm}_s{seed}.pt')
            group = queries[:4]
            report['restore_error'] = float(abs(cache.predict(net,group,refs,bank)-scores[:4]).max())
            values = batch([prepare_record(r) for r in group],'cuda')
            native_bank = batch([prepare_record(r) for r in refs],'cuda')
            with torch.no_grad():
                native_bank['prefix']=net.prefix(native_bank['images'],stop=2)
                native = forward_batch(net,values,native_bank,pair_mask(group,refs,'cuda')).cpu().numpy()
            report['native_error']=float(abs(native-scores[:4]).max())
            assert max(report['restore_error'],report['native_error'])<1e-4
            write(out/f'{arm}_s{seed}.json',report)
            print('COMPLETE',dataset,arm,seed,report['after'],controls,flush=True)
            del net,values,native_bank


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['private','oss2'],required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[17])
    args=parser.parse_args();main(args.dataset,args.seeds)
