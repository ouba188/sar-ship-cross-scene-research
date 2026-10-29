"""Query-discriminative M3 updates with class-bias-invariant pair ranking."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'m3_source_replay_20261030'))
from diagnose import *
from collections import Counter
from itertools import combinations
HERE = Path(__file__).resolve().parent


def paired_loss(z, y):
    left, right = z[::2], z[1::2]
    a, b = y[::2,None], y[1::2,None]
    gap = (left.gather(1,a)-left.gather(1,b))-(right.gather(1,a)-right.gather(1,b))
    return F.softplus(-gap).mean()


def pair_pools(rows):
    counts = Counter(str(r['mmsi']) for r in rows)
    pairs, probabilities, summary = [], [], []
    for a,b in combinations(sorted({r['label'] for r in rows}),2):
        ij = [(i,j) for i,r in enumerate(rows) if r['label']==a
              for j,s in enumerate(rows) if s['label']==b and r['pass_group'] != s['pass_group']
              and str(r['mmsi']) != str(s['mmsi'])]
        if not ij:
            summary.append(dict(a=a,b=b,pairs=0))
            continue
        weight = np.array([1/(counts[str(rows[i]['mmsi'])]*counts[str(rows[j]['mmsi'])]) for i,j in ij])
        pairs.append(np.array(ij,dtype=int)); probabilities.append(weight/weight.sum())
        summary.append(dict(a=a,b=b,pairs=len(ij),left_ships=len({rows[i]['mmsi'] for i,j in ij}),
            right_ships=len({rows[j]['mmsi'] for i,j in ij}),
            left_passes=len({rows[i]['pass_group'] for i,j in ij}),right_passes=len({rows[j]['pass_group'] for i,j in ij})))
    return pairs, probabilities, summary


def trainable(net):
    net.eval().requires_grad_(False)
    selected = []
    for n,p in net.named_parameters():
        if n.startswith('m3.') and '.reader' not in n:
            p.requires_grad_(True)
            selected.append((n,p))
    return selected


@torch.no_grad()
def source_pair_metrics(z, y, pools):
    z = np.asarray(z)
    losses, accuracies = [], []
    for ij in pools:
        i,j = ij[:,0],ij[:,1]
        a,b = y[i],y[j]
        d = z[i,a]-z[i,b]-z[j,a]+z[j,b]
        losses.append(float(np.logaddexp(0,-d).mean()))
        accuracies.append(float((d>0).mean()))
    return dict(pair_logistic=float(np.mean(losses)),pair_rank_accuracy=float(np.mean(accuracies)))


def restore(path):
    blob = torch.load(path,map_location='cpu',weights_only=True)
    net = original(blob['dataset'],blob['seed'])
    net.load_state_dict(blob['state'],strict=True)
    return net.eval().requires_grad_(False)


def main(dataset,seeds):
    setup()
    cache,(_,train,queries,refs) = data(dataset)
    bank = cache.take(refs)
    pp, pw, coverage = pair_pools(train)
    assert all(r.get('pass_group') and r.get('mmsi') for r in train)
    write(HERE/dataset/'pair_coverage.json',dict(classes=len({r['label'] for r in train}),
        source_count=len(train),ships=len({r['mmsi'] for r in train}),passes=len({r['pass_group'] for r in train}),
        pairs=coverage,missing_class_pairs=[r for r in coverage if r['pairs']==0]))
    y = np.array([r['label'] for r in queries])
    sy = np.array([r['label'] for r in train])
    steps = 6*((len(train)+15)//16)
    for seed in seeds:
        net = original(dataset,seed)
        before = cache.predict(net,queries,refs,bank)
        teacher = cache.predict(net,train,refs,bank)
        initial_source = source_pair_metrics(teacher,sy,pp)
        del net
        rng = np.random.default_rng(seed)
        schedule = [np.concatenate([pp[k][rng.choice(len(pp[k]),p=pw[k])]
                    for k in rng.integers(len(pp),size=8)]) for _ in range(steps)]
        write(HERE/dataset/'exposures'/f'schedule_s{seed}.json',
              [[str(train[i]['id']) for i in ix] for ix in schedule])
        for arm in ('ce','rank','hybrid'):
            out = HERE/dataset/'training'
            out.mkdir(parents=True,exist_ok=True)
            if (out/f'{arm}_s{seed}.json').exists():
                raise FileExistsError(out/f'{arm}_s{seed}.json')
            net = original(dataset,seed)
            selected = trainable(net)
            names = {n for n,p in selected}
            fixed = {n:v.detach().cpu().clone() for n,v in net.state_dict().items() if n not in names}
            opt = torch.optim.AdamW([p for n,p in selected],lr=1e-4,weight_decay=.01)
            curve = []; start = time.perf_counter()
            for step,ix in enumerate(schedule):
                q = [train[i] for i in ix]
                labels = torch.tensor(sy[ix],device='cuda')
                target = torch.tensor(teacher[ix],device='cuda')
                opt.zero_grad(set_to_none=True)
                z = cache.score(net,q,refs,bank)
                ce = F.cross_entropy(z,labels)
                rank = paired_loss(z,labels)
                kl = F.kl_div(z.log_softmax(1),target.log_softmax(1),log_target=True,reduction='none').sum(1)
                keep = (kl*target.argmax(1).eq(labels)).mean()
                if step==0:
                    gradients = {}
                    for key,term in [('ce',ce),('rank',rank)]:
                        gs = torch.autograd.grad(term,[p for n,p in selected],retain_graph=True,allow_unused=True)
                        gradients[key] = {n:float(g.norm()) for (n,p),g in zip(selected,gs) if g is not None}
                task = ce if arm=='ce' else rank if arm=='rank' else .5*(ce+rank)
                loss = task+.1*keep
                assert torch.isfinite(loss)
                loss.backward();opt.step()
                curve.append(dict(step=step+1,ce=float(ce.detach()),rank=float(rank.detach()),keep=float(keep.detach())))
                if (step+1)%((len(train)+15)//16)==0:
                    print('PAIR',dataset,arm,seed,curve[-1],flush=True)
            assert all(torch.equal(net.state_dict()[n].detach().cpu(),v) for n,v in fixed.items())
            net.requires_grad_(False)
            score = cache.predict(net,queries,refs,bank)
            source = cache.predict(net,train,refs,bank)
            pred = HERE/dataset/'predictions'
            pred.mkdir(parents=True,exist_ok=True)
            np.savez(pred/f'{arm}_s{seed}.npz',score=score,baseline=before,y=y,ids=np.array([str(r['id']) for r in queries]))
            np.savez(pred/f'{arm}_source_s{seed}.npz',score=source,baseline=teacher,y=sy,ids=np.array([str(r['id']) for r in train]))
            torch.save(dict(dataset=dataset,seed=seed,arm=arm,state={n:v.detach().cpu() for n,v in net.state_dict().items()}),out/f'{arm}_s{seed}.pt')
            report = dict(dataset=dataset,arm=arm,seed=seed,steps=steps,batch_size=16,source_count=len(train),
                query_count=len(queries),reference_ids=[str(r['id']) for r in refs],reference_count=len(refs),lr=1e-4,
                trainable_names=sorted(names),trainable_parameters=sum(p.numel() for n,p in selected),frozen_unchanged=True,
                baseline=metrics(before,y),after=metrics(score,y),source_before=metrics(teacher,sy),source_after=metrics(source,sy),
                pair_before=initial_source,pair_after=source_pair_metrics(source,sy,pp),gradients=gradients,curve=curve,
                seconds=time.perf_counter()-start,evaluation_used=False)
            del net, opt, selected, fixed, z, loss, task, rank, ce, keep, gs
            net = restore(out/f'{arm}_s{seed}.pt')
            q = queries[:4]
            report['restore_error'] = float(abs(cache.predict(net,q,refs,bank)-score[:4]).max())
            values = batch([prepare_record(r) for r in q],'cuda')
            native_bank = batch([prepare_record(r) for r in refs],'cuda')
            with torch.no_grad():
                native_bank['prefix'] = net.prefix(native_bank['images'],stop=2)
                native = forward_batch(net,values,native_bank,pair_mask(q,refs,'cuda')).cpu().numpy()
            report['native_error'] = float(abs(native-score[:4]).max())
            assert max(report['restore_error'],report['native_error'])<1e-4
            write(out/f'{arm}_s{seed}.json',report)
            print('COMPLETE',dataset,arm,seed,report['after'],report['pair_after'],flush=True)
            del net, values, native_bank


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['private','oss2'],required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[17])
    args=p.parse_args();main(args.dataset,args.seeds)
