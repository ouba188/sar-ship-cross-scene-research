"""Bounded SAR-only student pilot after a positive source teacher-kernel gate."""

from case_relation_headroom import *
from full_teacher_data import SOURCE, PREFIX, bank_from_extra
import torch
from torch.nn import functional as F
from torchvision.models import resnet18
from full_teacher_ops import FullExpertGeometry
from unified_head_expert import _conditions
from aligned_residual_ops import residual_scores
from scene_training import sample_weights

HEADROOM = OUT
OUT = BASE/'case_relation_student_20261012'
ARMS = ('task', 'score', 'relation', 'joint')
STEPS = 64


def tensor(x, dtype=torch.float64):
    return torch.as_tensor(x, dtype=dtype, device='cuda')


def relation_loss(student, teacher, ships, weights):
    """SPKD-inspired sample relations, excluding self and same-ship edges."""
    mask = ships[:,None] != ships[None,:]
    sk = (student@student.T)*mask
    tk = (teacher@teacher.T)*mask
    sn = F.normalize(sk, p=2, dim=1)
    tn = F.normalize(tk, p=2, dim=1)
    row = (sn-tn).square().sum(1)
    return (row*weights).sum()/weights.sum()


def grad_stats(task, score_loss, rel_loss, parameters):
    gradients = [torch.autograd.grad(v,parameters,retain_graph=True) for v in (task,score_loss,rel_loss)]
    norms = [sum(g.double().square().sum() for g in group).sqrt() for group in gradients]
    def cosine(i,j):
        dot = sum((a.double()*b.double()).sum() for a,b in zip(gradients[i],gradients[j]))
        return float(dot/(norms[i]*norms[j]).clamp_min(1e-30))
    return dict(task_norm=float(norms[0]),score_norm=float(norms[1]),relation_norm=float(norms[2]),
                task_score_cosine=cosine(0,1),task_relation_cosine=cosine(0,2),
                score_relation_cosine=cosine(1,2))


def source_readers(context, meta, obs_ids, nfit, folds, dest):
    manifest = pd.read_csv(SOURCE/'manifest.csv',usecols=['mmsi'])
    source_ships = manifest.iloc[context.source_rows].mmsi.to_numpy()
    mm = meta.iloc[obs_ids].reset_index(drop=True)
    held_ships = mm.iloc[nfit:].mmsi.unique()
    pos = pd.Index(context.source_rows).get_indexer(mm.manifest_row)
    heads, scores = [], np.zeros((nfit,8))
    for k in range(3):
        val = folds == k
        excluded = np.r_[held_ships,mm.iloc[:nfit].loc[val,'mmsi'].unique()]
        use = ~np.isin(source_ships,excluded)
        head = fit_head(context.augmented[use],context.source_labels[use],context.transforms)
        heads.append(head)
        scores[val] = context.augmented[pos[:nfit][val]]@head['coef']+head['intercept']
    teacher, _, multi, _, _ = aggregate(mm.iloc[:nfit], scores,
                                       np.full((nfit,6),np.nan), np.zeros(4))
    np.savez_compressed(dest/'source_readers.npz',coef=np.stack([h['coef'] for h in heads]),
        intercept=np.stack([h['intercept'] for h in heads]),fit_prior=scores,teacher_scores=teacher)
    return heads,scores,teacher,multi,pos


def extract(tail,prefix,raw,bank,u,geometry):
    values, extras = [], []
    with torch.inference_mode():
        for lo in range(0,len(prefix),128):
            extra = tail(prefix[lo:lo+128].float()).mean((2,3))
            _,phi = geometry(raw[lo:lo+128],extra,bank[lo:lo+128],u[lo:lo+128])
            values.append(phi.cpu().numpy()); extras.append(extra.cpu().numpy())
    return np.concatenate(values),np.concatenate(extras)


def freeze_predictions(phi,prior,ids,meta,dest):
    index = pd.Index(ids)
    for draw_seed in SEEDS:
        for draw in range(3):
            with np.load(BASE/'incidence_transport_20261011'/f'{draw_seed}_source_{draw}_predictions.npz') as z:
                so,qo = z['support_obs_ids'],z['query_obs_ids']
            s,q = index.get_indexer(so),index.get_indexer(qo)
            sy = meta.iloc[so].class_id.to_numpy()
            result = score(phi@phi[s].T,prior,s,sy)[0][q]
            np.savez_compressed(dest/f'{draw_seed}_{draw}.npz',query_obs_ids=qo,scores=result)


def geometry_diagnostics(phi,raw_phi,teacher_phi,mm,nfit):
    result = {}
    cols = r18_columns()
    for role,ix in [('fit',np.arange(nfit)),('held',np.arange(nfit,len(mm)))]:
        # Fixed subsets bound diagnostic memory; no subset selection by outcomes.
        ix = ix[np.linspace(0,len(ix)-1,min(256,len(ix)),dtype=int)]
        ships = tensor(pd.factorize(mm.iloc[ix].mmsi)[0],torch.long)
        w = tensor(np.ones(len(ix)))
        a,b,c = tensor(phi[ix,:-1]),tensor(raw_phi[ix,:-1]),tensor(teacher_phi[ix,:-1])
        result[role+'_relation_mse'] = float(relation_loss(a,c,ships,w))
        result[role+'_raw_relation_mse'] = float(relation_loss(b,c,ships,w))
        result[role+'_r18_teacher_mse'] = float(np.mean((phi[ix][:,cols]-teacher_phi[ix][:,cols])**2))
    return result


def train(arm,seed,tail,initial,prefix,raw,bank,u,geometry,folds,heads,fit_prior,
          teacher_scores,teacher_phi,multi,mm,nfit,dest):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    tail.load_state_dict(initial); tail.eval()
    optimizer = torch.optim.AdamW(tail.parameters(),lr=1e-5,weight_decay=1e-4)
    coef = tensor(np.stack([h['coef'] for h in heads]))
    intercept = tensor(np.stack([h['intercept'] for h in heads]))
    y = mm.class_id.to_numpy()[:nfit]
    ship = pd.factorize(mm.mmsi)[0][:nfit]
    counts = np.bincount(y,minlength=8)
    cw = tensor(len(y)/(8*counts))
    sw = sample_weights(mm.iloc[:nfit])
    teacher_full = tensor(teacher_phi[:nfit])
    teacher = teacher_full[:,:-1]
    rng = np.random.default_rng(seed)
    order = np.resize(rng.permutation(3),STEPS)
    history, coefficient = [], None
    start = time.perf_counter()
    for step,k in enumerate(order):
        pool = np.flatnonzero(folds == k)
        s = rng.choice(pool,64,replace=False)
        candidates = pool[~np.isin(ship[pool],ship[s])]
        q = rng.choice(candidates,64,replace=False)
        assert not set(ship[s]) & set(ship[q])
        t = rng.choice(np.flatnonzero(multi),64,replace=False)
        ix = tensor(np.r_[s,q,t],torch.long)
        extra = tail(prefix[ix].float()).mean((2,3))
        aug,phi = geometry(raw[ix],extra,bank[ix],u[ix])
        # Kernel-only intervention: source class priors are fixed across all arms.
        logits = residual_scores(phi[:64],tensor(fit_prior[s]),tensor(y[s],torch.long),
                                 phi[64:128],tensor(fit_prior[q]))
        qy = tensor(y[q],torch.long)
        task = (F.cross_entropy(logits,qy,reduction='none')*cw[qy]).sum()/cw[qy].sum()
        tf = tensor(folds[t],torch.long)
        predicted = torch.einsum('bi,bij->bj',aug[128:],coef[tf])+intercept[tf]
        wanted = tensor(teacher_scores[t])
        delta = predicted-predicted.mean(1,keepdim=True)-wanted+wanted.mean(1,keepdim=True)
        tw = tensor(sw[t])
        score_loss = (delta.square().mean(1)*tw).sum()/tw.sum()
        rel = relation_loss(phi[128:,:-1],teacher[t],tensor(ship[t],torch.long),tw)
        if arm == 'readout':
            with torch.no_grad():
                teacher_logits = residual_scores(teacher_full[s],tensor(fit_prior[s]),tensor(y[s],torch.long),
                                                  teacher_full[q],tensor(fit_prior[q]))
            change = logits-teacher_logits
            change = change-change.mean(1,keepdim=True)
            rel = (change.square().mean(1)*cw[qy]).sum()/cw[qy].sum()
        if step in (0,31,63):
            diag = grad_stats(task,score_loss,rel,list(tail.parameters()))
            write(dest/f'gradient_{step:02d}.json',diag)
            if step == 0:
                coefficient = .5*diag['score_norm']/max(diag['relation_norm'],1e-20)
                write(dest/'LOSS_WEIGHT.json',dict(score=.5,relation=coefficient,
                    objective='adapted_score' if arm == 'readout' else 'normalized_gram',
                    rule='Initial relation gradient norm matched to 0.5 score teacher; fixed thereafter',
                    joint='Half score and half relation teacher budget; no held selection'))
        teaching = {'task':0.*score_loss,'score':.5*score_loss,
                    'relation':coefficient*rel,'joint':.25*score_loss+.5*coefficient*rel,
                    'readout':coefficient*rel}[arm]
        loss = task+teaching
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite student loss')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(tail.parameters(),5.)
        optimizer.step()
        history.append(dict(step=step+1,task=float(task.detach()),score=float(score_loss.detach()),
            relation=float(rel.detach()),total=float(loss.detach()),gradient=float(norm),
            seconds=time.perf_counter()-start))
        if (step+1)%16 == 0:
            print('TRAIN',dest.parent.parent.name,seed,arm,history[-1],flush=True)
    torch.save({k:v.detach().cpu() for k,v in tail.state_dict().items()},dest/'layer4.pt')
    write(dest/'history.json',history)
    del optimizer


def run_student(port,meta,folds,nfit,incidence,prefix_cache):
    dest = OUT/port; dest.mkdir(parents=True,exist_ok=True)
    with np.load(HEADROOM/port/'maps.npz') as z:
        ids,prior,raw_phi,teacher_phi = (z[k] for k in ('obs_ids','prior','single','equal_r18'))
    mm = meta.iloc[ids].reset_index(drop=True)
    context = ExpertContext(ROOT/'learned_frequency_residual_20261009/tasks/self24ports'/port)
    context._prepare_source()
    heads,fit_prior,teacher_scores,multi,pos = source_readers(context,meta,ids,nfit,folds,dest)
    prefix = tensor(np.array(prefix_cache[mm.manifest_row.to_numpy()]),torch.float16)
    raw = tensor(np.array(context.raw[mm.manifest_row.to_numpy()]))
    bank = tensor(np.array(context.bank[pos])); u = tensor(incidence[mm.manifest_row.to_numpy()])
    geometry = FullExpertGeometry(context.geometry).cuda()
    tail = resnet18(weights=None).layer4.cuda().eval()
    initial = torch.load(ROOT/'module_completion_20261008/ports_projection'/port/'joint/layer4.pt',
                         map_location='cpu',weights_only=True)
    tail.load_state_dict(initial)
    initial_phi,initial_extra = extract(tail,prefix,raw,bank,u,geometry)
    np.testing.assert_array_equal(initial_extra,context.extra[pos])
    refreshed = bank_from_extra(context.bank[pos],initial_extra,context.geometry)
    numeric_phi = numpy_phi(refreshed,context.raw[mm.manifest_row.to_numpy()],initial_extra,
                            incidence[mm.manifest_row.to_numpy()],context.geometry)
    np.testing.assert_allclose(initial_phi,numeric_phi,atol=2e-7,rtol=2e-7)
    cache_gap = float(np.max(abs(initial_phi-raw_phi)))
    write(dest/'INITIAL_PARITY.json',dict(max_phi_error=float(np.max(abs(initial_phi-numeric_phi))),
        old_cached_bank_gap=cache_gap,extra_features_exact=True,
        cause='Old cached SSL PCA coordinates versus recomputation from identical R18 features'))
    ref = dest/'initial'; ref.mkdir(exist_ok=True)
    freeze_predictions(initial_phi,prior,ids,meta,ref)
    for draw_seed in SEEDS:
        for draw in range(3):
            with np.load(ref/f'{draw_seed}_{draw}.npz') as a, np.load(
                    HEADROOM/port/f'{draw_seed}_{draw}_single_native.npz') as b:
                np.testing.assert_array_equal(a['scores'].argmax(1),b['scores'].argmax(1))
    write(ref/'geometry.json',geometry_diagnostics(initial_phi,raw_phi,teacher_phi,mm,nfit))
    for seed in SEEDS:
        for arm in ARMS:
            ad = dest/str(seed)/arm; ad.mkdir(parents=True,exist_ok=True)
            if (ad/'FROZEN.json').exists():
                print('REUSE',port,seed,arm,flush=True); continue
            train(arm,seed,tail,initial,prefix,raw,bank,u,geometry,folds,heads,fit_prior,
                  teacher_scores,teacher_phi,multi,mm,nfit,ad)
            phi,extra = extract(tail,prefix,raw,bank,u,geometry)
            np.save(ad/'extra.npy',extra)
            freeze_predictions(phi,prior,ids,meta,ad)
            write(ad/'geometry.json',geometry_diagnostics(phi,raw_phi,teacher_phi,mm,nfit))
            write(ad/'FROZEN.json',dict(query_labels_used=False,steps=STEPS,prior_fixed=True))
    del context,prefix,raw,bank,u,geometry,tail
    gc.collect(); torch.cuda.empty_cache()


def student_report(meta):
    rows,events = [],[]
    for port in PORTS:
        for seed in SEEDS:
            for draw_seed in SEEDS:
                for draw in range(3):
                    key = f'{draw_seed}_{draw}.npz'
                    with np.load(OUT/port/str(seed)/'task'/key) as z:
                        bp = z['scores'].argmax(1); ids = z['query_obs_ids']
                    truth = meta.iloc[ids].class_id.to_numpy()
                    for arm in ('initial',*ARMS):
                        path = OUT/port/'initial'/key if arm == 'initial' else OUT/port/str(seed)/arm/key
                        with np.load(path) as z:
                            np.testing.assert_array_equal(z['query_obs_ids'],ids)
                            pred = z['scores'].argmax(1)
                        rows.append(dict(context=port,seed=seed,draw_seed=draw_seed,draw=draw,arm=arm,
                            **metric(truth,pred),rescue=int(np.sum((bp!=truth)&(pred==truth))),
                            harm=int(np.sum((bp==truth)&(pred!=truth)))))
                        for j in np.flatnonzero((bp==truth)!=(pred==truth)):
                            row = meta.iloc[ids[j]]
                            events.append(dict(context=port,seed=seed,draw_seed=draw_seed,draw=draw,arm=arm,
                                obs_id=int(ids[j]),mmsi=str(row.mmsi),port=row.port,class_id=int(truth[j]),
                                before=int(bp[j]),after=int(pred[j]),rescue=bool(pred[j]==truth[j])))
    frame = pd.DataFrame(rows); frame.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(events).to_csv(OUT/'events.csv',index=False)
    summary = frame.groupby('arm')[['oa','ba','rescue','harm']].mean()
    summary.to_csv(OUT/'summary.csv')
    frame.groupby(['context','seed','arm'])[['oa','ba']].mean().to_csv(OUT/'context_seed.csv')
    write(OUT/'COMPLETE.json',dict(neural_fits=24,steps=64,seeds=list(SEEDS),target_runs=0,
        inference_no_new_metadata=True,source_expert_kernel_only=True,main_model_unchanged=True,
        full_fusion_not_evaluated=True))
    print('STUDENT RESULTS',summary.to_string(),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ports',nargs='+',default=list(PORTS))
    parser.add_argument('--report-only',action='store_true')
    args = parser.parse_args()
    meta = pd.read_csv(BASE/'observations.csv')
    if args.report_only:
        student_report(meta); return
    OUT.mkdir(exist_ok=True)
    write(OUT/'PROTOCOL.json',dict(date='2026-10-12',stage='Student exploitability after teacher-kernel benefit',
        arms=list(ARMS),contexts=list(PORTS),seeds=list(SEEDS),steps=64,lr=1e-5,batch=192,
        training='Only1296fit observations; 3vessel-fold episodes, support64/query64 disjoint ships plus64teachers',
        teacher='Equal other-product history, product-balanced; no motion/quality weighting',
        scores='Source vessel-OOF teacher heads exclude all302heldships; centered MSE',
        relation='SPKD-inspired row-L2 Gram across different ships, current full teacher geometry minus bias',
        weighting='Relation gradient norm matched initially to0.5scoreMSE and frozen; joint splits teaching budget',
        task='Actual full expert residual ridge8; FIXED source prior isolates case geometry',
        checkpoint='Fixed step64; no held selection, two genuine training seeds',
        scope='Train R18layer4 only; existing original prior remains fixed, not refitted full-model integration',
        inference='Single SAR observation with existing SAR acquisition incidence; no new AIS/MMSI/history',
        limits='Historical source encoder/geometry exposure; three contexts reuse same823held observations; no target/full-fusion confirmation'))
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    with np.load(BASE/'crossfit_evidence_20261011/source_folds.npz') as z:
        folds=z['assignment']; nfit=len(z['fit_obs_ids'])
    incidence=_conditions(); cache=np.load(PREFIX,mmap_mode='r')
    for port in args.ports:
        run_student(port,meta,folds,nfit,incidence,cache)
    if all((OUT/p/str(s)/a/'FROZEN.json').exists() for p in PORTS for s in SEEDS for a in ARMS):
        student_report(meta)


if __name__ == '__main__':
    main()
