"""Three matched full-expert continuations; knowledge is training-only."""

from full_teacher_data import *
import argparse
import gc
import pickle
import time
import torch
from torch.nn import functional as F
from torchvision.models import resnet18
from unified_head_expert import ExpertContext, _conditions
from unified_head_core import score, late
from unified_head_run import base_kernel
from aligned_residual_ops import residual_scores
from full_teacher_ops import FullExpertGeometry, numpy_phi, parity_check

SEED = 31415
STEPS = 64
QUERY_BATCH = 64
TEACHER_BATCH = 64


def extract(tail, prefix, rows):
    answer = np.empty((len(rows), 512), np.float32)
    tail.eval()
    with torch.inference_mode():
        for lo in range(0, len(rows), 128):
            ids = torch.as_tensor(rows[lo:lo+128], device='cuda')
            answer[lo:lo+len(ids)] = tail(prefix[ids].float()).mean((2, 3)).cpu().numpy()
    return answer


def tensor(value, dtype=torch.float64):
    return torch.as_tensor(value, dtype=dtype, device='cuda')


def source_episodes(context, bank, manifest, dest):
    locations = pd.Index(context.source_rows)
    labels = manifest.class_id.to_numpy()
    episodes = []
    for item in context.items:
        with np.load(item['path']) as f:
            ep = {k: f[k] for k in ('rows', 'train_rows', 'support', 'query', 'labels')}
        pos = locations.get_indexer(ep['train_rows'])
        assert (pos >= 0).all() and np.isin(ep['rows'], context.source_rows).all()
        assert not np.intersect1d(ep['train_rows'], ep['rows']).size
        h = fit_head(bank[pos], labels[ep['train_rows']], context.transforms)
        ep.update(coef=tensor(h['coef']), intercept=tensor(h['intercept']), held_port=item['port'])
        episodes.append(ep)
    write(dest/'EPISODE_PLAN.json', [dict(held_port=e['held_port'], train_n=len(e['train_rows']),
        query_n=len(e['query']), supports=e['rows'][e['support']].tolist()) for e in episodes])
    return episodes


def gradient_diagnostic(task, teacher, parameters):
    ga = torch.autograd.grad(task, parameters, retain_graph=True)
    gb = torch.autograd.grad(.5*teacher, parameters, retain_graph=True)
    aa = sum((g.double().square().sum() for g in ga))
    bb = sum((g.double().square().sum() for g in gb))
    dot = sum(((a.double()*b.double()).sum() for a, b in zip(ga, gb)))
    return dict(task_gradient=float(aa.sqrt()), teaching_gradient=float(bb.sqrt()),
                gradient_cosine=float(dot/(aa*bb).sqrt().clamp_min(1e-30)))


def train_arm(arm, tail, initial, geometry, prefix, raw_gpu, bank_gpu, u_gpu,
              episodes, teachers, labels, source_rows, dest):
    tail.load_state_dict(initial)
    tail.eval()  # Running BN statistics, prefix and all geometry stay fixed.
    optimizer = torch.optim.AdamW(tail.parameters(), lr=1e-5, weight_decay=1e-4)
    teacher_coef = tensor(np.stack([h['coef'] for h in teachers['heads']]))
    teacher_intercept = tensor(np.stack([h['intercept'] for h in teachers['heads']]))
    active = np.flatnonzero(teachers['multi'])
    target = teachers['knowledge' if arm == 'knowledge_teacher' else 'equal']
    counts = np.bincount(labels[source_rows], minlength=8)
    weights = tensor(counts.sum()/(8*counts))
    rng = np.random.default_rng(SEED)
    schedule = np.resize(rng.permutation(len(episodes)), STEPS)
    history = []
    start = time.perf_counter()
    for step, ei in enumerate(schedule):
        ep = episodes[ei]
        qi = rng.choice(ep['query'], min(QUERY_BATCH, len(ep['query'])), replace=False)
        rows = ep['rows'][np.r_[ep['support'], qi]]
        ti = rng.choice(active, min(TEACHER_BATCH, len(active)), replace=False)
        tr = teachers['rows'][ti]
        ids = tensor(np.r_[rows, tr], torch.long)
        extra = tail(prefix[ids].float()).mean((2, 3))
        aug, phi = geometry(raw_gpu[ids], extra, bank_gpu[ids], u_gpu[ids])
        p = aug[:len(rows)]@ep['coef']+ep['intercept']
        sy = tensor(ep['labels'][ep['support']], torch.long)
        qy = tensor(ep['labels'][qi], torch.long)
        logits = residual_scores(phi[:64], p[:64], sy, phi[64:len(rows)], p[64:])
        task_loss = (F.cross_entropy(logits, qy, reduction='none')*weights[qy]).sum()/weights[qy].sum()
        fold = tensor(teachers['fold'][ti], torch.long)
        tp = torch.einsum('bi,bij->bj', aug[len(rows):], teacher_coef[fold])+teacher_intercept[fold]
        wanted = tensor(target[ti])
        difference = (tp-tp.mean(1, keepdim=True))-(wanted-wanted.mean(1, keepdim=True))
        tw = tensor(teachers['weights'][ti])
        teaching = (difference.square().mean(1)*tw).sum()/tw.sum()
        if step == 0:
            write(dest/'GRADIENT_DIAGNOSTIC.json', gradient_diagnostic(task_loss, teaching, list(tail.parameters())))
        loss = task_loss+(.5*teaching if arm != 'continued' else 0.*teaching)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite matched training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(tail.parameters(), 5.)
        optimizer.step()
        history.append(dict(step=step+1, episode=int(ei), task_loss=float(task_loss.detach()),
            teaching_loss=float(teaching.detach()), total=float(loss.detach()), gradient_norm=float(norm),
            elapsed=time.perf_counter()-start))
        if (step+1) % 16 == 0:
            print('TRAIN', dest.parent.name, arm, history[-1], flush=True)
    torch.save({k: v.detach().cpu() for k, v in tail.state_dict().items()}, dest/'layer4.pt')
    write(dest/'history.json', history)
    del optimizer


def target_components(context, data, extra, head, u):
    fixed = np.load(context.parent/'target_bank.npy')
    np.testing.assert_array_equal(np.load(context.parent/'target_rows.npy'), data['rows'])
    bank = bank_from_extra(fixed, extra, context.geometry)
    phi = numpy_phi(bank, context.raw[data['rows']], extra, u[data['rows']], context.geometry)
    prior = np.c_[bank, u[data['rows']]]@head['coef']+head['intercept']
    return prior, phi


def teacher_student_diagnostics(bank, rows, teachers, labels):
    pos = pd.Index(rows).get_indexer(teachers['rows'])
    p = np.empty_like(teachers['equal'])
    for k, head in enumerate(teachers['heads']):
        ix = teachers['fold'] == k
        p[ix] = bank[pos[ix]]@head['coef']+head['intercept']
    mask = teachers['multi']; y = labels[teachers['rows']][mask]
    records = {}
    for key in ('equal', 'knowledge'):
        d = p[mask]-teachers[key][mask]
        records[key+'_mse'] = float(np.mean((d-d.mean(1, keepdims=True))**2))
    records['source_teacher_cohort_ba'] = float(100*np.mean([np.mean(p[mask][y == c].argmax(1) == c) for c in np.unique(y)]))
    records['source_teacher_cohort_oa'] = float(100*np.mean(p[mask].argmax(1) == y))
    records['not_independent_validation'] = True
    return records


def freeze_predictions(context, data, base, extra, head, u, dest):
    prior, phi = target_components(context, data, extra, head, u)
    expert = np.empty((5, len(prior), 8))
    fusion = expert.copy()
    for draw, support in enumerate(data['support']):
        kernel = phi@phi[support].T
        expert[draw] = score(kernel, prior, support, data['support_labels'][draw])[0]
        fusion[draw] = late(base[draw], expert[draw])
    np.savez_compressed(dest/'predictions.npz', rows=data['rows'], support=data['support'],
        eval_masks=data['eval_masks'], expert=expert, fusion=fusion, prior=prior, phi=phi)
    return expert


def run_port(port, prefix, raw_gpu, u_gpu, manifest, u):
    dest = OUT/port
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'PREDICTIONS_FROZEN.json').exists():
        print('REUSE completed', port, flush=True)
        return
    folder = ROOT/'learned_frequency_residual_20261009/tasks/self24ports'/port
    context = ExpertContext(folder)
    context._prepare_source()
    write(dest/'OPERATOR_PARITY.json', parity_check(context, u))
    labels = manifest.class_id.to_numpy()
    data = dict(np.load(folder/'target.npz'))
    embed = dict(np.load(folder/'selective_embedding.npz'))
    np.testing.assert_array_equal(embed['rows'], data['rows'])
    base = []
    for draw, support in enumerate(data['support']):
        k = base_kernel(data, embed['embedding'].astype(float), support, context.info['eta'])
        base.append(score(k, data['prior'], support, data['support_labels'][draw])[0])
    base = np.stack(base)
    np.save(dest/'base_scores.npy', base)
    historical = np.load(ROOT/'frozen_comparison_20261010/targets/self24ports'/port/'policies.npz')
    base_mode = list(historical['modes']).index('ours_base')
    np.testing.assert_array_equal(base.argmax(2), historical['predictions'][:, base_mode])
    tail = resnet18(weights=None).layer4.cuda().eval()
    path = ROOT/'module_completion_20261008/ports_projection'/port/'joint/layer4.pt'
    initial = torch.load(path, map_location='cpu', weights_only=True)
    tail.load_state_dict(initial)
    print('INITIAL FEATURE EXTRACTION', port, len(context.source_rows), flush=True)
    initial_extra = extract(tail, prefix, context.source_rows)
    initial_target = extract(tail, prefix, data['rows'])
    source_bank = bank_from_extra(context.bank, initial_extra, context.geometry)
    augmented = np.c_[source_bank, u[context.source_rows]]
    head = fit_head(augmented, labels[context.source_rows], context.transforms)
    ref = dest/'no_training'; ref.mkdir(exist_ok=True)
    with (ref/'head.pkl').open('wb') as f:
        pickle.dump(head, f)
    expert = freeze_predictions(context, data, base, initial_target, head, u, ref)
    write(dest/'INITIAL_REPLAY.json', dict(source_feature_max_error=float(np.max(abs(initial_extra-context.extra))),
        target_expert_score_max_error=float(np.max(abs(expert-data['expert']))),
        target_expert_argmax_changes=int(np.sum(expert.argmax(2) != data['expert'].argmax(2))),
        baseline='Fresh float32 tail, original fixed geometry, refitted source prior; all arms share this recipe',
        sklearn_pickle='Numeric PCA buffers only; estimators are not refit through version-dependent sklearn APIs'))
    teachers = prepare_teachers(context, augmented, manifest, dest)
    write(ref/'STUDENT_DIAGNOSTIC.json', teacher_student_diagnostics(augmented, context.source_rows, teachers, labels))
    episodes = source_episodes(context, augmented, manifest, dest)
    banks = np.zeros((len(manifest), context.bank.shape[1]), np.float64)
    banks[context.source_rows] = context.bank
    banks[data['rows']] = np.load(context.parent/'target_bank.npy')
    bank_gpu = tensor(banks)
    geometry = FullExpertGeometry(context.geometry).cuda()
    for arm in ARMS:
        ad = dest/arm; ad.mkdir(exist_ok=True)
        if (ad/'predictions.npz').exists():
            print('REUSE trained arm', port, arm, flush=True)
            continue
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        train_arm(arm, tail, initial, geometry, prefix, raw_gpu, bank_gpu, u_gpu,
                  episodes, teachers, labels, context.source_rows, ad)
        print('REFIT AND EXTRACT', port, arm, flush=True)
        extra = extract(tail, prefix, context.source_rows)
        target_extra = extract(tail, prefix, data['rows'])
        np.save(ad/'source_extra.npy', extra)
        np.save(ad/'target_extra.npy', target_extra)
        new_bank = bank_from_extra(context.bank, extra, context.geometry)
        new_aug = np.c_[new_bank, u[context.source_rows]]
        head = fit_head(new_aug, labels[context.source_rows], context.transforms)
        with (ad/'head.pkl').open('wb') as f:
            pickle.dump(head, f)
        freeze_predictions(context, data, base, target_extra, head, u, ad)
        write(ad/'STUDENT_DIAGNOSTIC.json', teacher_student_diagnostics(new_aug, context.source_rows, teachers, labels))
        print('ARM FROZEN', port, arm, flush=True)
    write(dest/'PREDICTIONS_FROZEN.json', dict(arms=list(ARMS), final_step=STEPS,
        query_truth_used_for_training=False, target_labels=64, support_draws=5, neural_seeds=1))
    del tail, geometry, bank_gpu, episodes
    gc.collect(); torch.cuda.empty_cache()


def report():
    if not all((OUT/p/'PREDICTIONS_FROZEN.json').exists() for p in PORTS):
        raise RuntimeError('Freeze all three target ports before reading query truth')
    manifest = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id', 'mmsi', 'port'])
    rows, classes, events, attribution = [], [], [], []
    def metric(y, pred):
        return dict(oa=100*float(np.mean(pred == y)),
            ba=100*float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)])))
    for port in PORTS:
        saved = {a: dict(np.load(OUT/port/a/'predictions.npz')) for a in ('no_training', *ARMS)}
        reference = saved['continued']; truth = manifest.iloc[reference['rows']].class_id.to_numpy()
        base = np.load(OUT/port/'base_scores.npy')
        for arm, data in saved.items():
            for draw, ev in enumerate(data['eval_masks']):
                y = truth[ev]
                for endpoint in ('expert', 'fusion'):
                    before = reference[endpoint][draw, ev].argmax(1)
                    pred = data[endpoint][draw, ev].argmax(1)
                    rows.append(dict(port=port, arm=arm, draw=draw, endpoint=endpoint, **metric(y, pred),
                        rescue=int(np.sum((before != y)&(pred == y))), harm=int(np.sum((before == y)&(pred != y)))))
                    for c in np.unique(y):
                        m = y == c
                        classes.append(dict(port=port, arm=arm, draw=draw, endpoint=endpoint, class_id=int(c),
                            n=int(m.sum()), recall=100*float(np.mean(pred[m] == c)),
                            delta=100*float(np.mean((pred[m] == c).astype(float)-(before[m] == c)))))
                    if endpoint == 'fusion':
                        for j in np.flatnonzero((pred == y) != (before == y)):
                            rid = reference['rows'][ev][j]
                            events.append(dict(port=port, arm=arm, draw=draw, row=int(rid),
                                mmsi=str(manifest.iloc[rid].mmsi), class_id=int(y[j]),
                                before=int(before[j]), after=int(pred[j]), rescue=bool(pred[j] == y[j])))
        # Fixed readout interventions locate changes without choosing a target-best arm.
        for arm in ('equal_teacher', 'knowledge_teacher'):
            for p_arm, k_arm, name in (('continued','continued','control'),
                (arm,'continued','prior_only'), ('continued',arm,'kernel_only'), (arm,arm,'both')):
                for draw, support in enumerate(reference['support']):
                    phi = saved[k_arm]['phi']; prior = saved[p_arm]['prior']
                    sy = truth[support]
                    ep = score(phi@phi[support].T, prior, support, sy)[0]
                    if name in ('control', 'both'):
                        np.testing.assert_allclose(ep, saved[p_arm]['expert'][draw], atol=1e-10, rtol=1e-10)
                    ev = reference['eval_masks'][draw]
                    pred = late(base[draw], ep)[ev].argmax(1)
                    attribution.append(dict(port=port, arm=arm, path=name, draw=draw, **metric(truth[ev], pred)))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'class_metrics.csv', index=False)
    pd.DataFrame(events).to_csv(OUT/'error_events.csv', index=False)
    pd.DataFrame(attribution).to_csv(OUT/'prior_kernel_interventions.csv', index=False)
    summary = frame.groupby(['endpoint', 'arm'])[['oa','ba','rescue','harm']].mean()
    summary.to_csv(OUT/'summary.csv')
    frame.groupby(['port','endpoint','arm'])[['oa','ba']].mean().to_csv(OUT/'port_means.csv')
    write(OUT/'COMPLETE.json', dict(ports=list(PORTS), trained_arms=list(ARMS), neural_fits=9,
        neural_seeds=1, support_draws=5, steps=STEPS, all_predictions_before_query_readout=True,
        prediction_uses_motion_or_identity=False, acquisition_incidence_retained=True,
        independent_confirmation=False, original_model_untouched=True))
    print(summary.to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ports', nargs='+', default=list(PORTS))
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if args.report_only:
        report(); return
    self_test()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(date='2026-10-12', ports=list(PORTS), arms=list(ARMS), seed=SEED, steps=STEPS,
        query_batch=QUERY_BATCH, teacher_batch=TEACHER_BATCH, lr=1e-5, weight_decay=1e-4,
        task_loss='Class-balanced CE through current full expert, source support64 residual ridge8',
        teaching='0.5 centered-score MSE, source same-vessel other-whole-product OOF prior teachers',
        knowledge='4 bounded speed/double-direction inputs; no confidence; refit reader in full expert coordinates',
        coverage='Original 1296 fit observations intersect allowed outer source; no new target metadata',
        teacher_scope='Vessel-held linear source heads and quality readers; historical encoder/geometry not vessel-held; upstream head not nested in quality CV',
        prior_training='Source episode and teacher head coefficients frozen during continuation; refit final source prior in updated features for all arms',
        geometry='Original PCA/Qs/Qd/scales fixed; bank, complementary features and support kernel recomputed from new R18',
        inference='Original SAR chip and existing SAR acquisition incidence only; NO new AIS, identity or history input',
        primary='Equal softmax full fusion; original R50/frequency base unchanged; same5x64 supports',
        interpretation='Bounded full-model integration pilot, not causal-factor identification or independent confirmation',
        frozen_selection='Fixed final step64, no target selection or per-port hyperparameters')
    if (OUT/'PROTOCOL.json').exists():
        assert json.loads((OUT/'PROTOCOL.json').read_text(encoding='utf-8')) == protocol
    else:
        write(OUT/'PROTOCOL.json', protocol)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    manifest = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id','mmsi','port'])
    source = np.load(PREFIX, mmap_mode='r')
    prefix = torch.empty(source.shape, dtype=torch.float16, device='cuda')
    for lo in range(0, len(source), 512):
        prefix[lo:lo+512].copy_(torch.from_numpy(np.array(source[lo:lo+512])))
    print('PREFIX READY', tuple(prefix.shape), flush=True)
    raw_gpu = tensor(np.asarray(np.load(SOURCE/'pooled_features.npy', mmap_mode='r')), torch.float32)
    u = _conditions(); u_gpu = tensor(u)
    for port in args.ports:
        assert port in PORTS
        run_port(port, prefix, raw_gpu, u_gpu, manifest, u)
    del prefix, raw_gpu, u_gpu
    torch.cuda.empty_cache()
    if all((OUT/p/'PREDICTIONS_FROZEN.json').exists() for p in PORTS):
        report()


if __name__ == '__main__':
    main()
