"""Reuse the selected model; add only matched terminal adaptation without M3."""
from pathlib import Path
import argparse
import gc
import importlib.util
import json
import sys
import time

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
OLD = WORK / 'm1_guided_m2_20261106'
sys.path.insert(0, str(OLD))
import study as s

torch, np, pd, task = s.torch, s.np, s.pd, s.task


def write(name, data):
    task.write(ROOT / name, data)


@torch.no_grad()
def prepare():
    if (ROOT / 'cache/complete.json').exists():
        return
    rows, groups = s.prev.records()
    a, c = s.prev.arrays()
    net = s.prev.Network('class')
    net.restore(torch.load(s.CHECKPOINT, map_location='cpu', weights_only=True))
    net.eval().requires_grad_(False)
    saved = {p.stem: np.load(p, mmap_mode='r') for p in s.CACHE_ROOT.glob('*.npy')}
    seen = {}
    def head_hook(module, args):
        seen['features'] = args[0].detach()
    def response_hook(p):
        def hook(module, args, output):
            seen[('response', p)] = output.detach()
        return hook
    def readout_hook(p):
        def hook(module, args):
            seen[('x', p)] = args[0].detach()
        return hook
    handles = [net.base.head.register_forward_pre_hook(head_hook)]
    for p in range(2):
        handles += [net.m1[p].reader.register_forward_hook(response_hook(p)),
                    net.base.m2[p].register_forward_pre_hook(readout_hook(p))]
    probe = np.r_[groups['small'][:2], groups['validation'][:2], groups['evaluation'][:2]]
    score = net(a, c, probe)[0].cpu().numpy()
    errors = dict(score=float(abs(score-saved['score'][probe]).max()),
                  features=float(abs(seen['features'].cpu().numpy()-saved['features'][probe]).max()))
    assert max(errors.values()) < 1e-4, errors
    off = dict(c)
    off['delta'] = np.broadcast_to(np.zeros((1, 2, 1024, 10, 10), np.float32), c['delta'].shape)
    net(a, off, probe)
    for key in ('x', 'response'):
        value = torch.stack([seen[(key, p)] for p in range(2)], 1).cpu().numpy()
        errors['unchanged_'+key] = float(abs(value-saved[key][probe]).max())
        assert errors['unchanged_'+key] < 1e-4, errors
    (ROOT / 'cache').mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(ROOT/'cache/features_no_m3.npy', mode='w+',
                                   dtype=np.float32, shape=saved['features'].shape)
    out[:] = saved['features'][:]
    # The unused historical selection cohort is neither fitted nor evaluated.
    ids_all = np.r_[groups['small'], groups['validation'], groups['evaluation']]
    tick = time.perf_counter()
    for first in range(0, len(ids_all), 8):
        ids = ids_all[first:first+8]
        net(a, off, ids)
        out[ids] = seen['features'].cpu().numpy()
        if first % 160 == 0:
            print('PREPARE_NO_M3', first+len(ids), len(ids_all), flush=True)
    out.flush()
    assert np.isfinite(out[ids_all]).all()
    errors['feature_change_max'] = float(abs(out[ids_all]-saved['features'][ids_all]).max())
    assert errors['feature_change_max'] > 0
    for handle in handles:
        handle.remove()
    write('cache/complete.json', dict(n_recomputed=len(ids_all), errors=errors,
                                     seconds=time.perf_counter()-tick,
                                     no_m3='Zero fixed M3 residual before layer4 and deep pool; downstream recomputed',
                                     unused_selection_not_recomputed=True))
    print('PREPARED', errors, flush=True)
    del net, out, saved, a, c, off
    gc.collect()
    torch.cuda.empty_cache()


def fit():
    rows, groups = s.prev.records()
    cache = s.load_cache()
    off_features = np.load(ROOT/'cache/features_no_m3.npy', mmap_mode='r')
    off_tensor = torch.as_tensor(np.array(off_features, copy=True), device=task.DEVICE)
    original_features = cache['features']
    diagnostics = []
    y = np.array([r['label'] for r in rows])
    for seed in (17, 29, 43):
        net = s.RegionalGuidance('class').to(task.DEVICE).eval()
        net.load_state_dict(torch.load(OLD/f'class_s{seed}/final.pt', map_location=task.DEVICE, weights_only=True))
        for split in ('validation', 'evaluation'):
            ids = groups[split]
            cache['features'] = original_features
            actual = s.predict(net, cache, ids)
            old = np.load(OLD/f'class_s{seed}/{split}.npz')
            assert np.array_equal(old['ids'], ids)
            error = float(abs(actual-old['score']).max())
            assert error < 1e-4 and np.array_equal(actual.argmax(1), old['score'].argmax(1)), error
            cache['features'] = off_tensor
            off_score = s.predict(net, cache, ids)
            dest = ROOT / f'same_weights_s{seed}'
            dest.mkdir(exist_ok=True)
            np.savez_compressed(dest/(split+'.npz'), ids=ids, score=off_score)
            diagnostics.append(dict(seed=seed, split=split, replay_error=error,
                                    **task.metrics(y[ids], off_score)))
        del net
    write('same_weights.json', diagnostics)
    cache['features'] = off_tensor
    s.ROOT = ROOT / 'no_m3_refit'
    for seed in (17, 29, 43):
        s.train('class', seed, cache, rows, groups, freeze_head=False)
        new = json.loads((s.ROOT/f'class_s{seed}/orders.json').read_text())
        old = json.loads((OLD/f'class_s{seed}/orders.json').read_text())
        assert new == old, seed
    write('fit_status.json', dict(completed=True, new_fits=3, steps_per_fit=180,
                                  identical_orders=True, common_upstream_seed=17,
                                  original_runs_reused=True, formal_model_changed=False))


def metrics(y, score):
    result = task.metrics(y, score)
    p = np.exp(score-score.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    result['brier'] = float(((p-np.eye(4)[y])**2).sum(1).mean())
    return result


def analyze():
    rows, groups = s.prev.records()
    entries, summary, pairs, strata, events = {}, [], [], [], []
    roots = {'full': OLD, 'without_m3_refit': ROOT/'no_m3_refit'}
    for split in ('validation', 'evaluation'):
        ids = groups[split]
        y = np.array([rows[i]['label'] for i in ids])
        for seed in (17, 29, 43):
            paths = {name: path/f'class_s{seed}/{split}.npz' for name, path in roots.items()}
            paths.update(without_m3_same_weights=ROOT/f'same_weights_s{seed}/{split}.npz',
                         mean_readout=OLD/f'mean_s{seed}/{split}.npz',
                         content_readout=OLD/f'content_s{seed}/{split}.npz')
            for name, path in paths.items():
                z = np.load(path)
                assert np.array_equal(z['ids'], ids)
                entries[(split, seed, name)] = score = z['score']
                summary.append(dict(split=split, seed=seed, arm=name, **metrics(y, score)))
            full = entries[(split, seed, 'full')]
            good = full.argmax(1) == y
            for control in ('without_m3_refit', 'without_m3_same_weights', 'mean_readout', 'content_readout'):
                base = entries[(split, seed, control)]
                old = base.argmax(1) == y
                mm, bb = metrics(y, full), metrics(y, base)
                pairs.append(dict(split=split, seed=seed, control=control,
                                  rescues=int((good & ~old).sum()), harms=int((~good & old).sum()),
                                  **{'delta_'+key: mm[key]-bb[key] for key in ('oa', 'ba', 'nll', 'brier')}))
                if control != 'without_m3_refit':
                    continue
                for key in ('label', 'orbit'):
                    values = np.array([rows[i]['label'] if key == 'label' else task.orbit(rows[i]) for i in ids])
                    for val in np.unique(values):
                        mask = values == val
                        strata.append(dict(split=split, seed=seed, group=key, value=str(val), n=int(mask.sum()),
                                           rescues=int((good & ~old & mask).sum()), harms=int((~good & old & mask).sum()),
                                           delta_oa=100*float((good[mask].astype(float)-old[mask]).mean())))
                for pos in np.flatnonzero(good != old):
                    row = rows[ids[pos]]
                    events.append(dict(split=split, seed=seed, id=row['id'], mmsi=row['mmsi'], orbit=task.orbit(row),
                                       label=int(y[pos]), effect='rescue' if good[pos] else 'harm'))
    frame, pair = pd.DataFrame(summary), pd.DataFrame(pairs)
    frame.to_csv(ROOT/'summary.csv', index=False)
    pair.to_csv(ROOT/'paired.csv', index=False)
    pd.DataFrame(strata).to_csv(ROOT/'strata.csv', index=False)
    pd.DataFrame(events).to_csv(ROOT/'events.csv', index=False)
    averages = frame.groupby(['split', 'arm'])[['oa','ba','nll','brier']].agg(['mean','std'])
    averages.to_csv(ROOT/'averages.csv')
    pair.groupby(['split','control'])[['delta_oa','delta_ba','delta_nll','delta_brier','rescues','harms']].mean().to_csv(ROOT/'paired_means.csv')
    spec = importlib.util.spec_from_file_location('minimal_gallery', task.ROOT/'review_task.py')
    gallery = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gallery)
    gallery.ROOT = ROOT
    a, _ = s.prev.arrays()
    ids = groups['evaluation']
    y = np.array([rows[i]['label'] for i in ids])
    cases, statuses, used = [], [], set()
    for group in ('rescue','harm','stable_correct','stable_wrong'):
        choices = []
        for seed in (17,29,43):
            z = entries[('evaluation',seed,'full')]
            b = entries[('evaluation',seed,'without_m3_refit')]
            good, old = z.argmax(1)==y, b.argmax(1)==y
            mask = {'rescue':good&~old, 'harm':~good&old,
                    'stable_correct':good&old, 'stable_wrong':~good&~old}[group]
            delta = abs(gallery.margins(z,y)-gallery.margins(b,y))
            choices += [(float(delta[pos]),seed,int(pos)) for pos in np.flatnonzero(mask)]
        choices.sort(reverse=True)
        pick = next((v for v in choices if int(ids[v[2]]) not in used),None)
        statuses.append(dict(group=group,events=len(choices),selected=pick))
        if pick is None:
            continue
        _,seed,pos=pick
        idx=int(ids[pos]);used.add(idx)
        scores={name:entries[('evaluation',seed,name)] for name in ('full','without_m3_refit','without_m3_same_weights')}
        scores['baseline']=scores['without_m3_refit']
        case=gallery.draw_case(rows,a,groups['references'],idx,f'{group}_s{seed}',scores,y,pos,focus='full')
        case.update(seed=seed,split='evaluation')
        cases.append(case)
    write('cases.json',dict(cases=cases,groups=statuses,
                           selection='Largest absolute paired margin change, one unique observation per group; not prevalence'))
    import html
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>当前模型最小补实验</title>',
          '<style>body{font-family:Arial,sans-serif;max-width:1300px;margin:24px auto;padding:16px}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px}</style>',
          '<h1>当前组合：M3末阶段配套对照</h1><p>OSS2四分类；固定上游seed17；新增三个读出/分类头训练种子，不是全网络从头消融。</p>',
          '<p>正差表示完整组合更好；NLL/Brier负差更好。参考图来自固定参考库，但不是声称其注意力最大。</p>',
          averages.to_html(),pair.to_html(index=False)]
    for case in cases:
        body += ['<h2>'+html.escape(case['group']+' '+case['id'])+'</h2>',
                 '<img src="'+case['image']+'">','<pre>'+html.escape(json.dumps(case['evidence'],indent=2))+'</pre>']
    (ROOT/'index.html').write_text('\n'.join(body+['</html>']),encoding='utf-8')
    print(averages.to_string(),flush=True)
    print(pair.to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','fit','analyze','all'),default='all',nargs='?')
    args=parser.parse_args()
    task.current.setup(task.DEVICE)
    if args.action in ('prepare','all'):
        prepare()
    if args.action in ('fit','all'):
        fit()
    if args.action in ('analyze','all'):
        analyze()


if __name__=='__main__':
    main()
