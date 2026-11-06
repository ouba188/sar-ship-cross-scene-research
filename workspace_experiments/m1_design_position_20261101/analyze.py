"""Matched design/position comparisons, pathway checks and native-chip gallery."""
import html
from experiment import *
sys.path.insert(0, str(task.ROOT))
import review_task as gallery


def main():
    rows, train, ids, rid = task.records()
    y = np.array([rows[i]['label'] for i in ids])
    a = task.arrays()
    scores = {'anchor': np.load(ROOT/'anchor.npz')['score']}
    metrics, counts, pairs, source = [], [], [], []
    references_order = None
    for arm in ARMS:
        dest = ROOT/(arm+'_17')
        r = json.loads((dest/'receipt.json').read_text())
        order = json.loads((dest/'orders.json').read_text())
        if references_order is None:
            references_order = order
        assert order == references_order
        for mode in ('final', 'off', 'vv_only', 'vh_only', 'uniform_context', 'spatial_mean'):
            path = dest/(mode+'.npz')
            if not path.exists():
                continue
            with np.load(path) as z:
                assert np.array_equal(z['ids'], ids)
                name = arm if mode == 'final' else arm+'_'+mode
                scores[name] = z['score']
                metrics.append(dict(arm=arm, mode=mode, **task.metrics(y, z['score']),
                                    update_vv_mean=float(z['update_ratio'][:, 0].mean()),
                                    update_vh_mean=float(z['update_ratio'][:, 1].mean())))
        source.append(dict(arm=arm, **r['source'], gradient_norm_max=r['gradient_norm_max'], clipped_steps=r['clipped_steps']))
        with np.load(dest/'source_initial_probe.npz') as z:
            yy = np.array([rows[i]['label'] for i in z['ids']])
            av = np.asarray(a['available'][z['ids']]).astype(bool)
            for p in range(2):
                valid = av[:, p]
                m = task.metrics(yy[valid], z['auxiliary'][valid, p])
                source.append(dict(arm=arm, role='initial_source_aux_'+('VV','VH')[p], **m))
    comparisons = [(arm, 'none') for arm in ARMS[1:]]
    comparisons += [(arm, 'anchor') for arm in ARMS]
    comparisons += [('post_'+d, 'pre_'+d) for d in ('additive', 'class_gate', 'mean_gate')]
    for pos in ('pre', 'post'):
        comparisons += [(pos+'_class_gate', pos+'_'+d) for d in ('additive','mean_gate')]
        comparisons += [(pos+'_class_gate', pos+'_class_gate_'+mode) for mode in ('off','vv_only','vh_only','uniform_context','spatial_mean')]
    comparisons += [(arm, arm+'_off') for arm in ARMS[1:] if 'class_gate' not in arm]
    for candidate, control in comparisons:
        cc, bb = scores[candidate].argmax(1), scores[control].argmax(1)
        mc, mb = [task.metrics(y, scores[k]) for k in (candidate, control)]
        pairs.append(dict(candidate=candidate, control=control, rescues=int(((cc==y)&(bb!=y)).sum()),
                          harms=int(((cc!=y)&(bb==y)).sum()), changes=int((cc!=bb).sum()),
                          delta_oa=mc['oa']-mb['oa'], delta_ba=mc['ba']-mb['ba'], delta_nll=mc['nll']-mb['nll']))
    for name, z in scores.items():
        for cls in range(4):
            mask = y == cls
            counts.append(dict(arm=name, label=cls, n=int(mask.sum()), correct=int(((z.argmax(1)==y)&mask).sum())))
    pd.DataFrame(metrics).to_csv(ROOT/'metrics_all.csv', index=False)
    pd.DataFrame(pairs).to_csv(ROOT/'pairs.csv', index=False)
    pd.DataFrame(counts).to_csv(ROOT/'class_counts.csv', index=False)
    pd.DataFrame(source).to_csv(ROOT/'source_diagnostics.csv', index=False)
    focus = 'post_class_gate'
    good = scores[focus].argmax(1)==y
    no = scores['none'].argmax(1)==y
    mean = scores['post_mean_gate'].argmax(1)==y
    pre = scores['pre_class_gate'].argmax(1)==y
    groups = [('post_vs_none_rescue',good & ~no),('post_vs_none_harm',~good & no),
              ('post_vs_mean_rescue',good & ~mean),('post_vs_mean_harm',~good & mean),
              ('post_vs_pre_rescue',good & ~pre),('post_vs_pre_harm',~good & pre),
              ('stable_correct',good & no & mean & pre),('stable_wrong',~good & ~no & ~mean & ~pre)]
    selected, cases, status = set(), [], []
    rank = abs(gallery.margins(scores[focus],y)-gallery.margins(scores['none'],y))
    gallery.ROOT = ROOT
    view_scores = {k: scores[k] for k in ('anchor',)+ARMS}
    view_scores['baseline'] = scores['anchor']
    for group, mask in groups:
        pool = sorted(np.flatnonzero(mask), key=lambda i:-rank[i])
        pos = next((int(i) for i in pool if int(i) not in selected), None)
        status.append(dict(group=group, eligible=len(pool), selected=pos))
        if pos is None:
            continue
        selected.add(pos)
        case = gallery.draw_case(rows,a,rid,int(ids[pos]),group,view_scores,y,pos,focus=focus)
        case['pathways'] = {k: dict(prediction=int(z[pos].argmax()), margin=float(gallery.margins(z[pos:pos+1], y[pos:pos+1])[0]))
                            for k,z in scores.items() if k.startswith(focus)}
        cases.append(case)
    task.write(ROOT/'cases.json',dict(groups=status,cases=cases,identical_source_orders=True,seed=17))
    selected_metrics = [dict(arm='anchor',**task.metrics(y,scores['anchor']))]
    selected_metrics += [dict(arm=arm,**task.metrics(y,scores[arm])) for arm in ARMS]
    summary = pd.DataFrame(selected_metrics)
    summary.to_csv(ROOT/'summary17.csv',index=False)
    body = ['<html><meta charset="utf-8"><title>M1 design and placement</title>',
            '<style>body{font-family:Arial;margin:24px}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:5px}</style>',
            '<h1>M1 design x placement: OSS2</h1><p>Four classes, source719/development228, fixed32 references. Seed17, final epoch4. Historical development, not blind confirmation.</p>',
            '<p>M2/M3/head fixed; full M1 and final bottleneck conv3 trained. Only M1 moves. Native VV/VH crop/pad, no resizing of model inputs.</p>',
            summary.to_html(index=False),pd.DataFrame(pairs).to_html(index=False),
            '<p>Reference panels: two legal same-class and one confusing-class case selected by pooled cosine, not maximum learned attention.</p>']
    for case in cases:
        body += ['<h2>'+html.escape(case['group']+' '+case['id'])+'</h2>', '<img src="'+case['image']+'">',
                 '<pre>'+html.escape(json.dumps(dict(evidence=case['evidence'],pathways=case['pathways']),indent=2))+'</pre>']
    (ROOT/'index.html').write_text('\n'.join(body+['</html>']),encoding='utf-8')
    print(summary.to_string(index=False))
    print(pd.DataFrame(pairs).to_string(index=False))
    print(json.dumps(status))


if __name__ == '__main__':
    main()
