"""Paired classification outcomes and native query/reference evidence."""
import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_pilot import *
import review_task as gallery


def main():
    task.current.setup(task.DEVICE)
    rows, train, ids, rid = task.records()
    a = task.arrays()
    y = np.array([rows[i]['label'] for i in ids])
    scores = {'anchor': np.load(ROOT/'anchor.npz')['score']}
    for arm in ARMS:
        for mode in ('final', 'off', 'spatial_mean', 'vv_only', 'vh_only'):
            path = ROOT/arm/(mode+'.npz')
            if path.exists():
                with np.load(path) as p:
                    assert np.array_equal(ids, p['ids'])
                    scores[arm if mode == 'final' else arm+'_'+mode] = p['score']
    pairs = []
    comparisons = [('class_gate', k) for k in ('anchor', 'additive', 'bounded_add', 'mean_gate',
                                               'class_gate_off', 'class_gate_spatial_mean', 'class_gate_vv_only', 'class_gate_vh_only')]
    comparisons += [(k, k+'_off') for k in ('additive', 'bounded_add', 'mean_gate')]
    for candidate, control in comparisons:
        aa, bb = [scores[k].argmax(1) == y for k in (candidate, control)]
        ma, mb = [task.metrics(y, scores[k]) for k in (candidate, control)]
        pairs.append(dict(candidate=candidate, control=control, rescues=int((aa & ~bb).sum()),
                          harms=int((~aa & bb).sum()), delta_oa=ma['oa']-mb['oa'],
                          delta_ba=ma['ba']-mb['ba'], delta_nll=ma['nll']-mb['nll'],
                          changed=int((scores[candidate].argmax(1) != scores[control].argmax(1)).sum())))
    pd.DataFrame(pairs).to_csv(ROOT/'pairs.csv', index=False)
    counts = []
    for name, z in scores.items():
        for cls in range(4):
            mask = y == cls
            counts.append(dict(arm=name, label=cls, n=int(mask.sum()), correct=int(((z.argmax(1)==y)&mask).sum())))
    pd.DataFrame(counts).to_csv(ROOT/'class_counts.csv', index=False)
    stats = []
    for arm in ARMS:
        p = np.load(ROOT/arm/'final.npz')
        for pol, name in enumerate(('VV', 'VH')):
            r = p['update_ratio'][:, pol]
            stats.append(dict(arm=arm, polarization=name, update_norm_mean=float(r.mean()), update_norm_max=float(r.max())))
    pd.DataFrame(stats).to_csv(ROOT/'update_norms.csv', index=False)
    gate = scores['class_gate'].argmax(1)==y
    avg = scores['mean_gate'].argmax(1)==y
    old = scores['anchor'].argmax(1)==y
    groups = [('vs_mean_rescue', gate & ~avg), ('vs_mean_harm', ~gate & avg),
              ('vs_anchor_rescue', gate & ~old), ('vs_anchor_harm', ~gate & old),
              ('stable_correct', gate & avg & old), ('stable_wrong', ~gate & ~avg & ~old)]
    rank = abs(gallery.margins(scores['class_gate'], y)-gallery.margins(scores['mean_gate'], y))
    selected, cases, available = set(), [], []
    gallery.ROOT = ROOT
    visual_scores = {k: scores[k] for k in ('anchor', 'additive', 'bounded_add', 'class_gate', 'mean_gate')}
    visual_scores['baseline'] = scores['anchor']
    for name, mask in groups:
        pool = sorted(np.flatnonzero(mask), key=lambda i: -rank[i])
        pos = next((int(i) for i in pool if int(i) not in selected), None)
        available.append(dict(group=name, count=len(pool), selected=pos))
        if pos is None:
            continue
        selected.add(pos)
        cases.append(gallery.draw_case(rows, a, rid, int(ids[pos]), name, visual_scores, y, pos, focus='class_gate'))
    dump('cases.json', dict(groups=available, cases=cases))
    # Local response modulation is shown as a norm map, not a physical attribution.
    net = Pilot(anchor(), 'class_gate')
    net.outputs.load_state_dict(torch.load(ROOT/'class_gate/outputs.pt', map_location=task.DEVICE, weights_only=True))
    c = {p.stem: np.load(p, mmap_mode='r') for p in (ROOT/'cache').glob('*.npy')}
    for case in cases:
        idx = case['index']
        x = task.tensor(c['x'], [idx])
        ctx = task.tensor(c['context'], [idx])
        occ = task.tensor(c['occupancy'], [idx])
        fig, axes = plt.subplots(2, 3, figsize=(9, 6))
        views = task.data.prepare(rows[idx])
        for p in range(2):
            if views[p] is not None:
                axes[p, 0].imshow(views[p]['pixels'], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            with torch.no_grad():
                gate_value = net.gain[p]*net.outputs[p](ctx[:, p]).tanh()
                delta = gate_value*x[:, p]
            for ax, value in [(axes[p, 1], gate_value.abs().mean(1)[0]), (axes[p, 2], delta.norm(dim=1)[0])]:
                image = ax.imshow(value.cpu(), cmap='magma', interpolation='nearest')
                fig.colorbar(image, ax=ax, shrink=.6)
            for ax, title in zip(axes[p], ('Native chip', 'Mean absolute modulation', 'Update channel norm')):
                ax.set_title(('VV' if p == 0 else 'VH')+' / '+title, fontsize=8)
                ax.axis('off')
        fig.suptitle(case['id']+' / 10x10 feature grid, not pixel attribution', fontsize=9)
        fig.tight_layout()
        fig.savefig(ROOT/(case['group']+'_modulation.png'), dpi=130)
        plt.close(fig)
    summary = pd.read_csv(ROOT/'summary.csv')
    body = ['<html><meta charset="utf-8"><title>M1 role-separation pilot</title>',
            '<style>body{font-family:Arial;margin:24px}img{max-width:100%}td,th{padding:6px;border:1px solid #bbb}table{border-collapse:collapse}</style>',
            '<h1>M1 role-separation pilot</h1><p>OSS2 four classes. Historical development228, seed17, fixed epoch4. M2/M3/head/backbone frozen. Not independent confirmation.</p>',
            summary.to_html(index=False), pd.DataFrame(pairs).to_html(index=False),
            '<p>References shown are legal bank examples chosen by pooled cosine, not maximum learned attention. Maps show feature updates, not physical causality.</p>']
    for case in cases:
        body += ['<h2>'+html.escape(case['group']+' '+case['id'])+'</h2>',
                 '<img src="'+case['image']+'">', '<img src="'+case['group']+'_modulation.png">',
                 '<pre>'+html.escape(json.dumps(case['evidence'], indent=2))+'</pre>']
    (ROOT/'index.html').write_text('\n'.join(body+['</html>']), encoding='utf-8')
    print(summary.to_string(index=False))
    print(pd.DataFrame(pairs).to_string(index=False))
    print(json.dumps(available))


if __name__ == '__main__':
    main()
