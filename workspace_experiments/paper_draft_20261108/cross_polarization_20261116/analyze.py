"""Paired logit interaction, ship-cluster uncertainty, and native-chip panels."""
import csv
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'region_dependency_20261116'
WORK = ROOT.parent.parent
REGIONS = ('body', 'ends', 'side')
NAMES = ('Bulk', 'General cargo', 'Chemical/product tanker', 'Container')
SEEDS = (17, 29, 43)
TOLERANCE = 1e-4
plt.rcParams['svg.fonttype'] = 'none'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def table(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(values, ships):
    _, inverse = np.unique(ships, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=values)
    rng = np.random.default_rng(20261116)
    draws = rng.integers(0, len(counts), size=(1000, len(counts)))
    distribution = sums[draws].sum(1) / counts[draws].sum(1)
    return np.quantile(distribution, [.025, .975]).tolist()


def factorial(z, y):
    original = z[0].copy()
    original[np.arange(len(y)), y] = -np.inf
    rival = original.argmax(1)
    margins = z[:, np.arange(len(y)), y] - z[:, np.arange(len(y)), rival]
    interaction = margins[3] - margins[1] - margins[2] + margins[0]
    return margins, interaction, rival


def load(seed, region, ids, y):
    keep = 7 ^ (1 << REGIONS.index(region))
    files = [OLD / 'predictions' / f'full_s{seed}' / n for n in
             ('clean.npz', f'VV_mean50_keep{keep}.npz', f'VH_mean50_keep{keep}.npz')]
    files += [ROOT / 'predictions' / f'full_s{seed}' / f'both_{region}_mean50.npz']
    z = []
    for path in files:
        with np.load(path) as result:
            assert np.array_equal(result['ids'], ids) and np.array_equal(result['y'], y), path
            z.append(result['score'].astype(np.float64))
    return np.stack(z)


def main():
    # Detect a sign/order error against an exactly additive game and a known interaction.
    dummy = np.array([[[4., 1.]], [[5., 1.]], [[6., 1.]], [[7., 1.]]])
    assert factorial(dummy, np.array([0]))[1][0] == 0
    dummy[3, 0, 0] += 2
    assert factorial(dummy, np.array([0]))[1][0] == 2
    rows = read(WORK / 'm1_coverage_20261101/manifest.json')
    groups = read(WORK / 'm1_coverage_20261101/groups.json')
    ids = np.asarray(groups['validation'])
    y = np.asarray([rows[i]['label'] for i in ids])
    ships = np.asarray([str(rows[i]['mmsi']) for i in ids])
    passes = np.asarray([rows[i]['pass_group'] for i in ids])
    cache = ROOT.parent / 'recent_baselines_20261111/native_cache'
    av = np.load(cache / 'available.npy')
    masks = np.load(OLD / 'native_regions.npy', mmap_mode='r')
    details, summaries, seed_rows, pass_rows = [], [], [], []
    games = {}
    for r, region in enumerate(REGIONS):
        eligible = (av[ids] > 0).all(1) & masks[:, :, r].any((-2, -1)).all(1)
        stats = []
        for seed in SEEDS:
            z = load(seed, region, ids, y)
            m, interaction, rival = factorial(z, y)
            add = z[1] + z[2] - z[0]
            predictions = z.argmax(2)
            games[region, seed] = (z, m, interaction, rival, eligible, add)
            stats.append(interaction)
            seed_rows.append(dict(region=region, seed=seed, n=int(eligible.sum()),
                mean_interaction=float(interaction[eligible].mean()),
                mean_abs_interaction=float(abs(interaction[eligible]).mean()),
                clean_oa=float((predictions[0, eligible] == y[eligible]).mean() * 100),
                joint_oa=float((predictions[3, eligible] == y[eligible]).mean() * 100),
                additive_oa=float((add[eligible].argmax(1) == y[eligible]).mean() * 100),
                joint_vs_additive_disagreement=int((predictions[3, eligible] != add[eligible].argmax(1)).sum())))
            for j in np.flatnonzero(eligible):
                details.append(dict(region=region, seed=seed, index=int(ids[j]), id=rows[ids[j]]['id'],
                    mmsi=ships[j], pass_group=passes[j], label=int(y[j]), rival=int(rival[j]),
                    m00=float(m[0, j]), m10=float(m[1, j]), m01=float(m[2, j]), m11=float(m[3, j]),
                    interaction=float(interaction[j]), clean_pred=int(predictions[0, j]),
                    vv_pred=int(predictions[1, j]), vh_pred=int(predictions[2, j]),
                    both_pred=int(predictions[3, j]), additive_pred=int(add[j].argmax())))
        stats = np.stack(stats)
        for c in (-1, 0, 1, 2, 3):
            use = eligible & ((y == c) if c >= 0 else True)
            values = stats[:, use].mean(0)
            ci = interval(values, ships[use])
            summaries.append(dict(region=region, category='Overall' if c < 0 else NAMES[c],
                n=int(use.sum()), ships=len(set(ships[use])), mean_interaction=float(values.mean()),
                ci_low=ci[0], ci_high=ci[1], mean_abs_interaction=float(abs(stats[:, use]).mean()),
                above_numerical_tolerance=float((abs(stats[:, use]) > TOLERANCE).mean()),
                positive_fraction=float((stats[:, use] > TOLERANCE).mean()),
                sign_agreement_3seeds=float(((stats[:, use] > TOLERANCE).all(0) | (stats[:, use] < -TOLERANCE).all(0)).mean())))
        for group in np.unique(passes[eligible]):
            use = eligible & (passes == group)
            pass_rows.append(dict(region=region, pass_group=group, n=int(use.sum()),
                mean_interaction=float(stats[:, use].mean()), mean_abs_interaction=float(abs(stats[:, use]).mean())))
    table(ROOT / 'sample_effects.csv', details)
    table(ROOT / 'class_effects.csv', summaries)
    table(ROOT / 'seed_effects.csv', seed_rows)
    table(ROOT / 'pass_effects.csv', pass_rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, key, title in zip(axes, ('mean_interaction', 'mean_abs_interaction'), ('Signed interaction', 'Absolute interaction')):
        values = np.array([[s[key] for s in summaries if s['region'] == r and s['category'] != 'Overall'] for r in REGIONS])
        vmax = max(float(abs(values).max()), TOLERANCE)
        ax.imshow(values, cmap='coolwarm' if key == 'mean_interaction' else 'YlGnBu',
                  vmin=-vmax if key == 'mean_interaction' else 0, vmax=vmax)
        ax.set_xticks(range(4), ('Bulk', 'Cargo', 'Chem./product', 'Container'), fontsize=9)
        ax.set_yticks(range(3), REGIONS)
        ax.set_title(title + ' (fixed-rival logit margin)')
        for r in range(3):
            for c in range(4):
                ax.text(c, r, f'{values[r,c]:.1e}', ha='center', va='center')
    fig.suptitle('All observed interactions below numerical tolerance 1e-4; do not interpret signs', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .9))
    fig.savefig(ROOT / 'interaction_matrix.svg')
    fig.savefig(ROOT / 'interaction_matrix.png', dpi=180)
    plt.close(fig)
    make_gallery(rows, ids, y, groups, av, masks, cache, games)
    lines = ['# Cross-polarization 2x2 diagnostic', '', 'Three fixed seeds; paired dual-available OSS2 validation observations only. Not new module accuracy.',
        'All interactions are below FP32 replay tolerance 1e-4. Tiny raw bootstrap intervals excluding zero have NO substantive interpretation here. The implemented score is structurally additive across polarizations.', '',
        '|Region|Class|N|Mean interaction|Ship-bootstrap 95% interval|Mean absolute interaction|', '|---|---|---:|---:|---|---:|']
    lines += [f"|{s['region']}|{s['category']}|{s['n']}|{s['mean_interaction']:+.2e}|[{s['ci_low']:+.2e}, {s['ci_high']:+.2e}]|{s['mean_abs_interaction']:.2e}|" for s in summaries]
    lines += ['', '## Prediction sensitivity', '', '|Region|Seed|N|Original OA|Joint-operation OA|Additive-logit prediction OA|Joint/additive disagreements|', '|---|---:|---:|---:|---:|---:|---:|']
    lines += [f"|{s['region']}|{s['seed']}|{s['n']}|{s['clean_oa']:.2f}|{s['joint_oa']:.2f}|{s['additive_oa']:.2f}|{s['joint_vs_additive_disagreement']}|" for s in seed_rows]
    lines += ['', 'Additive logits are a diagnostic counterprediction, not a deployable classifier or trained comparator.',
              'A nonzero interaction identifies nonadditivity of this learned function under these operations, not a physical causal interaction or guaranteed PCEA benefit.',
              'Signed interactions may cancel across ships. Absolute effects and per-class/per-pass/seed records are also retained.']
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(summaries=summaries, seed_effects=seed_rows), ensure_ascii=False), flush=True)


def make_gallery(rows, ids, y, groups, av, masks, cache, games):
    pixels = np.load(cache / 'pixels.npy', mmap_mode='r')
    valid = np.load(cache / 'footprint.npy', mmap_mode='r')
    folder = ROOT / 'gallery'
    folder.mkdir(exist_ok=True)
    cases = []
    selected = set()
    for region in REGIONS:
        z, m, interaction, rival, live, add = games[region, 17]
        p0, p1 = z[0].argmax(1), z[3].argmax(1)
        states = {'rescued': (p0 != y) & (p1 == y), 'harmed': (p0 == y) & (p1 != y)}
        if region == 'body':
            states.update(stable_correct=(p0 == y) & (p1 == y), stable_wrong=(p0 != y) & (p1 != y))
        for state, state_mask in states.items():
            candidates = [int(j) for j in np.flatnonzero(live & state_mask) if int(ids[j]) not in selected]
            if not candidates:
                continue
            candidates.sort(key=lambda j: abs(m[3, j] - m[0, j]))
            j = candidates[len(candidates) // 2]
            selected.add(int(ids[j]))
            idx, r = int(ids[j]), REGIONS.index(region)
            item = rows[idx]
            x = np.asarray(pixels[idx], dtype=np.float32) / 255
            changed = x.copy()
            for p in range(2):
                mask = masks[j, p, r]
                changed[p, mask] = .5 * x[p, mask] + .5 * x[p, mask].mean()
            orbit = lambda row: '_'.join(row['pass_group'].split('_')[:2])
            legal = [i for i in groups['references'] if rows[i]['mmsi'] != item['mmsi'] and orbit(rows[i]) != orbit(item)]
            refs = []
            for c in (int(y[j]), int(rival[j])):
                possible = [i for i in legal if rows[i]['label'] == c]
                refs.append(next((i for i in possible if av[i].all()), possible[0] if possible else None))
            fig, axes = plt.subplots(2, 6, figsize=(14, 6))
            scale = max(float(abs(changed - x).max()), 1 / 255)
            for p in range(2):
                for ax in axes[p]:
                    ax.axis('off')
                # Keep the full native grid for common spatial scale; do not resize or restretch.
                for k, image in ((0, x[p]), (1, x[p]), (2, changed[p])):
                    axes[p, k].imshow(image, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
                overlay = np.zeros((160, 160, 4))
                overlay[masks[j, p, r]] = (.05, .7, .55, .4)
                axes[p, 1].imshow(overlay, interpolation='nearest')
                axes[p, 3].imshow(changed[p] - x[p], cmap='coolwarm', vmin=-scale, vmax=scale, interpolation='nearest')
                for k, ref in enumerate(refs, 4):
                    if ref is not None:
                        axes[p, k].imshow(pixels[ref, p], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                        axes[p, k].set_title(('Same-class' if k == 4 else 'Original rival') + ' ref\n' + rows[ref]['id'].split('_')[-1], fontsize=9)
                for k, title in enumerate((('VV' if p == 0 else 'VH') + ' original', region + ' mask', 'Joint-operation input', f'Difference +/-{scale:.3f}')):
                    axes[p, k].set_title(title, fontsize=9)
            madd = m[1, j] + m[2, j] - m[0, j]
            fig.suptitle(f"{item['id']} | {NAMES[y[j]]} | seed17 | {region} | {state}\n"
                         f"m00={m[0,j]:+.3f}, m10={m[1,j]:+.3f}, m01={m[2,j]:+.3f}, additive={madd:+.3f}, m11={m[3,j]:+.3f}, I={interaction[j]:+.3f}", fontsize=10)
            fig.text(.5, .02, 'Native 160 grid; fixed display scale. Reference examples are legal bank members, not ranked causal influence.', ha='center', fontsize=8)
            fig.subplots_adjust(left=.02, right=.99, bottom=.07, top=.83, hspace=.38, wspace=.12)
            stem = f'{region}_{state}_{idx}'
            fig.savefig(folder / (stem + '.png'), dpi=150)
            fig.savefig(folder / (stem + '.svg'))
            plt.close(fig)
            cases.append(dict(id=item['id'], index=idx, state=state, region=region, seed=17,
                true_class=NAMES[y[j]], original_prediction=NAMES[p0[j]], joint_prediction=NAMES[p1[j]],
                vv_prediction=NAMES[z[1,j].argmax()], vh_prediction=NAMES[z[2,j].argmax()],
                additive_prediction=NAMES[add[j].argmax()], fixed_rival=NAMES[rival[j]],
                margins=m[:,j].tolist(), additive_margin=float(madd), interaction=float(interaction[j]),
                reference_ids=[rows[i]['id'] if i is not None else None for i in refs],
                source_paths=item.get('source_paths'), image=stem + '.png', svg=stem + '.svg'))
    (folder / 'cases.json').write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding='utf-8')
    body = ['<!doctype html><meta charset="utf-8"><title>VV/VH factorial evidence</title>',
            '<style>body{font:16px system-ui;max-width:1450px;margin:24px auto}img{width:100%}section{border-top:1px solid #aaa;padding:20px 0}pre{white-space:pre-wrap}</style>',
            '<h1>VV/VH joint intervention: paired evidence</h1><p>Existing OSS2 validation, frozen weights. No new model training. All measured interactions are numerical roundoff: the current classifier is additive across polarizations. Cases are selected by median absolute joint margin change within outcome groups, not by roundoff magnitude.</p>',
            '<img src="../interaction_matrix.png"><p><a href="../interaction_matrix.svg">Editable matrix</a></p>']
    for case in cases:
        body += [f'<section><h2>{html.escape(case["id"])} / {case["region"]} / {case["state"]}</h2>',
                 f'<img src="{case["image"]}"><a href="{case["svg"]}">Editable SVG</a>',
                 '<pre>' + html.escape(json.dumps({k:v for k,v in case.items() if k not in ('source_paths', 'image', 'svg')}, ensure_ascii=False, indent=2)) + '</pre></section>']
    (folder / 'index.html').write_text('\n'.join(body), encoding='utf-8')


if __name__ == '__main__':
    main()
