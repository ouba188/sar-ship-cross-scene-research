"""Four frozen attention cases: response IG and matched response intervention."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '2')
import sys
sys.dont_write_bytecode = True
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torchvision.models import resnet50
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ATT = HERE.parent
sys.path.insert(0, str(ATT))
import common
from extract_attention import attention
sys.path.insert(0, str(ATT.parent / 'fixed_model_spatial_attribution'))
import attribute_fixed as old
run, parent = common.run, common.parent
SCORE, BASE = common.SCORE, common.BASE
IDS = ['pair_0005030', 'pair_0007310', 'pair_0003494', 'pair_0003504']


class AttentionReadout(old.FixedReadout):
    def __init__(self, ck, head, base, means, valid, y, rival):
        super().__init__(ck, head, base, means, valid, y, rival)
        self.tau = torch.as_tensor(ck['tau'], dtype=torch.float64)

    def statistics(self, r, weight):
        pols = []
        for p in range(2):
            if not bool(self.valid[p]):
                pols.append(r[:, p].sum((-2, -1))[..., None].expand(-1, -1, 2)*0)
                continue
            a, _, _ = attention(r[:, p].flatten(-2), weight[:, p].flatten(-2), self.tau[p])
            pols.append(a)
        return torch.stack(pols, 1).mean(0)

    def margin(self, r, weight):
        s = self.score(self.statistics(r, weight))
        return s[self.y]-s[self.rival]


def intervention(readout, r, weight, ig, identifier):
    rows, support, arrays = [], [], {}
    spatial_ig = ig.sum(2).abs().numpy()
    original = float(readout.margin(r, weight).detach())
    for p, pol in enumerate(['VV', 'VH']):
        if not bool(readout.valid[p]):
            continue
        masks = {kind: np.zeros((4, *r.shape[-2:]), bool) for kind in ('high', 'low')}
        for t in range(4):
            w = weight[t, p, 0].numpy().ravel()
            ids = np.flatnonzero(w > 0)
            rank = spatial_ig[t, p].ravel()
            n = max(1, int(np.ceil(.2*len(ids))))
            high = ids[np.lexsort((ids, -rank[ids]))[:n]]
            low = []
            for area in np.unique(w[high]):
                need = int((w[high] == area).sum())
                pool = ids[(w[ids] == area) & ~np.isin(ids, high)]
                low.extend(pool[np.lexsort((pool, rank[pool]))[:need]].tolist())
            if len(low) < n:
                pool = ids[~np.isin(ids, np.r_[high, low])]
                low.extend(pool[np.lexsort((pool, rank[pool]))[:n-len(low)]].tolist())
            low = np.asarray(low, int)
            assert len(low) == n and not np.intersect1d(high, low).size
            for kind, chosen in [('high', high), ('low', low)]:
                masks[kind][t].flat[chosen] = True
            support.append(dict(instance_id=identifier, pol=pol, rotation=t,
                valid_cells=len(ids), selected_cells=n, fraction=n/len(ids),
                high_mass=float(w[high].sum()), low_mass=float(w[low].sum()),
                mass_difference=float(w[high].sum()-w[low].sum()),
                weight_multiset_equal=bool(np.array_equal(np.sort(w[high]), np.sort(w[low]))),
                high_indices=json.dumps(high.tolist()), low_indices=json.dumps(low.tolist()),
                high_abs_ig=float(rank[high].sum()), low_abs_ig=float(rank[low].sum())))
        for kind in ('high', 'low'):
            changed = r.clone()
            mask = torch.tensor(masks[kind])[:, None]
            selected = changed[:, p].masked_fill(mask, 0)
            spatial_mean = (selected*weight[:, p]).sum((-2, -1), keepdim=True)
            # Projection changes all supported cells, not only the zeroed cells.
            changed[:, p] = torch.where(weight[:, p] > 0, selected-spatial_mean, selected)
            with torch.no_grad():
                score = readout.score(readout.statistics(changed, weight)).numpy()
            margin = float(score[readout.y]-score[readout.rival])
            rows.append(dict(instance_id=identifier, pol=pol, intervention=kind,
                original_margin=original, intervened_margin=margin, change=margin-original,
                prediction=int(score.argmax()), true_class=readout.y, fixed_rival=readout.rival,
                weighted_center_error=float((changed*weight).sum((-2, -1)).abs().max()),
                class_center_error=float(changed.mean(2).abs().max()),
                weighted_response_l2=float(((changed-r).square()*weight).sum().sqrt()),
                recentered=True))
            arrays[f'{pol}_{kind}_selected'] = masks[kind]
            arrays[f'{pol}_{kind}_r'] = changed.numpy()
            arrays[f'{pol}_{kind}_score'] = score
    return rows, support, arrays


def panel(row, q, data, mass, covered, scores, info, interventions):
    fig, axes = plt.subplots(2, 4, figsize=(19, 10), layout='constrained')
    limit = max(float(abs(mass).max()), 1e-10)
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#c8c8c8')
    for p, pol in enumerate(['VV', 'VH']):
        image, place = data['raw'][p], data['placements'][p]
        h, w = image.shape
        for ax in axes[p, :2]:
            ax.imshow(image, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            ax.set_xlabel('native pixel x'); ax.set_ylabel('native pixel y')
        axes[p, 0].set_title(f'{pol} original {w}x{h}; valid={bool(info[f"valid_{pol.lower()}"])}')
        gh, gw = mass[p].shape
        ex = np.linspace(0, 160, gw+1)-.5+place['crop_x']-place['pad_left']
        ey = np.linspace(0, 160, gh+1)-.5+place['crop_y']-place['pad_top']
        grid = np.ma.masked_where(~covered[p], mass[p])
        axes[p, 1].pcolormesh(ex, ey, grid, cmap=cmap, vmin=-limit, vmax=limit, alpha=.65)
        axes[p, 1].set_xlim(-.5, w-.5); axes[p, 1].set_ylim(h-.5, -.5)
        axes[p, 1].set_title(f'{pol} response IG, inverse C4\ncoarse bins, not pixel causes')
        im = axes[p, 2].imshow(grid, cmap=cmap, vmin=-limit, vmax=limit, interpolation='nearest')
        axes[p, 2].set_title(f'{pol} IG sum={mass[p].sum():+.6f}\ngray=zero pool weight; RF267')
        fig.colorbar(im, ax=axes[p, 2], shrink=.6)
    for name, score in scores.items():
        axes[0, 3].plot(range(8), score, marker='o', lw=1, label=name)
    axes[0, 3].set_xticks(range(8)); axes[0, 3].legend(fontsize=8)
    axes[0, 3].set_title('Actual frozen attention scores\nH0: raw h/b=0, not zero residual')
    labels, vals = ['r=0', 'real'], [info['zero_response_margin'], info['full_margin']]
    for x in interventions:
        labels.append(x['pol']+' '+x['intervention'])
        vals.append(x['intervened_margin'])
    axes[1, 3].bar(labels, vals, color=['#888888', '#248568', '#b14444', '#c6a36b', '#b14444', '#c6a36b'])
    axes[1, 3].axhline(0, color='black', lw=.7)
    axes[1, 3].tick_params(axis='x', labelrotation=45)
    axes[1, 3].set_title('Fixed-rival margin; high/low20%\nzero then recenter responses, not pixels')
    fig.suptitle(f'Two-sided attention266 / {row.instance_id} / {q.outcome} / seed17 fold{int(q.fold)} epoch{int(q.epoch)}\n'
        f'{row.class_name} / {row.port}; true={int(q.class_id)}, attention final rival={int(q.final_rival)}; '
        f'winner B/attention={int(q.base_prediction)}/{int(q.prediction)}\n'
        f'B / mu / valid / tau fixed; M1-M0={info["target_change"]:+.6f}; '
        f'IG={info["ig_sum"]:+.6f}; closure={info["closure_error"]:+.2e}; {info["steps"]} midpoints', fontsize=11)
    path = HERE/f'{q.outcome}.png'
    fig.savefig(path, dpi=110); plt.close(fig)
    return str(path)


def main():
    start = time.perf_counter()
    assert (HERE/'PROTOCOL_ZH.md').exists()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = resnet50(weights=None)
    backbone.load_state_dict(torch.load(old.extractor.WEIGHTS, map_location='cpu', weights_only=True))
    backbone = backbone.to(device).float().eval().requires_grad_(False)
    rf, stride = old.receptive_field(backbone)
    table, cache = parent.load_table(), common.cache_load()
    ac = dict(np.load(ATT/'attention_test.npz'))
    for k in ['instance_ids', 'reader_names', 'valid']:
        np.testing.assert_array_equal(ac[k], cache[k])
    np.testing.assert_array_equal(cache['instance_ids'], table.instance_id)
    lookup = table.reset_index().set_index('instance_id', drop=False)
    cases = pd.read_csv(ATT/'test_review_cases.csv').set_index('instance_id')
    assert list(cases.index) == IDS
    preds = pd.read_csv(ATT/'test_predictions.csv', float_precision='round_trip')
    pred = preds[preds.instance_id.isin(IDS) & preds.seed.eq(17) & preds.arm.eq(common.ARM)].set_index('instance_id')
    assert len(pred) == 4 and pred.index.is_unique
    snapshots = {s['obs_id']: s for s in json.loads((run.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    names, valid = list(cache['reader_names']), cache['valid'].astype(bool)
    closures, all_interventions, supports, convergences, reviews, class_rows = [], [], [], [], [], []
    (HERE/'arrays').mkdir(exist_ok=True)
    for identifier in IDS:
        row, q = lookup.loc[identifier], pred.loc[identifier]
        for key in ['fold', 'epoch', 'outcome', 'final_rival', 'prediction']:
            assert q[key] == cases.loc[identifier, key]
        idx, fold, y, rival = int(row['index']), int(q.fold), int(q.class_id), int(q.final_rival)
        with np.load(parent.PARENT/'baseline/models'/f'fold{fold}_outer.npz') as z:
            W = z['W']
            assert idx not in z['fit']
        ck, head = common.restore(ATT/'models'/f'fold{fold}_seed17.pt')
        assert ck['epoch'] == q.epoch
        np.testing.assert_array_equal(ck['tau'], ac['tau'][fold])
        ri = names.index(f'fold{fold}_outer')
        assert ac['computed'][ri, idx]
        data = old.extract_case(backbone, row, snapshots[int(row.obs_id)], valid[idx], W, device)
        r, weight = torch.tensor(data['r']), torch.tensor(data['weight'])
        means, expected = cache['mean_pol'][ri, idx], ac['attention_pol'][ri, idx]
        mean_error = float(abs(data['mean'].mean(0)-means).max())
        assert mean_error < 1e-5
        base = q[BASE].to_numpy(float)
        readout = AttentionReadout(ck, head, base, means, valid[idx], y, rival)
        a1, a0 = readout.statistics(r, weight), readout.statistics(torch.zeros_like(r), weight)
        assert not a0.any()
        att_error = float(abs(a1.numpy()-expected).max())
        assert att_error < 1e-5
        xn, _, _ = run.inputs(base[None], means[None], expected[None], valid[idx][None], ck['mu'], ck['scale'])
        input_error = float((readout.inputs(torch.tensor(expected))-xn).abs().max())
        assert input_error < 1e-6
        with torch.no_grad():
            s1, s0 = readout.score(a1).numpy(), readout.score(a0).numpy()
            scache = readout.score(torch.tensor(expected)).numpy()
            dm, h1 = [v.numpy() for v in readout.branches(a1)]
            dm0, h0 = [v.numpy() for v in readout.branches(a0)]
        assert np.array_equal(dm, dm0)
        saved = q[SCORE].to_numpy(float)
        cache_error, fresh_error = float(abs(scache-saved).max()), float(abs(s1-saved).max())
        assert cache_error < 1e-6 and fresh_error < 2e-5
        assert s1.argmax() == q.prediction
        assert np.argmax(np.where(np.arange(8) == y, -np.inf, s1)) == rival
        target = float((s1[y]-s1[rival])-(s0[y]-s0[rival]))
        tolerance = 1e-5+1e-4*abs(target)
        for steps in [64, 128]:
            ig = old.integrate(readout, r, weight, steps)
            total, error = float(ig.sum()), float(ig.sum())-target
            convergences.append(dict(instance_id=identifier, steps=steps, target_change=target,
                ig_sum=total, closure_error=error, tolerance=tolerance, passed=abs(error)<=tolerance))
            if abs(error) <= tolerance:
                break
        ig_array = ig.numpy()
        assert not ig_array[np.broadcast_to(data['weight']==0, ig_array.shape)].any()
        mass, covered, aligned = old.aligned_map(ig_array, data['weight'])
        if ck['epoch'] == 0:
            assert total == 0 and not ig_array.any()
        info = dict(instance_id=identifier, outcome=q.outcome, fold=fold, seed=17, epoch=int(q.epoch),
            true_class=y, fixed_rival=rival, steps=steps, target_change=target, ig_sum=total,
            closure_error=error, tolerance=tolerance, passed=abs(error)<=tolerance,
            VV_IG=float(mass[0].sum()), VH_IG=float(mass[1].sum()),
            tau_vv=float(ck['tau'][0]), tau_vh=float(ck['tau'][1]),
            cache_mean_error=mean_error, cache_attention_error=att_error,
            cache_score_error=cache_error, fresh_score_error=fresh_error, descriptor_error=input_error,
            B_margin=float(base[y]-base[rival]), mean_margin=float(dm[y]-dm[rival]),
            H0_margin=float(h0[y]-h0[rival]), Hreal_margin=float(h1[y]-h1[rival]),
            zero_response_margin=float(s0[y]-s0[rival]), full_margin=float(s1[y]-s1[rival]),
            branch_closure_error=float((base+dm+h1-s1).max()),
            valid_vv=bool(valid[idx, 0]), valid_vh=bool(valid[idx, 1]),
            grid_h=r.shape[-2], grid_w=r.shape[-1], receptive_field=rf, output_stride=stride)
        interventions, support, intervention_arrays = intervention(readout, r, weight, ig, identifier)
        assert max(x['weighted_center_error'] for x in interventions) < 1e-10
        assert max(x['class_center_error'] for x in interventions) < 1e-10
        all_interventions.extend(interventions); supports.extend(support)
        for t in range(4):
            for p, pol in enumerate(['VV', 'VH']):
                for c in range(8):
                    class_rows.append(dict(instance_id=identifier, rotation=t, pol=pol,
                        response_class=c, ig_sum=float(ig_array[t, p, c].sum())))
        path = panel(row, q, data, mass, covered,
            {'B': base, 'B+mean': base+dm, 'B+mean+H0': s0, 'attention': s1}, info, interventions)
        reviews.append(dict(**info, panel=path, image_vv=row.image_vv, image_vh=row.image_vh))
        closures.append(info)
        np.savez_compressed(HERE/'arrays'/f'{q.outcome}.npz', r=data['r'], weight=data['weight'],
            ig=ig_array, aligned_ig=aligned, aligned_mass=mass, covered=covered,
            h_b=a1.numpy(), cached_h_b=expected, fixed_mean=means, tau=ck['tau'],
            base=base, full_score=s1, zero_response_score=s0, mean_delta=dm,
            attention0_delta=h0, attentionreal_delta=h1, **intervention_arrays)
        print(identifier, 'epoch', ck['epoch'], 'rival', rival, 'change', target,
            'IG', total, 'error', error, 'steps', steps, 'score_replay', fresh_error, flush=True)
    for name, rows in [('closure', closures), ('interventions', all_interventions),
        ('intervention_support', supports), ('ig_convergence', convergences),
        ('review_cases', reviews), ('class_attribution', class_rows)]:
        pd.DataFrame(rows).to_csv(HERE/f'{name}.csv', index=False)
    summary = dict(cases=4, new_fits=0, seconds=time.perf_counter()-start,
        all_completeness_passed=all(x['passed'] for x in closures),
        max_steps=max(x['steps'] for x in closures),
        max_closure_error=max(abs(x['closure_error']) for x in closures),
        max_score_replay_error=max(x['fresh_score_error'] for x in closures),
        max_attention_replay_error=max(x['cache_attention_error'] for x in closures),
        matched_weight_multisets=all(x['weight_multiset_equal'] for x in supports),
        max_intervention_area_difference=max(abs(x['mass_difference']) for x in supports),
        interventions=len(all_interventions), theoretical_RF=rf, physical_intervention=False)
    (HERE/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
