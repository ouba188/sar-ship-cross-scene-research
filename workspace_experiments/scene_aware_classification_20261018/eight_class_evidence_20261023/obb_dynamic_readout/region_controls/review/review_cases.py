"""Bounded selected-head case replay; no fitting or backbone extraction."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '2')
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

HERE = Path(__file__).resolve().parent
CONTROL = HERE.parent
N = CONTROL.parent
sys.path.insert(0, str(CONTROL))
import run_controls as controls
old = controls.old
c = sys.modules['common']
sys.path.insert(0, str(N / 'data_review'))
from review_selected import one_info
SCORE, BASE = old.SCORE, old.BASE


def select_cases(pred):
    full = pred[pred.arm.eq('dynamic') & pred.seed.eq(17)].copy()
    keys = ['fold', 'instance_id']
    records, addenda = [], []
    for arm in ('body', 'shuffled'):
        other = pred[pred.arm.eq(arm) & pred.seed.eq(17)].set_index(keys)
        assert not other.index.duplicated().any() and not full.duplicated(keys).any()
        paired = other.loc[pd.MultiIndex.from_frame(full[keys])]
        np.testing.assert_array_equal(full.class_id, paired.class_id)
        d = full.copy()
        d['other_prediction'] = paired.prediction.to_numpy()
        d['other_epoch'] = paired.epoch.to_numpy()
        dc, oc = d.prediction.eq(d.class_id), d.other_prediction.eq(d.class_id)
        groups = [('rescued', dc & ~oc), ('harmed', ~dc & oc)]
        if arm == 'body':
            groups += [('CC', dc & oc), ('WW', ~dc & ~oc)]
        for outcome, mask in groups:
            group = d[mask].sort_values(['obs_id', 'instance_id', 'fold'])
            active = group[group.epoch.gt(0) | group.other_epoch.gt(0)]
            result = dict(comparison=arm, outcome=outcome, phase='original', group_n=len(group), active_n=len(active), available=bool(len(group)))
            if len(group):
                chosen = (active if arm == 'body' and len(active) else group).iloc[0]
                result.update(obs_id=int(chosen.obs_id), instance_id=chosen.instance_id, fold=int(chosen.fold),
                              dynamic_epoch=int(chosen.epoch), other_epoch=int(chosen.other_epoch),
                              allgroup_first_obs=int(group.iloc[0].obs_id),
                              fallback=not bool(chosen.epoch > 0 or chosen.other_epoch > 0),
                              selection='active OR first obs_id' if arm == 'body' and len(active) else 'allgroup first obs_id')
            records.append(result)
            if arm == 'body' and outcome in ('rescued', 'harmed'):
                both = group[group.epoch.gt(0) & group.other_epoch.gt(0)]
                extra = dict(comparison=arm, outcome=outcome, phase='both_active_addendum',
                             group_n=len(group), active_n=len(both), available=bool(len(both)))
                if len(both):
                    chosen = both.iloc[0]
                    extra.update(obs_id=int(chosen.obs_id), instance_id=chosen.instance_id, fold=int(chosen.fold),
                                 dynamic_epoch=int(chosen.epoch), other_epoch=int(chosen.other_epoch),
                                 allgroup_first_obs=int(group.iloc[0].obs_id), fallback=False,
                                 selection='authorized addendum: BOTH active first obs_id')
                addenda.append(extra)
    return records + addenda


def independent_raw(base, info, arm, obs):
    summary = info['summary'][:, :, 1:4].copy()
    if arm == 'shuffled':
        permutation = np.random.default_rng([4711, int(obs)]).permutation(3)
        summary = summary[:, :, permutation]
    elif arm in ('body', 'body_end', 'body_side'):
        keep = {'body': [0], 'body_end': [0, 1], 'body_side': [0, 2]}[arm]
        for r in set(range(3)) - set(keep):
            summary[:, :, r] = 0
    cols = [base - base.mean(1, keepdims=True)]
    for p in range(2):
        for r in range(3):
            for stat in range(3):
                v = summary[:, p, r, :, stat]
                cols += [v, v - v.mean(1, keepdims=True)]
    quality = old.common_quality(info)
    return np.concatenate((np.stack(cols, -1), np.broadcast_to(quality[:, None], (1, 8, 32))), -1)


def render(row, cases, scores, epochs, raw_inputs, info, table, snapshots, valid):
    arms = list(scores)
    y = int(row.class_id)
    primary = cases[0]
    rival = primary['fixed_rival']
    fig, axes = plt.subplots(2, 4, figsize=(21, 10), layout='constrained')
    colors = ['#38db68', '#efae24', '#28c4f0']
    palette = {'B': '#5c5c5c', 'dynamic': '#167e65', 'body': '#ba7327', 'shuffled': '#934c94'}
    for p, pol in enumerate(c.POL):
        canvas, fp, g, place = c.native(table, snapshots, int(row.name), p)
        with Image.open(row['image_' + pol]) as image:
            raw = np.asarray(image.convert('L'))
        axes[p, 0].imshow(raw, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        if g is not None:
            u = np.array([np.cos(g[4]), np.sin(g[4])]) * g[2] / 2
            v = np.array([-np.sin(g[4]), np.cos(g[4])]) * g[3] / 2
            center = g[:2] - [place['pad_left'] - place['crop_x'], place['pad_top'] - place['crop_y']]
            corners = np.array([center-u-v, center+u-v, center+u+v, center-u+v])
            axes[p, 0].add_patch(Polygon(corners, fill=False, edgecolor=colors[0], lw=1))
        axes[p, 0].set_title(f'{pol.upper()} original {raw.shape[1]}x{raw.shape[0]}; valid={bool(valid[int(row.name), p])}')
        axes[p, 1].imshow(canvas, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        mask = c.soft_regions(g) * fp
        for r in range(3):
            rgba = np.zeros((160, 160, 4))
            rgba[..., :3] = matplotlib.colors.to_rgb(colors[r])
            rgba[..., 3] = .5 * mask[r]
            axes[p, 1].imshow(rgba, interpolation='nearest')
        flags = info['region_valid'][0, p, 1:4].astype(int).tolist()
        axes[p, 1].set_title(f'{pol.upper()} native160 actual regions\nbody/end/side valid={flags}; RF35')
        labels = [f'{r}:{s}' for r in ('body', 'end', 'side') for s in ('mu', 'h', 'b')]
        x = np.arange(9)
        count = len(raw_inputs)
        for j, (arm, raw) in enumerate(raw_inputs.items()):
            values = [raw[0, y, 1+p*18+k*2] - raw[0, rival, 1+p*18+k*2] for k in range(9)]
            axes[p, 2].bar(x + (j-(count-1)/2)*.8/count, values, width=.8/count, color=palette[arm], label=arm)
        axes[p, 2].axhline(0, color='black', lw=.6)
        axes[p, 2].set_xticks(x, labels, rotation=75, fontsize=7)
        axes[p, 2].legend(fontsize=7)
        axes[p, 2].set_title(f'{pol.upper()} actual input slot: class {y} minus {rival}\nraw units before scaler, NOT score attribution', fontsize=9)
        axes[p, 0].set_xlabel('original pixel coordinates')
        axes[p, 1].set_xlabel('crop/pad coordinates; green body / amber end / cyan side')
    for arm, s in scores.items():
        axes[0, 3].plot(range(8), s, marker='o', lw=1, label=arm, color=palette[arm])
    axes[0, 3].set_xticks(range(8)); axes[0, 3].legend(fontsize=8)
    axes[0, 3].set_title(f'Actual selected scores; true={y}')
    lines = ['Fixed-rival margins and own top1 margins:']
    for case in cases:
        lines += [f'{case["comparison"]}/{case["outcome"]}: rival={case["fixed_rival"]}']
        for arm in ('B', 'dynamic', case['comparison']):
            s = scores[arm]
            own = np.max(np.where(np.arange(8) == y, -np.inf, s))
            lines += [f'  {arm}: pred{s.argmax()}, fixed={s[y]-s[case["fixed_rival"]]:+.4f}, top1={s[y]-own:+.4f}']
    lines += ['', 'Selected epochs: ' + ', '.join(f'{a}={e}' for a, e in epochs.items()),
              'Shuffled slot source order: ' + str(np.random.default_rng([4711, int(row.obs_id)]).permutation(3).tolist()),
              'Quality remains truly ordered. Epoch0 means B fallback.',
              'Slot bars are model inputs, not physical causes.']
    axes[1, 3].axis('off')
    axes[1, 3].text(0, 1, '\n'.join(lines), transform=axes[1, 3].transAxes, va='top', fontsize=9)
    labels = '; '.join(f'full vs {v["comparison"]}: {v["outcome"]} ({v["phase"]})' for v in cases)
    fig.suptitle(f'TEST seed17 / obs{row.obs_id} / {row.instance_id} / fold{primary["fold"]}\n'
                 f'{row.class_name} / {row.port} / {labels}\nFixed first-ID selection; no retraining or region intervention', fontsize=11)
    path = HERE / f'case_{int(row.obs_id)}.png'
    fig.savefig(path, dpi=115); plt.close(fig)
    return str(path)


def main():
    tick = time.perf_counter(); torch.set_num_threads(2)
    assert (CONTROL / 'test_complete.json').exists()
    pred = pd.read_csv(CONTROL / 'test_predictions.csv', dtype={'mmsi': str}, float_precision='round_trip')
    selected = select_cases(pred)
    pd.DataFrame(selected).to_csv(HERE / 'case_selection.csv', index=False)
    table, valid, snapshots = c.load_data()
    indexed = pred[pred.seed.eq(17)].set_index(['arm', 'fold', 'instance_id'])
    todo = pd.DataFrame([r for r in selected if r['available']])
    cases_out, checks, slots, infos, heads = [], [], [], {}, {}
    for (fold, identifier), selected_group in todo.groupby(['fold', 'instance_id'], sort=False):
        cases = selected_group.to_dict('records')
        q = indexed.loc[('dynamic', fold, identifier)]
        idx = int(q.row_index); row = table.iloc[idx]; y = int(row.class_id)
        assert row.instance_id == identifier and int(row.obs_id) == int(q.obs_id)
        if fold not in infos:
            infos[fold] = old.load_info(int(fold), 'test', table)
        loc = int(np.flatnonzero(infos[fold]['indices'] == idx)[0])
        info = one_info(infos[fold], loc)
        base = q[BASE].to_numpy(float)[None]
        scores = {'B': base[0]}; epochs = {}; raw_by_arm = {}
        arms = ['dynamic'] + list(dict.fromkeys(case['comparison'] for case in cases))
        for arm in arms:
            key = (fold, arm)
            if key not in heads:
                directory = N if arm == 'dynamic' else CONTROL
                ck = torch.load(directory / 'models' / f'fold{fold}_{arm}_seed17.pt', map_location='cpu', weights_only=False)
                model = old.Head(69).eval().requires_grad_(False)
                model.load_state_dict(ck['state']); heads[key] = ck, model
            ck, model = heads[key]
            raw = controls.raw_inputs(base, info, arm, [int(row.obs_id)])
            np.testing.assert_array_equal(raw, independent_raw(base, info, arm, int(row.obs_id)))
            x, _, _ = old.standardize(raw, ck['mu'], ck['scale'])
            with torch.inference_mode():
                score = base[0] + model(x).numpy()[0].astype(float)
            saved = indexed.loc[(arm, fold, identifier)]
            np.testing.assert_array_equal(base[0], saved[BASE].to_numpy(float))
            error = float(abs(score - saved[SCORE].to_numpy(float)).max())
            assert error < 1e-6 and ck['epoch'] == saved.epoch
            if ck['epoch'] == 0:
                np.testing.assert_array_equal(score, base[0])
            checks.append(dict(instance_id=identifier, obs_id=int(row.obs_id), fold=fold, arm=arm,
                               epoch=ck['epoch'], score_replay_error=error, independent_input_exact=True))
            scores[arm] = score; epochs[arm] = ck['epoch']; raw_by_arm[arm] = raw
            for p, pol in enumerate(c.POL):
                for r, region in enumerate(c.REGIONS):
                    for cls in range(8):
                        start = 1+p*18+r*6
                        slots.append(dict(instance_id=identifier, obs_id=int(row.obs_id), fold=fold, arm=arm,
                                          pol=pol, slot_region=region, response_class=cls,
                                          mu=raw[0, cls, start], h=raw[0, cls, start+2], b=raw[0, cls, start+4],
                                          actual_region_valid=bool(info['region_valid'][0, p, 1+r]),
                                          slot_source_region=(c.REGIONS[np.random.default_rng([4711, int(row.obs_id)]).permutation(3)[r]] if arm=='shuffled' else region)))
        for case in cases:
            other = case['comparison']
            dp, op = int(scores['dynamic'].argmax()), int(scores[other].argmax())
            rival = op if case['outcome'] == 'rescued' else dp
            if case['outcome'] == 'CC':
                rival = int(np.argmax(np.where(np.arange(8) == y, -np.inf, scores['dynamic'])))
            assert rival != y
            case.update(true_class=y, class_name=row.class_name, port=row.port, fixed_rival=rival,
                        base_prediction=int(base.argmax()), dynamic_prediction=dp, other_prediction=op,
                        image_vv=row.image_vv, image_vh=row.image_vh)
            for name, score in [('base', base[0]), ('dynamic', scores['dynamic']), ('other', scores[other])]:
                top = int(np.argmax(np.where(np.arange(8) == y, -np.inf, score)))
                case[name+'_fixed_margin'] = float(score[y] - score[rival])
                case[name+'_top1_margin'] = float(score[y] - score[top])
                case[name+'_top_rival'] = top
        panel = render(row, cases, scores, epochs, raw_by_arm, info, table, snapshots, valid)
        for case in cases:
            case['panel'] = panel; cases_out.append(case)
        print(int(row.obs_id), [(v['comparison'], v['outcome']) for v in cases], epochs, flush=True)
    for name, rows in [('cases', cases_out), ('replay_checks', checks), ('case_slots', slots)]:
        pd.DataFrame(rows).to_csv(HERE / f'{name}.csv', index=False)
    receipt = dict(comparisons=len(cases_out), unique_panels=len(set(v['panel'] for v in cases_out)), new_fits=0,
                   new_backbone_extractions=0, seconds=time.perf_counter()-tick,
                   max_score_replay_error=max(v['score_replay_error'] for v in checks),
                   actual_image_view_complete=False)
    (HERE / 'case_summary.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
