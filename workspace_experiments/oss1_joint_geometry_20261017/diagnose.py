"""Read-only outcome decomposition and auditable native-chip case panels."""
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from common import HERE, FULL, MANIFEST, folds, public
from local_reader import LocalReader


def main():
    torch.set_num_threads(2)
    manifest = pd.read_csv(MANIFEST)
    det = pd.read_csv(HERE / 'detections.csv').set_index('global_row')
    geo = pd.read_csv(HERE / 'geometry.csv').set_index('global_row')
    coverage = pd.read_csv(HERE / 'region_coverage.csv')
    frames, support_rows, attention_rows, source_rows = [], [], [], []
    for fold, cache, episode, prefix in folds():
        p = pd.read_csv(HERE / fold / 'predictions.csv')
        frames.append(p[p.policy.eq('source_selected')].copy())
        data = public.load_fold('oss1', fold, cache, episode, prefix, manifest,
                                np.load(FULL / 'oss1_c4.npy'), np.load(FULL / 'oss1_zero.npy'))
        for (origin, draw), s in data['support'].items():
            counts = np.bincount(data['labels'][s], minlength=8)
            for row in data['rows'][data['query']]:
                label = int(manifest.iloc[row].class_id)
                support_rows.append(dict(fold=fold, origin=origin, draw=draw, global_row=int(row),
                                         same_class_support=int(counts[label])))
        with np.load(HERE / fold / 'local_features.npz') as f:
            qrows = data['rows'][data['query']]
            body = torch.tensor(f['body'][qrows])
            tokens = torch.tensor(f['joint_tokens'][qrows])
            positions = torch.tensor(f['joint_positions'][qrows])
        for seed in (0, 1, 2):
            reader = LocalReader().eval()
            state = torch.load(HERE / fold / f'joint_{seed}' / 'reader.pt',
                               map_location='cpu', weights_only=False)
            reader.load_state_dict(state['selected'])
            with torch.no_grad():
                _, att = reader(body, tokens, positions)
            for row, weights in zip(qrows, att.numpy()):
                attention_rows.append(dict(fold=fold, seed=seed, global_row=int(row),
                    top_token=int(weights.argmax()), entropy=float(-(weights * np.log(weights.clip(1e-12))).sum()),
                    **{f'w{k}': float(v) for k, v in enumerate(weights)}))
        for arm in ('body_only', 'image_ring', 'ship', 'ship_radar_encoding', 'joint'):
            for seed in (0, 1, 2):
                history = json.loads((HERE / fold / f'{arm}_{seed}' / 'training.json').read_text())
                best = min(history, key=lambda x: x['selection_brier'])
                source_rows.append(dict(fold=fold, arm=arm, seed=seed,
                    old_source_brier=history[0]['alpha_risks']['0.0'],
                    chosen_source_brier=best['selection_brier'],
                    delta_source_brier=best['selection_brier']-history[0]['alpha_risks']['0.0']))
    pred = pd.concat(frames, ignore_index=True).merge(pd.DataFrame(support_rows),
        on=['fold', 'origin', 'draw', 'global_row'], validate='many_to_one')
    pred['net_base'] = pred.prediction.eq(pred.truth).astype(float) - pred.baseline.eq(pred.truth).astype(float)
    pred['class_name'] = manifest.class_name.to_numpy()[pred.global_row]
    pred['support_group'] = pd.cut(pred.same_class_support, [-1, 0, 1, 2, 16], labels=['0', '1', '2', '3+'])
    pred['axis_valid'] = det.loc[pred.global_row, 'axis_valid'].to_numpy()
    angle = np.degrees(det.loc[pred.global_row, 'ship_axis'].to_numpy()) % 180
    angle = np.minimum(angle, 180-angle)
    pred['axis_angle_bin'] = pd.cut(angle, [-.001, 15, 30, 45, 60, 75, 90],
                                  labels=['0-15', '15-30', '30-45', '45-60', '60-75', '75-90'])
    pred.to_csv(HERE / 'CASE_DIAGNOSTICS.csv', index=False)
    class_rows = []
    for (arm, origin, fold, label), z in pred.groupby(['arm', 'origin', 'fold', 'class_name']):
        class_rows.append(dict(arm=arm, origin=origin, fold=fold, class_name=label,
            query_ships=z.global_row.nunique(), paired_rescues=int((z.net_base>0).sum()),
            paired_harms=int((z.net_base<0).sum()),
            baseline_recall=100*z.baseline.eq(z.truth).mean(),
            recall=100*z.prediction.eq(z.truth).mean(), delta_recall_pp=100*z.net_base.mean()))
    classes = pd.DataFrame(class_rows)
    classes.to_csv(HERE / 'BASE_CLASS_EFFECTS.csv', index=False)
    subgroups = []
    for key in ('support_group', 'axis_angle_bin', 'axis_valid'):
        for (arm, origin, fold, value), z in pred.groupby(['arm', 'origin', 'fold', key], observed=True):
            subgroups.append(dict(arm=arm, origin=origin, fold=fold, group_key=key, group=str(value),
                query_ships=z.global_row.nunique(), paired_occurrences=len(z),
                delta_oa_pp=100*z.net_base.mean(),
                delta_true_probability=(z.true_probability-z.baseline_true_probability).mean()))
    pd.DataFrame(subgroups).to_csv(HERE / 'SUBGROUP_EFFECTS.csv', index=False)
    pd.DataFrame(source_rows).to_csv(HERE / 'SOURCE_SELECTION_GAINS.csv', index=False)
    attention = pd.DataFrame(attention_rows)
    attention.to_csv(HERE / 'JOINT_QUERY_ATTENTION.csv', index=False)
    joint = pred[(pred.arm=='joint') & (pred.origin=='cross')]
    cases = joint.groupby('global_row').agg(net_base=('net_base', 'mean'),
        paired_rescues=('net_base', lambda x: int((x>0).sum())),
        paired_harms=('net_base', lambda x: int((x<0).sum()))).reset_index()
    cases = cases.sort_values(['net_base', 'global_row'], ascending=[False, True])
    pos = cases[cases.net_base>0].head(3)
    neg = cases[cases.net_base<0].sort_values(['net_base', 'global_row']).head(3)
    selected = pd.concat([pos, neg], ignore_index=True)
    selected['class_name'] = manifest.class_name.to_numpy()[selected.global_row]
    selected.to_csv(HERE / 'OUTCOME_SELECTED_CASES.csv', index=False)
    fig, axes = plt.subplots(len(selected), 2, figsize=(8, 3*len(selected)), layout='constrained')
    for i, case in selected.iterrows():
        row = int(case.global_row)
        d = det.loc[row]
        theta = d.ship_axis
        co, si = np.cos(theta), np.sin(theta)
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * [d.length/2, d.width/2]
        corners = corners @ np.array([[co, si], [-si, co]]) + [d.cx, d.cy]
        for j, pol in enumerate(('vv', 'vh')):
            with Image.open(manifest.loc[row, 'chip_path_'+pol]) as im:
                image = np.asarray(im.convert('L'))
            ax = axes[i, j]
            ax.imshow(image, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            ax.add_patch(Polygon(corners, fill=False, edgecolor='lime', lw=.8))
            ax.plot(d.cx+np.array([-1, 1])*d.length*co,
                    d.cy+np.array([-1, 1])*d.length*si, color='cyan', lw=.7)
            ax.axhline(d.cy, color='yellow', lw=.6, alpha=.7)
            ax.set_title(f'{row} {case.class_name} {pol.upper()} {image.shape}\n'
                         f'paired rescues/harms {case.paired_rescues:.0f}/{case.paired_harms:.0f}; '
                         f'incidence {geo.loc[row, "incidence_deg"]:.1f} deg', fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Outcome-selected diagnostics, not representative accuracy evidence\n'
                 'Green: detected box; cyan: unsigned ship axis; yellow: range axis', fontsize=10)
    fig.savefig(HERE / 'outcome_cases.png', dpi=140)
    plt.close(fig)
    seeds = pd.read_csv(HERE / 'SEED_TABLE.csv')
    seeds = seeds[(seeds.policy=='source_selected') & (seeds.origin=='cross')]
    print('SEED_SD', seeds.groupby('arm')[['delta_oa', 'delta_ba']].agg(['mean', 'std']).to_string())
    print('JOINT_CLASS', classes[(classes.arm=='joint') & (classes.origin=='cross')].to_string(index=False))
    print('JOINT_SUPPORT', joint.groupby('support_group', observed=True).agg(
        paired_occurrences=('net_base', 'size'), net_correct=('net_base', 'sum'),
        delta_oa=('net_base', 'mean')).to_string())
    print('ATTENTION', attention.groupby('fold').agg(entropy=('entropy', 'mean'),
          top_token_count=('top_token', 'nunique')).to_string())
    print('COVERAGE', coverage[coverage.global_row.isin(joint.global_row.unique())].groupby('arm').mean_coverage.describe().to_string())
    print('CASES', selected.to_string(index=False))


if __name__ == '__main__':
    main()
