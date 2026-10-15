"""Report fixed comparisons and bounded source/target failure diagnosis."""

import numpy as np
import pandas as pd

from run_episodic_spatial_pooling import OUT, PORTS, ARMS, read_json


def main():
    assert read_json(OUT / 'COMPLETE.json')['status'] == 'complete'
    overall = pd.read_csv(OUT / 'target_equal_port_means.csv').set_index('arm')
    ports = pd.read_csv(OUT / 'target_port_means.csv')
    source = pd.read_csv(OUT / 'source_metrics.csv')
    learned = pd.read_csv(OUT / 'learning_diagnostics.csv')
    lines = ['# Episodic Spatial Pooling: Completed Pilot', '',
        'Six fixed development target ports; five original64-label draws each; 5% expert handoff. '
        'All target policies froze before query-label readout. The original main model was not modified.', '',
        '| Arm | OA % | BA % | Expert BA % | Mean Admissions |',
        '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        r = overall.loc[arm]
        lines.append(f'| {arm} | {100*r.oa:.4f} | {100*r.ba:.4f} | {100*r.expert_ba:.4f} | {r.admissions:.2f} |')
    lines.extend(['', 'Metrics average five draws within each port, then weight ports equally. '
        'Overlapping draws are not independent replications. The six-port means must not be compared to the full23-port means.', '',
        '## Fixed Comparisons', '', '| Treatment minus control | OA pp | BA pp | Expert BA pp |',
        '|---|---:|---:|---:|'])
    pairs = [('fixed_ordinary', 'matched_refit'), ('fixed_stable', 'matched_refit'),
        ('learned_ordinary', 'fixed_ordinary'), ('learned_stable', 'fixed_stable'),
        ('learned_stable', 'matched_refit'), ('learned_stable', 'original'),
        ('learned_stable', 'learned_ordinary')]
    for a, b in pairs:
        d = 100 * (overall.loc[a] - overall.loc[b])
        lines.append(f'| {a} - {b} | {d.oa:+.4f} | {d.ba:+.4f} | {d.expert_ba:+.4f} |')
    pivot = ports.pivot(index='port', columns='arm', values='ba')
    lines.extend(['', '| Port | Learned stable vs matched BA pp | Learned stable vs fixed stable BA pp |',
                  '|---|---:|---:|'])
    for port in PORTS:
        r = pivot.loc[port]
        lines.append(f'| {port} | {100*(r.learned_stable-r.matched_refit):+.4f} | {100*(r.learned_stable-r.fixed_stable):+.4f} |')
    lines.extend(['', '## Mechanism Actually Run', '',
        '1. Reuse frozen R50 four-token maps, centered within each chip and projected onto128 PCA components. '
        'The ordinary or candidate-stable source direction for the source-base predicted top-two classes supplies fixed token logits.',
        '2. Add a shared128->16 tanh->1 token scorer. Its output layer starts at zero, so initialization exactly retains the nonuniform fixed attention. '
        'This avoids a zero residual kernel with no first-order training signal.',
        '3. Softmax over four tokens yields a mean-zero spatial residual, scaled by the prior source RMS. '
        'Add only its .25-weight kernel to the original expert kernel. Do not add the previous auxiliary source-score correction.',
        '4. Refit the same class-balanced ridge8 correction from64 support labels. Backpropagate balanced source-query cross-entropy '
        'through the differentiable linear solve to train pooling. Original expert prior, backbone and base scores remain fixed.',
        '5. Source episodes at indices divisible by3 are held for epoch selection; the others train pooling. '
        'Eight epochs, AdamW .005, weight decay .001, no hyperparameter sweep. Choose maximum equal-source-port validation BA, '
        'tie minimum balanced CE; epoch0 is eligible. Query sampling is at most32/class for training and64/class for validation.',
        '6. Fit the original four-confidence binary source gate separately for each arm. Apply the unchanged disagreement, '
        'observed-support-class, positive-gate-score and top5% handoff rules.', '',
        'These heads have2080 trainable parameters. There is no physical or causal interpretation claim.', '',
        '## Minimal Root-Cause Diagnosis', '',
        '| Source split | Arm | Expert BA % | Final BA % |', '|---|---|---:|---:|'])
    means = source.groupby(['pool_split', 'arm'])[['expert_ba', 'ba']].mean()
    for (split, arm), row in means.iterrows():
        lines.append(f'| {split} | {arm} | {100*row.expert_ba:.4f} | {100*row.ba:.4f} |')
    lines.extend(['', 'The validation rows here use all available source-query examples; checkpoint selection used a fixed balanced subset. '
        'Neither table is an independent test: validation selected the checkpoint, and source gate metrics reuse gate fitting rows.', '',
        '| Target fold | Direction | Selected epoch | Validation BA change pp | Maximum training gradient norm |',
        '|---|---|---:|---:|---:|'])
    for r in learned.itertuples():
        lines.append(f'| {r.port} | {r.kind} | {r.selected_epoch} | '
                     f'{100*(r.selected_val_ba-r.initial_val_ba):+.4f} | {r.max_gradient:.5f} |')
    lines.extend(['',
        '- Implementation: five focused tests pass, including comparison with the existing NumPy ridge solver, '
        'finite-difference checking of gradients through support adaptation, live pooling gradients, and support/query disjoint sampling.',
        '- Integration: original expert replay is checked for132 source episodes and30 target draws. '
        'The source prior is no longer altered; only the support kernel changes. The three previously diagnosed '
        'Antwerp-outer source episodes reproduce kernel-only expert BA: ordinary44.6369%, stable44.6592%. '
        'No attention uniformity/no-gradient explanation remains.',
        '- Effect: removing the auxiliary prior eliminates the prior pilot\'s roughly5pp expert-BA regression. '
        'Nevertheless, kernel-only fixed pooling remains slightly below the matched unaugmented expert. '
        'Learned pooling improves some source expert metrics but does not increase target expert BA in either direction relative to fixed pooling.',
        '- Routing: learned-stable final BA is only about+.005pp versus its matched gate-refit control, with slightly higher OA. '
        'Versus the historical original, BA rises slightly but OA falls slightly. This is not a meaningful new overall improvement.',
        '- Remaining uncertainty: source-to-target transfer of the learned token weighting is not established. '
        'The limited four-token representation, the predicted-pair selector, and the strength/expressiveness of the extra kernel remain possible limits; '
        'this run does not isolate one as the unique cause. Do not claim the entire direction is impossible.', '',
        '## Boundary And Decision', '',
        'Outer-source-only encoder, geometry, candidate directions and scaling include the meta-validation source ports. '
        'Only the pooling optimizer excludes their query labels. Gates reuse all source episodes, including pooling-training queries. '
        'This is not fully nested OOF and is a repeatedly used development set, not new confirmation. '
        'No target query labels choose hyperparameters or checkpoints; target fitting uses only each original64-label support.', '',
        'Keep this as a bounded negative/tiny-effect development experiment. Do not promote it, combine it into the base-only pilot, '
        'or expand to other datasets without a better justified recipe. Preserve all earlier negative results.', ''])
    (OUT / 'FINDINGS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(OUT / 'FINDINGS.md')


if __name__ == '__main__':
    main()
