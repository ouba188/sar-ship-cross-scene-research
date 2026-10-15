"""Report the completed fixed pilot without selecting or promoting any arm."""

import numpy as np
import pandas as pd

from base_only_preserved_expert import ARMS, PORTS
from run_base_only_preserved_expert import OUT, ROOT, read_json
from sar_joint_completion import write_json


def table(frame, columns, percentages=()):
    lines = ['| ' + ' | '.join(columns) + ' |', '|' + '|'.join(['---']*len(columns)) + '|']
    for _, row in frame.iterrows():
        values = []
        for key in columns:
            value = row[key]
            if key in percentages:
                values.append(f'{100*value:.4f}')
            elif isinstance(value, (float, np.floating)):
                values.append(f'{value:.4f}')
            else:
                values.append(str(value))
        lines.append('| ' + ' | '.join(values) + ' |')
    return lines


def main():
    read_json(OUT / 'COMPLETE.json')
    df = pd.read_csv(OUT / 'metrics.csv')
    ports = pd.read_csv(OUT / 'port_means.csv')
    overall = pd.read_csv(OUT / 'equal_port_means.csv').set_index('arm').loc[list(ARMS)]
    source = pd.read_csv(OUT / 'source_crossfit_metrics.csv')
    sm = source.groupby(['port', 'arm']).mean(numeric_only=True).groupby('arm').mean()
    deltas = []
    for level, grouping in [('overall', []), ('port', ['port']), ('draw', ['draw'])]:
        groups = [('all', df)] if not grouping else df.groupby(grouping[0])
        for group, data in groups:
            means = data.groupby('arm').mean(numeric_only=True)
            for arm, control in [('matched_baseline', 'original'), ('stable_only', 'matched_baseline'),
                                 ('joint', 'matched_baseline'), ('joint', 'stable_only'),
                                 ('stable_only', 'original'), ('joint', 'original')]:
                delta = means.loc[arm]-means.loc[control]
                deltas.append(dict(level=level, group=group, arm=arm, control=control,
                    delta_oa_pp=100*delta.oa, delta_ba_pp=100*delta.ba,
                    delta_base_ba_pp=100*delta.base_ba, delta_expert_ba_pp=100*delta.expert_ba,
                    delta_handoff_ba_pp=100*(delta.ba-delta.base_ba)))
    delta = pd.DataFrame(deltas)
    delta.to_csv(OUT / 'deltas.csv', index=False)
    sm.to_csv(OUT / 'source_crossfit_equal_port_means.csv')
    prior = pd.read_csv(ROOT / 'protected_drift_activation_20261008/metrics.csv')
    prior = prior[prior.port.isin(PORTS) & prior.arm.isin(ARMS[1:])].rename(
        columns={'acc': 'oa', 'base_acc': 'base_oa', 'expert_acc': 'expert_oa'})
    compare_fields = ['oa', 'ba', 'base_oa', 'base_ba', 'expert_oa', 'expert_ba']
    paired = df[df.arm.isin(ARMS[1:])].merge(prior, on=['port', 'draw', 'arm'],
        suffixes=('_preserved', '_both_changed'), validate='one_to_one')
    if len(paired) != 90:
        raise ValueError('Expected all90 corresponding prior/current arm-draw pairs')
    for key in ['base_oa', 'base_ba', 'cap', 'query_rows']:
        np.testing.assert_allclose(paired[key + '_preserved'], paired[key + '_both_changed'], atol=1e-12, rtol=0)
    for key in compare_fields:
        paired['delta_' + key + '_pp'] = 100*(paired[key + '_preserved']-paired[key + '_both_changed'])
    paired.to_csv(OUT / 'preservation_vs_both_changed_draws.csv', index=False)
    preserve_ports = paired.groupby(['port', 'arm']).mean(numeric_only=True)
    preserve_ports.to_csv(OUT / 'preservation_vs_both_changed_ports.csv')
    preserve_means = preserve_ports.groupby('arm').mean()
    preserve_means.to_csv(OUT / 'preservation_vs_both_changed_means.csv')
    fields = ['oa','ba','base_oa','base_ba','expert_oa','expert_ba']
    lines = ['# Base-only activation + preserved historical expert', '',
        'Client-local run label: 2026-10-09 (directory label retained). COMPLETE: six fixed ports, '
        'all five original64-label draws, 5% expert handoff. '
        'All six target policies froze before this run read target query outcomes. No sweep, expansion, or promotion. '
        'Historical expert scores and the complete original-reference policy remain exactly unchanged.', '',
        '## Equal-port results', '',
        'All OA/BA columns below are percentages; means first average five draws per port, then weight six ports equally. '
        'This difficult six-port subset must not be compared directly with full23-port means.', '']
    lines += table(overall.reset_index(), ['arm',*fields,'rescues','harms','admissions'], fields)
    lines += ['', '## Paired changes', '', 'All deltas are percentage points.', '']
    lines += table(delta[delta.level == 'overall'], ['arm','control','delta_oa_pp','delta_ba_pp',
        'delta_base_ba_pp','delta_expert_ba_pp','delta_handoff_ba_pp'])
    lines += ['', '## Direct expert-preservation comparison', '',
        'Read-only comparison with the completed protected_drift_activation_20261008 cached metrics, '
        'restricted to these identical six ports and five draws. Corresponding base OA/BA, query counts '
        'and handoff quotas match exactly. Both recipes fit a compatible source gate; therefore this '
        'tests preserved-expert-plus-compatible-gate versus changed-expert-plus-compatible-gate, not '
        'expert replacement under a single fixed gate. No new training or target selection was performed.', '',
        '| Arm | Both changed OA % | Preserved OA % | OA delta pp | Both changed BA % | Preserved BA % | BA delta pp | Expert BA delta pp |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS[1:]:
        r = preserve_means.loc[arm]
        lines.append(f'| {arm} | {100*r.oa_both_changed:.4f} | {100*r.oa_preserved:.4f} | '
            f'{r.delta_oa_pp:+.4f} | {100*r.ba_both_changed:.4f} | {100*r.ba_preserved:.4f} | '
            f'{r.delta_ba_pp:+.4f} | {r.delta_expert_ba_pp:+.4f} |')
    lines += ['',
        'Preservation raises final BA only slightly relative to the corresponding both-changed recipes '
        '(about0.04 pp for stable-only/joint), while lowering OA by about0.50-0.56 pp. '
        'It preserves higher standalone expert BA, but does NOT establish a better overall handoff system. '
        'The stronger preserved expert need not be more helpful on the particular5% disagreements selected. '
        'This direct result weakens, rather than confirms, a broad claim that preserving the historical '
        'expert is the missing route to a material gain.', '']
    lines += ['', '## Source-only bounded diagnosis', '',
        'These diagnostic gates use three blocked episode folds (episode index modulo3), excluding each held '
        'episode from gate fitting. Upstream source geometry, encoder and precomputed head predictions stay fixed; '
        'this is gate-only crossfit, NOT a fully nested end-to-end validation. No diagnostic selected a recipe.', '']
    lines += table(sm.reset_index(), ['arm','oa','ba','base_ba','expert_ba','rescues','harms','admissions'], fields)
    diagnosis = []
    for arm in ('stable_only','joint'):
        d = overall.loc[arm]-overall.loc['matched_baseline']
        sd = sm.loc[arm]-sm.loc['matched_baseline']
        pair = ports.pivot(index='port', columns='arm', values='ba')
        differences = pair[arm]-pair['matched_baseline']
        diagnosis.append(dict(arm=arm, target_ba_delta_pp=100*d.ba, target_base_ba_delta_pp=100*d.base_ba,
            target_handoff_ba_delta_pp=100*(d.ba-d.base_ba), source_crossfit_ba_delta_pp=100*sd.ba,
            source_base_ba_delta_pp=100*sd.base_ba, source_handoff_ba_delta_pp=100*(sd.ba-sd.base_ba),
            positive_ports=int((differences>1e-12).sum()), negative_ports=int((differences < -1e-12).sum())))
        lines.append(f'- {arm}: target base BA {100*d.base_ba:+.4f} pp; additional handoff contribution '
            f'{100*(d.ba-d.base_ba):+.4f} pp; final BA {100*d.ba:+.4f} pp. '
            f'Source base BA {100*sd.base_ba:+.4f} pp; gate-crossfit final BA {100*sd.ba:+.4f} pp. '
            f'Target ports up/down: {(differences>1e-12).sum()}/{(differences < -1e-12).sum()}.')
    lines += ['',
        'Bounded finding: stable-only has positive mean target BA in all five draws but only three of six '
        'ports improve, and mean OA falls. Its much larger source gain does not transfer at the same magnitude. '
        'Because source candidate geometry used inner-held labels, that source gain cannot be treated as '
        'an unbiased estimate of unseen-port gain. Joint versus stable-only reduces base BA on both the '
        'source diagnostic and target readout: the additional attenuation harms this base readout, while '
        'keeping the expert fixed cannot restore all of the lost benefit. This does not prove that Qd is '
        'purely useful or purely nuisance information. No implementation defect is established, and the '
        'remaining split between source geometry optimism and actual port heterogeneity is unresolved.', '',
        'The additive final-minus-base decomposition identifies where measured gains/losses enter the pipeline, '
        'not a causal attribution. An unchanged expert can still have a different marginal handoff benefit when '
        'base decisions and eligibility change. Source crossfit distinguishes a source-visible loss from a '
        'target-only reversal; it does not identify a unique latent mechanism.', '',
        'Implementation checks: all132 source episodes reproduce the original protected-activation base OA/BA; '
        'all90 cached target base score arrays independently replay exactly; all30 target expert score arrays '
        'and original policies are exactly unchanged. Row IDs, original64 supports, query masks and quotas match. '
        'Five focused tests cover explicit views, independent primal ridge equivalence, original binary gate '
        'objective, expert nonmutation/5% routing and corrupted support/row alignment. '
        'Thus no observed score-reconstruction, wrong-row, changed-expert or budget error explains any negative result.', '',
        '## Pipeline and boundaries', '',
        '1. Reuse source-discovered candidate Qs and protected drift Qd from the completed20261008 run; '
        'these are candidate stable directions, not proven causal invariants.',
        '2. Matched baseline refits only the ordinary base source head. Stable-only appends source-standardized '
        'Qs coordinates to that head and64-label residual features. Joint also attenuates16 orthogonal drift '
        'directions by0.5. No expert source head, expert kernel or expert support correction is changed.',
        '3. Reuse exact saved target base scores; rebuild source base heads using original purged train rows. '
        'Refit each compatible four-confidence gate using source eligible informative rescue/harm disagreements, '
        'equal episode/class weighting and LR C1.',
        '4. Among query disagreements whose expert class is present in the64 supports and whose gate score '
        'is positive, hand off at most floor(0.05*Nquery), using row-ID ties.',
        '5. Qs/Qd/PCA/scales and encoder use all outer-source data including inner-held source labels. Only '
        'source readout fits are inner-held. Existing data have repeatedly supported development; neither '
        'the six ports nor overlapping draws constitute independent confirmation.', '',
        'The historical reference retains old base and old gate. The matched baseline intentionally includes '
        'base refitting, so historical-reference improvement alone is not the stable-module effect. '
        'No same-dimensional generic extra branch is tested here, so even positive stable activation does not '
        'establish exclusive stability causation. Main model is not replaced.', '',
        '## Every port', '']
    lines += table(ports, ['port','arm',*fields,'admissions'], fields)
    lines += ['', '## Every draw and port', '']
    lines += table(df.sort_values(['port','draw','arm']), ['port','draw','arm',*fields,'admissions'], fields)
    lines += ['', '## Files', '',
        '- metrics.csv:120 port/draw/arm records, including OA, BA, base/expert metrics, rescues/harms and quotas.',
        '- port_means.csv, equal_port_means.csv, draw_means.csv, deltas.csv: grouped results and paired deltas.',
        '- source_crossfit_metrics.csv:396 held-source arm records; source_fit_metrics.csv is fitting-sample gate diagnostic only.',
        '- preservation_vs_both_changed_{draws,ports,means}.csv: direct matched comparison to prior both-changed cached recipes.',
        '- ports/*/policies.npz: frozen target policies; one shared historical expert score array per port.',
        '- ports/*/*_source_scores.npz: source episodes, offsets, labels, base/expert scores and gate inputs.',
        '- PROTOCOL.json, ALL_FROZEN.json, COMPLETE.json, DIAGNOSIS.json: scope and completion records.', '']
    (OUT / 'FINDINGS.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(OUT / 'DIAGNOSIS.json', dict(contrasts=diagnosis, source_gate_crossfit_only=True,
        implementation_mismatch_found=False, no_unique_causal_root_identified=True, no_promotion=True))
    print(overall[['oa','ba','base_ba','expert_ba']].to_string())
    print(delta[delta.level == 'overall'].to_string(index=False))
    print('PRESERVED minus BOTH CHANGED:')
    print(preserve_means[['delta_oa_pp', 'delta_ba_pp', 'delta_base_ba_pp', 'delta_expert_ba_pp']].to_string())
    print('REPORT: ' + str(OUT / 'FINDINGS.md'))


if __name__ == '__main__':
    main()
