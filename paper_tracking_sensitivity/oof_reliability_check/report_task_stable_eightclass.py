"""Separate mixed target support from all-eight-class target evaluation."""

import pandas as pd

from task_stable_eightclass import OUT, PARENT, SOURCE


def main():
    frame = pd.read_csv(SOURCE / 'manifest.csv', usecols=['port', 'class_id'])
    counts = frame.groupby('port').class_id.nunique()
    pm = pd.read_csv(OUT / 'port_means.csv')
    history = pd.read_csv(PARENT / 'port_means.csv')
    evaluated = set(pm.port)
    groups = dict(all23=evaluated,
                  all8_target12={p for p in evaluated if counts[p] == 8},
                  partial_target11={p for p in evaluated if counts[p] < 8})
    summaries, originals = [], []
    for name, ports in groups.items():
        part = pm[pm.port.isin(ports)].groupby(['arm', 'selector', 'attenuation']).mean(numeric_only=True).reset_index()
        part.insert(0, 'group', name)
        part.insert(1, 'ports', len(ports))
        summaries.append(part)
        h = history[history.port.isin(ports)].groupby('arm')[['acc', 'ba']].mean().reset_index()
        h.insert(0, 'group', name)
        h.insert(1, 'ports', len(ports))
        originals.append(h)
    result = pd.concat(summaries, ignore_index=True)
    result.to_csv(OUT / 'coverage_group_means.csv', index=False)
    original = pd.concat(originals, ignore_index=True)
    original.to_csv(OUT / 'coverage_group_original_models.csv', index=False)
    print(original.to_string(index=False))
    print(result[result.selector.isin(['reference', 'predicted_top2'])][
        ['group', 'arm', 'selector', 'attenuation', 'full_oa', 'full_ba']].to_string(index=False))


if __name__ == '__main__':
    main()
