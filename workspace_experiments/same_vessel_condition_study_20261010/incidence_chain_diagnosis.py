"""Read frozen chain outputs: effect scale and whether predictions actually change."""

from incidence_factor_chain import OUT, BASE, np, pd


def main():
    table = pd.read_csv(OUT/'representation_results.csv')
    table = table.loc[table.pairs > 0].copy()
    energy = table.angle_energy*table.slope_norm**2
    table['held_amplitude_ratio_diagnostic'] = table.signed_alignment/energy
    table.to_csv(OUT/'response_amplitude_diagnosis.csv', index=False)
    meta = pd.read_csv(BASE/'observations.csv')
    rows = []
    for space, seed in [('raw', -1), ('condition', 31415), ('condition', 27182)]:
        with np.load(OUT/f'{space}_{seed}_original_pair_scores.npz') as z:
            original, ids = z['scores'], z['ids']
        with np.load(OUT/f'{space}_{seed}_incidence_pair_scores.npz') as z:
            changed = z['scores']
            np.testing.assert_array_equal(z['ids'], ids)
        delta = changed-original
        y = meta.iloc[ids.ravel()].class_id.to_numpy().reshape(ids.shape)
        a, b = original.argmax(-1), changed.argmax(-1)
        rows.append(dict(space=space, seed=seed, score_delta_rms=float(np.sqrt((delta**2).mean())),
            score_delta_max=float(abs(delta).max()), changed_endpoint_occurrences=int((a != b).sum()),
            affected_ships=int(meta.iloc[ids[(a != b)]].mmsi.nunique()),
            rescued_endpoint_occurrences=int(((a != y)&(b == y)).sum()),
            harmed_endpoint_occurrences=int(((a == y)&(b != y)).sum())))
    result = pd.DataFrame(rows)
    result.to_csv(OUT/'decision_path_effects.csv', index=False)
    print(result.to_string(index=False))
    print(table.query('group == "held"')[['space', 'seed', 'held_amplitude_ratio_diagnostic']].to_string(index=False))


if __name__ == '__main__':
    main()
