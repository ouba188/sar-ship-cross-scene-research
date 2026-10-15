"""Posthoc score accounting prompted by gains in an unsupported class."""

from run_mechanism import c, BUDGET, OUT, DATASETS, CachedStrongFeatures, np, pd


def main():
    rows = []
    for ds in DATASETS:
        directory = BUDGET/'strong_external'/ds/ds
        manifest = pd.read_csv(c.read_json(directory/'PREDICTIONS_FROZEN.json')['target_manifest'])
        features = CachedStrongFeatures(ds)
        for draw in range(5):
            old = c.archive(directory/f'{draw}_16.npz')
            q, y = old['query_ids'], manifest.iloc[old['query_ids']].class_id.to_numpy(int)
            prior = features.prior[q]
            raw, changed = old['expert_raw'], old['expert_initial']
            wrong = raw.copy()
            wrong[np.arange(len(y)), y] = -np.inf
            rival = wrong.argmax(1)
            at = np.arange(len(y))
            margins = {name: score[at, y]-score[at, rival]
                       for name, score in [('prior', prior), ('raw', raw), ('initial', changed)]}
            for cls in np.unique(y):
                mask = y==cls
                rows.append(dict(dataset=ds, budget=16, draw=draw, class_id=int(cls),
                    class_name=str(manifest.iloc[q[mask][0]].class_name), query_n=int(mask.sum()),
                    support_n=int(np.sum(old['support_labels']==cls)),
                    prior_margin=float(margins['prior'][mask].mean()),
                    raw_margin=float(margins['raw'][mask].mean()),
                    initial_margin=float(margins['initial'][mask].mean()),
                    raw_support_margin=float((margins['raw']-margins['prior'])[mask].mean()),
                    initial_support_margin=float((margins['initial']-margins['prior'])[mask].mean()),
                    margin_gain=float((margins['initial']-margins['raw'])[mask].mean()),
                    raw_true_class_support_score=float((raw-prior)[mask, cls].mean()),
                    initial_true_class_support_score=float((changed-prior)[mask, cls].mean())))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'prior_margin.csv', index=False)
    frame.groupby(['dataset', 'budget', 'class_id', 'class_name']).mean(numeric_only=True).to_csv(OUT/'prior_margin_summary.csv')
    c.write(OUT/'PRIOR_MARGIN_NOTE.json', dict(
        stage='Posthoc descriptive decomposition; no new candidate or training',
        trigger='F_B16 fishing gains despite zero fishing supports in all five draws',
        scope='Both datasets B16, all classes and five existing support draws',
        reference_rival='Highest RAW expert score among wrong classes, fixed for prior/raw/initial',
        limitation='Group-average fixed-rival expert margins; not a causal explanation of every fusion decision'))
    print(frame.groupby(['dataset', 'class_name'])[['support_n','raw_support_margin','initial_support_margin','margin_gain']].mean().to_string())


if __name__ == '__main__':
    with c.threadpool_limits(limits=2):
        main()
