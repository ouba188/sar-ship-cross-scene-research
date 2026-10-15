"""Exploratory class-conditional display-cue ordering, all class pairs retained."""

from crossfit import *
from itertools import combinations
from sklearn.metrics import roc_auc_score
from image_diagnostics import DST, FIELDS


def main():
    metadata = pd.read_csv(OUT / "audit/metadata.csv", low_memory=False)
    extra = pd.read_csv(DST / "oss1.csv")
    one = metadata[metadata.dataset == "oss1"].drop(columns=FIELDS).merge(extra[["sample_id"]+FIELDS],
        on="sample_id", how="left", validate="one_to_one")
    frame = pd.concat([metadata[metadata.dataset != "oss1"], one], ignore_index=True)
    records = []
    groups = [(dataset, "all", values) for dataset, values in frame.groupby("dataset")]
    source = frame[frame.dataset == "self24ports"]
    groups += [("self24ports", port, values) for port, values in source.groupby("actual_port")]
    for dataset, task, part in groups:
        for field in FIELDS:
            values = pd.to_numeric(part[field], errors="coerce")
            for a, b in combinations(range(8), 2):
                ix = part.class_id.isin([a, b]) & values.notna()
                y = part.loc[ix, "class_id"].to_numpy(int)
                na, nb = int(np.sum(y == a)), int(np.sum(y == b))
                if min(na, nb) < 10:
                    continue
                records.append(dict(dataset=dataset, task=task, descriptor=field, class_a=a, class_b=b,
                    n_a=na, n_b=nb, auc_b_higher=float(roc_auc_score(y == b, values[ix])),
                    median_a=float(values[ix][y == a].median()), median_b=float(values[ix][y == b].median())))
    pd.DataFrame(records).to_csv(DST / "class_pair_ordering.csv", index=False)
    write(DST / "CLASS_SIGNAL_COMPLETE.json", dict(all_pairs=28, minimum_per_class=10,
        direction="AUC is P(descriptor(class_b)>descriptor(class_a)) with half-credit for ties; NOT a trained classifier",
        selection="All pairs and all five descriptors retained, exploratory after outcomes",
        meaning="Reversal establishes changed class/display-proxy association, not model reliance or a physical causal mechanism"))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
