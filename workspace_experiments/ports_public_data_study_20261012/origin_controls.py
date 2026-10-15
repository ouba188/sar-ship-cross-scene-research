"""Post-hoc source-origin and same-acquisition controls for one cue reversal."""

from crossfit import *
from sklearn.metrics import roc_auc_score


def main():
    fields = ["instance_id", "class_id", "data_origin", "product", "port"]
    source = pd.read_csv(SOURCE / "manifest.csv", usecols=fields)
    meta = pd.read_csv(OUT / "audit/metadata.csv", low_memory=False)
    source = source.merge(meta[meta.dataset == "self24ports"][["sample_id", "contrast_proxy"]],
        left_on="instance_id", right_on="sample_id", validate="one_to_one")
    source.groupby(["data_origin", "class_id"]).contrast_proxy.agg(["count", "mean", "median"]).reset_index().to_csv(
        OUT / "image_diagnostics/origin_class_contrast.csv", index=False)
    records = []

    def add(dataset, scope, group, part, minimum):
        part = part[part.class_id.isin([1, 4])]
        ns = part.class_id.value_counts()
        if len(ns) == 2 and ns.min() >= minimum:
            records.append(dict(dataset=dataset, scope=scope, group=group, n_fishing=int(ns[1]),
                n_container=int(ns[4]), auc_container_higher=float(roc_auc_score(part.class_id == 4, part.contrast_proxy))))

    for origin, part in source.groupby("data_origin"):
        add("self24ports", "origin", origin, part, 10)
    for product, part in source.groupby("product"):
        add("self24ports", "same_product", product, part, 3)
    oss1 = pd.read_csv(OUT / "image_diagnostics/oss1.csv", usecols=["sample_id", "contrast_proxy"])
    for dataset in PUBLIC_TASKS:
        part = meta[meta.dataset == dataset]
        if dataset == "oss1":
            part = part.drop(columns="contrast_proxy").merge(oss1, on="sample_id", validate="one_to_one")
        for product, values in part.groupby("product"):
            add(dataset, "same_product", product, values, 3)
    pd.DataFrame(records).to_csv(OUT / "image_diagnostics/origin_product_pair_controls.csv", index=False)
    write(OUT / "image_diagnostics/ORIGIN_CONTROLS_COMPLETE.json", dict(pair=["fishing_vessel", "container_ship"],
        selected_after="Exploratory all-pair contrast-ordering result", model_fit=False,
        scope="Data association only; within-origin and within-product controls do not identify causal preprocessing effects"))


if __name__ == "__main__":
    main()
