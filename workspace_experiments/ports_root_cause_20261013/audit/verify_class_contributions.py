"""Check two interpretation-critical classes against already audited recalls."""

import json

import numpy as np
import pandas as pd

from verify_predictions import OUT, ROOT, TASKS


def main():
    classes = pd.read_csv(OUT / "independent_classes.csv")
    classes = classes[(classes.source_condition != "archive") & (classes.scope == "original") &
                      (classes.arm == "R") & (classes["head"] == "expert")].copy()
    classes["cell"] = classes.source_condition.str[0].str.upper()+classes.target_condition.str[0].str.upper()
    keys = ["fold", "task", "dataset", "draw", "scope", "arm", "head"]
    classes["class_count"] = classes.groupby(keys+["cell"]).class_id.transform("nunique")
    selected = classes[classes.class_id.isin([1, 5])]
    rows = []
    for contrast, old_cell in (("source_coordination", "OD"), ("matched_net", "OO")):
        new = selected[selected.cell == "DD"]
        old = selected[selected.cell == old_cell]
        joined = new.merge(old, on=keys+["class_id"], suffixes=("_new", "_old"), validate="one_to_one")
        np.testing.assert_array_equal(joined.class_count_new, joined.class_count_old)
        out = joined[keys+["class_id"]].copy()
        out["contrast"] = contrast
        out["delta_recall"] = joined.recall_new-joined.recall_old
        out["ba_contribution"] = out.delta_recall/joined.class_count_new
        out["risk_gain_contribution"] = (joined.brier_old-joined.brier_new)/joined.class_count_new
        rows.append(out)
    audit = pd.concat(rows, ignore_index=True)
    main_table = pd.read_csv(ROOT / "class_effects.csv")
    mask = [(task in TASKS.get(fold, ())) for fold, task in zip(main_table.fold, main_table.task)]
    main_table = main_table[mask]
    main_table = main_table[(main_table.scope == "original") & (main_table.arm == "R") &
                            (main_table["head"] == "expert") & main_table.class_id.isin([1, 5]) &
                            main_table.contrast.isin(["source_coordination", "matched_net"])]
    joined = audit.merge(main_table, on=keys+["class_id", "contrast"], suffixes=("_audit", "_main"),
                         how="outer", indicator=True, validate="one_to_one")
    assert joined._merge.eq("both").all()
    errors = {}
    for field in ("delta_recall", "ba_contribution", "risk_gain_contribution"):
        np.testing.assert_allclose(joined[field+"_audit"], joined[field+"_main"], atol=1e-10, rtol=0)
        errors[field] = float(np.abs(joined[field+"_audit"]-joined[field+"_main"]).max())
    audit.to_csv(OUT / "independent_key_class_effects.csv", index=False)
    result = dict(status="passed", rows=len(audit), max_errors=errors,
                  scope="Previously selected six tasks, R/expert original queries, fishing/crude, coordination/net",
                  reused_independent_class_metrics=True, new_inference_or_fits=0)
    (OUT / "CLASS_EFFECT_AUDIT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
