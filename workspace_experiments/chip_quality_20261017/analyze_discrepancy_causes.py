"""Cross-tabulate existing review evidence; do not assign causal error labels."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "discrepancy_causes"


def main():
    OUT.mkdir(exist_ok=True)
    chips = pd.read_csv(HERE / "per_chip_quality.csv", low_memory=False, dtype={"mmsi": str})
    keys = ["global_row","instance_id","ais_length_m","ais_width_m","ais_size_scope","vessel_static_conflict",
            "ais_size_valid","crop_position_method_vv","crop_uncertainty_m_vv","crop_position_method_vh",
            "crop_uncertainty_m_vh","position_link_quality","ais_record_match_quality"]
    meta = pd.read_csv(HERE / "ais_metadata/ais_metadata_sidecar.csv", usecols=keys, low_memory=False)
    frame = chips.merge(meta,on=["global_row","instance_id"],validate="one_to_one")
    assert len(frame) == 38091
    frame["ais_extreme_length"] = frame.ais_length_m.gt(500) | frame.ais_length_m.le(5)
    frame["ais_limit_pattern"] = (frame.ais_length_m.isin([511,1022]) | frame.ais_width_m.isin([63,126])) & frame.ais_size_valid
    frame["ais_reference_review"] = frame.ais_extreme_length | frame.ais_limit_pattern | frame.vessel_static_conflict
    frame["box_longer"] = ((frame.length_screen_vv & frame.length_ratio_vv.gt(2)) |
                            (frame.length_screen_vh & frame.length_ratio_vh.gt(2)))
    frame["box_shorter"] = ((frame.length_screen_vv & frame.length_ratio_vv.lt(.5)) |
                             (frame.length_screen_vh & frame.length_ratio_vh.lt(.5)))
    frame["flagged_view_edge"] = ((frame.length_screen_vv & frame.target_edge_vv) |
                                   (frame.length_screen_vh & frame.target_edge_vh))
    frame["flagged_view_nonunique_match"] = ((frame.length_screen_vv & ~frame.target_match_status_vv.eq("inside_unique_box")) |
                                               (frame.length_screen_vh & ~frame.target_match_status_vh.eq("inside_unique_box")))
    flagged = frame[frame.length_screen_any].copy()
    assert len(flagged) == 4558
    flagged.to_csv(OUT / "length_candidates_with_evidence.csv",index=False,encoding="utf-8-sig")
    boolcols = ["box_longer","box_shorter","vessel_static_conflict","ais_extreme_length","ais_limit_pattern",
                "ais_reference_review","multi_any_25","flagged_view_edge","flagged_view_nonunique_match"]
    summary = dict(flagged_pairs=len(flagged),comparable_pairs=int(frame.size_comparable_any.sum()),
                   overlapping_evidence_counts={k:int(flagged[k].sum()) for k in boolcols},
                   neither_ais_review_nor_multi=int((~flagged.ais_reference_review & ~flagged.multi_any_25).sum()),
                   ais_review_and_multi=int((flagged.ais_reference_review & flagged.multi_any_25).sum()),
                   box_longer_ais_length_le60=int((flagged.box_longer & flagged.ais_length_m.le(60)).sum()),
                   ais_scope=flagged.ais_size_scope.value_counts().to_dict())
    flagged.groupby(["ais_reference_review","multi_any_25"]).size().rename("n").to_csv(OUT / "overlap.csv")
    for col in ("class_name","ais_size_scope","crop_position_method_vv"):
        comparable = frame[frame.size_comparable_any].copy()
        comparable[col] = comparable[col].fillna("missing")
        table = comparable.groupby(col).agg(comparable=("global_row","size"),length_flags=("length_screen_any","sum"),
                                           ais_review=("ais_reference_review","sum"),multi=("multi_any_25","sum"))
        table["flag_fraction"] = table.length_flags/table.comparable
        table.to_csv(OUT / ("by_"+col+".csv"),encoding="utf-8-sig")
    both = frame[frame.size_comparable_vv & frame.size_comparable_vh].copy()
    views = pd.read_csv(HERE / "per_view_quality.csv",usecols=["global_row","pol","target_confidence"],low_memory=False)
    confidence = views.pivot(index="global_row",columns="pol",values="target_confidence")
    both = both.join(confidence,on="global_row")
    both["pol_length_ratio"] = both.target_length_m_vv / both.target_length_m_vh
    both["pol_disagree_2x"] = ((both.pol_length_ratio.lt(.5) | both.pol_length_ratio.gt(2)) &
                                 (both.target_length_m_vv-both.target_length_m_vh).abs().gt(30))
    summary["both_pols_comparable"] = len(both)
    summary["both_pols_length_flags"] = int(both.length_screen_any.sum())
    summary["pol_length_disagree_2x"] = int(both.pol_disagree_2x.sum())
    summary["flagged_with_pol_length_disagree_2x"] = int((both.pol_disagree_2x & both.length_screen_any).sum())
    both[both.pol_disagree_2x].sort_values("pol_length_ratio",ascending=False).to_csv(OUT / "polarization_disagreement.csv",index=False,encoding="utf-8-sig")
    for name,mask in (("nearest_uncertain", flagged.crop_position_method_vv.eq("nearest_uncertain")),
                      ("uncertainty_gt500m", flagged.crop_uncertainty_m_vv.gt(500))):
        summary[name] = int(mask.sum())
    (OUT / "summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
    print("POLARIZATION EXAMPLES")
    print(both[both.pol_disagree_2x][["global_row","mmsi","class_name","ais_length_m","target_length_m_vv","target_length_m_vh","vv","vh","multi_any_25"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
