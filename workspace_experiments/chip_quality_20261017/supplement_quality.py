"""Describe size-dependent bias and reviewable AIS value patterns, without relabeling."""
import json
import numpy as np
import pandas as pd
from analyze_quality import HERE, read_detections, gallery


def main():
    views = pd.read_csv(HERE / "per_view_quality.csv", low_memory=False, dtype={"mmsi":str})
    chips = pd.read_csv(HERE / "per_chip_quality.csv", dtype={"mmsi":str})
    columns = ["global_row","instance_id","mmsi","product","port","class_name","ais_length_m","ais_width_m",
               "ais_size_scope","ais_size_source","ais_source_line","vessel_static_conflict","ais_size_valid"]
    ais = pd.read_csv(HERE / "ais_metadata/ais_metadata_sidecar.csv", usecols=columns, dtype={"mmsi":str}, low_memory=False)
    ais["extreme_total_length_review"] = ais.ais_length_m.gt(500) | ais.ais_length_m.le(5)
    ais["encoding_limit_pattern_review"] = (ais.ais_length_m.isin([511,1022]) | ais.ais_width_m.isin([63,126])) & ais.ais_size_valid
    flagged = ais[ais.extreme_total_length_review | ais.encoding_limit_pattern_review | ais.vessel_static_conflict]
    flagged.to_csv(HERE / "ais_attribute_review.csv", index=False, encoding="utf-8-sig")
    primary = views[views.primary_size_eligible].copy()
    primary["ais_length_bin_m"] = pd.cut(primary.ais_length_m,[0,30,60,100,200,500,1000,np.inf])
    grouped = primary.groupby(["pol","ais_length_bin_m"],observed=True).agg(n=("global_row","size"),
        length_ratio_median=("length_ratio","median"),width_ratio_median=("width_ratio","median"),
        length_screen_n=("length_screen","sum"))
    grouped.to_csv(HERE / "size_by_length_bin.csv", encoding="utf-8-sig")
    cautious = primary.merge(ais[["global_row","encoding_limit_pattern_review","extreme_total_length_review"]],on="global_row",validate="many_to_one")
    cautious = cautious[cautious.ais_size_scope.eq("exact_observation_record") & ~cautious.vessel_static_conflict &
                        ~cautious.encoding_limit_pattern_review & ~cautious.extreme_total_length_review]
    cautious.groupby("pol").agg(n=("global_row","size"),length_screen_n=("length_screen","sum"),
        width_screen_n=("width_screen","sum"),length_ratio_median=("length_ratio","median"),
        width_ratio_median=("width_ratio","median")).to_csv(HERE / "size_exact_unconflicted_reference.csv",encoding="utf-8-sig")
    pattern = ais[ais.encoding_limit_pattern_review].groupby(["ais_length_m","ais_width_m"]).agg(n=("global_row","size"),ships=("mmsi","nunique"))
    pattern.sort_values("n",ascending=False).to_csv(HERE / "ais_limit_patterns.csv", encoding="utf-8-sig")
    detected = read_detections(HERE / "full/canvas512/detections.jsonl")
    gallery(chips,views,detected,[34358,29930,38063],"multi_ship_selected","Multiple spatially separated ship-like targets")
    large = primary[primary.pol.eq("vv") & primary.ais_length_m.ge(511)].sort_values("length_ratio").groupby("port",sort=False).head(1)
    gallery(chips,views,detected,large.global_row.head(6),"ais_extreme_length_examples","Very large AIS length: record/association review required")
    gallery(chips,views,detected,[8340,1398,29237],"size_ambiguity_selected","Size mismatch has different possible origins; OBB is not hull truth")
    metadata = dict(static_conflict_rows=int(ais.vessel_static_conflict.sum()),
        static_conflict_distinct_mmsi=ais.loc[ais.vessel_static_conflict,"mmsi"].nunique(),
        large_ais_rows=int(ais.ais_length_m.gt(500).sum()),tiny_ais_rows=int(ais.ais_length_m.le(5).sum()),
        encoding_limit_pattern_rows=int(ais.encoding_limit_pattern_review.sum()),
        encoding_limit_pattern_mmsi=ais.loc[ais.encoding_limit_pattern_review,"mmsi"].nunique(),
        n_511_63=int((ais.ais_length_m.eq(511)&ais.ais_width_m.eq(63)).sum()),
        n_1022_126=int((ais.ais_length_m.eq(1022)&ais.ais_width_m.eq(126)).sum()))
    (HERE / "attribute_summary.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    reviews = [
        (34358,"multi_ship_selected","Four separated compact ship-like responses in both views; visible multi-object case; identities unverified"),
        (29930,"multi_ship_selected","Three separated differently oriented ship-like responses in both views"),
        (38063,"multi_ship_selected","Three separated ship-like responses; scale unavailable, keep pixel units"),
        (13525,"multi_ship_examples","Other candidates detected but bright central label-target response not matched at 0.25"),
        (9304,"multi_ship_one_view","VH extra box near the extension of the central target; possible response splitting/artifact, not confirmed second ship"),
        (11717,"multi_ship_one_view","Two parallel extended responses; image alone does not establish whether second physical ship or ambiguity response"),
        (8340,"size_ambiguity_selected","AIS total 2x2m; long weak vertical response inside OBB; both AIS field and image measurement require review"),
        (1398,"size_ambiguity_selected","Strong extended VV streak; OBB apparent length is not clean hull extent; VH much less extended"),
        (29237,"size_ambiguity_selected","Large OBB covers weak extended VV response; VH lacks accepted target; detector/response failure plausible"),
        (17484,"size_agreement_examples","Length close to AIS in both views but width still much larger; agreement not identity verification"),
    ]
    pd.DataFrame(reviews,columns=["global_row","figure","visual_observation"]).to_csv(HERE / "visual_review_notes.csv",index=False,encoding="utf-8-sig")
    print(json.dumps(metadata,indent=2))


if __name__ == "__main__":
    main()
