"""Measure canvas sensitivity on screened candidates, without changing thresholds."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analyze_quality import read_detections, usable, analyze_image, load_manifest, HERE


def main():
    base = pd.read_csv(HERE / "per_view_quality.csv", keep_default_na=False, low_memory=False).set_index(["global_row", "pol"])
    chips = pd.read_csv(HERE / "per_chip_quality.csv")
    manifest = load_manifest().to_dict("records")
    metadata = pd.read_csv(HERE / "ais_metadata/ais_metadata_sidecar.csv", dtype=str, keep_default_na=False).set_index("instance_id")
    alternative = read_detections(HERE / "recheck_rows/canvas1024/detections.jsonl")
    wanted = pd.read_csv(HERE / "recheck_rows.csv").global_row.tolist()
    assert set(alternative) == {(i, pol) for i in wanted for pol in ("vv", "vh")}
    comparisons = []
    for (i, pol), rec in alternative.items():
        a = base.loc[(i,pol)]
        b = analyze_image(rec, manifest[i], metadata.loc[manifest[i]["instance_id"]].to_dict())
        length_a, length_b = pd.to_numeric(a.target_length_px, errors="coerce"), b["target_length_px"]
        comparisons.append(dict(global_row=i,pol=pol,n_512=a.n_25,n_1024=b["n_25"],
            target_match_512=a.target_match_status,target_match_1024=b["target_match_status"],
            length_screen_512=a.length_screen,length_screen_1024=b["length_screen"],
            width_screen_512=a.width_screen,width_screen_1024=b["width_screen"],
            length_px_512=length_a,length_px_1024=length_b,
            relative_length_change=abs(length_b/length_a-1) if length_a > 0 else np.nan))
    c = pd.DataFrame(comparisons)
    c.to_csv(HERE / "canvas_recheck_per_view.csv", index=False, encoding="utf-8-sig")
    rows = []
    for i, g in c.groupby("global_row"):
        original = chips.iloc[i]
        rows.append(dict(global_row=i,original_multi=bool(original.multi_any_25),
            multi_1024=bool((g.n_1024 >= 2).any()),
            original_length_flag=bool(original.length_screen_any),length_flag_1024=bool(g.length_screen_1024.any()),
            original_width_flag=bool(original.width_screen_any),width_flag_1024=bool(g.width_screen_1024.any())))
    p = pd.DataFrame(rows)
    p.to_csv(HERE / "canvas_recheck_per_chip.csv", index=False, encoding="utf-8-sig")
    stable_multi = p[p.original_multi & p.multi_1024]
    stable_length = p[p.original_length_flag & p.length_flag_1024]
    summary = dict(rechecked_chips=len(p),rechecked_images=len(c),same_object_count_images=int((c.n_512 == c.n_1024).sum()),
        original_multi_rechecked=int(p.original_multi.sum()),multi_retained_1024=len(stable_multi),
        original_length_flags=int(p.original_length_flag.sum()),length_flags_retained_1024=len(stable_length),
        original_width_flags=int(p.original_width_flag.sum()),width_flags_retained_1024=int((p.original_width_flag & p.width_flag_1024).sum()),
        common_target_length_relative_difference_q50_q95=c.relative_length_change.quantile([.5,.95]).tolist())
    (HERE / "canvas_recheck_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
