"""Independent metadata pairing and one-fold frozen-prediction comparison."""

import json

import numpy as np
import pandas as pd

from verify_predictions import OUT, ROOT, SENSITIVITY, archive, folder, metric_values


def main():
    metadata = pd.read_csv(ROOT / "dataset_audit/public_metadata.csv", dtype=str, keep_default_na=False)
    metadata = metadata[metadata.dataset == "oss2"].copy()
    manifest = pd.read_csv(SENSITIVITY / "opensarship2_transfer_20261007/target_manifest.csv",
                           dtype=str, keep_default_na=False).set_index("sample_id")
    assert len(metadata) == 3745 and metadata.sample_id.is_unique
    assert set(metadata.sample_id) == set(manifest.index)
    np.testing.assert_array_equal(metadata.identity, manifest.loc[metadata.sample_id, "mmsi_or_ship_id"])
    np.testing.assert_array_equal(metadata.class_id, manifest.loc[metadata.sample_id, "class_id"])
    product_fields = metadata["product"].str.extract(r"^(S1[A-Z])_.*_(\d{6})_([0-9A-F]{6})_[0-9A-F]{4}(?:\(1\))?$")
    assert product_fields.notna().all().all()
    metadata["take"] = product_fields.agg("|".join, axis=1)
    np.testing.assert_array_equal(metadata["take"], metadata.datatake_key)
    metadata["exact_time"] = pd.to_datetime(metadata.sar_time.str.replace(".", " ", regex=False),
                                             format="%Y/%m/%d %H:%M:%S", utc=True)
    np.testing.assert_array_equal(metadata.exact_time, pd.to_datetime(metadata.sar_datetime, utc=True))
    keys = ["identity", "take"]
    cardinality = metadata.groupby(keys+["product_type"]).size().unstack(fill_value=0)
    unique = cardinality[(cardinality.GRDH == 1) & (cardinality.SLC == 1)].reset_index()[keys]
    grdh = metadata[metadata.product_type == "GRDH"].merge(unique, on=keys)
    slc = metadata[metadata.product_type == "SLC"].merge(unique, on=keys)
    pairs = grdh.merge(slc, on=keys, suffixes=("_grdh", "_slc"), validate="one_to_one")
    unique_n = len(pairs)
    pairs = pairs[pairs.exact_time_grdh == pairs.exact_time_slc].copy()
    exported = pd.read_csv(ROOT / "dataset_audit/oss2_grdh_slc_pairs.csv", dtype=str, keep_default_na=False)
    idkeys = ["sample_id_grdh", "sample_id_slc"]
    assert set(map(tuple, pairs[idkeys].to_numpy())) == set(map(tuple, exported[idkeys].to_numpy()))
    assert len(pairs) == len(exported) == 234
    same_label = pairs.class_id_grdh == pairs.class_id_slc
    aligned = exported.set_index(idkeys).loc[pd.MultiIndex.from_frame(pairs[idkeys])]
    np.testing.assert_array_equal(same_label.to_numpy(), aligned.same_label.eq("True").to_numpy())
    for side in ("grdh", "slc"):
        np.testing.assert_array_equal(pairs["noise_xml_"+side].to_numpy(), aligned["noise_xml_"+side].to_numpy())
    same_n = int(same_label.sum())
    assert unique_n == 370 and same_n == 233
    eligible_pairs = pairs.loc[same_label].copy()
    results, query_counts = [], []
    reference = None
    for cell, source, target in (("OO", "original", "original"), ("OD", "original", "display"),
                                 ("DO", "display", "original"), ("DD", "display", "display")):
        path = folder(source, target, 2, "oss2")
        saved = archive(path / "predictions.npz")
        if reference is None:
            reference = saved
        for field in ("ids", "support", "eval_masks"):
            np.testing.assert_array_equal(saved[field], reference[field])
        lookup = pd.Index(saved["ids"])
        g = lookup.get_indexer(eligible_pairs.sample_id_grdh)
        s = lookup.get_indexer(eligible_pairs.sample_id_slc)
        assert (g >= 0).all() and (s >= 0).all()
        y = manifest.loc[eligible_pairs.sample_id_grdh, "class_id"].to_numpy(int)
        np.testing.assert_array_equal(y, manifest.loc[eligible_pairs.sample_id_slc, "class_id"].to_numpy(int))
        probability = np.load(path / "probabilities.npy", mmap_mode="r")
        ai, hi = list(saved["arms"]).index("T"), list(saved["heads"]).index("expert")
        clean = eligible_pairs.noise_xml_grdh.eq("0").to_numpy() & eligible_pairs.noise_xml_slc.eq("0").to_numpy()
        for draw, query in enumerate(saved["eval_masks"]):
            both = query[g] & query[s]
            if cell == "OO":
                all_g = lookup.get_indexer(pairs.sample_id_grdh)
                all_s = lookup.get_indexer(pairs.sample_id_slc)
                query_counts.append(dict(draw=draw, exact_pair_both_query=int(np.sum(query[all_g] & query[all_s])),
                                         same_label_both_query=int(both.sum()),
                                         same_label_both_query_clean=int(np.sum(both & clean))))
            for scope, mask in (("paired_all", both), ("paired_both_clean", both & clean)):
                yy = y[mask]
                gp = saved["predictions"][draw, ai, hi, g[mask]]
                sp = saved["predictions"][draw, ai, hi, s[mask]]
                gm, _, w = metric_values(yy, gp, np.asarray(probability[draw, ai, hi, g[mask]], float))
                sm, _, _ = metric_values(yy, sp, np.asarray(probability[draw, ai, hi, s[mask]], float))
                results.append(dict(fold=2, cell=cell, draw=draw, scope=scope, arm="T", head="expert",
                                    pairs=len(yy), classes=len(np.unique(yy)), grdh_ba=gm["ba"], slc_ba=sm["ba"],
                                    slc_minus_grdh_ba=sm["ba"]-gm["ba"], grdh_oa=gm["oa"], slc_oa=sm["oa"],
                                    grdh_brier=gm["brier_macro"], slc_brier=sm["brier_macro"],
                                    grdh_wrong_slc_right_ba=100*(w@((gp != yy) & (sp == yy))),
                                    grdh_right_slc_wrong_ba=100*(w@((gp == yy) & (sp != yy))),
                                    disagreement_percent=100*np.mean(gp != sp)))
    table = pd.DataFrame(results)
    main_table = pd.read_csv(ROOT / "product_pairs.csv")
    main_table = main_table[(main_table.fold == 2) & (main_table.arm == "T") & (main_table["head"] == "expert")]
    keys = ["fold", "cell", "draw", "scope", "arm", "head"]
    compared = table.merge(main_table, on=keys, validate="one_to_one", how="outer", indicator=True,
                            suffixes=("_audit", "_main"))
    assert compared._merge.eq("both").all()
    errors = {}
    for key in table.columns.difference(keys):
        np.testing.assert_allclose(compared[key+"_audit"], compared[key+"_main"], atol=1e-10, rtol=0)
        errors[key] = float(np.abs(compared[key+"_audit"]-compared[key+"_main"]).max())
    pd.DataFrame(query_counts).to_csv(OUT / "independent_product_pair_counts.csv", index=False)
    table.to_csv(OUT / "independent_product_pair_metrics.csv", index=False)
    receipt = dict(status="passed", unique_datatake_pairs=unique_n, exact_time_pairs=len(pairs),
                   same_label_pairs=same_n, label_conflicts=len(pairs)-same_n,
                   alignment="Rebuilt from metadata identity and product-parsed datatake; raw SAR timestamp reparsed",
                   metric_scope="OSS2 fold2, T expert, four cells, five draws, paired_all and paired_both_clean",
                   compared_rows=len(table), max_errors=errors, query_pair_counts=query_counts,
                   limitations="Natural pairs; raw XML extraction not repeated; no independent product-causal claim")
    (OUT / "PRODUCT_PAIR_AUDIT.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
