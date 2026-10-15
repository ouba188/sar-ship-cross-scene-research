"""Same-ship, same-SAR-time GRDH/SLC diagnostic with unchanged full-task supports."""

from settings import *
from analyze_factorial import CELLS,load,weighted_events
from report import metrics


def main():
    assert (OUT / "PREDICTIONS_FROZEN.json").exists()
    pairs = pd.read_csv(OUT / "dataset_audit/oss2_grdh_slc_pairs.csv",keep_default_na=False)
    pairs = pairs[pairs.same_label].copy()
    records,detail,classes = [],[],[]
    for fold in FOLDS:
        for cell,conditions in CELLS.items():
            saved,prob = load(fold,"oss2",*conditions)
            ix = pd.Index(saved["ids"])
            grdh,slc = ix.get_indexer(pairs.sample_id_grdh),ix.get_indexer(pairs.sample_id_slc)
            assert (grdh >= 0).all() and (slc >= 0).all()
            yall = labels("oss2",saved["ids"])
            np.testing.assert_array_equal(yall[grdh],yall[slc])
            for draw,mask in enumerate(saved["eval_masks"]):
                eligible = mask[grdh] & mask[slc]
                for scope,selection in (("paired_all",eligible),("paired_both_clean",eligible &
                    (pairs.noise_xml_grdh.astype(str) == "0").to_numpy() & (pairs.noise_xml_slc.astype(str) == "0").to_numpy())):
                    if not selection.any():
                        continue
                    g,s = grdh[selection],slc[selection]
                    y = yall[g]
                    meta = dict(fold=fold,cell=cell,draw=draw,scope=scope)
                    for ai,arm in enumerate(ARMS):
                        for hi,head in enumerate(HEADS):
                            gp,sp = saved["predictions"][draw,ai,hi,g],saved["predictions"][draw,ai,hi,s]
                            gm = metrics(y,gp,np.asarray(prob[draw,ai,hi,g],float))
                            sm = metrics(y,sp,np.asarray(prob[draw,ai,hi,s],float))
                            w,rescue,harm = weighted_events(y,gp,sp)
                            records.append(dict(**meta,arm=arm,head=head,pairs=len(y),classes=len(np.unique(y)),
                                grdh_ba=gm["ba"],slc_ba=sm["ba"],slc_minus_grdh_ba=sm["ba"]-gm["ba"],
                                grdh_oa=gm["oa"],slc_oa=sm["oa"],
                                grdh_brier=gm["brier_macro"],slc_brier=sm["brier_macro"],
                                grdh_wrong_slc_right_ba=w[rescue].sum(),grdh_right_slc_wrong_ba=w[harm].sum(),
                                disagreement_percent=100*np.mean(gp != sp)))
                            for c in np.unique(y):
                                take = y == c
                                classes.append(dict(**meta,arm=arm,head=head,class_id=int(c),pairs=int(take.sum()),
                                    grdh_recall=100*np.mean(gp[take] == c),slc_recall=100*np.mean(sp[take] == c)))
                            if scope == "paired_all" and arm in ("R","T") and head == "expert":
                                for j,(gi,si) in enumerate(zip(g,s)):
                                    detail.append(dict(**meta,arm=arm,head=head,sample_id_grdh=saved["ids"][gi],
                                        sample_id_slc=saved["ids"][si],class_id=int(y[j]),grdh_correct=bool(gp[j] == y[j]),
                                        slc_correct=bool(sp[j] == y[j])))
    table = pd.DataFrame(records)
    table.to_csv(OUT / "product_pairs.csv",index=False)
    pd.DataFrame(detail).to_csv(OUT / "product_pair_events.csv",index=False)
    pd.DataFrame(classes).to_csv(OUT / "product_pair_classes.csv",index=False)
    keys = ["cell","scope","arm","head"]
    measures = [k for k in table.columns if k not in keys+["fold","draw"]]
    summary = table.groupby(keys)[measures].mean().reset_index()
    summary.to_csv(OUT / "product_pair_summary.csv",index=False)
    np.testing.assert_allclose(table.slc_minus_grdh_ba,table.grdh_wrong_slc_right_ba-table.grdh_right_slc_wrong_ba,atol=1e-10,rtol=0.)
    write(OUT / "PRODUCT_PAIR_COMPLETE.json",dict(same_label_pairs=len(pairs),rows=len(table),events=len(detail),
        supports_unchanged=True,model_fits=0,natural_pair_not_randomized=True))
    print(summary[(summary.scope == "paired_all") & (summary["head"] == "expert")].to_string(index=False))


if __name__ == "__main__":
    main()
