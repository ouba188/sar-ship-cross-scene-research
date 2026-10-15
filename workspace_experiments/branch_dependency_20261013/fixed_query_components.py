"""Locate query-side score changes with the T support coefficients fixed."""

from support_query_factorial import *


def main():
    data = {c: run.target("oss2", c) if c == "original" else slc_geometry.data_for("oss2", c)
            for c in CONDITIONS}
    original = data["original"]
    reference = np.unique(original["support"])
    locations = [np.searchsorted(reference, s) for s in original["support"]]
    n = len(original["ids"])
    replays = []
    for fold in (2, 3):
        with (PREVIOUS / f"fits/original/fold_{fold}/geometry.pkl").open("rb") as f:
            bundle = pickle.load(f)
        refs = {c: run.encode(d, reference, bundle) for c, d in data.items()}
        coefficients = {}
        for c, f in refs.items():
            kernel = f["r"] @ f["r"].T + 1. + f["t"] @ f["t"].T
            for draw, loc in enumerate(locations):
                coefficients[c, draw] = run.evaluate.coefficients(kernel[np.ix_(loc,loc)],
                    f["pr"][loc], original["support_labels"][draw], 8.)
        probabilities = np.lib.format.open_memmap(OUT / f"fold_{fold}/component_probabilities.npy",
            mode="w+", dtype=np.float32, shape=(2, 2, 8, 5, n, 8))
        factorial = np.load(OUT / f"fold_{fold}/probabilities.npy", mmap_mode="r")
        for vi, variant in enumerate(CONDITIONS[1:]):
            for lo in range(0, n, 512):
                part = slice(lo, min(lo + 512, n))
                fq = [run.encode(data[c], part, bundle) for c in ("original", variant)]
                for si, support in enumerate(("original", variant)):
                    fr = refs[support]
                    kr = [f["r"] @ fr["r"].T + 1. for f in fq]
                    kt = [f["t"] @ fr["t"].T for f in fq]
                    for mask in range(8):
                        prior_bit, r_bit, t_bit = mask & 1, (mask >> 1) & 1, (mask >> 2) & 1
                        kernel = kr[r_bit] + kt[t_bit]
                        for draw, loc in enumerate(locations):
                            logits = fq[prior_bit]["pr"] + kernel[:,loc] @ coefficients[support,draw]
                            probabilities[vi,si,mask,draw,part] = softmax(logits, axis=1)
            probabilities.flush()
            for si, support_i in enumerate((0, vi + 1)):
                for mask, query_i in ((0,0), (7,vi+1)):
                    a = np.asarray(probabilities[vi,si,mask])
                    b = np.asarray(factorial[support_i,query_i,:,1])
                    np.testing.assert_allclose(a,b,atol=2e-7,rtol=0.)
                    np.testing.assert_array_equal(a.argmax(-1),b.argmax(-1))
                    replays.append(dict(fold=fold,variant=variant,support=CONDITIONS[support_i],
                                        mask=mask,max_error=float(np.max(np.abs(a-b)))))
        print("COMPONENT_FROZEN",fold,flush=True)
    (OUT / "COMPONENT_FROZEN.json").write_text(json.dumps(dict(replays=replays,
        bit_order=["query_prior","query_r50_kernel","query_r18_kernel"],
        fixed_coefficients_within_support=True,query_labels_used_for_fit=False),indent=2),encoding="utf-8")
    yall = run.labels("oss2", original["ids"])
    frame = metadata("oss2", original["ids"])
    slc = frame.scene_id.str.contains("_SLC_",regex=False).to_numpy()
    rows, classrows = [], []
    for fold in (2,3):
        probs = np.load(OUT / f"fold_{fold}/component_probabilities.npy",mmap_mode="r")
        for draw,ev in enumerate(original["eval_masks"]):
            for scope, selected in (("full",ev),("SLC",ev & slc),("GRDH",ev & ~slc)):
                y = yall[selected]
                for vi,variant in enumerate(CONDITIONS[1:]):
                    baseline = probs[vi,0,0,draw,selected].argmax(1)
                    for si,support in enumerate(("original",variant)):
                        for mask in range(8):
                            p = np.asarray(probs[vi,si,mask,draw,selected],float)
                            pred = p.argmax(1)
                            w,rescue,harm = weighted_events(y,baseline,pred)
                            meta = dict(fold=fold,draw=draw,scope=scope,variant=variant,support=support,
                                mask=mask,prior=bool(mask & 1),r50_kernel=bool(mask & 2),r18_kernel=bool(mask & 4))
                            rows.append(dict(**meta,ba=w[pred == y].sum(),delta_ba=w[rescue].sum()-w[harm].sum(),
                                oa=100*np.mean(pred == y),rescue=w[rescue].sum(),harm=w[harm].sum()))
                            for cls in np.unique(y):
                                cm = y == cls
                                classrows.append(dict(**meta,class_id=int(cls),n=int(cm.sum()),
                                    recall=100*np.mean(pred[cm] == cls),
                                    contribution=w[rescue & cm].sum()-w[harm & cm].sum()))
    keys = ["scope","variant","support","mask","prior","r50_kernel","r18_kernel"]
    for name,records,group,values in (("components",rows,keys,["ba","delta_ba","oa","rescue","harm"]),
                                    ("component_classes",classrows,keys+["class_id"],["n","recall","contribution"])):
        df = pd.DataFrame(records)
        df.to_csv(OUT / (name+".csv"),index=False)
        df.groupby(group)[values].mean().reset_index().to_csv(OUT / (name+"_summary.csv"),index=False)
    print(pd.read_csv(OUT / "components_summary.csv").query(
        "scope == 'SLC' and variant == 'range_geometry'").to_string(index=False),flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
