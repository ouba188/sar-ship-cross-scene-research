"""Shuffle matched branch response changes within class/acquisition blocks."""

from product_pair_chain import *


def stats(a,b,start,y):
    changed = a+b
    pred = (start+changed).argmax(-1)
    da,db = margin(a,y),margin(b,y)
    return dict(ba=100*macro((pred == y).astype(float),y),oa=100*np.mean(pred == y),
                cancel=np.mean(1-np.abs(da+db)/(np.abs(da)+np.abs(db)+1e-12)),
                opposite=100*np.mean(da*db < 0))


def main():
    pairs = pd.read_csv(OUT / "product_pairs_frozen.csv",keep_default_na=False)
    with np.load(OUT / "product_cohort.npz") as z:
        ev,g,s = z["eval_masks"],z["grdh"],z["slc"]
    common = ev.all(0)
    eligible = common[g] & common[s]
    clean = (pairs.noise_xml_grdh.astype(str) == "0") & (pairs.noise_xml_slc.astype(str) == "0")
    scores = {}
    for fold in (2,3):
        with np.load(OUT / f"product_scores_fold_{fold}.npz") as z:
            scores[fold] = (z["a"],z["b"])
    random_rows, observed_rows, coverage = [],[],[]
    for scope,mask in (("common_all",eligible),("common_clean",eligible & clean.to_numpy())):
        selected = pairs[mask].reset_index(drop=True)
        y = selected.class_id_grdh.to_numpy(int)
        blocks = list(selected.groupby(["class_id_grdh","datatake_key"]).indices.values())
        movable = sum(len(x) for x in blocks if len(x)>1)
        coverage.append(dict(scope=scope,pairs=len(selected),identities=selected.identity.nunique(),
            datatakes=selected.datatake_key.nunique(),classes=len(np.unique(y)),
            blocks=len(blocks),exchangeable_pairs=movable))
        states = []
        for fold,(aa,bb) in scores.items():
            for draw in range(5):
                a,b = aa[draw,1,mask]-aa[draw,0,mask],bb[draw,1,mask]-bb[draw,0,mask]
                start = aa[draw,0,mask]+bb[draw,0,mask]
                state = (a,b,start)
                states.append(state)
                observed_rows.append(dict(scope=scope,fold=fold,draw=draw,**stats(*state,y)))
        rng = np.random.default_rng(20261013)
        for iteration in range(500):
            order = np.arange(len(selected))
            for block in blocks:
                order[block] = rng.permutation(block)
            values = [stats(a,b[order],start,y) for a,b,start in states]
            random_rows.append(dict(scope=scope,iteration=iteration,
                **{k:np.mean([v[k] for v in values]) for k in values[0]}))
    observed = pd.DataFrame(observed_rows)
    null = pd.DataFrame(random_rows)
    observed.to_csv(OUT / "correspondence_observed.csv",index=False)
    null.to_csv(OUT / "correspondence_shuffled.csv",index=False)
    pd.DataFrame(coverage).to_csv(OUT / "correspondence_coverage.csv",index=False)
    results = []
    for scope,df in null.groupby("scope"):
        obs = observed[observed.scope == scope]
        for metric in ("ba","oa","cancel","opposite"):
            actual = obs[metric].mean()
            results.append(dict(scope=scope,metric=metric,actual=actual,shuffled_mean=df[metric].mean(),
                actual_minus_shuffled=actual-df[metric].mean(),
                shuffled_q025=df[metric].quantile(.025),shuffled_q975=df[metric].quantile(.975)))
    table = pd.DataFrame(results)
    table.to_csv(OUT / "correspondence_summary.csv",index=False)
    print(pd.DataFrame(coverage).to_string(index=False),flush=True)
    print(table.to_string(index=False),flush=True)


if __name__ == "__main__":
    main()
