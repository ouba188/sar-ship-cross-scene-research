"""Check probability replay, old metric anchors, and noncollapsed stability."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
old = pd.read_csv(OUT.parent / "ports_root_cause_20261013/product_pairs.csv")
old = old[(old.cell == "OO") & (old.arm == "T") & (old["head"] == "expert")]
new = pd.read_csv(OUT / "product_swaps.csv")
errors = []
for scope, previous_scope in (("all","paired_all"),("both_clean","paired_both_clean")):
    previous = old[old.scope == previous_scope].set_index(["fold","draw"])
    for swap, prefix in (("GG","grdh"),("SS","slc")):
        current = new[(new.scope == scope) & (new.swap == swap)].set_index(["fold","draw"])
        for metric in ("ba","oa"):
            diff = current[metric] - previous[prefix+"_"+metric]
            errors.extend(abs(diff).tolist())
            np.testing.assert_allclose(diff,0,atol=1e-10,rtol=0.)
events = pd.read_csv(OUT / "product_pair_events.csv")
np.testing.assert_allclose(events.mean_correct,.5*(events.grdh_correct+events.slc_correct),atol=0,rtol=0.)
np.testing.assert_allclose(1-events.flip,events.both_correct+events.stable_wrong,atol=0,rtol=0.)
np.testing.assert_allclose(events.grdh_correct-events.slc_correct,
                           events.lost_correct-events.gained_correct,atol=0,rtol=0.)
replays = json.loads((OUT / "PRODUCT_FROZEN.json").read_text(encoding="utf-8"))["replays"]
receipt = dict(status="passed",old_metric_anchors=len(errors),max_metric_error=max(errors),
               replay_contexts=len(replays),max_probability_error=max(x["max_error"] for x in replays),
               event_rows=len(events),endpoint_and_correct_stability_identities=True)
(OUT / "PRODUCT_VERIFICATION.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
print(json.dumps(receipt,indent=2))
