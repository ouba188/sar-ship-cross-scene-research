# Source-only alignment diagnostic

|Data|Arm|Orders|OA|BA|NLL|Brier|
|---|---|---|---|---|---|---|
|private|saved_baseline|3|45.000000|35.733663|2.430989|0.840963|
|private|ce|3|43.958333|35.034117|2.393069|0.844141|
|private|rank|3|44.583333|34.947949|2.422526|0.835941|
|private|hybrid|3|44.791667|35.757473|2.386091|0.838890|
|private|ce_guard|3|44.375000|35.355184|2.416271|0.834798|
|private|rank_guard|3|44.375000|34.650330|2.430546|0.840252|
|private|hybrid_guard|3|45.000000|35.707689|2.417807|0.835269|
|oss2|saved_baseline|3|58.479532|49.636699|1.381566|0.618582|
|oss2|ce|3|58.479532|49.778014|1.402460|0.620205|
|oss2|rank|3|58.771930|49.746849|1.380658|0.616830|
|oss2|hybrid|3|58.625731|49.873799|1.398300|0.619320|
|oss2|ce_guard|3|58.625731|49.732484|1.382595|0.618802|
|oss2|rank_guard|3|58.479532|49.636699|1.377971|0.616504|
|oss2|hybrid_guard|3|58.625731|49.924055|1.381249|0.618227|

Shift uses source_adapt labels for class balance and source predictions only. No query centering and no target prior estimation.
This is output alignment, not evidence that matching learned new physical semantics. Final evaluation remains unused.
