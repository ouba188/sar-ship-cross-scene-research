# Query-common shift diagnostic

Not a deployable candidate: the common offset is estimated on development predictions, without labels.
Energy fraction concerns changes in centered logits, not physical information or explained classification accuracy.
This measures the new training-induced change, not the entire original M3 contribution.

|Data|Seed|Arm|Shared energy|Same prediction|Same among changed|Actual OA/BA|Common-only OA/BA|
|---|---|---|---|---|---|---|---|
|private|17|adapt_only|0.931153|157/160|12/13|45.000000/34.678108|43.750000/33.581169|
|private|17|source_mix|0.923984|158/160|12/12|45.625000/35.249536|45.000000/35.045455|
|private|29|adapt_only|0.930696|156/160|15/17|43.125000/35.567254|42.500000/34.628479|
|private|29|source_mix|0.929496|157/160|11/12|43.125000/35.567254|41.875000/34.791744|
|private|43|adapt_only|0.935750|158/160|9/9|44.375000/36.782158|43.125000/36.006648|
|private|43|source_mix|0.936492|159/160|9/9|44.375000/36.782158|43.750000/36.210730|
|oss2|17|adapt_only|0.965002|225/228|12/14|57.894737/49.179624|58.771930/51.128649|
|oss2|17|source_mix|0.944335|225/228|11/12|57.894737/49.760794|58.771930/51.128649|
|oss2|29|adapt_only|0.968527|227/228|7/7|57.894737/49.211911|58.333333/50.298867|
|oss2|29|source_mix|0.968371|227/228|6/7|57.017544/47.837598|57.456140/48.124954|
|oss2|43|adapt_only|0.955600|226/228|6/7|59.210526/49.849092|59.210526/49.849092|
|oss2|43|source_mix|0.971437|228/228|11/11|58.771930/50.129991|58.771930/50.129991|
