# Gradient and update-magnitude diagnostic

|Dataset|Model|Orders|OA|BA|NLL|Brier|
|---|---|---|---|---|---|---|
|private|saved_baseline|3|45.000000|35.733663|2.430989|0.840963|
|private|adapt_only|3|44.166667|35.675840|2.394498|0.844893|
|private|source_mix|3|44.375000|35.866316|2.402862|0.845614|
|private|norm_control|3|44.166667|35.339105|2.396897|0.841648|
|oss2|saved_baseline|3|58.479532|49.636699|1.381566|0.618582|
|oss2|adapt_only|3|58.333333|49.413542|1.410364|0.622839|
|oss2|source_mix|3|57.894737|49.242794|1.405884|0.621768|
|oss2|norm_control|3|58.333333|49.534456|1.403968|0.621708|

Gradient comparisons are at seed17 complete-model endpoints; no development gradient enters training.
A cosine is a local direction diagnostic, not a classification result or a causal explanation.

private:
{"adapt_fit": -0.5536736845970154, "adapt_development": 0.16500142216682434, "fit_development": 0.4794900417327881}
Matched present classes: {"adapt_fit": -0.5063130855560303, "adapt_development": 0.2189769744873047, "fit_development": 0.5850614905357361}

oss2:
{"adapt_fit": -0.1320662796497345, "adapt_development": 0.9005833864212036, "fit_development": -0.23897358775138855}
Matched present classes: {"adapt_fit": -0.1320662796497345, "adapt_development": 0.9005833864212036, "fit_development": -0.23897358775138855}
