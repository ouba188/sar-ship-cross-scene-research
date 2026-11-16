# Cross-polarization 2x2 diagnostic

Three fixed seeds; paired dual-available OSS2 validation observations only. Not new module accuracy.
All interactions are below FP32 replay tolerance 1e-4. Tiny raw bootstrap intervals excluding zero have NO substantive interpretation here. The implemented score is structurally additive across polarizations.

|Region|Class|N|Mean interaction|Ship-bootstrap 95% interval|Mean absolute interaction|
|---|---|---:|---:|---|---:|
|body|Overall|206|-4.30e-08|[-1.21e-07, +2.98e-08]|6.22e-07|
|body|Bulk|82|-1.11e-07|[-1.98e-07, -1.87e-08]|5.16e-07|
|body|General cargo|14|+1.79e-07|[+1.89e-08, +3.35e-07]|5.00e-07|
|body|Chemical/product tanker|23|+6.01e-08|[-9.89e-08, +2.26e-07]|4.90e-07|
|body|Container|87|-4.21e-08|[-1.77e-07, +9.76e-08]|7.78e-07|
|ends|Overall|206|+4.66e-09|[-7.10e-08, +8.12e-08]|6.25e-07|
|ends|Bulk|82|+6.16e-09|[-7.49e-08, +8.87e-08]|5.05e-07|
|ends|General cargo|14|+9.82e-08|[-1.21e-07, +3.24e-07]|4.63e-07|
|ends|Chemical/product tanker|23|+1.78e-07|[+2.72e-08, +3.34e-07]|5.23e-07|
|ends|Container|87|-5.76e-08|[-1.92e-07, +8.09e-08]|7.92e-07|
|side|Overall|206|-3.82e-08|[-1.23e-07, +4.79e-08]|6.21e-07|
|side|Bulk|82|+3.18e-08|[-7.44e-08, +1.44e-07]|5.14e-07|
|side|General cargo|14|+3.65e-08|[-1.28e-07, +1.88e-07]|4.26e-07|
|side|Chemical/product tanker|23|+1.12e-07|[-7.03e-08, +3.07e-07]|5.80e-07|
|side|Container|87|-1.56e-07|[-3.10e-07, -9.87e-09]|7.64e-07|

## Prediction sensitivity

|Region|Seed|N|Original OA|Joint-operation OA|Additive-logit prediction OA|Joint/additive disagreements|
|---|---:|---:|---:|---:|---:|---:|
|body|17|206|68.93|55.83|55.83|0|
|body|29|206|67.96|56.80|56.80|0|
|body|43|206|63.59|51.46|51.46|0|
|ends|17|206|68.93|66.50|66.50|0|
|ends|29|206|67.96|67.48|67.48|0|
|ends|43|206|63.59|62.62|62.62|0|
|side|17|206|68.93|66.99|66.99|0|
|side|29|206|67.96|68.45|68.45|0|
|side|43|206|63.59|62.14|62.14|0|

Additive logits are a diagnostic counterprediction, not a deployable classifier or trained comparator.
A nonzero interaction identifies nonadditivity of this learned function under these operations, not a physical causal interaction or guaranteed PCEA benefit.
Signed interactions may cancel across ships. Absolute effects and per-class/per-pass/seed records are also retained.