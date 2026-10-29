# M3 repair fixed-endpoint results

Historical development160 / 3 passes / 7 present classes. Source-fit continuation only. No new blind test.
Same-checkpoint compare_only/value_only swaps are diagnostics, not retrained contenders.

| Condition | OA | BA | NLL | Brier |
|---|---:|---:|---:|---:|
| after_control | 40.8333 +/- 0.9547 | 34.2501 +/- 2.3728 | 2.6666 +/- 0.0972 | 0.9047 +/- 0.0155 |
| after_fast_compare | 40.8333 +/- 0.9547 | 34.2501 +/- 2.3728 | 2.6671 +/- 0.0968 | 0.9047 +/- 0.0156 |
| after_metric | 40.8333 +/- 0.9547 | 34.2501 +/- 2.3728 | 2.6648 +/- 0.0977 | 0.9043 +/- 0.0158 |
| before_control | 45.0000 +/- 1.0825 | 35.7337 +/- 0.7401 | 2.4310 +/- 0.1622 | 0.8410 +/- 0.0181 |
| compare_only_control | 45.0000 +/- 1.0825 | 35.7337 +/- 0.7401 | 2.4310 +/- 0.1622 | 0.8410 +/- 0.0181 |
| compare_only_fast_compare | 45.0000 +/- 1.0825 | 35.7337 +/- 0.7401 | 2.4310 +/- 0.1622 | 0.8410 +/- 0.0181 |
| compare_only_metric | 45.0000 +/- 1.0825 | 35.7337 +/- 0.7401 | 2.4305 +/- 0.1624 | 0.8409 +/- 0.0180 |
| value_only_control | 40.8333 +/- 0.9547 | 34.2501 +/- 2.3728 | 2.6666 +/- 0.0972 | 0.9047 +/- 0.0155 |
| value_only_fast_compare | 40.8333 +/- 0.9547 | 34.2501 +/- 2.3728 | 2.6671 +/- 0.0969 | 0.9047 +/- 0.0156 |
| value_only_metric | 40.8333 +/- 0.9547 | 34.2501 +/- 2.3728 | 2.6672 +/- 0.0996 | 0.9047 +/- 0.0166 |

## Outcome events (three orders, not independent ships)

- after_control minus before_control: rescued=16, harmed=36, CC=180, WW=248; all prediction changes=83
- after_fast_compare minus after_control: rescued=0, harmed=0, CC=196, WW=284; all prediction changes=0
- after_metric minus after_control: rescued=0, harmed=0, CC=196, WW=284; all prediction changes=0
- after_metric minus before_control: rescued=16, harmed=36, CC=180, WW=248; all prediction changes=83
- compare_only_metric minus before_control: rescued=0, harmed=0, CC=216, WW=264; all prediction changes=1
- value_only_metric minus before_control: rescued=16, harmed=36, CC=180, WW=248; all prediction changes=83
