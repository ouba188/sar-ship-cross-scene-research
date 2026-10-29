# Source guards: paired CPU results

Source adaptation is training-internal. Development/OSS2 validation is reported separately. All comparisons use the same IDs, labels and original logits within seed. The old-M3 CE comparator has the same step/sample budget but different trainable parameters. No held/full evaluation labels are read. No expansion is inferred from source protection.

## source

| Dataset | Seed | Arm | OA | BA | Rescue | Harm | Correct margin drops | Mean correct margin delta |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| private | 17 | baseline | 61.764706 | 51.752646 | 0 | 0 | 0/84 | +0.000000 |
| private | 17 | aligned | 63.970588 | 60.550214 | 5 | 2 | 45/84 | -0.021457 |
| private | 17 | ce | 61.029412 | 58.315527 | 4 | 5 | 59/84 | -0.224354 |
| private | 17 | fd_lm | 63.235294 | 52.679843 | 2 | 0 | 40/84 | +0.000300 |
| private | 17 | margin | 65.441176 | 61.013177 | 5 | 0 | 43/84 | -0.003205 |
| oss2 | 17 | baseline | 75.518672 | 70.378051 | 0 | 0 | 0/182 | +0.000000 |
| oss2 | 17 | aligned | 74.273859 | 69.418457 | 1 | 4 | 130/182 | -0.061665 |
| oss2 | 17 | ce | 76.348548 | 73.215027 | 11 | 9 | 93/182 | -0.060106 |
| oss2 | 17 | fd_lm | 75.933610 | 71.339590 | 1 | 0 | 94/182 | -0.003690 |
| oss2 | 17 | margin | 75.518672 | 70.796111 | 1 | 1 | 93/182 | +0.001010 |

| Dataset | Seed | Guard | Avoided aligned harm | Introduced harm | Retained aligned rescue | Lost aligned rescue | Additional rescue |
|---|---:|---|---:|---:|---:|---:|---:|
| private | 17 | fd_lm | 2 | 0 | 2 | 3 | 0 |
| private | 17 | margin | 2 | 0 | 5 | 0 | 0 |
| oss2 | 17 | fd_lm | 4 | 0 | 1 | 0 | 0 |
| oss2 | 17 | margin | 3 | 0 | 1 | 0 | 0 |

| Dataset | Seed | Comparison | Delta OA pp | Delta BA pp | Rescue/Harm | Top1 changes |
|---|---:|---|---:|---:|---|---:|
| private | 17 | fd_lm - baseline | +1.470588 | +0.927198 | 2/0 | 2 |
| private | 17 | fd_lm - aligned | -0.735294 | -7.870370 | 2/3 | 6 |
| private | 17 | fd_lm - ce | +2.205882 | -5.635684 | 6/3 | 13 |
| private | 17 | margin - baseline | +3.676471 | +9.260531 | 5/0 | 6 |
| private | 17 | margin - aligned | +1.470588 | +0.462963 | 2/0 | 2 |
| private | 17 | margin - ce | +4.411765 | +2.697650 | 6/0 | 9 |
| private | 17 | margin - fd_lm | +2.205882 | +8.333333 | 3/0 | 4 |
| oss2 | 17 | fd_lm - baseline | +0.414938 | +0.961538 | 1/0 | 1 |
| oss2 | 17 | fd_lm - aligned | +1.659751 | +1.921132 | 4/0 | 5 |
| oss2 | 17 | fd_lm - ce | -0.414938 | -1.875438 | 9/10 | 25 |
| oss2 | 17 | margin - baseline | +0.000000 | +0.418060 | 1/1 | 2 |
| oss2 | 17 | margin - aligned | +1.244813 | +1.377654 | 3/0 | 4 |
| oss2 | 17 | margin - ce | -0.829876 | -2.418916 | 8/10 | 24 |
| oss2 | 17 | margin - fd_lm | -0.414938 | -0.543478 | 0/1 | 1 |

## development

| Dataset | Seed | Arm | OA | BA | Rescue | Harm | Correct margin drops | Mean correct margin delta |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| private | 17 | baseline | 45.625000 | 34.928108 | 0 | 0 | 0/73 | +0.000000 |
| private | 17 | aligned | 45.625000 | 34.928108 | 0 | 0 | 41/73 | +0.000650 |
| private | 17 | ce | 44.375000 | 34.474026 | 2 | 4 | 37/73 | -0.044574 |
| private | 17 | fd_lm | 45.000000 | 34.724026 | 0 | 1 | 46/73 | -0.009027 |
| private | 17 | margin | 45.625000 | 34.928108 | 0 | 0 | 40/73 | +0.002792 |
| oss2 | 17 | baseline | 57.894737 | 50.123534 | 0 | 0 | 0/132 | +0.000000 |
| oss2 | 17 | aligned | 57.894737 | 50.704704 | 1 | 1 | 101/132 | -0.062209 |
| oss2 | 17 | ce | 57.017544 | 48.611368 | 4 | 6 | 55/132 | +0.031646 |
| oss2 | 17 | fd_lm | 57.894737 | 50.123534 | 0 | 0 | 60/132 | +0.001472 |
| oss2 | 17 | margin | 57.894737 | 50.123534 | 0 | 0 | 64/132 | +0.013490 |

| Dataset | Seed | Guard | Avoided aligned harm | Introduced harm | Retained aligned rescue | Lost aligned rescue | Additional rescue |
|---|---:|---|---:|---:|---:|---:|---:|
| private | 17 | fd_lm | 0 | 1 | 0 | 0 | 0 |
| private | 17 | margin | 0 | 0 | 0 | 0 | 0 |
| oss2 | 17 | fd_lm | 1 | 0 | 0 | 1 | 0 |
| oss2 | 17 | margin | 1 | 0 | 0 | 1 | 0 |

| Dataset | Seed | Comparison | Delta OA pp | Delta BA pp | Rescue/Harm | Top1 changes |
|---|---:|---|---:|---:|---|---:|
| private | 17 | fd_lm - baseline | -0.625000 | -0.204082 | 0/1 | 2 |
| private | 17 | fd_lm - aligned | -0.625000 | -0.204082 | 0/1 | 2 |
| private | 17 | fd_lm - ce | +0.625000 | +0.250000 | 4/3 | 14 |
| private | 17 | margin - baseline | +0.000000 | +0.000000 | 0/0 | 2 |
| private | 17 | margin - aligned | +0.000000 | +0.000000 | 0/0 | 0 |
| private | 17 | margin - ce | +1.250000 | +0.454082 | 4/2 | 12 |
| private | 17 | margin - fd_lm | +0.625000 | +0.204082 | 1/0 | 2 |
| oss2 | 17 | fd_lm - baseline | +0.000000 | +0.000000 | 0/0 | 0 |
| oss2 | 17 | fd_lm - aligned | +0.000000 | -0.581170 | 1/1 | 3 |
| oss2 | 17 | fd_lm - ce | +0.877193 | +1.512165 | 6/4 | 14 |
| oss2 | 17 | margin - baseline | +0.000000 | +0.000000 | 0/0 | 0 |
| oss2 | 17 | margin - aligned | +0.000000 | -0.581170 | 1/1 | 3 |
| oss2 | 17 | margin - ce | +0.877193 | +1.512165 | 6/4 | 14 |
| oss2 | 17 | margin - fd_lm | +0.000000 | +0.000000 | 0/0 | 0 |

## Files and limits

Each scope/dataset contains endpoints.csv, paired.csv, paired_means.csv, class_protection.csv, protection_vs_rescue.csv, id_seed.csv and summary.json. Correct-margin statistics always use original-baseline-correct queries. Protection decomposition conditions on original correctness and then compares with aligned. Avoided harm and lost rescue are distinct outcomes, not interchangeable gains. Per-case files retain every arm/seed and actual native references. Renderer status is not a claim of visual review. The authorized round ends at seed17: no 29/43 or final evaluation is scheduled by this script.
