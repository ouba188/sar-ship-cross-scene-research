# Episodic Spatial Pooling: Completed Pilot

Six fixed development target ports; five original64-label draws each; 5% expert handoff. All target policies froze before query-label readout. The original main model was not modified.

| Arm | OA % | BA % | Expert BA % | Mean Admissions |
|---|---:|---:|---:|---:|
| original | 44.5655 | 45.0082 | 48.6416 | 138.83 |
| matched_refit | 44.4739 | 45.0519 | 48.6416 | 138.83 |
| fixed_ordinary | 44.4438 | 44.9900 | 48.6018 | 138.83 |
| fixed_stable | 44.4772 | 45.0303 | 48.5694 | 138.83 |
| learned_ordinary | 44.5567 | 45.0351 | 48.4520 | 138.83 |
| learned_stable | 44.5473 | 45.0572 | 48.5548 | 138.83 |

Metrics average five draws within each port, then weight ports equally. Overlapping draws are not independent replications. The six-port means must not be compared to the full23-port means.

## Fixed Comparisons

| Treatment minus control | OA pp | BA pp | Expert BA pp |
|---|---:|---:|---:|
| fixed_ordinary - matched_refit | -0.0302 | -0.0619 | -0.0398 |
| fixed_stable - matched_refit | +0.0033 | -0.0216 | -0.0723 |
| learned_ordinary - fixed_ordinary | +0.1130 | +0.0452 | -0.1498 |
| learned_stable - fixed_stable | +0.0700 | +0.0269 | -0.0145 |
| learned_stable - matched_refit | +0.0733 | +0.0053 | -0.0868 |
| learned_stable - original | -0.0182 | +0.0490 | -0.0868 |
| learned_stable - learned_ordinary | -0.0095 | +0.0220 | +0.1028 |

| Port | Learned stable vs matched BA pp | Learned stable vs fixed stable BA pp |
|---|---:|---:|
| Antwerp-Bruges | +0.0701 | +0.0000 |
| Jebel Ali | +0.0530 | +0.0750 |
| Port Klang | +0.0095 | +0.0251 |
| Santos | -0.0852 | -0.0302 |
| Singapore | -0.0194 | +0.0891 |
| Tanger Med | +0.0035 | +0.0023 |

## Mechanism Actually Run

1. Reuse frozen R50 four-token maps, centered within each chip and projected onto128 PCA components. The ordinary or candidate-stable source direction for the source-base predicted top-two classes supplies fixed token logits.
2. Add a shared128->16 tanh->1 token scorer. Its output layer starts at zero, so initialization exactly retains the nonuniform fixed attention. This avoids a zero residual kernel with no first-order training signal.
3. Softmax over four tokens yields a mean-zero spatial residual, scaled by the prior source RMS. Add only its .25-weight kernel to the original expert kernel. Do not add the previous auxiliary source-score correction.
4. Refit the same class-balanced ridge8 correction from64 support labels. Backpropagate balanced source-query cross-entropy through the differentiable linear solve to train pooling. Original expert prior, backbone and base scores remain fixed.
5. Source episodes at indices divisible by3 are held for epoch selection; the others train pooling. Eight epochs, AdamW .005, weight decay .001, no hyperparameter sweep. Choose maximum equal-source-port validation BA, tie minimum balanced CE; epoch0 is eligible. Query sampling is at most32/class for training and64/class for validation.
6. Fit the original four-confidence binary source gate separately for each arm. Apply the unchanged disagreement, observed-support-class, positive-gate-score and top5% handoff rules.

These heads have2080 trainable parameters. There is no physical or causal interpretation claim.

## Minimal Root-Cause Diagnosis

| Source split | Arm | Expert BA % | Final BA % |
|---|---|---:|---:|
| train | fixed_ordinary | 56.4462 | 51.3121 |
| train | fixed_stable | 56.3920 | 51.3561 |
| train | learned_ordinary | 56.9623 | 51.4497 |
| train | learned_stable | 56.5857 | 51.1513 |
| train | matched_refit | 56.4822 | 51.3585 |
| validation | fixed_ordinary | 58.1373 | 52.4356 |
| validation | fixed_stable | 58.1538 | 52.4457 |
| validation | learned_ordinary | 58.3259 | 52.4752 |
| validation | learned_stable | 58.2866 | 52.4590 |
| validation | matched_refit | 58.2167 | 52.4426 |

The validation rows here use all available source-query examples; checkpoint selection used a fixed balanced subset. Neither table is an independent test: validation selected the checkpoint, and source gate metrics reuse gate fitting rows.

| Target fold | Direction | Selected epoch | Validation BA change pp | Maximum training gradient norm |
|---|---|---:|---:|---:|
| Antwerp-Bruges | ordinary | 7 | +0.0482 | 0.09656 |
| Antwerp-Bruges | stable | 0 | +0.0000 | 0.09601 |
| Jebel Ali | ordinary | 5 | +0.0179 | 0.11556 |
| Jebel Ali | stable | 4 | +0.2665 | 0.09628 |
| Port Klang | ordinary | 5 | +0.7638 | 0.12055 |
| Port Klang | stable | 1 | +0.3163 | 0.08622 |
| Santos | ordinary | 6 | +0.3225 | 0.20869 |
| Santos | stable | 1 | +0.1383 | 0.15164 |
| Singapore | ordinary | 5 | +0.1108 | 0.15337 |
| Singapore | stable | 5 | +0.0123 | 0.17505 |
| Tanger Med | ordinary | 5 | +0.1259 | 0.10528 |
| Tanger Med | stable | 1 | +0.1515 | 0.08502 |

- Implementation: five focused tests pass, including comparison with the existing NumPy ridge solver, finite-difference checking of gradients through support adaptation, live pooling gradients, and support/query disjoint sampling.
- Integration: original expert replay is checked for132 source episodes and30 target draws. The source prior is no longer altered; only the support kernel changes. The three previously diagnosed Antwerp-outer source episodes reproduce kernel-only expert BA: ordinary44.6369%, stable44.6592%. No attention uniformity/no-gradient explanation remains.
- Effect: removing the auxiliary prior eliminates the prior pilot's roughly5pp expert-BA regression. Nevertheless, kernel-only fixed pooling remains slightly below the matched unaugmented expert. Learned pooling improves some source expert metrics but does not increase target expert BA in either direction relative to fixed pooling.
- Routing: learned-stable final BA is only about+.005pp versus its matched gate-refit control, with slightly higher OA. Versus the historical original, BA rises slightly but OA falls slightly. This is not a meaningful new overall improvement.
- Remaining uncertainty: source-to-target transfer of the learned token weighting is not established. The limited four-token representation, the predicted-pair selector, and the strength/expressiveness of the extra kernel remain possible limits; this run does not isolate one as the unique cause. Do not claim the entire direction is impossible.

## Boundary And Decision

Outer-source-only encoder, geometry, candidate directions and scaling include the meta-validation source ports. Only the pooling optimizer excludes their query labels. Gates reuse all source episodes, including pooling-training queries. This is not fully nested OOF and is a repeatedly used development set, not new confirmation. No target query labels choose hyperparameters or checkpoints; target fitting uses only each original64-label support.

Keep this as a bounded negative/tiny-effect development experiment. Do not promote it, combine it into the base-only pilot, or expand to other datasets without a better justified recipe. Preserve all earlier negative results.
