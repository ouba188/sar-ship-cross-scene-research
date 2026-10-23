# Learned conditional reconstruction protocol

Written before training, 2026-10-24. Scope: this learned/ directory only. No Git, raw-image review, external data, or hyperparameter search. Parent handles literature and image review. This is a bounded algorithm learnability correction, not a strong novelty claim.

## Rationale and fixed comparisons

Parent fixed stage completed 30 heads. Clean full OA/BA = 58.8316/40.1223%; full+excess = 59.0378/39.3581%: OA +0.2062 pp but BA -0.7642 pp. Utility diagnostic shows positive within-image held-edge reconstruction, yet even the high-skill case can align more strongly with a different-class reference. Local predictability is not class specificity. Prior reference-relative metric retained a modest BA advantage; it motivates learning relative evidence without injecting an uncalibrated fixed score, not a guarantee of transfer or novelty. Existing two modules remain the clean full baseline.

Center frozen PCA32 unit tokens B,N over legal paired edges, then transform without renormalizing: B'=B Mb, N'=N Mn. Mb=I+Ub Vb.T, Mn=I+Un Vn.T, each U,V has shape 32x4. Shared across all groups, images and classes; 512 transform parameters plus four coefficients. A CPU generator with seed17 initializes Ub then Un with Normal(0,.01); Vb,Vn=0. a=.25. The three configurations17/29/43 retain their existing frozen full logits and use this same deterministic transform initialization, matching the earlier relative-stage convention.

For each query/reference/group: A=Bq Bs.T (Bs Bs.T+lambda I)^-1, lambda=max(||Bs||F^2/ms,1e-8), pred=A Ns. T=(2<Nq,pred>-||pred||F^2)/mq. V=||Ns||F^2/(ms-1)*(||A||F^2-||A1||^2/ms)/mq. Scores: transfer=T, excess=T+V. Padded invalid edges are zero; references and queries with fewer than two legal edges are excluded. Valid references are averaged equally within class; valid classes are centered per group.

Five arms: independent, transfer, excess, excess_shuffle101, excess_shuffle307. Independent is the mean of body-to-body AND neighbor-to-neighbor ridge reconstruction improvements, so both transforms are active. This deliberately differs from parent's neighbor-only independent arm. Shuffles replay existing edge.edges on the complete original row order with each fixed seed, then center; they do not change valid-edge sets or content multisets.

For each arm, compute utility-only RMS of class-centered scores at identity and freeze it. Ztheta uses that same divisor. Prediction=full+delta; delta=sum_g a_g*(Ztheta-Zidentity), recentered across classes. Initial delta is exactly zero; V data gradients are nonzero. Identity cache is detached, not a trained baseline. Never fit against ordinary anchor.

## Training and selection

15 fits = 5 arms x configurations17/29/43. Same utility644, selection612, evaluation485, frozen bank64. Adam lr.01, 200 full-utility optimizer steps, batch128 gradient accumulation. Per-sample weight=.5/644+.5/(8*class_count_utility[y]); all six batches contribute before one update. This equals dep.joint_loss over the complete utility, including the final short batch. Regularizer once per update: .01||a||^2+.01*(||Mb-I||F^2+||Mn-I||F^2)/32. a is regularized toward zero, as in the existing relative study.

Checkpoints0/20/50/100/200; exact existing J=(OA+BA)/2 on selection612, strict improvement, earliest checkpoint retained on ties. Evaluation485 only after the configuration/arm checkpoint is selected. No evaluation-driven tuning, lambda grid, new classifier branch, or external confirmation. This reused development cohort is not independent blind confirmation.

Vectorize all four groups and 64 references on GPU. Solve 8x8 padded systems; zero support rows make padding algebraically harmless. Compute reconstruction scores with small Gram products equivalent to explicit A Ns, avoiding a large prediction tensor. No per-batch CPU scalar extraction. Inference and numeric smoke checks may transfer results at bounded checkpoints.

## Required checks and outputs

Before training: identity transfer/excess agree with parent on actual utility rows and explicit partial masks; missing groups remain zero. Exact reference permutation mean of excess is zero on a small scalar case. Compare differentiable scores against unpadded scalar reconstruction. Independent equals mean(body and neighbor reconstruction), with nonzero data gradients for Vb and Vn. Initial delta is exactly zero. Compare accumulated global-weight loss with dep.joint_loss. Check representative transform gradients by finite difference, then train.

Save selected U/V/a/Mb/Mn and all checkpoint states; utility/selection curves; selected role metrics OA/BA/NLL/balanced_NLL/Brier, per-class metrics and rescue/harm/net against full, frozen scales, selected prediction/feature caches, summary and receipt. Document trace API and cache axes for parent image analysis. Preserve negative results; no claim that algebra or source gains establish a physical mechanism.
