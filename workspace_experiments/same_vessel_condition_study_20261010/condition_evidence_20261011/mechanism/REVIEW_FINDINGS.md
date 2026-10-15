# Independent Diagnostic Closeout

## Findings

- Scope stayed at TWO source-selected candidates: center_border_log_ratio/magnitude and middle_energy_share/signed. No additional hypothesis scan, classification deployment, or corrective sweep.
- Center magnitude is a FRAGILE LOCAL CANDIDATE, not an established invariant or causal factor. Excluding BOTH held-pair products from response fitting preserves aggregate MSE reductions of 11.647436% versus anchor, 8.483379% versus capacity, and 1.576817% versus sham. Original values were 7.137363%, 4.257157%, and 3.457771%.
- Removing held ship403086000 (three pairs; product/chemical tanker, Jebel Ali), WITHOUT refitting but WITH class/ship/pair reweighting, changes gains to -6.951341%, -14.988359%, and -0.212230%. This is the only one of43 single-ship removals that flips all three gains negative. Reweighting matters: these are not unchanged-weight contribution subtractions.
- Middle-share signed prediction remains negative after product exclusion: -21.928409%, -28.993776%, -25.402521% against the same controls. Do not advance this candidate from this round.
- Product exclusions comprise158 unique unordered product pairs and1264 small response fits, retaining339-384 training pairs,80-88 paired ships, and8 classes. Only the response layer excludes products; the historical CNN is not product-heldout. Both-new-scene evidence remains4 pairs/2ships/1class.
- Center large-angle>=5 versus near-angle<2 outcome comparisons use only port/class cells covering BOTH bins:10 fitting cells and5 held cells. Four of five held cells have greater large-gap magnitude. Common-ship comparisons also yield4/5, but Busan changes sign; the surviving positive cells differ. Coverage is sparse and does not identify causation.

## Review Boundaries

- No held metric enters source candidate selection. No B image evidence or true label enters response predictor inputs. Fit and evaluation use the same class/ship/pair weighting; capacity and condition arms have equal column counts, not identical hypothesis spaces.
- Source response CV does not nest upstream classifier fitting or confusion-pair screening. This developmental limitation is documented, not an independent full-pipeline generalization guarantee.
- Deterioration uses abs(d),d*d,mid,abs(d)*u. At fixed A evidence/context, swapping A/B angles leaves the conditional design unchanged: this family cannot express directional margin deterioration that depends on the sign of the angle change. Its negative result therefore does NOT rule out all angle-aware risk predictors. The separate next_view_error design is direction-sensitive; its negative result likewise concerns only the tested fixed family and data. No corrective sweep was performed.
- Magnitude describes evidence volatility, not classification error, reliability, or extracted causal factors.

## Tests Already Run

Working directory: the same_vessel_condition_study_20261010 project directory containing the test module.

Exact PowerShell command (same command for both completed runs):

```powershell
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -m unittest -v test_condition_evidence_response
```

Initial nine response checks completed before the main study. Exact terminal summary:

```text
----------------------------------------------------------------------
Ran 9 tests in 0.731s

OK
```

After the requested diagnostic scope was added, three focused checks were added for product exclusion, gain-contribution accounting, and comparable-cell coverage. Exact terminal summary:

```text
----------------------------------------------------------------------
Ran 12 tests in 0.924s

OK
```

The other nine checks cover synthetic signed response, synthetic condition-dependent magnitude, excluded-observation isolation for all arms, no-B-input prediction, equal-column controls, distinct magnitude/drop/error semantics, the documented deterioration symmetry, weight consistency, and held-truth-independent risk inputs/candidate pairs.

## Final Non-numerical Change

The diagnostic script now computes logaddexp context only at exported available observation rows. Other rows remain NaN, avoiding an unused-row warning. All used-row expressions and saved diagnostics are unchanged. No experiment or test rerun was performed for this scoped change.

The main agent separately reported112 exact exported prediction replays. Its float64-before-score-difference replay correction did not change main fitting; this was not an additional diagnostic experiment here.
