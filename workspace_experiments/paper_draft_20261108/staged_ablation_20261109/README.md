# Staged retraining of current SAR modules

See `ROUND_FINDINGS_ZH.md` for completed findings and `PROTOCOL_ZH.md` for the pre-training design. This experiment keeps the formal working model unchanged. It trains the surviving current operators from a common ImageNet initialization; it is not an exact replay of every historically replaced operator.

Local execution order (existing ClearSAR environment):

1. `intervention_inputs.py` prepares frozen-prefix source intervention inputs.
2. `run_round.py` runs bounded development selection and the three-seed matched repetitions.
3. `select_and_evaluate.py --evaluate` reloads frozen choices, checks development replay, then evaluates.
4. `analyze.py` produces paired, class, scene and availability results and the initial native-input gallery.
5. `followup.py` performs the fixed-weight M3 switch and selected outcome inspection without retraining.
6. `make_gallery.py` assembles the local reviewed evidence page.

The code reuses the immediately preceding `../experiment_completion_20261109` implementation and its documented current model/data dependencies. It is a scoped experiment archive, not a new self-contained replacement package.

All 21 fits and 18 selected-model evaluations completed. Three unselected development scouts are retained locally. Every selected fit has its own model-specific intervention teacher and all five stages. Training-loop timing excludes teacher creation and some preparation/evaluation operations; it is not end-to-end runtime or inference latency.

Git archive includes code, protocol, aggregate results and analysis reports. Weights, arrays, original images, local galleries, per-sample teacher records and raw manifests remain local. The local gallery is `index.html`; its accompanying path/score records remain in `case_records.json` and `followup_cases.json`.
