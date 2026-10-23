# Seed17 Competitive Matching Cases

Scope: source evaluation485, reused development split. Coupled versus fixed.
Cases are minimum obs_id within each outcome, not estimates of population prevalence.

| Outcome | Group size | Selected obs | Shown rival |
| --- | ---: | ---: | --- |
| rescued | 8 | 2080 | fixed_actual_error |
| harmed | 4 | 5405 | coupled_actual_error |
| CC | 288 | 150 | coupled_highest_nontrue_not_prediction |
| WW | 185 | 220 | coupled_actual_error |

## Cached Evidence And Interpretation

case_margins.csv distinguishes a fixed comparison rival from each arm's strongest wrong class.
local_scores.csv and query_weights.csv preserve all eight candidates for the selected queries.
image_records.json records actual query/reference identities, paths, placement and mask masses.
The displayed refs come directly from scores.npz; no support reselection is performed.
Local scores and query masses alone do not establish transport changes or nonlinear-head attribution.

## Required Visual Follow-Up

For each case, record visible VV/VH facts, true/rival support availability and evidence changes.
Separate those observations from proposed causes; leave unexplained changes explicitly unresolved.
No physical mechanism or visual explanation is generated automatically by this script.
