"""Prepare editable condition matrices from saved CSVs only (Python stdlib)."""

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCE = ROOT / "same_vessel_gallery_20261019/causal_frequency_side_20261024"
OUT = HERE / "condition_data"
R26 = "round26_polarization_coverage"
R22 = "round22_classification_coverage"
POLS = ("VV", "VH")
REGIONS = ("body", "side", "axis_end")
SCALES = ("fine", "mid")
DEFINITION = "[z_y(I(x))-z_c0(I(x))]-[z_y(x)-z_c0(x)]; c0=original joint-head strongest wrong class"
MODEL = "historical frozen R50 stem + C4 mean/max descriptors + class-balanced ridge diagnostic"
MEASURES = ("delta_margin", "level_delta_margin", "residual_delta_margin")
KEYS = ("scope", "dose", "readout", "polarization", "region", "scale", "class_id")


def read_csv(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalize(rows, experiment, class_names):
    output = []
    seen = set()
    for line, row in enumerate(rows, start=2):
        arm = row["arm"] if experiment == R26 else "joint"
        key = (row["obs_id"], row["pol"], row["region"], row["band"], row["dose"], arm)
        if key in seen:
            raise ValueError(f"Duplicate observation-condition-readout in {experiment}: {key}")
        seen.add(key)
        cid = int(row["class_id"])
        delta = row["delta"] if experiment == R26 else row["delta_fixed"]
        output.append(dict(
            experiment=experiment, obs_id=row["obs_id"], mmsi=row["mmsi"],
            overpass=row["overpass"], fold=int(row["fold"]), class_id=cid,
            class_name=class_names[cid], readout=arm, polarization=row["pol"].upper(),
            region={"end": "axis_end"}.get(row["region"], row["region"]),
            scale={"middle": "mid"}.get(row["band"], row["band"]), dose=float(row["dose"]),
            rival=int(row["rival"]), delta_margin=float(delta),
            level_delta_margin=float(row["level_delta"]),
            residual_delta_margin=float(row["residual"]), l2=float(row["l2"]),
            rescued=int(row["rescued"] == "True"), harmed=int(row["harmed"] == "True"),
            old=(row["old_case"] if experiment == R26 else row["previous_pilot_ship"]) == "True",
            source_csv=f"{experiment}/" + ("effects.csv.gz" if experiment == R26 else "interventions.csv"),
            source_csv_line=line,
        ))
    return output


def aggregate(rows, experiment):
    scopes = (("all_eligible", False), ("new_coverage", True)) if experiment == R26 else (
        ("all", False), ("expansion_only", True))
    groups = defaultdict(list)
    for scope, exclude_old in scopes:
        for row in rows:
            if exclude_old and row["old"]:
                continue
            key = (scope,) + tuple(row[k] for k in KEYS[1:])
            groups[key].append(row)
    results = []
    for key, group in sorted(groups.items()):
        record = dict(zip(KEYS, key))
        ships = defaultdict(list)
        for row in group:
            ships[row["mmsi"]].append(row)
        for metric in MEASURES:
            record[metric] = fmean(fmean(r[metric] for r in ship) for ship in ships.values())
        record.update(
            class_name=group[0]["class_name"], n_observations=len(group), n_ships=len(ships),
            n_overpasses=len({r["overpass"] for r in group}),
            n_folds=len({r["fold"] for r in group}), rescued=sum(r["rescued"] for r in group),
            harmed=sum(r["harmed"] for r in group),
            small_n_under_20_ships=len(ships) < 20,
            min_l2=min(r["l2"] for r in group), max_l2=max(r["l2"] for r in group),
            aggregation="mean over ships of within-ship observation means",
            metric_definition=DEFINITION, units="raw ridge score margin; not probability or percentage points",
            model=MODEL, dataset="OSS1 historical 8-class diagnostic cohort" if experiment == R26 else "historical six-port same-vessel candidate cohort",
            actual_readout="joint VV+VH images + 4 geometry features" if key[2] == "joint" else f"single {key[3]} image + 4 geometry features",
            region_source="end" if key[4] == "axis_end" else key[4],
            scale_source="middle" if key[5] == "mid" else key[5],
            source_csv=group[0]["source_csv"], experiment=experiment,
        )
        results.append(record)
    return results


def matrix(rows, metric):
    classes = sorted({(r["class_id"], r["class_name"]) for r in rows})
    lookup = {(r["readout"], r["polarization"], r["region"], r["scale"], r["class_id"]): r for r in rows}
    result = []
    for arm in ("joint", "single"):
        if not any(r["readout"] == arm for r in rows):
            continue
        for pol in POLS:
            for region in REGIONS:
                for scale in SCALES:
                    record = dict(readout=arm, polarization=pol, region=region, scale=scale,
                                  condition=f"{pol} / {region} / {scale}")
                    for cid, name in classes:
                        record[name] = lookup[arm, pol, region, scale, cid][metric]
                    result.append(record)
    return result


def verify(raw26, raw22, norm26, norm22, cells26, cells22):
    """Detect wrong margins, unmatched readouts, weights, counts or source mapping."""
    checks = []

    def check_values(name, pairs, tolerance=1e-12):
        errors = [abs(float(actual) - float(expected)) for actual, expected in pairs]
        if not errors or max(errors) > tolerance:
            raise ValueError(f"{name}: max error {max(errors) if errors else 'no comparisons'}")
        checks.append(dict(check=name, comparisons=len(errors), max_absolute_error=max(errors), tolerance=tolerance))

    base26 = {(r["obs_id"], r["arm"]): r for r in read_csv(SOURCE / R26 / "baseline_predictions.csv")}
    base22 = {r["obs_id"]: r for r in read_csv(SOURCE / R22 / "baseline.csv")}
    for label, raw, base in ((R26, raw26, base26), (R22, raw22, base22)):
        comparisons = []
        for row in raw:
            arm = row.get("arm", "joint")
            b = base[(row["obs_id"], "joint" if arm == "joint" else row["pol"])] if label == R26 else base[row["obs_id"]]
            y, rival = row["class_id"], row["rival"]
            expected = float(row[f"score_{y}"]) - float(row[f"score_{rival}"]) - float(b[f"score_{y}"]) + float(b[f"score_{rival}"])
            comparisons.append((row["delta"] if label == R26 else row["delta_fixed"], expected))
            if label == R26 and int(rival) != int(base[(row["obs_id"], "joint")]["own_rival"]):
                raise ValueError("Saved class pair is not the original joint rival")
        check_values(label + " fixed-pair delta reconstructed from saved scores", comparisons)

    pairs = defaultdict(dict)
    for row in norm26:
        pairs[(row["obs_id"], row["polarization"], row["region"], row["scale"], row["dose"])][row["readout"]] = row
    for pair in pairs.values():
        if set(pair) != {"joint", "single"}:
            raise ValueError("Missing readout partner")
        for field in ("rival", "mmsi", "class_id", "l2"):
            if pair["joint"][field] != pair["single"][field]:
                raise ValueError(f"Readout pairs disagree on {field}")
    checks.append(dict(check="same observation, operation, L2 and fixed rival across readouts", pairs=len(pairs), passed=True))

    lookup26 = {tuple(r[k] for k in KEYS): r for r in cells26}
    comparisons = []
    for report in read_csv(SOURCE / R26 / "class_detail.csv"):
        for arm in ("joint", "single"):
            key = (report["scope"], float(report["dose"]), arm, report["pol"].upper(),
                   {"end": "axis_end"}.get(report["region"], report["region"]),
                   {"middle": "mid"}.get(report["band"], report["band"]), int(report["class_id"]))
            row = lookup26[key]
            comparisons.extend([(row["delta_margin"], report["delta_" + arm]),
                                (row["residual_delta_margin"], report["residual_" + arm]),
                                (row["n_observations"], report["n"]), (row["n_ships"], report["ships"]),
                                (row["n_overpasses"], report["passes"])])
    check_values("round26 all 288 high-dose class/readout cells vs class_detail.csv", comparisons)
    lookup22 = {tuple(r[k] for k in KEYS): r for r in cells22}
    comparisons = []
    for report in read_csv(SOURCE / R22 / "by_class.csv"):
        key = (report["scope"], float(report["dose"]), "joint", report["pol"].upper(),
               {"end": "axis_end"}.get(report["region"], report["region"]),
               {"middle": "mid"}.get(report["band"], report["band"]), int(report["class_id"]))
        row = lookup22[key]
        comparisons.extend([(row["delta_margin"], report["ship_mean"]), (row["residual_delta_margin"], report["residual_mean"]),
                            (row["n_observations"], report["observations"]), (row["n_ships"], report["ships"])])
    check_values("round22 all 336 class cells vs by_class.csv", comparisons)

    # Overall ship means must be recomputed from observations, not averaged over class cells.
    comparisons = []
    for report in read_csv(SOURCE / R22 / "summary.csv"):
        selected = [r for r in norm22 if (report["scope"] == "all" or not r["old"])
                    and r["polarization"].lower() == report["pol"] and r["dose"] == float(report["dose"])
                    and r["region"] == {"end": "axis_end"}.get(report["region"], report["region"])
                    and r["scale"] == {"middle": "mid"}.get(report["band"], report["band"])]
        field = "delta_margin" if report["effect"] == "delta_fixed" else "residual_delta_margin"
        ships = defaultdict(list)
        for row in selected:
            ships[row["mmsi"]].append(row[field])
        comparisons.append((fmean(fmean(v) for v in ships.values()), report["ship_mean"]))
    check_values("round22 overall means vs summary.csv (includes E06 report anchors)", comparisons)

    anchors = []
    for cid in (0, 3, 4, 2, 5, 1):
        joint = lookup26["new_coverage", 1.0, "joint", "VH", "body", "fine", cid]
        single = lookup26["new_coverage", 1.0, "single", "VH", "body", "fine", cid]
        anchors.append(dict(class_name=joint["class_name"], observations=joint["n_observations"], ships=joint["n_ships"],
                            joint=joint["delta_margin"], single=single["delta_margin"],
                            joint_minus_single=joint["delta_margin"] - single["delta_margin"]))
    return dict(status="passed", checks=checks, round26_report_section3_anchors=anchors,
                action_on_failure="Stop export and correct selection, weights or margin field; never rerun a model.")


README = """# Condition effect data for editable figures

## Recommended figure

Use `condition_long.csv` (144 cells), or `condition_matrix.csv` (24 rows x 6 class columns).
Two facets: `joint` and `single`. Each facet contains all 12 combinations of VV/VH,
body/side/axis_end, and fine/mid. All six observed classes, negative effects, near-zero
effects, and positive effects are retained. The single facet means single VV for VV
rows and single VH for VH rows; it is not one fixed single-polarization head across rows.
Both single and joint heads retain four geometry features. Historical diagnostic,
NOT the current Full model. Both facets may share a zero-centered color scale.

Primary cohort: round26 `new_coverage`, high preset dose=1.0, 1165 observations,
969 vessels, 22 overpasses. This excludes six old observations, not all old vessels:
five retained observations belong to four old-case vessels; only 965 identities are new.
No selection uses effect sign or predicted success. Main rows use vessel-equal means:
first average all observations of each MMSI in the cell, then average those vessels.
Repeated doses, conditions and heads are never counted as independent samples.

`condition_n_ships_matrix.csv` and `condition_n_observations_matrix.csv` align exactly
with the effect matrix. The long table also gives overpass/fold counts, actual readout,
model, dataset, source CSV, metric definition, L2 range, and rescue/harm event counts.
Show small samples explicitly: crude-oil tanker has 11 vessels and fishing has 4;
`small_n_under_20_ships` is a descriptive plotting flag, not a statistical cutoff.
Missing classes tug/offshore supply have no eligible observations, not zero effect.

## One margin definition

For each observation let c0 be the strongest incorrect class under the ORIGINAL JOINT
head. For either readout h:
  delta_m_h = [z_h,y(I(x)) - z_h,c0(I(x))] - [z_h,y(x) - z_h,c0(x)].
Round26 uses saved `delta`; round22 uses saved `delta_fixed`. Positive values mean
this attenuation helps that fixed true-versus-rival score difference. They are raw
ridge decision-score margins, not probabilities or OA/BA percentage points.
`own_delta` and `decision_margin_delta` are deliberately not used in the matrices.
Readout changes include the already-fitted alternative diagnostic head, not just
removing a branch at inference; this preparation never fits or loads any model.

The saved same-mean control appears separately as `level_delta_margin` and
`residual_delta_margin = delta_margin - level_delta_margin` in long tables. The main
matrix uses raw delta_m only. A residual is not pure frequency causality: the control
matches mean change but not L2 or the full pixel distribution.

## Operation, dose and geometry

fine = I - Gaussian_sigma1(I); mid = Gaussian_sigma1(I) - Gaussian_sigma2(I),
sigma1=1 and sigma2=2 native pixels, reflect boundary. These are overlapping local
spatial-scale components, not independent electromagnetic frequency bands.
For mask M: taper=min(distance_transform(M)/2,1)*M;
d=clip(I-0.5*taper*component,0,1)-I; I'=I+dose*T*d/||d||2.
Round26 T=0.5*minimum proposal L2 over BOTH polarizations, all three regions, both
scales, and own/common masks. Only common-mask operations are evaluated. Dose 1.0
therefore means the higher preset matched-L2 dose, NOT complete removal or identical
absolute L2 across observations. All 12 conditions have equal L2 within an observation.
Round22 uses its own polarization mask and matches L2 across six region/scale
conditions within each polarization separately. Its matrix must remain separate.

body: detection OBB mask. axis_end maps raw `end`: outer strips extending 8 native
pixels beyond the axis ends, with transverse extent width/2+2, excluding body.
side: lengthwise span |u|<=length/2 and width/2<|v|<=width/2+8, excluding body.
Round26 uses intersection of corresponding VV/VH masks (>=8 pixels per region).
axis_end is not an identified bow/stern or wake. mid maps raw `middle`.
Original native crops/padding and geometry are retained; no new image processing occurs.

## Other deliverables

- `condition_all_doses_scopes_long.csv`: all 576 round26 aggregate cells, both doses
  (0.5/1.0), all-eligible and new-coverage scopes, both readouts. Do not pool scopes.
- `round26_selected_observations.csv`: the 27960 main-scope effect records, with MMSI,
  observation/fold/rival, L2 and original CSV line for traceability. Local data only.
- `condition_readout_difference_long.csv`: 72 paired ship-mean joint-minus-single
  effects. This is an alternative contrast, not the raw delta_m facet.
- `round22_condition_long.csv`: all 336 cells, 7 classes, two doses and scopes;
  `round22_condition_matrix.csv` and matching count matrices show only dose=1,
  expansion_only (209 observations, 104 ships, 31 overpasses, six ports).
  Historical full_global diagnostic, not the current model or the entire 24-port set.
- `validation.json`: score-based definition checks, paired-readout checks and exact
  comparisons against original class_detail/by_class/summary CSVs. No new CI is made.
- `provenance.json`: source files, code locations, definitions, exclusions and output map.

## Reproduce

Run `python prepare_conditions.py` from this figure directory, or use its absolute path.
Only Python's standard library is needed. The script reads existing CSV/CSV.GZ files
and writes only this condition_data directory. It imports no original experiment
modules, reads no weights/features/images, and performs no inference or training.
Validation fails before export if definitions, counts or report agreement disagree.
The original per-class high-dose estimates reproduce within 1e-12. Class composition
and small cells remain visible; no significance or natural physical cause is inferred.
Existing case reviews are the original rounds' CASE_REVIEW_ZH.md; this data-only task
does not claim a new raw-chip inspection or a newly run experiment.
"""


def main():
    names = {int(r["class_id"]): r["class_name"] for r in read_csv(SOURCE / R26 / "cohort.csv")}
    raw26 = read_csv(SOURCE / R26 / "effects.csv.gz")
    raw22 = read_csv(SOURCE / R22 / "interventions.csv")
    norm26, norm22 = normalize(raw26, R26, names), normalize(raw22, R22, names)
    cells26, cells22 = aggregate(norm26, R26), aggregate(norm22, R22)
    validation = verify(raw26, raw22, norm26, norm22, cells26, cells22)
    primary = [r for r in cells26 if r["scope"] == "new_coverage" and r["dose"] == 1.0]
    secondary = [r for r in cells22 if r["scope"] == "expansion_only" and r["dose"] == 1.0]
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv("condition_long.csv", primary)
    write_csv("condition_all_doses_scopes_long.csv", cells26)
    write_csv("round22_condition_long.csv", cells22)
    for prefix, rows in (("condition", primary), ("round22_condition", secondary)):
        for suffix, metric in (("matrix", "delta_margin"), ("n_ships_matrix", "n_ships"),
                               ("n_observations_matrix", "n_observations")):
            write_csv(f"{prefix}_{suffix}.csv", matrix(rows, metric))
    selected = [r for r in norm26 if not r["old"] and r["dose"] == 1.0]
    write_csv("round26_selected_observations.csv", selected)
    singles = {(r["polarization"], r["region"], r["scale"], r["class_id"]): r for r in primary if r["readout"] == "single"}
    differences = []
    for r in primary:
        if r["readout"] == "joint":
            s = singles[r["polarization"], r["region"], r["scale"], r["class_id"]]
            differences.append({k: r[k] for k in ("scope", "dose", "polarization", "region", "scale", "class_id", "class_name", "n_ships", "n_observations", "n_overpasses")} |
                               dict(joint_delta_margin=r["delta_margin"], single_delta_margin=s["delta_margin"],
                                    joint_minus_single=r["delta_margin"] - s["delta_margin"], metric_definition=DEFINITION,
                                    source_csv=r["source_csv"]))
    write_csv("condition_readout_difference_long.csv", differences)
    provenance = dict(
        preparation_date="2026-11-11", script=str(Path(__file__).resolve()), source_root=str(SOURCE),
        primary=dict(experiment=R26, scope="new_coverage", dose=1.0, cells=len(primary),
                     observations=len({r["obs_id"] for r in selected}), vessels=len({r["mmsi"] for r in selected}),
                     overpasses=len({r["overpass"] for r in selected}), raw_records=len(raw26),
                     definition=DEFINITION, model=MODEL, arms=["joint", "single corresponding polarization + geometry"]),
        secondary=dict(experiment=R22, scope="expansion_only", dose=1.0, raw_records=len(raw22),
                       reason_separate="different data, fitted heads and mask/L2-matching protocol"),
        aggregation="within class/condition/readout/scope/dose, mean observations within MMSI, then mean MMSIs",
        units="raw ridge decision-score margin", no_current_full=True,
        no_training=True, no_inference=True, runtime_input_types=["csv", "csv.gz"],
        selection="fixed existing scopes and both preset doses; no sign/outcome selection; main uses dose 1",
        display_mappings=dict(region={"end": "axis_end"}, scale={"middle": "mid"}),
        class_mapping=names,
        input_csvs=[f"{R26}/{f}" for f in ("cohort.csv", "effects.csv.gz", "baseline_predictions.csv", "class_detail.csv")] +
                   [f"{R22}/{f}" for f in ("interventions.csv", "baseline.csv", "by_class.csv", "summary.csv")],
        code_references=dict(
            fixed_rival_and_readouts="polarization_coverage.py:evaluate, lines 117-151",
            original_ship_aggregation="polarization_coverage.py:analyze, lines 225-243",
            operation_and_dose="polarization_attribution.py:make_changes, lines 30-65",
            region_and_scale="classification_joint_probe.py:regions/bands, lines 19-31; joint_probe.py:masks_for, lines 72-81",
            round22_aggregation="classification_joint_coverage.py:estimate/analyze, lines 136-168",
            single_head_definition="source_readout_probe.py:fit, lines 19-37"),
        report_references=[f"{R26}/FINDINGS_ZH.md", f"{R26}/PROTOCOL_ZH.md", f"{R22}/FINDINGS_ZH.md", f"{R22}/PROTOCOL_ZH.md",
                           str(HERE.parent / "problem_evidence_synthesis_20261111/RESEARCH_EVIDENCE_ZH.md")],
        caveats=["Historical weak diagnostic, not current Full", "Single image readouts retain geometry",
                 "Fixed joint rival differs from single own rival and dynamic rival",
                 "High relative dose does not mean equal absolute L2 across observations",
                 "Overpasses share three heads; not independent training repetitions",
                 "New coverage excludes six old observations, not all old identities",
                 "Class composition and small-N classes matter", "Residual is not a pure physical frequency effect"],
        outputs={p.name: str(p) for p in sorted(OUT.glob("*.csv"))},
        validation=validation,
    )
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(README, encoding="utf-8")
    print(json.dumps(dict(status="complete", primary_cells=len(primary), round26_all_cells=len(cells26),
                          round22_all_cells=len(cells22), checks=len(validation["checks"]), output=str(OUT)), ensure_ascii=True))


if __name__ == "__main__":
    main()
