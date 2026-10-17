"""Bounded, exact-identity OSS factor recovery. No imagery or model execution."""

from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import math
import re

from lxml import etree as ET
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.interpolate import CubicHermiteSpline


HERE = Path(__file__).resolve().parent
WORK = HERE.parent.parent
GEOMETRY = WORK / "oss1_joint_geometry_20261017/geometry.csv"
OSS2_MANIFEST = Path("E:/Hermes/paper_tracking/sensitivity/opensarship2_transfer_20261007/target_manifest.csv")
OSS2_INVENTORY = WORK / "oss1_polar_environment_20261017/oss2_native_inventory.csv"
CENTER = re.compile(r"_x(\d+)_y(\d+)(?:_(vv|vh))?\.tif$", re.I)
TO_ECEF = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
NAN = math.nan
KNOT_TO_MPS = 1852.0 / 3600.0
INCIDENCE_CHECK_DEG = 1.0


def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else NAN
    except (TypeError, ValueError):
        return NAN


def attributes(node):
    return {n.attrib["name"]: n.attrib.get("value", "") for n in node.findall("attrib")}


def values(node):
    return {n.tag: (n.text or "").strip() for n in node} if node is not None else {}


def epoch(text, fmt):
    try:
        return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).timestamp()
    except (ValueError, TypeError):
        return NAN


def iso(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat().replace("+00:00", "Z") if math.isfinite(seconds) else ""


def encoded(raw, minimum, maximum, scale=1.0):
    value = number(raw)
    valid = math.isfinite(value) and value.is_integer() and minimum <= value <= maximum
    return (value / scale if valid else NAN), valid


def positive(raw):
    v = number(raw)
    return v if v > 0 else NAN


def ais_fields(ais, sar):
    out = {"ais_" + key.lower() + "_raw": value for key, value in ais.items()}
    for field, raw, lo, hi, scale in (
        ("sog_knots", "Sog", 0, 1022, 10),
        ("cog_deg", "Cog", 0, 3599, 10),
        ("true_heading_deg", "True_Head", 0, 359, 1),
        ("nav_status", "Nav_Status", 0, 14, 1),
        ("draught_m", "Draught", 1, 255, 10),
    ):
        val, valid = encoded(ais.get(raw), lo, hi, scale)
        out["ais_" + field] = val
        stem = {"sog_knots": "sog", "cog_deg": "cog", "true_heading_deg": "true_heading", "draught_m": "draught"}.get(field, field)
        out["ais_" + stem + "_valid"] = valid
    out["ais_sog_mps"] = out["ais_sog_knots"] * KNOT_TO_MPS
    out["ais_sog_censored"] = number(ais.get("Sog")) == 1022
    out["ais_sog_point_valid"] = out["ais_sog_valid"] and not out["ais_sog_censored"]
    out["ais_draught_censored"] = number(ais.get("Draught")) == 255
    out["ais_draught_point_valid"] = out["ais_draught_valid"] and not out["ais_draught_censored"]
    for k in ("Length", "Width"):
        out["ais_" + k.lower() + "_m"] = positive(ais.get(k))
        out["ais_" + k.lower() + "_valid"] = math.isfinite(out["ais_" + k.lower() + "_m"])
    for key in ("Longitude", "Latitude", "Projected_Longitude", "Projected_Latitude"):
        out["ais_" + key.lower() + "_deg"] = number(ais.get(key))
    a = number(ais.get("Time"))
    a = a if 0 < a < 4102444800 else NAN
    s = epoch(sar.get("Time_UTC", ""), "%Y/%m/%d.%H:%M:%S")
    out.update(ais_time_epoch_s=a, ais_time_utc=iso(a), sar_time_raw=sar.get("Time_UTC", ""),
               sar_time_epoch_s=s, sar_time_utc=iso(s), ais_minus_sar_seconds=a-s,
               ais_abs_timegap_s=abs(a-s), ais_timegap_valid=math.isfinite(a-s),
               ais_static_update_time_utc="", ais_static_update_time_known=False)
    for bound in (60, 120, 300):
        out[f"ais_within_{bound}s"] = math.isfinite(a-s) and abs(a-s) <= bound
    return out


def load_product(dataset, scene, folder):
    ship_path = folder / "Ship.xml"
    # Same structured recovery used by the existing dataset provenance audit.
    decoded = ship_path.read_bytes().decode("utf-8", errors="replace")
    parser = ET.XMLParser(recover=True, encoding="utf-8")
    root = ET.fromstring(decoded.encode("utf-8"), parser)
    ships = {}
    for ordinal, node in enumerate(root.findall("ship")):
        sar, ais = values(node.find("SARShipInformation")), values(node.find("AISShipInformation"))
        key = (int(sar["Center_x"]), int(sar["Center_y"]))
        if key in ships:
            raise ValueError(f"Ambiguous XML center: {scene}/{key}")
        ships[key] = dict(ordinal=ordinal, sar=sar, ais=ais,
                          marine=values(node.find("MarineTrafficInformation")),
                          match=values(node.find("MatchInformation")),
                          noise=values(node.find("ChipNoise")))
    metadata = next((folder / n for n in ("Metedata.xml", "Metadata.xml", scene + ".xml", scene.removesuffix("(1)") + ".xml") if (folder / n).is_file()), None)
    attrs, bands, orbit_raw = {}, {}, []
    if metadata:
        ar = ET.parse(str(metadata)).getroot().find("Abstracted_Metadata")
        attrs = attributes(ar)
        bands = {n.tag: attributes(n) for n in ar if n.tag.startswith("Band_")}
        orbit_node = ar.find("Orbit_State_Vectors")
        orbit_raw = [attributes(n) for n in orbit_node] if orbit_node is not None else []
        # One OSS2 local folder has a duplicate-download suffix, retained as scene_id.
        expected = scene.removesuffix("(1)")
        if attrs.get("PRODUCT") != expected:
            raise ValueError(f"Product ID mismatch: {scene}: {attrs.get('PRODUCT')}")
    vectors = []
    for raw in orbit_raw:
        vectors.append([epoch(raw.get("time", ""), "%d-%b-%Y %H:%M:%S.%f")]
                       + [number(raw.get(axis + "_pos")) for axis in "xyz"]
                       + [number(raw.get(axis + "_vel")) for axis in "xyz"])
    vectors = np.asarray(vectors, dtype=float).reshape(-1, 7)
    if len(vectors):
        vectors = vectors[np.argsort(vectors[:, 0])]
    orbit_valid = bool(len(vectors) >= 2 and np.isfinite(vectors).all()
                       and np.all(np.diff(vectors[:, 0]) > 0))
    interpolator = None
    kin_error = NAN
    if orbit_valid:
        interpolator = CubicHermiteSpline(vectors[:, 0] - vectors[0, 0], vectors[:, 1:4], vectors[:, 4:7], extrapolate=False)
        finite_velocity = np.diff(vectors[:, 1:4], axis=0) / np.diff(vectors[:, 0])[:, None]
        kin_error = float(np.max(np.linalg.norm(finite_velocity - (vectors[:-1, 4:7] + vectors[1:, 4:7]) / 2, axis=1)))
    pol = sorted({b.get("polarization") for b in bands.values() if b.get("polarization") in ("VV", "VH", "HH", "HV")}
                 | {attrs.get(f"mds{i}_tx_rx_polar") for i in range(1, 5) if attrs.get(f"mds{i}_tx_rx_polar") in ("VV", "VH", "HH", "HV")})
    kind = "GRDH" if "_GRDH_" in scene else "SLC"
    p = dict(dataset=dataset, scene_id=scene, product=attrs.get("PRODUCT", scene.removesuffix("(1)")),
             product_type=kind, ship_xml_path=str(ship_path), metadata_xml_path=str(metadata) if metadata else "",
             product_metadata_available=metadata is not None, product_polarizations="|".join(pol),
             product_polarization_valid=bool(pol), product_platform=attrs.get("MISSION", ""),
             product_mode=attrs.get("ACQUISITION_MODE", ""), product_native_type=attrs.get("PRODUCT_TYPE", ""),
             product_pass=attrs.get("PASS", ""), product_antenna_pointing=attrs.get("antenna_pointing", ""),
             product_sample_type=attrs.get("SAMPLE_TYPE", ""), product_processing_system=attrs.get("Processing_system_identifier", ""),
             product_processing_time_raw=attrs.get("PROC_TIME", ""), product_map_projection=attrs.get("map_projection", ""),
             product_range_spacing_semantics="ground_range" if attrs.get("srgr_flag") == "1" else "slant_range" if attrs.get("srgr_flag") == "0" else "unknown",
             product_resolution_m=NAN, product_resolution_known=False,
             product_first_line_time_utc=iso(epoch(attrs.get("first_line_time", ""), "%d-%b-%Y %H:%M:%S.%f")),
             product_last_line_time_utc=iso(epoch(attrs.get("last_line_time", ""), "%d-%b-%Y %H:%M:%S.%f")),
             orbit_vector_count=len(vectors), orbit_vectors_valid=orbit_valid,
             orbit_kinematic_error_mps=kin_error,
             orbit_span_start_utc=iso(vectors[0, 0]) if orbit_valid else "",
             orbit_span_end_utc=iso(vectors[-1, 0]) if orbit_valid else "",
             orbit_max_step_s=float(np.max(np.diff(vectors[:, 0]))) if orbit_valid else NAN,
             xml_parser_message_count=len(parser.error_log), xml_replacement_count=decoded.count("\ufffd"),
             xml_replacement_fields="|".join(sorted({n.tag for n in root.iter() if "\ufffd" in (n.text or "")})),
             xml_ship_count=len(ships), sar_distinct_timestamps=len({s["sar"].get("Time_UTC") for s in ships.values()}))
    for output, source in {
        "frequency_mhz": "radar_frequency", "range_spacing_m": "range_spacing", "azimuth_spacing_m": "azimuth_spacing",
        "range_looks": "range_looks", "azimuth_looks": "azimuth_looks", "relative_orbit": "REL_ORBIT",
        "absolute_orbit": "ABS_ORBIT", "orbit_cycle": "orbit_cycle", "line_time_interval_s": "line_time_interval",
        "prf_hz": "pulse_repetition_frequency", "range_bandwidth_mhz": "range_bandwidth", "azimuth_bandwidth_hz": "azimuth_bandwidth",
        "is_terrain_corrected": "is_terrain_corrected", "srgr_flag": "srgr_flag", "multilook_flag": "multilook_flag",
        "abs_calibration_flag": "abs_calibration_flag", "range_sampling_rate_mhz": "range_sampling_rate",
    }.items():
        val = number(attrs.get(source))
        p["product_" + output] = NAN if val == 99999 else val
    p["product_frequency_hz"] = p["product_frequency_mhz"] * 1e6
    return dict(summary=p, ships=ships, vectors=vectors, interpolator=interpolator,
                raw=dict(attributes=attrs, bands=bands, orbit_vectors=orbit_raw,
                         xml_parser_messages=[str(e) for e in parser.error_log]))


def orbit_at(product, seconds):
    vec = product["vectors"]
    spline = product["interpolator"]
    if spline is None or not math.isfinite(seconds) or not vec[0, 0] <= seconds <= vec[-1, 0]:
        return None
    t = seconds - vec[0, 0]
    return np.asarray(spline(t)), np.asarray(spline(t, 1))


def local_basis(lon_deg, lat_deg):
    lon, lat = np.radians([lon_deg, lat_deg])
    return np.asarray([[-np.sin(lon), np.cos(lon), 0],
                       [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
                       [np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]])


def look_geometry(position, velocity, longitude, latitude):
    target = np.asarray(TO_ECEF.transform(longitude, latitude, 0.0))
    basis = local_basis(longitude, latitude)
    difference = position - target
    distance = float(np.linalg.norm(difference))
    los = basis @ (difference / distance)
    flight = basis @ velocity
    horizontal = float(np.linalg.norm(los[:2]))
    if not np.isfinite(los).all() or los[2] <= 0 or horizontal <= 0 or np.linalg.norm(flight[:2]) <= 0:
        return None
    h = los[:2] / horizontal
    f = flight[:2] / np.linalg.norm(flight[:2])
    return dict(orbit_los_east_unit=float(los[0]), orbit_los_north_unit=float(los[1]), orbit_los_up_unit=float(los[2]),
                orbit_los_horizontal_norm=horizontal, orbit_los_horizontal_east_unit=float(h[0]),
                orbit_los_horizontal_north_unit=float(h[1]), orbit_los_bearing_deg=float(np.degrees(np.arctan2(h[0], h[1])) % 360),
                orbit_flight_east_unit=float(f[0]), orbit_flight_north_unit=float(f[1]),
                orbit_flight_bearing_deg=float(np.degrees(np.arctan2(f[0], f[1])) % 360),
                orbit_incidence_deg=float(np.degrees(np.arccos(np.clip(los[2], -1, 1)))),
                orbit_slant_range_m=distance, orbit_horizontal_los_dot_flight=float(h @ f))


def ground_motion(speed, course, geometry):
    angle = math.radians(course)
    east, north = speed * math.sin(angle), speed * math.cos(angle)
    horizontal = east * geometry["orbit_los_horizontal_east_unit"] + north * geometry["orbit_los_horizontal_north_unit"]
    los = east * geometry["orbit_los_east_unit"] + north * geometry["orbit_los_north_unit"]
    flight = east * geometry["orbit_flight_east_unit"] + north * geometry["orbit_flight_north_unit"]
    return dict(ground_east_mps=east, ground_north_mps=north,
                ground_los_horizontal_signed_mps=horizontal, ground_los_horizontal_abs_mps=abs(horizontal),
                ground_range_away_signed_mps=-horizontal, ground_range_abs_mps=abs(horizontal),
                ground_los_3d_signed_mps=los, ground_los_3d_abs_mps=abs(los),
                ground_alongflight_signed_mps=flight, ground_alongflight_abs_mps=abs(flight))


GEOMETRY_NUMERIC = tuple(look_geometry(np.array([7078137., 700000., 0.]), np.array([0., 0., 7500.]), 0, 0))
MOTION_NUMERIC = tuple(ground_motion(1, 0, {k: 0.0 for k in GEOMETRY_NUMERIC}))


def recover_one(dataset, row, prod, key):
    ship = prod["ships"][key]
    sar, ais, marine = ship["sar"], ship["ais"], ship["marine"]
    identity = str(row["mmsi"] if dataset == "oss1" else row["mmsi_or_ship_id"])
    if identity != ais.get("MMSI"):
        raise ValueError(f"MMSI mismatch: {row['sample_id']}")
    if dataset == "oss1" and (int(row["xml_ship_index"]) != ship["ordinal"] or row["sample_id"] != f"{row['scene_id']}:x{key[0]}:y{key[1]}"):
        raise ValueError(f"OSS1 identity/order mismatch: {row['sample_id']}")
    out = {k: row[k] for k in ("global_row", "target_manifest_row", "sample_id", "class_id", "class_name")}
    out.update(prod["summary"])
    out.update(mmsi=identity, center_x=key[0], center_y=key[1], xml_ship_index=ship["ordinal"], identity_match_valid=True)
    out.update(ais_fields(ais, sar))
    first = epoch(prod["raw"]["attributes"].get("first_line_time", ""), "%d-%b-%Y %H:%M:%S.%f")
    last = epoch(prod["raw"]["attributes"].get("last_line_time", ""), "%d-%b-%Y %H:%M:%S.%f")
    out["sar_time_within_product_span"] = first <= out["sar_time_epoch_s"] <= last
    out["sar_time_minus_product_midpoint_s"] = out["sar_time_epoch_s"] - (first + last) / 2
    out.update(sar_center_longitude=number(sar.get("Center_longitude")), sar_center_latitude=number(sar.get("Center_latitude")),
               sar_manual_length_m=positive(sar.get("Manual_Length")), sar_manual_length_raw=sar.get("Manual_Length", ""),
               sar_chip_direction_deg=number(sar.get("Chip_Direction")), sar_north_direction_deg=number(sar.get("North_Direction")),
               sar_incidence_deg=number(sar.get("Incidence")), sar_elevation_deg=number(sar.get("Elevation")),
               sar_slant_range_time_raw=sar.get("Slant_Range_Time", sar.get("SlantRangeTime", "")),
               marine_length_overall_m=positive(marine.get("Length_overall")), marine_breadth_extreme_m=positive(marine.get("Breadth_extreme")),
               marine_length_overall_raw=marine.get("Length_overall", ""), marine_breadth_extreme_raw=marine.get("Breadth_extreme", ""),
               marine_deadweight=positive(marine.get("Deadweight")), marine_gross_tonnage=positive(marine.get("Gross_tonnage")),
               marine_year_built=positive(marine.get("Year_built")), marine_elaborated_type=marine.get("Elaborated_type", ""),
               match_distance_error_m=number(ship["match"].get("Distance_error")),
               match_length_error_m=number(ship["match"].get("Length_error")),
               imported_heading_deg=number(row.get("heading_deg")), imported_mixed_length_m=number(row.get("length_m")),
               freeboard_m=NAN, freeboard_valid=False, stationary_duration_s=NAN, stationary_duration_valid=False,
               sar_target_precise_time_known=False, full_chip_georeference_valid=False,
               orbit_geometry_time_basis="Ship.xml reported SAR timestamp; product-constant, not per-pixel time",
               orbit_target_height_m=0.0)
    out["incidence_deg"] = number(row.get("incidence_deg", out["sar_incidence_deg"]))
    out["incidence_table_xml_difference_deg"] = out["incidence_deg"] - out["sar_incidence_deg"]
    if dataset == "oss1" and abs(out["incidence_table_xml_difference_deg"]) > 1e-9:
        raise ValueError(f"Geometry/XML incidence mismatch: {row['sample_id']}")
    for name in ("sar_manual_length_m", "marine_length_overall_m", "marine_breadth_extreme_m"):
        out[name.removesuffix("_m") + "_valid"] = math.isfinite(out[name])
    out.update({k: NAN for k in (*GEOMETRY_NUMERIC, *MOTION_NUMERIC)})
    out.update(orbit_interpolation_valid=False, orbit_geometry_valid=False, orbit_incidence_check_valid=False,
               orbit_incidence_minus_table_deg=NAN, orbit_incidence_within_1deg=False,
               ground_vector_valid=out["ais_sog_point_valid"] and out["ais_cog_valid"],
               ground_projection_valid=False, orbit_geometry_status="orbit_missing_or_time_outside_span")
    state = orbit_at(prod, out["sar_time_epoch_s"])
    if state is not None:
        out["orbit_interpolation_valid"] = True
        lon, lat = out["sar_center_longitude"], out["sar_center_latitude"]
        geo = look_geometry(*state, lon, lat) if -180 <= lon <= 180 and -90 <= lat <= 90 else None
        if geo:
            out.update(geo)
            out["orbit_geometry_valid"] = True
            out["orbit_geometry_status"] = "at_reported_product_time"
            out["orbit_incidence_check_valid"] = 0 < out["incidence_deg"] < 90
            out["orbit_incidence_minus_table_deg"] = geo["orbit_incidence_deg"] - out["incidence_deg"]
            out["orbit_incidence_within_1deg"] = abs(out["orbit_incidence_minus_table_deg"]) <= INCIDENCE_CHECK_DEG
            out["ground_projection_valid"] = out["ground_vector_valid"] and out["orbit_incidence_within_1deg"] and out["sar_time_within_product_span"]
            if not out["sar_time_within_product_span"]:
                out["orbit_geometry_status"] = "reported_sar_time_outside_product_acquisition"
            if out["ground_projection_valid"]:
                out.update(ground_motion(out["ais_sog_mps"], out["ais_cog_deg"], geo))
        else:
            out["orbit_geometry_status"] = "invalid_target_or_below_horizon"
    if out["ground_vector_valid"]:
        c = math.radians(out["ais_cog_deg"])
        out["ground_east_mps"] = out["ais_sog_mps"] * math.sin(c)
        out["ground_north_mps"] = out["ais_sog_mps"] * math.cos(c)
    return out


def recover(dataset):
    source = GEOMETRY if dataset == "oss1" else OSS2_MANIFEST
    frame = pd.read_csv(source, dtype={"sample_id": str, "mmsi": str, "mmsi_or_ship_id": str})
    if dataset == "oss2":
        frame.insert(0, "global_row", np.arange(len(frame)))
        inventory = pd.read_csv(OSS2_INVENTORY, dtype={"sample_id": str, "mmsi": str})
        if len(frame) != 3745 or frame.sample_id.duplicated().any() or inventory.sample_id.duplicated().any():
            raise ValueError("Expected unique 3745-row OSS2 audit cohort")
        aligned = inventory.set_index("sample_id").loc[frame.sample_id]
        if not (np.array_equal(aligned["product"].to_numpy(), frame.scene_id.to_numpy()) and
                np.array_equal(aligned.mmsi.to_numpy(), frame.mmsi_or_ship_id.to_numpy())):
            raise ValueError("OSS2 prior inventory product/MMSI mismatch")
    elif len(frame) != 1349 or frame.scene_id.nunique() != 22 or not frame.product_type.eq("GRDH").all():
        raise ValueError("Expected current 1349-row/22-product OSS1 GRDH cohort")
    if not np.array_equal(frame.global_row, np.arange(len(frame))) or frame.sample_id.duplicated().any():
        raise ValueError("Nonunique sample IDs or noncanonical input row order")
    products = {}
    for scene, group in frame.groupby("scene_id", sort=False):
        first = group.iloc[0]
        folder = Path(first.ship_xml_path).parent if dataset == "oss1" else Path(first.source_path_vv).parent.parent
        products[scene] = load_product(dataset, scene, folder)
    records = []
    for row in frame.to_dict("records"):
        if dataset == "oss1":
            key = (int(row["center_x"]), int(row["center_y"]))
        else:
            vv, vh = Path(row["source_path_vv"]), Path(row["source_path_vh"])
            a, b = CENTER.search(vv.name), CENTER.search(vh.name)
            if vv.parent.parent.name != row["scene_id"] or vh.parent.parent.name != row["scene_id"] or not a or not b or a.group(1, 2) != b.group(1, 2):
                raise ValueError(f"OSS2 native center/product mismatch: {row['sample_id']}")
            key = tuple(map(int, a.group(1, 2)))
            inv = aligned.loc[row["sample_id"]]
            match = CENTER.search(Path(inv.raw_path).name)
            if not match or tuple(map(int, match.group(1, 2))) != key:
                raise ValueError(f"OSS2 prior inventory center mismatch: {row['sample_id']}")
        records.append(recover_one(dataset, row, products[row["scene_id"]], key))
    out = pd.DataFrame(records)
    if not np.array_equal(out.sample_id, frame.sample_id) or not np.array_equal(out.global_row, frame.global_row):
        raise ValueError("Output row alignment changed")
    out.to_csv(HERE / f"{dataset}_factor_metadata.csv", index=False)
    return out, products


def coverage(frame):
    fields = [c for c in frame if c.endswith("_valid") or c.endswith("_censored") or c.startswith("ais_within_")]
    fields += ["sar_time_within_product_span", "orbit_incidence_within_1deg"]
    rows = []
    groups = [("all", "all", frame)]
    groups += [(col, str(key), group) for col in ("product_type", "class_name") for key, group in frame.groupby(col, sort=False)]
    for grouping, group_id, group in groups:
        for field in fields:
            valid = group[field].fillna(False).astype(bool)
            subset = group.loc[valid]
            rows.append(dict(dataset=group.dataset.iloc[0], grouping=grouping, group=group_id, field=field,
                             rows=len(group), valid_rows=int(valid.sum()), ships=group.mmsi.nunique(),
                             valid_ships=subset.mmsi.nunique(), products=group.scene_id.nunique(),
                             valid_products=subset.scene_id.nunique(), fraction=float(valid.mean())))
    return rows


def protocol():
    return dict(
        scope="Metadata only; no images, training, new downloads, root scripts or brain modifications",
        alignment={"oss1": str(GEOMETRY), "oss2": str(OSS2_MANIFEST), "oss2_prior_inventory": str(OSS2_INVENTORY),
                   "keys": "preserve input order; exact scene folder + center_x/y + MMSI; OSS1 XML ordinal checked"},
        ais={"sog": "integer raw 0..1022 divided by 10 knots; 1023 missing; 1022 is >=102.2 knots, excluded from point-vector projections",
             "cog": "integer raw 0..3599 divided by 10 degrees clockwise from true north; 3600 missing; low-speed direction not necessarily meaningful",
             "true_heading": "0..359 degrees, 511 missing; independent from COG and SAR Chip_Direction",
             "nav_status": "raw code retained; 15 undefined; 0..14 encoded categories include reserved statuses, not a validated activity label",
             "draught": "raw 1..254 /10 metres; 0 missing; 255 lower bound >=25.5m; outside 0..255 invalid; update time unknown",
             "dimensions": "AIS Length/Width metres, MarineTraffic Length_overall/Breadth_extreme metres and SAR Manual_Length metres remain separate; no fallback",
             "timegap": "AIS UTC epoch - Ship.xml SAR UTC; signed seconds; no timegap excluded automatically; 60/120/300s flags are sensitivity views",
             "static_time": "AIS dynamic timestamp is not evidence of dimensions/draught update time"},
        geometry={"method": "Scipy CubicHermiteSpline uses original timed ECEF position metres and velocity m/s, no extrapolation; pyproj WGS84 lon/lat to ECEF at ellipsoid height 0m",
                  "time": "evaluate at reported Ship.xml SAR time, constant per product; not recovered per-target azimuth time; geographic ground proxy only",
                  "los": "target to satellite unit vector transformed into target local east,north,up; horizontal normalized LOS also retained",
                  "incidence": "acos(up component), compared to geometry.csv incidence for OSS1 / raw XML for OSS2; absolute residual <=1deg is a coarse prespecified consistency flag, not accuracy certification",
                  "flight": "normalize satellite ECEF velocity projected onto target local horizontal plane; not PASS-derived bearing",
                  "ground": "SOG vector: east=speed*sin(COG), north=speed*cos(COG), vertical=0; dot horizontal LOS for ground range-direction proxy; dot 3D LOS for slant-LOS ground-motion contribution; dot flight for alongflight; abs columns discard sign",
                  "positive_signs": "LOS toward satellite; range_away opposite horizontal LOS; alongflight in projected satellite flight direction",
                  "validity": "point SOG + coded COG + orbit bracket + finite above-horizon geometry + reported SAR timestamp inside product acquisition interval + incidence residual <=1deg; does NOT certify simultaneity or physical motion truth",
                  "not_claimed": ["STW", "measured SAR velocity", "measured Doppler shift", "stationary duration", "precise per-pixel acquisition time", "full chip georeference", "image-plane geographic direction mapping"]},
        product={"frequency": "SNAP AbstractMetadata radar_frequency MHz -> Hz", "pixel_spacing": "range spacing uses srgr_flag=1 ground vs 0 slant; azimuth spacing stored separately; neither is effective resolution",
                 "processing": "raw metadata flags describe native product; imported displayed chips may have separate scaling/cropping; raw attributes/bands retained in product_metadata_raw.json",
                 "local_duplicate_folder": "OSS2 scene ending (1) retained; metadata PRODUCT is checked against exact name minus that observed suffix"},
        xml={"method": "existing audit lxml structured parser with recover=True and UTF8 replacement; no source files edited; errors and affected fields recorded per product",
             "evidence": str(WORK / "ports_root_cause_20261013/dataset_audit/audit_local.py")},
        references=[str(WORK / "ports_root_cause_20261013/dataset_audit/BUNDLED_README_TEXT.md"),
                    "https://www.navcen.uscg.gov/ais-class-a-reports",
                    "https://www.navcen.uscg.gov/ais-class-a-static-voyage-message-5",
                    "https://raw.githubusercontent.com/senbox-org/snap-engine/master/snap-engine-utilities/src/main/java/org/esa/snap/engine_utilities/datamodel/AbstractMetadata.java"],
        provenance_only=["imported_heading_deg is SAR Chip_Direction, not AIS heading", "imported_mixed_length_m may include SAR fallback, never independent truth"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("oss1", "oss2", "all"), default="all")
    args = parser.parse_args()
    tables, prods = [], {}
    for dataset in (("oss1", "oss2") if args.dataset == "all" else (args.dataset,)):
        frame, products = recover(dataset)
        tables.append(frame)
        prods.update({dataset + ":" + k: p for k, p in products.items()})
        print(json.dumps(dict(dataset=dataset, rows=len(frame), products=frame.scene_id.nunique(),
                              orbit_geometry_valid=int(frame.orbit_geometry_valid.sum()),
                              ground_projection_valid=int(frame.ground_projection_valid.sum())), ensure_ascii=False), flush=True)
    pd.DataFrame([r for t in tables for r in coverage(t)]).to_csv(HERE / "coverage.csv", index=False)
    pd.DataFrame([p["summary"] for p in prods.values()]).to_csv(HERE / "products.csv", index=False)
    per_product = []
    for frame in tables:
        for scene, group in frame.groupby("scene_id", sort=False):
            r = dict(dataset=group.dataset.iloc[0], scene_id=scene, product_type=group.product_type.iloc[0],
                     rows=len(group), ships=group.mmsi.nunique(), classes=group.class_id.nunique(),
                     ais_timegap_min_s=group.ais_minus_sar_seconds.min(), ais_timegap_max_s=group.ais_minus_sar_seconds.max(),
                     incidence_abs_error_max_deg=group.orbit_incidence_minus_table_deg.abs().max())
            r.update({c: int(group[c].sum()) for c in ("ais_sog_valid", "ais_cog_valid", "ais_true_heading_valid", "ais_draught_valid", "orbit_geometry_valid", "ground_projection_valid")})
            per_product.append(r)
    pd.DataFrame(per_product).to_csv(HERE / "product_coverage.csv", index=False)
    (HERE / "product_metadata_raw.json").write_text(json.dumps({k: p["raw"] for k, p in prods.items()}, indent=2, ensure_ascii=False), encoding="utf-8")
    (HERE / "PROTOCOL.json").write_text(json.dumps(protocol(), indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
