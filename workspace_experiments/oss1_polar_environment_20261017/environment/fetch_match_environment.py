"""Bounded, cached ERA5/GLORYS12 matching; all outputs stay beside this script."""

from __future__ import annotations

import argparse
from collections.abc import MutableMapping
import csv
from datetime import datetime, timezone
import json
import math
import netrc
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
import numpy as np
import pandas as pd
import requests
import xarray as xr

GEOMETRY = ROOT.parents[1] / "oss1_joint_geometry_20261017" / "geometry.csv"
PRODUCT = "GLOBAL_MULTIYEAR_PHY_001_030"
MARINE_DATASET = "cmems_mod_glo_phy_my_0.083deg_P1D-m_202311"
STAC_ROOT = "https://stac.marine.copernicus.eu/metadata/"
GOOGLE_ERA5 = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
CDS_ARCO = "https://arco.datastores.ecmwf.int/cadl-arco-geo-003/arco/reanalysis_era5_single_levels/wav/geoChunked.zarr"
VARIABLES = {
    "era5_u10": ("era5", "10m_u_component_of_wind", "m s-1"),
    "era5_v10": ("era5", "10m_v_component_of_wind", "m s-1"),
    "era5_hs": ("era5", "significant_height_of_combined_wind_waves_and_swell", "m"),
    "era5_mean_wave_period": ("era5", "mean_wave_period", "s"),
    "era5_mean_wave_direction": ("era5", "mean_wave_direction", "degree_from_north_clockwise_FROM"),
    "cmems_uo": ("cmems", "uo", "m s-1"),
    "cmems_vo": ("cmems", "vo", "m s-1"),
    "cmems_thetao": ("cmems", "thetao", "degree_C_potential_temperature"),
    "cmems_so": ("cmems", "so", "1e-3_practical_salinity"),
}


def write_json(name, data):
    (ROOT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def utc(value):
    return pd.Timestamp(value).isoformat().replace("+00:00", "Z")


def credentials():
    """Use only documented config locations; never emit credential values."""
    import cdsapi

    result = {"cds_client_configured": False, "marine_config_present": False, "marine_netrc_present": False}
    try:
        cdsapi.Client(quiet=True, timeout=10, retry_max=0)
        result["cds_client_configured"] = True
    except Exception as exc:
        result["cds_client_exception_type"] = type(exc).__name__
    marine_dir = Path(os.environ.get("COPERNICUSMARINE_SERVICE_CREDENTIALS_DIRECTORY", Path.home() / ".copernicusmarine"))
    result["marine_config_present"] = any((marine_dir / n).exists() for n in [".copernicusmarine-credentials", ".copernicusmarine-credentials.json"])
    result["marine_env_pair_present"] = all(os.environ.get(k) for k in ["COPERNICUSMARINE_SERVICE_USERNAME", "COPERNICUSMARINE_SERVICE_PASSWORD"])
    for path in [Path.home() / ".netrc", Path.home() / "_netrc"]:
        if path.exists():
            try:
                config = netrc.netrc(str(path))
                result["marine_netrc_present"] |= any(bool(config.authenticators(host)) for host in ["auth.marine.copernicus.eu", "my.cmems-du.eu"])
            except (OSError, netrc.NetrcParseError):
                result["marine_netrc_parse_error"] = True
    return result


class Downloader:
    """Record actual HTTP response bytes and enforce the user-requested bound."""

    def __init__(self, max_mib, offline):
        self.limit = int(max_mib * 1024**2)
        self.offline = offline
        self.bytes = 0
        self.session = requests.Session()
        self.session.trust_env = True
        self.log_path = ROOT / "download_log.csv"
        self.fields = ["utc", "source", "key", "url", "http_status", "bytes", "kind"]

    def fetch(self, source, base, key):
        target = ROOT / "cache" / source / key
        if target.exists():
            return target.read_bytes()
        if self.offline:
            raise FileNotFoundError("uncached " + source + "/" + key)
        url = base.rstrip("/") + "/" + key
        kind = "metadata" if key.endswith((".json", ".zmetadata", ".zattrs", ".zarray", ".md", ".cfg")) else "array_chunk"
        with self.session.get(url, timeout=(10, 45), stream=True) as response:
            size = int(response.headers.get("Content-Length", 0))
            if size > 16 * 1024**2 or self.bytes + size > self.limit:
                raise RuntimeError("download_byte_budget_exceeded")
            parts = []
            received = 0
            for part in response.iter_content(65536):
                received += len(part)
                self.bytes += len(part)
                if received > 16 * 1024**2 or self.bytes > self.limit:
                    raise RuntimeError("download_byte_budget_exceeded")
                parts.append(part)
            data = b"".join(parts)
            new = not self.log_path.exists()
            with self.log_path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fields)
                if new:
                    writer.writeheader()
                writer.writerow(dict(utc=datetime.now(timezone.utc).isoformat(), source=source, key=key,
                                     url=url, http_status=response.status_code, bytes=received, kind=kind))
            if response.status_code == 404:
                raise KeyError(key)
            response.raise_for_status()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return data


class CachedHTTPStore(MutableMapping):
    def __init__(self, downloader, source, base):
        self.downloader, self.source, self.base = downloader, source, base
        self.metadata = json.loads(self[".zmetadata"])["metadata"]

    def __getitem__(self, key):
        if hasattr(self, "metadata") and key in self.metadata:
            return json.dumps(self.metadata[key]).encode()
        return self.downloader.fetch(self.source, self.base, key)

    def __iter__(self):
        return iter(self.metadata)

    def __len__(self):
        return len(self.metadata)

    def __setitem__(self, key, value):
        raise TypeError("read-only remote store")

    def __delitem__(self, key):
        raise TypeError("read-only remote store")


def load_manifest(path):
    frame = pd.read_csv(path, dtype={"sample_id": str, "acquisition": str})
    if "sar_acquisition_time" in frame:
        frame["dataset"] = "OSS1"
        frame["time"] = pd.to_datetime(frame.sar_acquisition_time, format="%Y/%m/%d.%H:%M:%S", utc=True)
        frame["lat"] = frame.sar_center_latitude
        frame["lon"] = frame.sar_center_longitude
    else:
        frame["time"] = pd.to_datetime(frame.time, utc=True)
    columns = ["dataset", "sample_id", "acquisition", "time", "lat", "lon"]
    columns += [c for c in ["mmsi", "class_name", "product"] if c in frame]
    frame = frame[columns].copy()
    if frame[["dataset", "sample_id", "acquisition", "time", "lat", "lon"]].isna().any().any():
        raise ValueError("missing required manifest values")
    if frame.duplicated(["dataset", "sample_id"]).any():
        raise ValueError("duplicate dataset/sample_id")
    if not (frame.lat.between(-90, 90).all() and frame.lon.between(-180, 180).all()):
        raise ValueError("invalid sample coordinates")
    return frame


def requests_inventory(frame):
    records, recipes = [], []
    for (dataset, acquisition), group in frame.groupby(["dataset", "acquisition"], sort=False):
        times = sorted(group.time.dt.round("h").unique())
        days = sorted(group.time.dt.floor("D").unique())
        record = dict(dataset=dataset, acquisition=acquisition, sample_count=len(group),
                      sar_start_utc=utc(group.time.min()), sar_end_utc=utc(group.time.max()),
                      south=float(group.lat.min()), north=float(group.lat.max()),
                      west=float(group.lon.min()), east=float(group.lon.max()),
                      era5_nearest_hours_utc=json.dumps([utc(t) for t in times]),
                      cmems_dates=json.dumps([str(pd.Timestamp(t).date()) for t in days]),
                      era5_wind_grid_deg=0.25, cds_standard_wave_grid_deg=0.5,
                      google_wave_delivery_grid_deg=0.25, cmems_grid_deg=1/12,
                      cmems_depth_selection="shallowest_original_layer_no_deeper_fill",
                      cmems_dataset_version=MARINE_DATASET,
                      spatial_method="nearest_original_delivery_grid_no_wet_cell_search")
        records.append(record)
        for t in times:
            t = pd.Timestamp(t)
            recipes.append(dict(dataset=dataset, acquisition=acquisition, service="CDS",
                                dataset_id="reanalysis-era5-single-levels", request=dict(
                                    product_type=["reanalysis"], variable=[v[1] for v in VARIABLES.values() if v[0] == "era5"],
                                    year=[f"{t.year:04}"], month=[f"{t.month:02}"], day=[f"{t.day:02}"],
                                    time=[t.strftime("%H:%M")], data_format="netcdf", download_format="unarchived",
                                    area=[math.ceil(group.lat.max()*2)/2, math.floor(group.lon.min()*2)/2,
                                          math.floor(group.lat.min()*2)/2, math.ceil(group.lon.max()*2)/2])))
        for t in days:
            recipes.append(dict(dataset=dataset, acquisition=acquisition, service="CopernicusMarine",
                                request=dict(dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m", dataset_version="202311",
                                             variables=["uo", "vo", "thetao", "so"],
                                             minimum_longitude=float(group.lon.min())-1/12,
                                             maximum_longitude=float(group.lon.max())+1/12,
                                             minimum_latitude=float(group.lat.min())-1/12,
                                             maximum_latitude=float(group.lat.max())+1/12,
                                             start_datetime=utc(t), end_datetime=utc(t+pd.Timedelta(hours=23, minutes=59, seconds=59)),
                                             minimum_depth=0, maximum_depth=1)))
    pd.DataFrame(records).to_csv(ROOT / "acquisition_requests.csv", index=False)
    write_json("authenticated_request_recipes.json", recipes)


def distance_km(lat, lon, grid_lat, grid_lon):
    a, b = np.deg2rad(lat), np.deg2rad(grid_lat)
    h = np.sin((b-a)/2)**2 + np.cos(a)*np.cos(b)*np.sin(np.deg2rad(grid_lon-lon)/2)**2
    return 6371.0088*2*np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def nearest_indices(coords, values, cyclic=False):
    coords = np.asarray(coords)
    difference = np.abs(np.asarray(values)[:, None] - coords[None, :])
    if cyclic:
        difference = np.minimum(difference, 360-difference)
    return difference.argmin(axis=1)


def error_label(exc):
    if isinstance(exc, requests.HTTPError):
        return "http_" + str(exc.response.status_code)
    return type(exc).__name__ + (":download_byte_budget_exceeded" if "download_byte_budget_exceeded" in str(exc) else "")


def open_source(dl, source, url):
    store = CachedHTTPStore(dl, source, url)
    ds = xr.open_zarr(store, consolidated=True, chunks=None, zarr_format=2)
    inventory = dict(url=url, attrs=dict(ds.attrs), sizes=dict(ds.sizes),
                     variables={name:dict(attrs=dict(ds[name].attrs), dimensions=list(ds[name].dims),
                                          zarr=store.metadata[name+"/.zarray"])
                                for name in ds.data_vars if name in [v[1] for v in VARIABLES.values()] + ["land_sea_mask"]},
                     coordinate_first_last={name:[str(ds[name].values[0]),str(ds[name].values[-1])]
                                            for name in ["time", "latitude", "longitude", "elevation"] if name in ds})
    write_json(source+"_source_inventory.json", inventory)
    return ds


def match_source(frame, dl, source, url, max_acquisitions):
    fields = {k:v for k,v in VARIABLES.items() if v[0] == source}
    out = pd.DataFrame(index=frame.index)
    for name in fields:
        out[name] = np.nan
        out[name+"_status"] = "not_requested"
    info = dict(source=source, status="started", matched_acquisitions=0)
    try:
        ds = open_source(dl, source, url)
    except Exception as exc:
        label = error_label(exc)
        for name in fields:
            out[name+"_status"] = label
        return out, dict(source=source, status=label)
    time_index = pd.DatetimeIndex(ds.time.values)
    latitudes = ds.latitude.values.astype(np.float64)
    longitudes = ds.longitude.values.astype(np.float64)
    depth = None
    if source == "cmems":
        surface_index = int(np.abs(ds.elevation.values).argmin())
        depth = float(abs(ds.elevation.values[surface_index]))
    lsm = None
    for number, ((dataset, acquisition), group) in enumerate(frame.groupby(["dataset", "acquisition"], sort=False)):
        if max_acquisitions and number >= max_acquisitions:
            break
        selection_times = group.time.dt.round("h") if source == "era5" else group.time.dt.floor("D")
        for wanted_time, part in group.groupby(selection_times, sort=False):
            ti = int(time_index.get_indexer([wanted_time.tz_localize(None)], method="nearest")[0])
            actual = pd.Timestamp(time_index[ti]).tz_localize("UTC")
            # Daily fields are selected by date; never use nearest-day rounding at 22 UTC.
            if (source == "era5" and actual != wanted_time) or (source == "cmems" and actual.date() != wanted_time.date()):
                for name in fields:
                    out.loc[part.index, name+"_status"] = "requested_time_not_available"
                continue
            yi = nearest_indices(latitudes, part.lat)
            source_lons = np.mod(part.lon.to_numpy(), 360) if np.min(longitudes) >= 0 else part.lon.to_numpy()
            xi = nearest_indices(longitudes, source_lons, cyclic=True)
            y = latitudes[yi]
            x = (longitudes[xi]+180) % 360-180
            distance = distance_km(part.lat.to_numpy(), part.lon.to_numpy(), y, x)
            indexers = dict(time=ti, latitude=xr.DataArray(yi, dims="sample"), longitude=xr.DataArray(xi, dims="sample"))
            if source == "cmems":
                indexers["elevation"] = surface_index
            metadata = dict(product_time_utc=utc(actual), grid_lat=y, grid_lon=x, match_distance_km=distance,
                            grid_dlat_deg=float(np.median(np.abs(np.diff(latitudes)))),
                            grid_dlon_deg=float(np.median(np.abs(np.diff(longitudes)))),
                            temporal_semantics="hourly_instantaneous_nearest_hour" if source == "era5" else "daily_mean_not_instantaneous",
                            depth_m=depth if depth is not None else np.nan,
                            time_offset_seconds=(actual-part.time).dt.total_seconds().to_numpy(),
                            support_start_utc=utc(actual) if source == "era5" else utc(wanted_time),
                            support_end_utc=utc(actual) if source == "era5" else utc(wanted_time+pd.Timedelta(days=1)),
                            support_window_basis="instantaneous" if source == "era5" else "PUM_1.7_p9_midnight_to_midnight_centered_noon;ARCO_coordinate_at_midnight",
                            support_center_utc=utc(actual) if source == "era5" else utc(wanted_time+pd.Timedelta(hours=12)),
                            source_url=url,
                            dataset_version="google_arco_full_37_1h_0p25_v3" if source == "era5" else MARINE_DATASET)
            for name, value in metadata.items():
                out.loc[part.index, source+"_"+name] = value
            out.loc[part.index, source+"_shared_cell_time_id"] = [f"{dataset}|{source}|{actual.isoformat()}|{a:.7f}|{b:.7f}|{depth}" for a,b in zip(y,x)]
            for name, (_, variable, _) in fields.items():
                try:
                    values = ds[variable].isel(indexers).values.astype(float)
                    out.loc[part.index, name] = values
                    out.loc[part.index, name+"_status"] = np.where(np.isfinite(values), "actual", "source_missing_at_nearest_cell")
                except Exception as exc:
                    out.loc[part.index, name+"_status"] = error_label(exc)
                    info["last_error"] = error_label(exc)
            if source == "era5":
                try:
                    if lsm is None:
                        lsm = ds.land_sea_mask.isel(time=ti).load()
                        info["land_sea_mask_source_time_utc"] = utc(actual)
                    out.loc[part.index, "era5_atmospheric_land_fraction"] = lsm.isel(latitude=xr.DataArray(yi,dims="sample"),longitude=xr.DataArray(xi,dims="sample")).values
                except Exception as exc:
                    info["land_sea_mask_error"] = error_label(exc)
        info["matched_acquisitions"] += 1
        result = {name:int((out.loc[group.index,name+"_status"]=="actual").sum()) for name in fields}
        print(json.dumps(dict(source=source, acquisition=acquisition, n=len(group), actual=result, network_bytes=dl.bytes)), flush=True)
    info["status"] = "finished"
    info["depth_m"] = depth
    return out, info


def export_tables(frame, parts, source_status, dl, auth):
    merged = pd.concat([frame, *parts], axis=1)
    merged["era5_wind_speed_m_s"] = np.hypot(merged.era5_u10, merged.era5_v10)
    merged["era5_wind_from_deg"] = (np.degrees(np.arctan2(-merged.era5_u10, -merged.era5_v10))+360)%360
    merged["cmems_current_speed_m_s"] = np.hypot(merged.cmems_uo, merged.cmems_vo)
    merged["cmems_current_to_deg"] = (np.degrees(np.arctan2(merged.cmems_uo, merged.cmems_vo))+360)%360
    merged["era5_wave_delivery_note"] = "Google_CDS_requested_0.25deg_regridding;standard_CDS_wave_grid_0.5deg;not_finer_physical_resolution"
    merged["coastal_policy"] = "retain_nearest_cell_missing;no_wet_cell_search;atmospheric_LSM_is_not_wave_mask"
    merged.to_csv(ROOT / "sample_environment.csv", index=False)
    long_rows = []
    for name,(source,variable,units) in VARIABLES.items():
        subset = merged[["dataset", "sample_id", "acquisition", "time", "lat", "lon"]].copy()
        subset["variable"] = name
        subset["source_variable"] = variable
        subset["value"] = merged[name]
        subset["status"] = merged[name+"_status"]
        subset["units"] = units
        subset["source"] = source
        for field in ["product_time_utc", "support_start_utc", "support_end_utc", "support_center_utc", "support_window_basis", "temporal_semantics", "depth_m", "grid_lat", "grid_lon", "grid_dlat_deg", "grid_dlon_deg", "match_distance_km", "time_offset_seconds", "dataset_version", "source_url", "shared_cell_time_id"]:
            subset[field] = merged.get(source+"_"+field, np.nan)
        long_rows.append(subset)
    long_frame = pd.concat(long_rows, ignore_index=True)
    long_frame.to_csv(ROOT / "sample_environment_long.csv", index=False)
    coverage = []
    for (dataset, acquisition), group in merged.groupby(["dataset", "acquisition"],sort=False):
        row = dict(dataset=dataset, acquisition=acquisition, samples=len(group))
        for name in VARIABLES:
            row[name+"_actual"] = int((group[name+"_status"]=="actual").sum())
        coverage.append(row)
    pd.DataFrame(coverage).to_csv(ROOT / "coverage_by_acquisition.csv", index=False)
    counts = {name:merged[name+"_status"].value_counts().to_dict() for name in VARIABLES}
    log = pd.read_csv(dl.log_path) if dl.log_path.exists() else pd.DataFrame(columns=dl.fields)
    data_variables = [v[1] for v in VARIABLES.values()] + ["land_sea_mask"]
    variable_mask = log["key"].str.split("/").str[0].isin(data_variables)
    actual_bytes = int(log.loc[(log.http_status==200)&(log.kind=="array_chunk")&variable_mask,"bytes"].sum())
    status = dict(state="completed_matching_attempt", updated_utc=datetime.now(timezone.utc).isoformat(),
                  manifest_samples=len(frame), acquisition_count=frame.groupby(["dataset","acquisition"]).ngroups,
                  actual_coverage=counts, source_status=source_status, credentials=auth,
                  bytes_downloaded_this_run=dl.bytes, bytes_downloaded_logged_total=int(log["bytes"].sum()),
                  actual_array_chunk_bytes_downloaded_total=actual_bytes,
                  cache_bytes=sum(p.stat().st_size for p in (ROOT/"cache").rglob("*") if p.is_file()),
                  run_download_limit_bytes=dl.limit,
                  sample_rows_with_all_nine_variables=int(np.logical_and.reduce([merged[n+"_status"].eq("actual").to_numpy() for n in VARIABLES]).sum()),
                  no_new_training=True, predictions_remain_SAR_only=True,
                  storage_overfetch="ERA5: single-hour global 721x1440 chunks; CMEMS: 2081-day x 1-layer x 16x16 chunks; only required chunks are fetched, logical point selection retains sample date/location.",
                  boundary="Coarse regional background; shared grid/time records are not independent chip observations.")
    write_json("status.json",status)
    print(json.dumps(status,ensure_ascii=True),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=GEOMETRY)
    parser.add_argument("--max-download-mib", type=float, default=320)
    parser.add_argument("--max-acquisitions", type=int, default=0)
    parser.add_argument("--offline", action="store_true")
    args=parser.parse_args()
    frame=load_manifest(args.manifest)
    frame.to_csv(ROOT/"normalized_manifest.csv",index=False)
    requests_inventory(frame)
    auth=credentials()
    write_json("access_check.json",auth)
    dl=Downloader(args.max_download_mib,args.offline)
    product=json.loads(dl.fetch("documents",STAC_ROOT,PRODUCT+"/product.stac.json"))
    dataset=json.loads(dl.fetch("documents",STAC_ROOT,PRODUCT+"/"+MARINE_DATASET+"/dataset.stac.json"))
    cmems_url=dataset["assets"]["geoChunked"]["href"]
    write_json("provenance.json",dict(era5=dict(product="ERA5",doi="10.24381/cds.adbb2d47",host="Google Research ARCO ERA5 public mirror",url=GOOGLE_ERA5,
                                              producer_documentation="https://github.com/google-research/arco-era5",regrid_config="https://github.com/google-research/arco-era5/blob/main/raw/daily/era5_sl_hourly.cfg"),
                                      cmems=dict(product=PRODUCT,dataset=MARINE_DATASET,doi=product["sci:doi"],host="Copernicus Marine official STAC-linked cloud storage",url=cmems_url)))
    parts,statuses=[],[]
    # Marine first establishes actual independent access without CDS credentials.
    for source,url in [("cmems",cmems_url),("era5",GOOGLE_ERA5)]:
        part,status=match_source(frame,dl,source,url,args.max_acquisitions)
        parts.append(part)
        statuses.append(status)
    export_tables(frame,parts,statuses,dl,auth)


if __name__=="__main__":
    main()
