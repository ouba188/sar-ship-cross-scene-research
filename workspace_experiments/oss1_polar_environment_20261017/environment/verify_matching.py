"""Offline verification against packed Zarr values, independent of xarray decoding."""

import json
import sys

import fetch_match_environment as match
import numpy as np
import pandas as pd
import zarr


def decoded_times(array):
    unit, origin = array.attrs["units"].split(" since ", 1)
    units = {"hours": "h", "days": "D", "seconds": "s", "milliseconds": "ms"}
    return pd.to_datetime(array[:], unit=units[unit], origin=origin, utc=True)


def main():
    root = match.ROOT
    frame = pd.read_csv(root/"sample_environment.csv")
    manifest = pd.read_csv(root/"normalized_manifest.csv")
    requested = pd.read_csv(root/"acquisition_requests.csv")
    assert len(frame) == len(manifest) == 1349
    assert frame[["dataset", "sample_id"]].equals(manifest[["dataset", "sample_id"]])
    assert len(requested) == frame.acquisition.nunique() == 22
    for name in match.VARIABLES:
        assert frame[name].notna().equals(frame[name+"_status"].eq("actual")), name
        assert set(frame[name+"_status"]) <= {"actual", "source_missing_at_nearest_cell"}, name
    sar_time = pd.to_datetime(frame.time, utc=True)
    era_time = pd.to_datetime(frame.era5_product_time_utc, utc=True)
    marine_start = pd.to_datetime(frame.cmems_support_start_utc, utc=True)
    marine_end = pd.to_datetime(frame.cmems_support_end_utc, utc=True)
    assert ((era_time-sar_time).abs() <= pd.Timedelta(minutes=30)).all()
    assert ((marine_start <= sar_time) & (sar_time < marine_end)).all()
    assert ((marine_end-marine_start) == pd.Timedelta(days=1)).all()
    assert np.isclose(frame.cmems_depth_m, 0.49402499198913574).all()
    for source in ["era5", "cmems"]:
        for name,(s,_,_) in match.VARIABLES.items():
            if s == source:
                assert frame.groupby(source+"_shared_cell_time_id")[name].nunique().max() <= 1

    # Three actual geographic groups plus every distinct source-missing pattern.
    selected = [0, int(frame.index[frame.lat.between(29,32)][0]), int(frame.index[frame.lat<25][0])]
    for source, var in [("cmems", "cmems_uo"), ("era5", "era5_hs")]:
        missing = frame.loc[frame[var].isna()].drop_duplicates(source+"_shared_cell_time_id")
        selected.extend(missing.head(1).index.tolist())
    selected = list(dict.fromkeys(selected))
    dl = match.Downloader(0, True)
    evidence = []
    for source in ["cmems", "era5"]:
        inventory = json.loads((root/(source+"_source_inventory.json")).read_text(encoding="utf-8"))
        group = zarr.open_consolidated(match.CachedHTTPStore(dl,source,inventory["url"]),mode="r")
        time = decoded_times(group["time"])
        lat=group["latitude"][:].astype(np.float64)
        lon=group["longitude"][:].astype(np.float64)
        for index in selected:
            row = frame.loc[index]
            actual_time = pd.Timestamp(row[source+"_product_time_utc"])
            ti = int(np.flatnonzero(time==actual_time)[0])
            yi = int(np.flatnonzero(np.isclose(lat,row[source+"_grid_lat"],atol=1e-7,rtol=0))[0])
            normalized_lon = (lon+180)%360-180
            xi = int(np.flatnonzero(np.isclose(normalized_lon,row[source+"_grid_lon"],atol=1e-7,rtol=0))[0])
            assert abs(lat[yi]-row.lat) <= abs(float(lat[1]-lat[0]))/2+2e-5
            assert abs(normalized_lon[xi]-row.lon) <= abs(float(lon[1]-lon[0]))/2+2e-5
            if source=="cmems":
                zi=int(np.abs(group["elevation"][:]).argmin())
                ix=(ti,zi,yi,xi)
            else:
                ix=(ti,yi,xi)
            values={}
            for name,(s,variable,_) in match.VARIABLES.items():
                if s!=source:
                    continue
                arr=group[variable]
                assert list(arr.attrs["_ARRAY_DIMENSIONS"]) == (["time","elevation","latitude","longitude"] if source=="cmems" else ["time","latitude","longitude"])
                raw=arr[ix]
                valid=bool(np.isfinite(raw) and raw!=arr.fill_value)
                value=float(raw)*arr.attrs.get("scale_factor",1)+arr.attrs.get("add_offset",0) if valid else None
                if value is None:
                    assert pd.isna(row[name]) and row[name+"_status"]=="source_missing_at_nearest_cell"
                else:
                    assert np.isclose(row[name],value,rtol=1e-7,atol=1e-8), (name,row[name],value)
                values[name]={"packed_raw":float(raw) if np.isfinite(raw) else None,"decoded":value,"csv_value":None if pd.isna(row[name]) else float(row[name])}
            evidence.append(dict(sample_id=row.sample_id, acquisition=row.acquisition, source=source,
                                 sar_time_utc=row.time, sar_lat=float(row.lat),sar_lon=float(row.lon),
                                 product_time_utc=actual_time.isoformat(), source_indices=list(ix),
                                 grid_lat=float(lat[yi]),grid_lon=float(normalized_lon[xi]),
                                 match_distance_km=float(row[source+"_match_distance_km"]), values=values))
    summary=dict(result="passed", verification="Offline original packed Zarr scalar values vs xarray-derived CSV; coordinate/time/depth/fill/status/shared-cell consistency",samples=1349,
                 acquisitions=22, checked_examples=len(selected), checked_source_examples=len(evidence),network_bytes=dl.bytes,
                 selected_sample_ids=frame.loc[selected,"sample_id"].tolist(), examples=evidence)
    match.write_json("verification.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k!="examples"},ensure_ascii=True))


if __name__=="__main__":
    main()
