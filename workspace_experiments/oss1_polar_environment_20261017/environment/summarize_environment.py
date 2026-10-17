"""Summarize source coverage and shared cells, without fitting or inference inputs."""

import json

import fetch_match_environment as match
import numpy as np
import pandas as pd


def main():
    root=match.ROOT
    d=pd.read_csv(root/"sample_environment.csv")
    summary={"samples":len(d),"acquisitions":d.acquisition.nunique(),"variables":{},"sources":{}}
    for name in match.VARIABLES:
        values=d[name].dropna()
        summary["variables"][name]=dict(actual=int(values.size),missing=int(d[name].isna().sum()),
                                        min=float(values.min()),median=float(values.median()),max=float(values.max()))
        if name=="era5_mean_wave_direction":
            del summary["variables"][name]["median"]
    cells=[]
    for source in ["era5","cmems"]:
        fields=[n for n,(s,_,_) in match.VARIABLES.items() if s==source]
        key=source+"_shared_cell_time_id"
        for cell,g in d.groupby(key,sort=False):
            row=dict(source=source,shared_cell_time_id=cell,dataset=g.dataset.iloc[0],
                     product_time_utc=g[source+"_product_time_utc"].iloc[0],
                     grid_lat=float(g[source+"_grid_lat"].iloc[0]),grid_lon=float(g[source+"_grid_lon"].iloc[0]),
                     sample_count=len(g),vessel_count=g.mmsi.nunique() if "mmsi" in g else None,
                     acquisition_count=g.acquisition.nunique(),class_count=g.class_name.nunique() if "class_name" in g else None)
            for f in fields:
                row[f]=g[f].iloc[0]
            cells.append(row)
        valid=np.logical_and.reduce([d[f].notna().to_numpy() for f in fields])
        distance=d[source+"_match_distance_km"]
        summary["sources"][source]=dict(unique_cell_times=int(d[key].nunique()),
                                        unique_valid_all_variable_cell_times=int(d.loc[valid,key].nunique()),
                                        max_samples_sharing_cell_time=int(d.groupby(key).size().max()),
                                        distance_km_median=float(distance.median()),distance_km_max=float(distance.max()))
    pd.DataFrame(cells).to_csv(root/"unique_environment_records.csv",index=False)
    variable_names=list(match.VARIABLES)
    missing=d[variable_names].isna().any(axis=1)
    columns=["dataset","sample_id","acquisition","time","lat","lon","class_name","mmsi",
             "era5_grid_lat","era5_grid_lon","cmems_grid_lat","cmems_grid_lon","era5_atmospheric_land_fraction"]
    columns=[c for c in columns if c in d]+variable_names+[c+"_status" for c in variable_names]
    d.loc[missing,columns].to_csv(root/"missing_environment_samples.csv",index=False)
    acquisition_rows=[]
    for (dataset,acquisition),g in d.groupby(["dataset","acquisition"],sort=False):
        row=dict(dataset=dataset,acquisition=acquisition,samples=len(g),sar_time=g.time.iloc[0])
        for name,(source,_,_) in match.VARIABLES.items():
            if name=="era5_mean_wave_direction":
                continue
            # The scene summary uses distinct cells, not repeated copies per ship.
            unique=g.drop_duplicates(source+"_shared_cell_time_id")[name]
            row[name+"_unique_cell_mean"]=unique.mean()
            row[name+"_valid_cells"]=int(unique.notna().sum())
        for source in ["era5","cmems"]:
            row[source+"_unique_cells"]=g[source+"_shared_cell_time_id"].nunique()
        acquisition_rows.append(row)
    pd.DataFrame(acquisition_rows).to_csv(root/"acquisition_environment_summary.csv",index=False)
    summary["all_nine_actual_samples"]=int((~missing).sum())
    summary["missing_any_samples"]=int(missing.sum())
    summary["wave_missing_with_atmospheric_land_fraction_gt_half"]=int((d.era5_hs.isna()&(d.era5_atmospheric_land_fraction>0.5)).sum())
    summary["wave_missing_with_atmospheric_land_fraction_le_half"]=int((d.era5_hs.isna()&(d.era5_atmospheric_land_fraction<=0.5)).sum())
    summary["atmospheric_land_fraction_gt_half_samples"]=int((d.era5_atmospheric_land_fraction>0.5).sum())
    summary["mean_wave_direction_aggregation"]="not_averaged;retain_native_FROM_degrees_per_cell"
    match.write_json("environment_summary.json",summary)
    print(json.dumps(summary,ensure_ascii=True,indent=2))


if __name__=="__main__":
    main()
