"""Join historical F_0921 scene-port associations to the current chip cohort."""

import json
from pathlib import Path

import pandas as pd


OUT = Path(__file__).resolve().parent
PAPER = Path("E:/Hermes/paper_tracking")
MANIFEST = PAPER / "sensitivity/f0921_transfer_20261007/target_manifest.csv"
METADATA = OUT.parent / "external_scene_transfer_20261013/f0921_metadata.csv"
SCENE_PORT = PAPER / "f0921_scene_port.json"


def main():
    raw = json.loads(SCENE_PORT.read_text(encoding="utf-8"))["scene_port"]
    scene_port = {}
    for path, port in raw.items():
        scene = Path(path.replace("\\", "/")).stem
        if scene in scene_port and scene_port[scene] != port:
            raise ValueError(f"Conflicting historical port for {scene}")
        scene_port[scene] = port

    chips = pd.read_csv(MANIFEST, dtype={"sample_id": str, "scene": str, "mmsi": str})
    meta = pd.read_csv(METADATA, dtype={"sample_id": str, "scene": str, "identity": str})
    joined = chips.merge(
        meta[["sample_id", "scene", "acquisition", "acquisition_known"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_meta"),
    )
    if len(joined) != len(chips) or not joined.scene.eq(joined.scene_meta).all():
        raise ValueError("Current manifest and acquisition metadata do not align")

    joined["historical_scene_port"] = joined.scene.map(scene_port).fillna("")
    agreed = joined.historical_scene_port.ne("") & joined.historical_scene_port.eq(
        joined.nearest24
    )
    distance = pd.to_numeric(joined.dist24_km, errors="coerce")
    for radius in (25, 50):
        joined[f"port_vicinity_{radius}km"] = joined.historical_scene_port.where(
            agreed & distance.le(radius), ""
        )

    columns = [
        "sample_id", "scene", "acquisition", "acquisition_known", "mmsi",
        "class_id", "lon", "lat", "historical_scene_port", "nearest24",
        "dist24_km", "port_vicinity_25km", "port_vicinity_50km",
    ]
    joined[columns].to_csv(OUT / "f0921_scene_port_recovered.csv", index=False)

    summary = {
        "rows": int(len(joined)),
        "historical_scene_count": len(scene_port),
        "historical_scene_matched_rows": int(joined.historical_scene_port.ne("").sum()),
        "historical_scene_matched_current_scenes": int(
            joined.loc[joined.historical_scene_port.ne(""), "scene"].nunique()
        ),
        "port_vicinity_counts": {
            str(radius): {
                str(k): int(v)
                for k, v in joined.loc[joined[f"port_vicinity_{radius}km"].ne(""),
                                       f"port_vicinity_{radius}km"].value_counts().items()
            }
            for radius in (25, 50)
        },
        "acquisition_known_rows": int(joined.acquisition_known.eq(True).sum()),
        "scene_port_nearest24_disagreements": int(
            (joined.historical_scene_port.ne("") & ~agreed).sum()
        ),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
