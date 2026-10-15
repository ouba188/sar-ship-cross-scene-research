"""Recover actual chip origins from native OSS1 XML; no outcome-based selection."""

from replication_context import *
from lxml import etree as ET
from PIL import Image
from types import SimpleNamespace
from transforms_regions import region_boxes


def main():
    frame = rc.run.pilot.frame("oss1").reset_index(drop=True)
    details = pd.read_csv(rc.PREVIOUS / "dataset_audit/public_metadata.csv", keep_default_na=False)
    details = details[details.dataset.eq("oss1")].set_index("sample_id").loc[frame.sample_id]
    frame["identity"] = details.identity.astype(str).to_numpy()
    frame["datatake_key"] = details.datatake_key.to_numpy()
    frame["noise_xml"] = details.noise_xml.to_numpy()
    records = []
    for scene, group in frame.groupby("scene_id", sort=False):
        xml = Path(group.path_vv.iloc[0]).parent.parent / "Ship.xml"
        raw = xml.read_bytes().decode("utf-8", errors="replace").encode("utf-8")
        root = ET.fromstring(raw, ET.XMLParser(recover=True))
        ships = {}
        for s in root.findall(".//ship"):
            sar = s.find("SARShipInformation")
            fields = {k: int(sar.findtext(k)) for k in (
                "Head_x", "Head_y", "Tail_x", "Tail_y", "Center_x", "Center_y",
                "UpperLeft_x", "UpperLeft_y", "LowerRight_x", "LowerRight_y")}
            key = fields["Center_x"], fields["Center_y"]
            if key in ships:
                raise ValueError("Ambiguous XML center")
            fields["mmsi"] = s.findtext("AISShipInformation/MMSI").strip()
            ships[key] = fields
        for row in group.itertuples():
            source = ships[int(row.center_x), int(row.center_y)]
            if str(row.mmsi_or_ship_id) != source["mmsi"]:
                raise ValueError("MMSI/XML mismatch")
            with Image.open(row.path_vv) as vv, Image.open(row.path_vh) as vh:
                w, h = vv.size
                assert vh.size == (w, h)
            x0 = min(source["UpperLeft_x"], source["LowerRight_x"])
            y0 = min(source["UpperLeft_y"], source["LowerRight_y"])
            ew = abs(source["UpperLeft_x"]-source["LowerRight_x"])+1
            eh = abs(source["UpperLeft_y"]-source["LowerRight_y"])+1
            exact = (w, h) == (ew, eh)
            local = dict(head_x=source["Head_x"]-x0, head_y=source["Head_y"]-y0,
                         tail_x=source["Tail_x"]-x0, tail_y=source["Tail_y"]-y0,
                         center_x=w//2, center_y=h//2)
            _, _, valid, _, length, _ = region_boxes(SimpleNamespace(**local), (h, w), .20)
            valid = bool(valid and exact)
            if not valid:
                # Invalid mapping remains unchanged by the shared transform, never re-centered by guess.
                local["head_x"] = local["tail_x"] = -1
                local["head_y"] = local["tail_y"] = -1
            records.append(dict(sample_id=row.sample_id, native_h=h, native_w=w,
                xml_origin_x=x0, xml_origin_y=y0, xml_extent_w=ew, xml_extent_h=eh,
                exact_xml_extent=exact, region_valid=valid, headtail_length=length, **local))
    table = pd.DataFrame(records).set_index("sample_id").loc[frame.sample_id].reset_index()
    for key in table.columns:
        if key != "sample_id":
            frame[key] = table[key].to_numpy()
    folder = OUT / "oss1"
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / "frame.csv", index=False)
    table.to_csv(folder / "localization.csv", index=False)
    write(folder / "LOCALIZATION.json", dict(rows=len(frame), valid=int(table.region_valid.sum()),
        invalid=table.loc[~table.region_valid].to_dict("records"),
        exact_xml_extent=int(table.exact_xml_extent.sum()),
        region_definition="same head-tail bbox padding .20, wider .35; native XML origin subtraction",
        noise_missing_not_clean=True, images_modified=False))
    print("OSS1_LOCALIZATION", len(frame), int(table.region_valid.sum()), flush=True)


if __name__ == "__main__":
    main()
