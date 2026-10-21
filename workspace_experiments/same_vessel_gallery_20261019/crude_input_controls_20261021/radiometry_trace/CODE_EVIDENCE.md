# Source Code Evidence

Read-only excerpts of local source files. These document the stored pipeline; no remote runtime version or per-scene stretch limits are claimed.

## E:\临时会话\asf_ais_first_screen_20260913\server_zip_pipeline.sh

```text
4: base=/root/autodl-tmp/sar_download_v1
5: zipdir="$base/zips"
6: manifest="$base/server_geocode_manifest.csv"
7: tool=/root/autodl-tmp/v5_server_toolkit_20260912/scripts/sentinel1_geocode_manifest.py
8: work="$base/v5"
```

```text
54:     /usr/bin/python3 "$tool" \
55:       --manifest "$one_manifest" \
56:       --output-root "$work/geocode" \
57:       --log-dir "$work/logs/geocode" \
58:       --pol ALL --pixel-size 10 --clip-percent 2 --gdal-threads 1 \
59:       >> "$work/logs/geocode_${key}.log" 2>&1 || true
60: 
61:     if awk -F, -v key="$key" '$1 == key && $2 == "ok" {found=1} END {exit !found}' "$work/logs/geocode/geocode_results.csv" 2>/dev/null; then
62:       touch "$work/processed/$key.ok"
63:       printf '%s GEOCODED %s\n' "$(date -Is)" "$key" >> "$work/logs/pipeline.log"
64:       if rm -f -- "$z"; then
65:         printf '%s REMOVED_ZIP %s\n' "$(date -Is)" "$key" >> "$work/logs/pipeline.log"
66:       else
67:         printf '%s REMOVE_ZIP_FAILED %s\n' "$(date -Is)" "$key" >> "$work/logs/pipeline.log"
68:       fi
69:     else
70:       printf '%s FAILED geocode %s\n' "$(date -Is)" "$key" >> "$work/logs/pipeline.log"
71:     fi
72:   done
73: 
74:   [ "$found_work" -eq 1 ] || sleep 30
```

## E:\临时会话\asf_ais_first_screen_20260913\server2_zip_pipeline.sh

```text
4: old_base=/root/autodl-fs/asf_download_v2_allports
5: new_base=/root/autodl-tmp/asf_download_v2_tmp
6: zipdirs=("$old_base/zips" "$new_base/zips")
7: manifest="$old_base/server2_geocode_manifest.csv"
8: tool=/root/autodl-fs/v5_server_toolkit_20260912/scripts/sentinel1_geocode_manifest.py
9: old_work="$old_base/v5"
10: work="$new_base/v5"
11: 
```

```text
76:       if awk -F, -v key="$key" '$1 == key && $2 == "ok" {found=1} END {exit !found}' "$work/logs/geocode/geocode_results.csv" 2>/dev/null; then
77:         touch "$work/processed/$key.ok"
78:         printf '%s GEOCODED %s\n' "$(date -Is)" "$key" >> "$work/logs/pipeline.log"
79:         if rm -f -- "$z"; then
80:           printf '%s REMOVED_ZIP %s\n' "$(date -Is)" "$key" >> "$work/logs/pipeline.log"
81:         else
82:           printf '%s REMOVE_ZIP_FAILED %s\n' "$(date -Is)" "$key" >> "$work/logs/pipeline.log"
83:         fi
84:       else
85:         printf '%s FAILED geocode %s\n' "$(date -Is)" "$key" >> "$work/logs/pipeline.log"
86:       fi
87:     done
88:   done
89: 
90:   [ "$found_work" -eq 1 ] || sleep 30
91: done
```

## E:\临时会话\v5_server_toolkit_20260912\scripts\sentinel1_geocode_manifest.py

```text
84: def create_8bit(gdal, src16: Path, dst8: Path, clip_percent: float) -> None:
85:     ds = gdal.Open(str(src16))
86:     if ds is None:
87:         raise RuntimeError(f"could not open {src16}")
88:     band = ds.GetRasterBand(1)
89:     lo_bound, hi_bound = [float(v) for v in band.ComputeRasterMinMax(True)]
90:     if hi_bound <= lo_bound:
91:         lo, hi = lo_bound, lo_bound + 1.0
92:     else:
93:         buckets = 4096
94:         hist = band.GetHistogram(lo_bound, hi_bound, buckets, include_out_of_range=0, approx_ok=1)
95:         total = max(1, sum(hist))
96:         low_target = total * clip_percent / 100.0
97:         high_target = total * (100.0 - clip_percent) / 100.0
98:         cumulative = 0
99:         lo_idx = 0
100:         hi_idx = buckets - 1
101:         for idx, count in enumerate(hist):
102:             cumulative += count
103:             if cumulative >= low_target:
104:                 lo_idx = idx
105:                 break
106:         cumulative = 0
107:         for idx, count in enumerate(hist):
108:             cumulative += count
109:             if cumulative >= high_target:
110:                 hi_idx = idx
111:                 break
112:         step = (hi_bound - lo_bound) / buckets
113:         lo = lo_bound + lo_idx * step
114:         hi = lo_bound + max(hi_idx, lo_idx + 1) * step
115:     dst8.parent.mkdir(parents=True, exist_ok=True)
116:     out_ds = gdal.Translate(
117:         str(dst8),
118:         ds,
119:         outputType=gdal.GDT_Byte,
120:         scaleParams=[[lo, hi, 0, 255]],
121:         creationOptions=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"],
122:     )
123:     if out_ds is None:
124:         raise RuntimeError(f"gdal.Translate failed for {src16}")
125:     out_ds.GetRasterBand(1).SetNoDataValue(0)
126:     out_ds.FlushCache()
127:     out_ds = None
128:     ds = None
```

```text
148:         return {"status": "failed", "error": f"no selected polarization {pol}"}
149: 
150:     first_ds = gdal.Open(vsizip_path(zip_path, selected[0][0]))
151:     if first_ds is None:
152:         return {"status": "failed", "error": "could not open first measurement"}
153:     epsg = utm_epsg(first_ds)
154:     first_ds = None
155: 
156:     made = []
157:     skipped = []
158:     for member, member_pol in selected:
159:         dst8 = out_utm / member_pol / "int8" / f"{product_key}_{member_pol}_UTM_8bit.tif"
160:         if dst8.exists() and dst8.stat().st_size > 0:
161:             skipped.append(str(dst8))
162:             continue
163:         tmp16 = out_utm / member_pol / "int16" / f"{product_key}_{member_pol}_UTM.tif"
164:         tmp16.parent.mkdir(parents=True, exist_ok=True)
165:         log(f"[warp] {product_key} {member_pol} {epsg}")
166:         result = gdal.Warp(
167:             str(tmp16),
168:             vsizip_path(zip_path, member),
169:             dstSRS=epsg,
170:             xRes=pixel_size,
171:             yRes=pixel_size,
172:             resampleAlg="near",
173:             format="GTiff",
174:             outputType=gdal.GDT_UInt16,
175:             multithread=False,
176:             warpMemoryLimit=128,
177:             warpOptions=[f"NUM_THREADS={gdal_threads}"],
178:             creationOptions=["TILED=YES", "BIGTIFF=IF_SAFER"],
179:         )
180:         if result is None:
181:             return {"status": "failed", "error": f"gdal.Warp failed for {member_pol}"}
182:         result = None
183:         create_8bit(gdal, tmp16, dst8, clip_percent)
184:         try:
185:             tmp16.unlink()
186:         except OSError:
187:             try:
188:                 gdal.Unlink(str(tmp16))
189:             except Exception:
190:                 pass
191:         for sidecar in tmp16.parent.glob(tmp16.name + ".*"):
192:             try:
193:                 sidecar.unlink()
194:             except OSError:
195:                 pass
196:         try:
197:             tmp16.parent.rmdir()
```

## E:\临时会话\v5_server_toolkit_20260912\scripts\tile_sarlabelimg_quadrants.py

```text
233: def write_quadrant(
234:     source: rasterio.io.DatasetReader,
235:     output: Path,
236:     quadrant: Quadrant,
237: ) -> None:
238:     os.makedirs(filesystem_path(output.parent), exist_ok=True)
239:     partial = output.with_suffix(output.suffix + ".partial")
240:     window = Window(quadrant.x0, quadrant.y0, quadrant.width, quadrant.height)
241:     profile = source.profile.copy()
242:     profile.update(
243:         width=quadrant.width,
244:         height=quadrant.height,
245:         transform=transform(window, source.transform),
246:         tiled=True,
247:         blockxsize=512,
248:         blockysize=512,
249:         compress="LZW",
250:         BIGTIFF="IF_SAFER",
251:     )
252:     with rasterio.open(filesystem_path(partial), "w", **profile) as target:
253:         target.write(source.read(window=window))
254:     with rasterio.open(filesystem_path(partial)) as check:
```

```text
318:     rows: list[dict[str, Any]] = []
319:     with rasterio.open(filesystem_path(tif)) as source:
320:         if source.count != 1 or source.dtypes[0] != "uint8":
321:             raise ValueError(f"expected one-band uint8 TIFF: {tif}")
322:         quadrants = build_grid(source.width, source.height, overlap, grid_size)
```

## E:\临时会话\build_v5_context_chips.py

```text
265:                 with Image.open(tiff_path) as image:
266:                     for row in group:
267:                         try:
268:                             obj = find_object(row, objects, by_uid)
269:                             cx = (float(row["left"]) + float(row["right"])) / 2
270:                             cy = (float(row["top"]) + float(row["bottom"])) / 2
271:                             side = crop_side(row, args.min_side, args.scale, args.max_side)
272:                             left, top, right, bottom = square_window(
273:                                 cx, cy, side, image.width, image.height
274:                             )
275:                             chip = image.crop((left, top, right, bottom))
276:                             destination = (
277:                                 image_root
278:                                 / row["port"]
279:                                 / row["class_name"]
280:                                 / Path(row["output_png"]).name
281:                             )
282:                             destination.parent.mkdir(parents=True, exist_ok=True)
283:                             if not destination.is_file():
284:                                 chip.save(destination, format="PNG", compress_level=3)
```

## E:\临时会话\build_v5_context_chips_gdal.py

```text
88:     objs = payload.get("objects", [])
89:     return objs, {str(o.get("obj_uid", "")).strip().lower(): o for o in objs if o.get("obj_uid")}
90: 
91: 
92: def write_png(dataset, out_path, left, top, right, bottom):
93:     width, height = right - left, bottom - top
94:     if width <= 0 or height <= 0:
95:         raise ValueError(f"invalid crop {left},{top},{right},{bottom}")
96:     band = dataset.ReadAsArray(left, top, width, height)
97:     if band is None:
98:         raise RuntimeError("GDAL ReadAsArray returned None")
99:     out_path.parent.mkdir(parents=True, exist_ok=True)
100:     if out_path.is_file():
101:         return
102:     drv = gdal.GetDriverByName("PNG")
103:     mem_drv = gdal.GetDriverByName("MEM")
104:     if drv is None or mem_drv is None:
105:         raise RuntimeError("GDAL PNG/MEM driver unavailable")
106:     # This GDAL build exposes PNG CreateCopy but not PNG Create.  Build a
107:     # temporary in-memory dataset, then encode it with CreateCopy.
108:     bands = 1 if getattr(band, "ndim", 2) == 2 else band.shape[0]
109:     mem = mem_drv.Create("", width, height, bands, dataset.GetRasterBand(1).DataType)
110:     if bands == 1:
111:         mem.GetRasterBand(1).WriteArray(band)
112:     else:
113:         for idx in range(bands):
114:             mem.GetRasterBand(idx + 1).WriteArray(band[idx])
115:     out = drv.CreateCopy(str(out_path), mem, strict=0, options=["ZLEVEL=3"])
116:     mem = None
117:     if out is None:
118:         raise RuntimeError(f"GDAL PNG CreateCopy failed for {out_path}")
119:     out.FlushCache()
120:     out = None
```

```text
169:                         cx = (float(row["left"]) + float(row["right"])) / 2
170:                         cy = (float(row["top"]) + float(row["bottom"])) / 2
171:                         side = crop_side(row, args.min_side, args.scale, args.max_side)
172:                         left, top, right, bottom = square_window(cx, cy, side, ds.RasterXSize, ds.RasterYSize)
173:                         dest = args.output_root / "images" / row["port"] / row["class_name"] / Path(row["output_png"]).name
174:                         write_png(ds, dest, left, top, right, bottom)
175:                         meta = (obj or {}).get("meta") or {}
176:                         out = {f: row.get(f, "") for f in BASE_FIELDS}
177:                         out.update({
178:                             "context_output_png": str(dest), "crop_left": left, "crop_top": top,
179:                             "crop_right": right, "crop_bottom": bottom, "crop_side": side,
180:                             "target_center_x": cx, "target_center_y": cy,
181:                             "redetect_status": meta.get("redetect_status", ""),
182:                             "redetect_model_conf": meta.get("redetect_model_conf", (obj or {}).get("conf", "")),
183:                             "position_method": meta.get("position_method", ""),
184:                             "abs_time_delta_seconds": meta.get("abs_time_delta_seconds", ""),
```
