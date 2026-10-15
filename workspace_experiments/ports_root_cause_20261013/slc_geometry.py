"""Metadata-fixed range correction versus equal-area wrong-axis compression."""

from settings import *
import argparse
import time
from PIL import Image,ImageDraw
from threadpoolctl import threadpool_limits
from report import metrics

VARIANTS = ("range_geometry","axis_sham")


def corrected(image, ratio, variant):
    a = np.asarray(image.convert("L"))
    h,w = a.shape
    if h != w or not 0 < ratio < 1:
        raise ValueError("Control requires the verified square native SLC grid")
    m = max(1,round(w*ratio))
    size = (m,h) if variant == "range_geometry" else (w,m)
    if variant not in VARIANTS:
        raise ValueError(variant)
    border = np.concatenate((a[0],a[-1],a[:,0],a[:,-1]))
    fill = int(np.rint(np.median(border)))
    resized = image.convert("L").resize(size,Image.Resampling.BILINEAR)
    canvas = Image.new("L",(w,h),fill)
    canvas.paste(resized,((w-size[0])//2,(h-size[1])//2))
    return canvas,dict(native_side=w,compressed_side=m,content_pixels=size[0]*size[1],fill=fill,ratio=ratio)


def extract_features():
    import torch
    import extract_inputs as extract
    f = pilot.frame("oss2")
    geo = pd.read_csv(OUT / "dataset_audit/oss2_slc_native_grid.csv").set_index("sample_id")
    indices = np.flatnonzero(f.scene_id.str.contains("_SLC_",regex=False))
    np.testing.assert_array_equal(np.sort(f.sample_id.iloc[indices]),np.sort(geo.index))
    expected = geo.range_spacing/np.sin(np.deg2rad(geo.incidence_deg))/geo.azimuth_spacing
    np.testing.assert_allclose(geo.range_compress_ratio,expected,atol=1e-12,rtol=0.)
    models = extract.encoders()
    old = INPUT / "features/oss2/original"
    preview_indices = indices[np.linspace(0,len(indices)-1,4,dtype=int)]
    preview = Image.new("RGB",(3*160,4*175),"white")
    draw = ImageDraw.Draw(preview)
    for variant in VARIANTS:
        dst = OUT / "geometry_features" / variant
        if (dst / "COMPLETE.json").exists():
            continue
        started = time.perf_counter()
        dst.mkdir(parents=True,exist_ok=True)
        arrays = {}
        for k in ("raw","c4","extra"):
            source = np.load(old / (k+".npy"),mmap_mode="r")
            arrays[k] = np.lib.format.open_memmap(dst / (k+".npy"),mode="w+",dtype=source.dtype,shape=source.shape)
            arrays[k][:] = source
        records = []
        for lo in range(0,len(indices),32):
            take = indices[lo:lo+32]
            batch = []
            for i in take:
                row = f.iloc[i]
                ratio = float(geo.loc[row.sample_id,"range_compress_ratio"])
                with Image.open(row.path_vv) as vv,Image.open(row.path_vh) as vh:
                    images = [corrected(im,ratio,variant) for im in (vv,vh)]
                    arrays128 = np.stack([np.asarray(im.resize((128,128),Image.Resampling.BILINEAR)) for im,info in images])
                    batch.append((arrays128,extract.TF(images[1][0].convert("RGB")),{}))
                    records.append(dict(sample_id=row.sample_id,**images[0][1],vh_fill=images[1][1]["fill"]))
                    if i in preview_indices:
                        r = int(np.flatnonzero(preview_indices == i)[0])
                        c = VARIANTS.index(variant)+1
                        preview.paste(images[0][0].resize((148,148)),(c*160,r*175+25))
                        draw.text((c*160+2,r*175+4),variant,fill="black")
                        if c == 1:
                            preview.paste(vv.convert("L").resize((148,148)),(0,r*175+25))
                            draw.text((2,r*175+4),"Original SLC",fill="black")
            with torch.inference_mode():
                values = extract.forward(batch,*models)
            for k,v in values.items():
                assert np.isfinite(v).all()
                arrays[k][take] = v
            if lo % 256 == 0:
                print("GEOMETRY_FEATURES",variant,min(lo+32,len(indices)),len(indices),flush=True)
        unmodified = np.setdiff1d(np.arange(len(f)),indices)
        for k,a in arrays.items():
            a.flush()
            np.testing.assert_array_equal(a[unmodified],np.load(old / (k+".npy"),mmap_mode="r")[unmodified])
        np.save(dst / "ids.npy",f.sample_id.to_numpy(str))
        pd.DataFrame(records).to_csv(dst / "controls.csv",index=False)
        write(dst / "COMPLETE.json",dict(variant=variant,slc_rows=len(indices),unchanged_grdh_rows=len(unmodified),
            seconds=time.perf_counter()-started,source_refit=False))
    a = pd.read_csv(OUT / "geometry_features/range_geometry/controls.csv")
    b = pd.read_csv(OUT / "geometry_features/axis_sham/controls.csv")
    for k in ("sample_id","native_side","compressed_side","content_pixels","fill","vh_fill","ratio"):
        np.testing.assert_array_equal(a[k],b[k])
    preview.save(OUT / "slc_geometry_controls.png")
    write(OUT / "GEOMETRY_FEATURES_COMPLETE.json",dict(slc_rows=999,equal_actual_pixel_area=True,
        equal_padding_values=True,grdh_features_unchanged=True,parameters_from_metadata=True))


def data_for(name,condition):
    assert name == "oss2"
    data = target(name,"original")
    folder = OUT / "geometry_features" / condition
    np.testing.assert_array_equal(np.load(folder / "ids.npy"),data["ids"])
    for k in ("raw","c4","extra"):
        data[k] = np.load(folder / (k+".npy"),mmap_mode="r")
    return data


def predict():
    import run_factorial as run
    assert (OUT / "GEOMETRY_FEATURES_COMPLETE.json").exists()
    run.target = data_for
    for fold in FOLDS:
        bundle = run.bundle_for(fold,"original")
        for variant in VARIANTS:
            run.predict(fold,"original",variant,"oss2",bundle)
    write(OUT / "GEOMETRY_PREDICTIONS_FROZEN.json",dict(contexts=4,variants=VARIANTS,source_refit=False,query_metrics_evaluated=False))


def analyze():
    from analyze_factorial import load,metadata,scopes,weighted_events
    assert (OUT / "GEOMETRY_PREDICTIONS_FROZEN.json").exists()
    rows,effects,classes,pairrows = [],[],[],[]
    pairs = pd.read_csv(OUT / "dataset_audit/oss2_grdh_slc_pairs.csv",keep_default_na=False)
    pairs = pairs[pairs.same_label]
    for fold in FOLDS:
        caches = {"original":load(fold,"oss2","original","original")}
        caches.update({v:load(fold,"oss2","original",v) for v in VARIANTS})
        saved = caches["original"][0]
        yall = labels("oss2",saved["ids"])
        f = metadata("oss2",saved["ids"])
        ix = pd.Index(saved["ids"])
        gi,si = ix.get_indexer(pairs.sample_id_grdh),ix.get_indexer(pairs.sample_id_slc)
        for other,prob in caches.values():
            for k in ("ids","support","eval_masks"):
                np.testing.assert_array_equal(other[k],saved[k])
            np.testing.assert_allclose(prob.sum(-1,dtype=float),1.,atol=1e-7,rtol=0.)
            grdh = f.scene_id.str.contains("_GRDH_",regex=False).to_numpy()
            np.testing.assert_array_equal(prob[:,:,0,grdh],caches["original"][1][:,:,0,grdh])
        for draw,ev in enumerate(saved["eval_masks"]):
            for scope,mask in scopes("oss2",f,saved,draw).items():
                y = yall[mask]
                for ai,arm in enumerate(ARMS):
                    for hi,head in enumerate(HEADS):
                        meta = dict(fold=fold,draw=draw,scope=scope,arm=arm,head=head)
                        values = {}
                        for variant,(pred,prob) in caches.items():
                            pp = pred["predictions"][draw,ai,hi,mask]
                            p = np.asarray(prob[draw,ai,hi,mask],float)
                            value = metrics(y,pp,p)
                            values[variant] = (value,pp,p)
                            rows.append(dict(**meta,variant=variant,**value))
                            for c in np.unique(y):
                                cm = y == c
                                classes.append(dict(**meta,variant=variant,class_id=int(c),n=int(cm.sum()),
                                    recall=100*np.mean(pp[cm] == c)))
                        for a,b in (("range_geometry","original"),("axis_sham","original"),("range_geometry","axis_sham")):
                            va,pa,proba = values[a]
                            vb,pb,probb = values[b]
                            w,rescue,harm = weighted_events(y,pb,pa)
                            net = w[rescue].sum()-w[harm].sum()
                            np.testing.assert_allclose(net,va["ba"]-vb["ba"],atol=1e-10,rtol=0.)
                            effects.append(dict(**meta,contrast=a+"-"+b,delta_ba=net,delta_oa=va["oa"]-vb["oa"],
                                risk_gain_macro=vb["brier_macro"]-va["brier_macro"],rescue_ba=w[rescue].sum(),harm_ba=w[harm].sum()))
            eligible = ev[gi] & ev[si]
            for scope,selected in (("paired_all",eligible),("paired_both_clean",eligible &
                (pairs.noise_xml_grdh.astype(str) == "0").to_numpy() & (pairs.noise_xml_slc.astype(str) == "0").to_numpy())):
                g,s = gi[selected],si[selected]
                y = yall[g]
                np.testing.assert_array_equal(y,yall[s])
                for variant,(pred,prob) in caches.items():
                    for ai,arm in enumerate(ARMS):
                        for hi,head in enumerate(HEADS):
                            gm = metrics(y,pred["predictions"][draw,ai,hi,g],np.asarray(prob[draw,ai,hi,g],float))
                            sm = metrics(y,pred["predictions"][draw,ai,hi,s],np.asarray(prob[draw,ai,hi,s],float))
                            pairrows.append(dict(fold=fold,draw=draw,scope=scope,variant=variant,arm=arm,head=head,pairs=len(y),
                                grdh_ba=gm["ba"],slc_ba=sm["ba"],grdh_oa=gm["oa"],slc_oa=sm["oa"],
                                grdh_brier=gm["brier_macro"],slc_brier=sm["brier_macro"],gap=sm["ba"]-gm["ba"]))
    for filename,records,keys in (("geometry_metrics",rows,["scope","arm","head","variant"]),
        ("geometry_effects",effects,["scope","arm","head","contrast"]),
        ("geometry_pairs",pairrows,["scope","arm","head","variant"])):
        table = pd.DataFrame(records)
        table.to_csv(OUT / (filename+".csv"),index=False)
        measures = [k for k in table.columns if k not in keys+["fold","draw"]]
        table.groupby(keys)[measures].mean().reset_index().to_csv(OUT / (filename+"_summary.csv"),index=False)
    pd.DataFrame(classes).to_csv(OUT / "geometry_classes.csv",index=False)
    write(OUT / "GEOMETRY_READOUT_COMPLETE.json",dict(metric_rows=len(rows),effect_rows=len(effects),pair_rows=len(pairrows),
        fixed_supports=True,equal_area_sham=True,no_parameter_search=True))
    print(pd.read_csv(OUT / "geometry_effects_summary.csv").query("scope in ['original','query_SLC'] and head == 'expert'").to_string(index=False),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage",choices=["extract","predict","analyze"])
    with threadpool_limits(limits=2):
        {"extract":extract_features,"predict":predict,"analyze":analyze}[parser.parse_args().stage]()
