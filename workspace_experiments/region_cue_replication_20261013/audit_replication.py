"""Bounded independent audit. CPU only; writes AUDIT*.json/md beside this file."""

import argparse
import ast
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parent
WORK = OUT.parent
OLD = WORK / "region_cue_probe_20261013"
CROSS = WORK / "ports_public_data_study_20261012"
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(OUT))
CONDITIONS = ("original", "target_flat", "outer_flat", "sham_flat", "target_shuffle",
              "outer_shuffle", "target_display", "outer_display", "wide_target_flat", "wide_outer_flat")
PATHS = ("R_query", "T_Aonly", "T_Bonly", "T_query", "R_joint", "T_joint")


def read(path):
    return pd.read_csv(path, keep_default_na=False, low_memory=False)


def save(name, value):
    (OUT / ("AUDIT_"+name+".json")).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def eq(actual, expected):
    np.testing.assert_array_equal(actual, expected)


def near(actual, expected, tol=1e-8):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tol)


def mask(shape, bounds):
    yy, xx = np.indices(shape)
    l, t, r, b = bounds
    return (xx >= l) & (xx < r) & (yy >= t) & (yy < b)


def expected_box(row, pol, wide=False):
    l, t, r, b = [float(row[f"bbox_{k}_{pol}"]) for k in ("x0", "y0", "x1", "y1")]
    pad = max(3, (.35 if wide else .20)*max(r-l, b-t))
    return (max(0, math.floor(l-pad)), max(0, math.floor(t-pad)),
            min(int(row["native_w_"+pol]), math.ceil(r+pad)),
            min(int(row["native_h_"+pol]), math.ceil(b+pad)))


def pixels():
    import port_transforms as port
    import transforms_regions as public
    import tifffile
    # Inspect the frozen CPU preprocessing configuration without loading model libraries.
    legacy = ast.parse((WORK / "ports_input_controls_20261012/extract_inputs.py").read_text(encoding="utf-8"))
    tf_node = next(n.value for n in legacy.body if isinstance(n,ast.Assign) and
                   any(isinstance(t,ast.Name) and t.id=="TF" for t in n.targets))
    assert ast.unparse(tf_node)=="transforms.Compose([transforms.Resize(128), transforms.ToTensor()])"
    tf = lambda image: np.asarray(image)
    print("PIXELS reading selected metadata",flush=True)
    frame = read(OUT / "source_region_metadata.csv")
    offset = ((frame.bbox_center_x_vv-frame.center_x_vv)**2+
              (frame.bbox_center_y_vv-frame.center_y_vv)**2)
    choices = [0, int(frame.index[~frame.native_pair_same_shape][0]), int(offset.idxmax()),
               int(frame.index[frame.sham_valid_vv & frame.sham_valid_vh][0]),
               int(frame.index[frame.sham_valid_vv ^ frame.sham_valid_vh][0])]
    results = []
    for index in dict.fromkeys(choices):
        row = frame.iloc[index]
        print("PIXEL_CASE",row.sample_id,flush=True)
        images = port.read_images(row)
        originals = [a.copy() for a in images]
        anchor = np.concatenate([np.asarray(Image.fromarray(a).resize((128,128), Image.Resampling.BILINEAR)).ravel()
                                 for a in images])
        lo, hi = np.quantile(anchor, [.01, .99])
        outputs = {}
        for condition in CONDITIONS:
            changed, info = port.transform_images(images, row, condition, index)
            outputs[condition] = changed
            for k, pol in enumerate(("vv", "vh")):
                a, b = images[k], changed[k]
                bounds = expected_box(row, pol, condition.startswith("wide_"))
                region = mask(a.shape, bounds)
                eq(tuple(info[p+"_"+pol] for p in ("x0","y0","x1","y1")), bounds)
                if condition == "original":
                    chosen = np.zeros(a.shape, bool)
                else:
                    zone, kind = condition.removeprefix("wide_").split("_")
                    if zone == "sham":
                        both = bool(row.sham_valid_vv and row.sham_valid_vh)
                        assert info["sham_valid"] == both
                        chosen = mask(a.shape, [row[f"sham_{p}_{pol}"] for p in ("x0","y0","x1","y1")]) if both else np.zeros(a.shape, bool)
                        assert not np.any(chosen & region)
                        if both:
                            assert chosen.sum() == region.sum()
                    else:
                        chosen = region if zone == "target" else ~region
                    if kind == "flat":
                        border = np.r_[a[0],a[-1],a[:,0],a[:,-1]]
                        eq(b[chosen], np.full(chosen.sum(), np.rint(np.median(border)), dtype=np.uint8))
                    elif kind == "shuffle":
                        eq(np.sort(a[chosen]), np.sort(b[chosen]))
                    elif kind == "display":
                        expected = a[chosen] if hi <= lo else np.rint(np.clip((a[chosen].astype(float)-lo)*255/(hi-lo),0,255)).astype(np.uint8)
                        eq(b[chosen], expected)
                eq(a[~chosen], b[~chosen])
                eq(a.shape, b.shape)
                near(info["edited_area_"+pol], chosen.mean())
                near(info["region_area_"+pol], region.mean())
            for a, b in zip(images, originals):
                eq(a, b)
        repeat, _ = port.transform_images(images, row, "target_shuffle", index)
        for k, a in enumerate(images):
            eq(repeat[k], outputs["target_shuffle"][k])
            full = a if hi <= lo else np.rint(np.clip((a.astype(float)-lo)*255/(hi-lo),0,255)).astype(np.uint8)
            eq(outputs["target_display"][k].astype(int)+outputs["outer_display"][k].astype(int)-a.astype(int), full)
        raw, rgb = port.prepared(images, tf)
        for k, a in enumerate(images):
            expected = np.asarray(Image.fromarray(a).resize((128,128), Image.Resampling.BILINEAR))
            eq(raw[k], expected)
        eq(rgb,np.repeat(images[1][:,:,None],3,axis=2))
        results.append(dict(sample_id=row.sample_id, manifest_row=index, shapes=[list(a.shape) for a in images],
                            region_area=[float(mask(a.shape,expected_box(row,p)).mean()) for a,p in zip(images,("vv","vh"))],
                            original_saturation=[float((a==255).mean()) for a in images]))
    print("PIXELS port cases passed; reading OSS1",flush=True)
    oss = read(OUT / "oss1/frame.csv")
    selected = list(dict.fromkeys([0, int(oss.native_w.idxmin()), int(oss.native_w.idxmax()), int(oss.index[~oss.region_valid][0])]))
    xml_cases = []
    for index in selected:
        row = oss.iloc[index]
        print("XML_CASE",row.sample_id,flush=True)
        path = Path(row.path_vv).parent.parent / "Ship.xml"
        root = ET.fromstring(path.read_bytes().decode("utf-8",errors="replace"))
        ship = root.findall(".//ship")[int(row.xml_ship_index)]
        sar = ship.find("SARShipInformation")
        val = lambda k: int(sar.findtext(k))
        assert str(row.identity) == ship.findtext("AISShipInformation/MMSI").strip()
        assert row.sample_id.endswith(f":x{val('Center_x')}:y{val('Center_y')}")
        x0 = min(val("UpperLeft_x"),val("LowerRight_x"))
        y0 = min(val("UpperLeft_y"),val("LowerRight_y"))
        # Independent TIFF decoder avoids this audit environment's Pillow TIFF exit.
        pair = np.stack([tifffile.imread(row.path_vv),tifffile.imread(row.path_vh)])
        assert pair.dtype==np.uint8 and pair.ndim==3
        h,w = pair.shape[1:]
        eq((w,h),(abs(val("UpperLeft_x")-val("LowerRight_x"))+1,abs(val("UpperLeft_y")-val("LowerRight_y"))+1))
        eq((row.xml_origin_x,row.xml_origin_y,row.center_x,row.center_y),(x0,y0,w//2,h//2))
        points = np.array([[val("Head_x")-x0,val("Head_y")-y0],[val("Tail_x")-x0,val("Tail_y")-y0]])
        valid = h==w and np.all(points>=0) and np.all(points[:,0]<w) and np.all(points[:,1]<h) and np.any(points[0]!=points[1])
        assert bool(row.region_valid) == bool(valid)
        if valid:
            eq(points.ravel(),[row.head_x,row.head_y,row.tail_x,row.tail_y])
        for condition in ("original","target_flat","outer_flat","sham_flat","wide_target_flat"):
            transformed, info = public.transform(pair,row,condition,index)
            if not valid:
                eq(transformed,pair)
            elif condition != "original":
                pad = max(3,(.35 if condition.startswith("wide_") else .20)*np.linalg.norm(points[0]-points[1]))
                bounds = np.r_[np.maximum(np.floor(points.min(0)-pad),0),np.minimum(np.ceil(points.max(0)+pad)+1,[w,h])].astype(int)
                eq([info[k] for k in ("x0","y0","x1","y1")],bounds)
        xml_cases.append(dict(sample_id=row.sample_id, xml=str(path), xml_ship_index=int(row.xml_ship_index),
                              origin=[x0,y0], native_shape=[h,w], actual_local_points=points.tolist(), valid=bool(valid)))
    record = dict(status="passed", port_cases=results, port_conditions=list(CONDITIONS), oss1_cases=xml_cases,
                  checks="actual pixels, outside-mask identity, flat value, per-pol histogram, shared display complement, deterministic shuffle, input immutability, native bbox coordinates, exact R50 bilinear128 pixels, native VH RGB handoff to R18 plus frozen Resize128/ToTensor configuration, direct XML origin/extent/endpoints and invalid no-op",
                  limitations="Small real-case audit, not all-image or full-XML scan; no GPU/feature re-extraction; large detection-box neighborhoods are not vessel segmentation; no cross-pol tuple preservation claim",
                  audit_environment="Pillow exited while decoding the first OSS1 TIFF in this environment; OSS1 native pixels independently decoded with tifffile. No parent files or reader were changed.")
    save("pixels",record)
    print("PIXELS PASSED",len(results),"port cases;",len(xml_cases),"OSS1 cases",flush=True)


def ba(values, labels):
    return float(np.mean([np.mean(values[labels==c]) for c in np.unique(labels)])*100)


def metrics(scope):
    tasks = [("oss1",0),("oss2",0)] if scope == "public" else [("Los Angeles",2),("Port Klang",3)]
    conditions = ("original","target_shuffle","outer_shuffle","target_display","outer_display","sham_flat")
    results, pending = [], []
    for name, fold in tasks:
        folder = OUT / "predictions" / name / f"fold_{fold}"
        if not (folder / "READOUT_COMPLETE.json").exists():
            pending.append(name)
            continue
        dataset = name if scope == "public" else "ports"
        framepath = OLD / "cohort.csv" if name == "oss2" else OUT / dataset / "frame.csv"
        frame = read(framepath)
        cache = OLD / "features" if name == "oss2" else OUT / "oss1/features" if name == "oss1" else Path("C:/CodexArtifacts/region_cue_replication_20261013/ports/features")
        cohort = read(folder / "cohort.csv")
        controls = read(folder / "regions.csv")
        with np.load(folder / "cohort.npz",allow_pickle=False) as z:
            data = {k:z[k] for k in ("ids","support","support_labels","eval_masks")}
        positions = frame.set_index("sample_id").index.get_indexer(data["ids"])
        assert (positions>=0).all()
        for condition in ("replay_original","target_shuffle"):
            assert (cache / condition / "COMPLETE.json").exists()
            ids = np.load(cache / condition / "ids.npy",allow_pickle=False)
            eq(ids,frame.sample_id.to_numpy(str))
            eq(ids[positions],data["ids"])
        eq(cohort.sample_id.to_numpy(str),data["ids"])
        eq(controls.sample_id.to_numpy(str),data["ids"])
        old = CROSS / f"fold_{fold}/targets/shared_frozen" / name
        with np.load(old / "predictions.npz",allow_pickle=False) as z:
            for k in ("ids","support","eval_masks"):
                eq(data[k],z[k])
            arms = list(z["arms"])
        oldp = np.load(old / "probabilities.npy",mmap_mode="r")
        archived = np.load(folder / "archived_baseline.npy",mmap_mode="r")
        baseline = np.load(folder / "baseline.npy",mmap_mode="r")
        replay_error = []
        for k, arm in enumerate(("R","T")):
            z = np.asarray(archived[:,k]); exp = np.exp(z-z.max(-1,keepdims=True)); p = (exp/exp.sum(-1,keepdims=True)).astype(np.float32)
            expected = oldp[0,:,arms.index(arm),0]
            near(p,expected,2e-7)
            eq(p.argmax(-1),expected.argmax(-1))
            replay_error.append(float(np.max(np.abs(p-expected))))
        y = cohort.class_id.to_numpy(int)
        eq(y[data["support"]],data["support_labels"])
        effects = read(folder / "path_effects.csv").set_index(["condition","draw","scope","path"],verify_integrity=True)
        utility = read(folder / "utility.csv").set_index(["condition","draw","scope","mode"],verify_integrity=True)
        checks=0
        for condition in conditions:
            scores = np.load(folder / (condition+".npy"),mmap_mode="r")
            for draw in (0,4):
                assert not data["eval_masks"][draw,data["support"][draw]].any()
                if condition == "original":
                    for pi,bi in enumerate((0,1,1,1,0,1)):
                        near(scores[draw,pi],baseline[draw,bi],1e-11)
                near(scores[draw,1]+scores[draw,2]-baseline[draw,1],scores[draw,3],1e-9)
                ev = data["eval_masks"][draw] & controls.region_valid.to_numpy(bool)
                for subgroup, selected in (("full",ev),("sham_valid",ev & controls.sham_valid.to_numpy(bool))):
                    if not selected.any():
                        continue
                    labels = y[selected]
                    r0 = baseline[draw,0,selected].argmax(-1)
                    t0 = baseline[draw,1,selected].argmax(-1)
                    for pi,path in enumerate(PATHS):
                        pred = scores[draw,pi,selected].argmax(-1)
                        ref = r0 if path.startswith("R_") else t0
                        saved = effects.loc[(condition,draw,subgroup,path)]
                        near([saved.n,saved.classes_present,saved.oa,saved.ba,saved.delta_ba,saved.flips_macro],
                             [len(labels),len(np.unique(labels)),100*np.mean(pred==labels),ba(pred==labels,labels),
                              ba(pred==labels,labels)-ba(ref==labels,labels),ba(pred!=ref,labels)])
                        checks+=6
                    for mode,ri,ti in (("query",0,3),("joint",4,5)):
                        r,t = scores[draw,ri,selected].argmax(-1),scores[draw,ti,selected].argmax(-1)
                        rescue,harm = (r!=labels)&(t==labels),(r==labels)&(t!=labels)
                        saved = utility.loc[(condition,draw,subgroup,mode)]
                        near([saved.r_ba,saved.t_ba,saved.rescue,saved.harm,saved.net],
                             [ba(r==labels,labels),ba(t==labels,labels),ba(rescue,labels),ba(harm,labels),ba(t==labels,labels)-ba(r==labels,labels)])
                        checks+=5
        flips = [int(((baseline[:,k].argmax(-1)!=archived[:,k].argmax(-1)) & data["eval_masks"]).sum()) for k in (0,1)]
        results.append(dict(task=name,fold=fold,draws=[0,4],conditions=list(conditions),checks=checks,
                            feature_id_alignment="full dataset frame and exact target archived IDs passed for replay_original/target_shuffle",
                            legacy_probability_max_error=replay_error,primary_vs_archived_query_flips=flips))
    value=dict(status="passed" if not pending else "pending",tasks=results,pending=pending,
               limitation="No new fits; scores/metrics replay only on named tasks, one public fold, two draws and six conditions; not full matrix")
    save(scope,value)
    print(scope.upper(),value["status"],"checks",sum(r["checks"] for r in results),"pending",pending,flush=True)


def aggregation():
    tree = ast.parse((OUT / "summarize_replication.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="equal_task_summary")
    namespace={"pd":pd}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),"equal_task_summary","exec"),namespace)
    synthetic=pd.DataFrame(dict(dataset=["ports"]*6,task=["a"]*3+["b"]*2+["c"],value=[0,0,30,20,40,80],n=[10000]*3+[100]*2+[10]))
    _,summary=namespace["equal_task_summary"](synthetic,[],["value"])
    near(summary.value.iloc[0],(10+30+80)/3)
    folder=OUT / "summary_public"
    raw=read(folder / "path_effects.csv")
    agg=read(folder / "path_effects_dataset_means.csv")
    checks=[]
    for name in ("oss1","oss2"):
        for condition in ("target_shuffle","outer_shuffle","target_display","outer_display"):
            selected=raw[(raw.dataset==name)&(raw.scope=="full")&(raw.path=="T_Bonly")&(raw.condition==condition)]
            assert len(selected)==25 and selected.fold.nunique()==5 and selected.draw.nunique()==5
            saved=agg[(agg.dataset==name)&(agg.scope=="full")&(agg.path=="T_Bonly")&(agg.condition==condition)].iloc[0]
            near(saved.delta_ba,selected.groupby("task").delta_ba.mean().mean())
            checks.append(dict(dataset=name,condition=condition,contexts=len(selected),delta_ba=float(saved.delta_ba)))
    port_checks=[]
    port_folder=OUT / "summary_all"
    port_conditions=("target_shuffle","outer_shuffle","target_display","outer_display")
    columns=["dataset","task","fold","draw","scope","condition","path","delta_ba"]
    chosen=[]
    for chunk in pd.read_csv(port_folder / "path_effects.csv",usecols=columns,chunksize=20000):
        selected=chunk[(chunk.dataset=="ports")&(chunk.scope=="full")&(chunk.path=="T_Bonly")&
                       chunk.condition.isin(port_conditions)]
        if len(selected): chosen.append(selected)
    port_raw=pd.concat(chosen,ignore_index=True)
    port_summary=read(port_folder / "path_effects_dataset_means.csv")
    port_means=read(port_folder / "path_effects_task_means.csv")
    for condition in port_conditions:
        selected=port_raw[port_raw.condition==condition]
        assert len(selected)==115 and selected.task.nunique()==23
        assert selected.groupby("task").size().eq(5).all()
        expected=selected.groupby("task").delta_ba.mean().sort_index()
        saved_tasks=port_means[(port_means.dataset=="ports")&(port_means.scope=="full")&
                              (port_means.path=="T_Bonly")&(port_means.condition==condition)].set_index("task").sort_index()
        eq(saved_tasks.index.to_numpy(),expected.index.to_numpy())
        near(saved_tasks.delta_ba.to_numpy(),expected.to_numpy())
        saved=port_summary[(port_summary.dataset=="ports")&(port_summary.scope=="full")&
                           (port_summary.path=="T_Bonly")&(port_summary.condition==condition)].iloc[0]
        near(saved.delta_ba,expected.mean())
        port_checks.append(dict(condition=condition,ports=23,contexts=115,delta_ba=float(saved.delta_ba)))
    save("aggregation",dict(status="passed",equal_port_synthetic_expected=40.,public_cells=checks,
                             actual_port_cells=port_checks,
                             limitation="Eight public cells and four actual equal-port B cells; unequal-task-size synthetic test; no full-matrix metric replay"))
    print("AGGREGATION PASSED",len(checks),"public cells;",len(port_checks),"actual port cells",flush=True)


def report():
    statuses=[]
    for name in ("pixels","public","aggregation","ports"):
        path=OUT / ("AUDIT_"+name+".json")
        statuses.append(f"- {name}: "+(json.loads(path.read_text(encoding="utf-8"))["status"] if path.exists() else "not run"))
    text="# Bounded Replication Audit\n\n"+"\n".join(statuses)+"\n\n"
    text+="Independent metric calculations do not import the parent evaluator. Pixel tests execute the actual transform under test against independently computed masks, pixel invariants and direct XML fields. All audit work is CPU-only. Only audit_replication.py and AUDIT*.json/md are written.\n\n"
    text+="Primary same-pipeline originals and legacy archived baselines remain separate. Tiny legacy/primary numerical differences and prediction flips are recorded, not treated as failed exact replay between deliberately different inputs. Cache lineage checks require exact ids.npy/frame/archived-target alignment.\n\n"
    text+="Limits: bbox neighborhoods can be large and include surrounding water, wake or sidelobes. Original saturation is not by itself a preprocessing error. Native per-polarization shuffles preserve each region's grayscale multiset, not VV/VH joint tuples on different masks. Shared public vessels/acquisitions prevent an independent-data replication claim. Aggregate directional repetition does not imply class-universal reliance; container-driven effects and counterexamples must remain in the main report. Removing shared query identities does not redesign supports. No class exclusions or new protocol choices are introduced here.\n\n"
    text+="Interpretation boundaries supplied by the main analysis (not additional audit recomputations): ports' B display contrast reverses the public pattern; matched target shuffling can increase T-R while reducing absolute T accuracy. Ports' sham-valid shuffle contrast reverses the full-cohort contrast (+0.745 versus -0.778 pp; 13 negative/8 positive/2 zero ports), with lower average class coverage (about 4.31 versus 6.07). Do not claim complete-mechanism replication or stability across all subsets. Public class counterexamples also remain part of the result.\n\n"
    text+="Audit environment: Pillow exited decoding the first OSS1 TIFF; four OSS1 native pairs were independently read with tifffile, then checked against original XML. This was confined to the auditor; parent readers and preprocessing were not changed. R50 bilinear128 pixels and the R18 native VH-to-RGB handoff were verified; the frozen Resize(128)/ToTensor configuration was checked without loading model libraries.\n\n"
    text+="Run with D:/Program_files/anaconda3/python.exe -B audit_replication.py pixels|public|aggregation|ports. Ports may be run later after the two named tasks finish; pending is not a pass.\n"
    (OUT / "AUDIT.md").write_text(text,encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("pixels","public","ports","aggregation"))
    stage=parser.parse_args().stage
    if stage=="pixels": pixels()
    elif stage=="aggregation": aggregation()
    else: metrics(stage)
    report()
