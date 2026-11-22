"""Private, deterministic image/evidence review after all nine test runs finish.

Does not train, select checkpoints, change budgets, or update model/config files.
The query's true label is used only for post-hoc case/support presentation.
"""
import argparse
from collections import defaultdict, deque
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
SEEDS = (17, 29, 43)
ARMS = ('baseline', 'disentangle', 'full_h')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): value.update(block)
    return value.hexdigest()


def checked_artifact(state, key, path):
    item = state.get('steps', {}).get(key, {})
    if item.get('status') != 'succeeded' or not path.is_file():
        raise RuntimeError(f'Completed artifact required: {key}')
    if digest(path) != item.get('output_sha256'):
        raise RuntimeError(f'Completed artifact changed: {path}')


def readiness(root):
    # Refuse before importing NumPy/Torch or reading any prediction content.
    expected = [root/'runs'/f'seed{s}'/f'{arm}_test.json' for s in SEEDS for arm in ARMS]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError('Case review remains closed; all nine test files are required: '+', '.join(missing))
    if (root/'.run_comparison.lock').exists():
        raise RuntimeError('Comparison is still running. Review only after its lock is released.')
    state = read_json(root/'state_local.json')
    for seed in SEEDS:
        for stage in ('baseline_stage1','baseline_base','disentangle_stage1','disentangle_base','full_h'):
            checked_artifact(state, f'seed{seed}/{stage}', root/'runs'/f'seed{seed}'/(stage+'.pt'))
        for arm in ARMS:
            checked_artifact(state, f'seed{seed}/{arm}_test', root/'runs'/f'seed{seed}'/(arm+'_test.json'))
    ledger = read_json(root/'ledger_local.json')
    manifest = root/'experiment_manifest_local.json'
    if digest(manifest) != ledger['inputs']['manifest']['sha256']:
        raise RuntimeError('Pinned manifest changed')
    code_root = Path(ledger['inputs']['code_root'])
    for relative, expected_hash in ledger['inputs']['code_sha256'].items():
        if digest(code_root/relative) != expected_hash:
            raise RuntimeError(f'Pinned code changed: {relative}')
    return state, ledger, code_root


def predictions(root, rows):
    expected = {r['id']: str(r['label']) for r in rows if r.get('development_split') == 'test'}
    arms = {}
    for arm in ARMS:
        result = read_json(root/'runs'/'seed17'/(arm+'_test.json'))
        values = result.get('predictions', [])
        by_id = {r['id']: r for r in values}
        if len(by_id) != len(values) or result.get('count') != len(expected):
            raise RuntimeError(f'{arm}: duplicate IDs or incorrect test count')
        if {sid: str(r['label']) for sid,r in by_id.items()} != expected:
            raise RuntimeError(f'{arm}: prediction IDs/labels disagree with the locked test split')
        arms[arm] = by_id
    return {sid: dict(id=sid, label=label, **{arm: arms[arm][sid]['prediction'] for arm in ARMS})
            for sid,label in expected.items()}


def choose_cases(records):
    classes = sorted({r['label'] for r in records.values()})
    categories = [('H_rescued', 'disentangle', 'full_h', False, True, 2),
                  ('H_harmed', 'disentangle', 'full_h', True, False, 2),
                  ('H_still_correct', 'disentangle', 'full_h', True, True, 2),
                  ('H_still_wrong', 'disentangle', 'full_h', False, False, 2),
                  ('disentangle_rescued', 'baseline', 'disentangle', False, True, 1),
                  ('disentangle_harmed', 'baseline', 'disentangle', True, False, 1)]
    selected, counts, used, cursor = [], {}, set(), 0
    for category, before, after, old_ok, new_ok, limit in categories:
        candidates = [r for r in records.values() if (r[before] == r['label']) == old_ok
                      and (r[after] == r['label']) == new_ok]
        counts[category] = dict(total=len(candidates), per_class={c: sum(r['label']==c for r in candidates)
                                                               for c in classes})
        queues = {c: deque(sorted((r for r in candidates if r['label']==c and r['id'] not in used),
                                 key=lambda r:r['id'])) for c in classes}
        picked = 0
        while picked < limit and any(queues.values()):
            for offset in range(len(classes)):
                position = (cursor+offset) % len(classes)
                if queues[classes[position]]:
                    row = queues[classes[position]].popleft()
                    selected.append(dict(row, category=category)); used.add(row['id']); picked += 1
                    cursor = (position+1) % len(classes)
                    break
        counts[category]['selected'] = picked
    return selected, counts


def verify_frozen(base, full, torch):
    if base['stage'] != 'base' or full['stage'] != 'h':
        raise RuntimeError('Expected disentangle_base and full_h checkpoints')
    for name in ('encoder_config','input_config','geometry_config','labels','source_ids','manifest_metadata'):
        if base[name] != full[name]: raise RuntimeError(f'Checkpoint interface differs: {name}')
    a, b = base['encoder'], full['encoder']
    if set(a) != set(b) or any(not torch.equal(a[k],b[k]) for k in a):
        raise RuntimeError('Frozen encoder/G tensor equality failed')
    keys = [k for k in base['h'] if k.startswith('h0.')]
    if not keys or set(keys) != {k for k in full['h'] if k.startswith('h0.')}:
        raise RuntimeError('h0 checkpoint keys differ')
    if any(not torch.equal(base['h'][k],full['h'][k]) for k in keys):
        raise RuntimeError('Frozen h0 tensor equality failed')
    return dict(encoder_and_g_tensors=len(a), h0_tensors=len(keys), exact_equality=True)


def actual_contributions(output, refs, classes, config, torch):
    """Sum omega[n] * pi[r,n] * evidence[r,n] / active_ports[c,n]."""
    detail, omega = output['details'][0], output['weights'][0]
    denominators = {}
    for (c, _), mixture in detail['mixtures'].items():
        active = torch.stack([detail['pairs'][ri]['valid'] for ri in mixture['indices']]).any(0)
        denominators[c] = denominators.get(c, torch.zeros_like(omega))+active.to(omega)
    reconstructed = torch.zeros(len(classes),device=omega.device,dtype=omega.dtype)
    contributions = []
    for (c, port), mixture in detail['mixtures'].items():
        for position, ri in enumerate(mixture['indices']):
            pair = detail['pairs'][ri]
            per_atom = omega*mixture['weights'][position]*pair['evidence']/denominators[c].clamp_min(1)
            value = per_atom.sum(); reconstructed[c] += value
            contributions.append(dict(reference_index=ri, reference_id=refs[ri]['id'],
                label=classes[c], port=str(port), evidence_contribution=float(value),
                logit_contribution=float(value*config.gamma/config.evidence_temperature),
                per_atom_contribution=per_atom.cpu().tolist(),
                mixture_pi=mixture['weights'][position].cpu().tolist(),
                pair_evidence=pair['evidence'].cpu().tolist(), valid=pair['valid'].cpu().tolist()))
    torch.testing.assert_close(reconstructed,output['evidence'][0],rtol=2e-5,atol=2e-6)
    return contributions, reconstructed.cpu().tolist()


def supporting_reference(label, contributions, pairs):
    values = sorted((r for r in contributions if r['label']==label and r['evidence_contribution']>0),
                    key=lambda r:(-r['evidence_contribution'],r['reference_id']))
    if not values: return None
    result = dict(values[0]); pair = pairs[result['reference_index']]
    result['pair_details'] = {k: pair[k].detach().cpu().tolist() for k in
        ('transport','beta','appearance','edge_valid','geometry_cost','rate','mean_support','stats')}
    result.update(direction=pair['direction'],direction_solves=pair['direction_solves'],
                  direction_identifiable=pair['direction_identifiable'])
    return result


def validation_predictions(root, state, rows):
    expected={r['id']:str(r['label']) for r in rows if r.get('development_split') in ('val','validation')}
    results={}
    for arm,stage in (('disentangle','disentangle_base_validation'),('full_h','full_h_validation')):
        path=root/'runs'/'seed17'/(stage+'.json')
        checked_artifact(state,f'seed17/{stage}',path)
        payload=read_json(path); values=payload.get('predictions',[])
        by_id={r['id']:r for r in values}
        if len(values)!=len(by_id) or payload.get('count')!=len(expected):
            raise RuntimeError(f'{stage}: incorrect validation count or duplicate IDs')
        if {sid:str(r['label']) for sid,r in by_id.items()}!=expected:
            raise RuntimeError(f'{stage}: validation identities/labels differ from manifest')
        results[arm]=by_id
    return results


def summarize_margins(observations, bound, np):
    """Aggregate only; no sample IDs, image paths or per-observation rows escape."""
    def distribution(values):
        if not values: return dict(count=0)
        array=np.asarray(values,dtype=float)
        return dict(count=len(values),mean=float(array.mean()),
                    quantiles={str(q):float(np.quantile(array,q)) for q in (0,.1,.25,.5,.75,.9,1)})
    wrong=[r for r in observations if not r['h0_correct']]
    correct=[r for r in observations if r['h0_correct']]
    def outcomes(part):
        return dict(count=len(part),rescued=sum(not r['h0_correct'] and r['h_correct'] for r in part),
                    harmed=sum(r['h0_correct'] and not r['h_correct'] for r in part),
                    still_correct=sum(r['h0_correct'] and r['h_correct'] for r in part),
                    still_wrong=sum(not r['h0_correct'] and not r['h_correct'] for r in part),
                    h_alone_correct=sum(r['h_alone_correct'] for r in part),
                    no_true_class_support=sum(not r['true_class_supported'] for r in part),
                    h0_absolute_margin=distribution([r['h0_margin'] for r in part]),
                    actual_true_vs_initial_competitor_correction=distribution([r['actual_correction'] for r in part]),
                    final_true_class_margin=distribution([r['final_true_margin'] for r in part]))
    cuts=sorted(set((0.,1.,3.,5.,10.,float(bound))))
    bins=[]
    for i,low in enumerate(cuts):
        high=cuts[i+1] if i+1<len(cuts) else None
        if high==0: continue
        # [0, first upper] then (lower, upper]; exact-bound observations stay below/outside separately.
        def member(row):
            value=row['h0_margin']
            return (value>=low if i==0 else value>low) and (high is None or value<=high)
        bins.append(dict(lower=low,upper=high,lower_inclusive=i==0,
                         h0_wrong=outcomes([r for r in wrong if member(r)]),
                         h0_correct=outcomes([r for r in correct if member(r)])))
    beyond=[r for r in wrong if r['h0_margin']>bound]
    inside=[r for r in wrong if r['h0_margin']<=bound]
    robust=[r for r in correct if r['h0_margin']>bound]
    return dict(count=len(observations),global_pairwise_fusion_bound=bound,
        h0_wrong=outcomes(wrong),h0_correct=outcomes(correct),
        h0_wrong_margin_strictly_above_bound=outcomes(beyond),
        h0_wrong_margin_at_or_below_bound=outcomes(inside),
        h0_correct_margin_strictly_above_bound=outcomes(robust),
        wrong_above_bound_fraction_of_h0_wrong=len(beyond)/len(wrong) if wrong else None,
        wrong_exactly_at_bound=sum(r['h0_margin']==bound for r in wrong),
        evidence_ordering_by_bound={name:{'h_alone_correct':outcomes([r for r in part if r['h_alone_correct']]),
                                        'h_alone_wrong':outcomes([r for r in part if not r['h_alone_correct']])}
                                   for name,part in (('above_bound',beyond),('at_or_below_bound',inside))},
        margin_bins=bins,per_class={label:outcomes([r for r in observations if r['label']==label])
                                  for label in sorted({r['label'] for r in observations})},
        bound_invariant_violations=dict(rescued_with_margin_above_bound_plus_1e_5=
            sum(r['h_correct'] and r['h0_margin']>bound+1e-5 for r in beyond),
            harmed_with_margin_above_bound_plus_1e_5=
            sum(not r['h_correct'] and r['h0_margin']>bound+1e-5 for r in robust)),
        interpretation=['E is bounded in [0,1]; gamma/temperature bounds any pairwise logit-margin change.',
            'Margin strictly above the bound cannot be corrected (or damaged for an initially correct prediction), apart from numerical ties.',
            'Being within this global bound does not prove that the available references/parameterization can achieve the required evidence.',
            'H-alone correctness describes evidence ordering; sufficient residual margin is a separate condition.',
            'This is validation-only diagnosis. No parameter, checkpoint or budget is selected or changed.'])


def margin_diagnostics(h, features, indices, rows, refs, ref_rows, reference_labels, expected,
                       classes, device, torch, np, legal_reference_mask):
    observations=[]; single_replays=0
    bound=h.config.gamma/h.config.evidence_temperature
    for start in range(0,len(indices),8):
        ids=indices[start:start+8]
        identities=[{k:rows[i][k] for k in ('vessel_id','group_id')} for i in ids]
        legal=torch.as_tensor(legal_reference_mask(identities,ref_rows),device=device)
        queries=[features[i].to(device) for i in ids]
        result=h(queries,refs,reference_labels,[r['port'] for r in ref_rows],legal)
        batch_values={k:result[k].detach().cpu() for k in ('z0','logits','evidence','support')}
        del result
        for position,index in enumerate(ids):
            row=rows[index]; sid=row['id']; y=classes.index(str(row['label']))
            z0=batch_values['z0'][position]; logits=batch_values['logits'][position]; evidence=batch_values['evidence'][position]
            support=batch_values['support'][position]
            expected_base=expected['disentangle'][sid]['prediction']; expected_h=expected['full_h'][sid]['prediction']
            if classes[int(z0.argmax())]!=expected_base or classes[int(logits.argmax())]!=expected_h:
                # Use the original one-query evaluation shape if batch GEMM roundoff changes a near tie.
                replay=h([queries[position]],refs,reference_labels,[r['port'] for r in ref_rows],legal[position:position+1])
                z0,logits,evidence,support=(replay[k][0].detach().cpu() for k in ('z0','logits','evidence','support'))
                single_replays+=1
            if classes[int(z0.argmax())]!=expected_base or classes[int(logits.argmax())]!=expected_h:
                raise RuntimeError('Validation replay disagrees with stored checkpoint predictions')
            if not torch.isfinite(z0).all() or not torch.isfinite(logits).all() or not torch.isfinite(evidence).all():
                raise RuntimeError('Nonfinite validation logits/evidence')
            if float(evidence.min()) < -1e-6 or float(evidence.max()) > 1+1e-6:
                raise RuntimeError('Evidence violates the fusion-bound contract')
            old=int(z0.argmax()); new=int(logits.argmax()); alternatives=[c for c in range(len(classes)) if c!=y]
            competitor=max(alternatives,key=lambda c:float(z0[c]))
            margin=float(z0[y]-z0[competitor]) if old==y else float(z0[old]-z0[y])
            observations.append(dict(label=str(row['label']),h0_correct=old==y,h_correct=new==y,
                h_alone_correct=int(evidence.argmax())==y,true_class_supported=bool(support[y]),
                h0_margin=margin,actual_correction=float(bound*(evidence[y]-evidence[competitor])),
                final_true_margin=float(logits[y]-logits[alternatives].max())))
    summary=summarize_margins(observations,bound,np)
    summary.update(scope='seed17 full validation set; frozen post-training replay',split='validation',seed=17,
                   gamma=h.config.gamma,evidence_temperature=h.config.evidence_temperature,
                   single_query_roundoff_replays=single_replays,query_labels_used_in_matching=False)
    return summary


def export_chips(row, directory, gallery, Image):
    result = {}
    token = hashlib.sha256(row['id'].encode('utf-8')).hexdigest()[:20]
    for pol in ('vv','vh'):
        view = row.get(pol)
        if not view or not view.get('alignment_valid',False):
            result[pol] = None; continue
        source = (directory/Path(view['image'])).resolve()
        with Image.open(source) as image:
            if image.format != 'PNG': raise RuntimeError(f'Expected original PNG chip: {source}')
            size, mode = image.size, image.mode
        target = gallery/'chips'/f'{token}_{pol}.png'
        target.parent.mkdir(parents=True,exist_ok=True)
        source_hash = digest(source)
        if target.exists() and digest(target) != source_hash: raise RuntimeError('Chip export collision')
        if not target.exists(): shutil.copy2(source,target)
        result[pol] = dict(path=target.relative_to(gallery).as_posix(),native_size=list(size),
                           native_mode=mode,sha256=source_hash,source_path=str(source))
    return result


def render_gallery(cases, gallery, Image):
    from PIL import ImageDraw, ImageFont
    gallery.mkdir(parents=True,exist_ok=True)
    font_path = Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'/'msyh.ttc'
    font = ImageFont.truetype(str(font_path),17) if font_path.is_file() else ImageFont.load_default()
    heading = ImageFont.truetype(str(font_path),21) if font_path.is_file() else font
    pages = []
    for start in range(0,len(cases),4):
        subset = cases[start:start+4]
        sizes = [info['native_size'] for case in subset
                 for key in ('chips','true_support_chips','confusing_support_chips')
                 for info in (case.get(key) or {}).values() if info]
        cell_width=max([220]+[s[0] for s in sizes]); cell_height=max([190]+[s[1] for s in sizes])
        panel_width=2*(cell_width+18)+18; page_width=3*panel_width+18; row_height=cell_height+150
        page = Image.new('RGB',(page_width,55+len(subset)*row_height),'#ffffff')
        draw = ImageDraw.Draw(page)
        draw.text((16,12),'OSS2 seed17 | native PNG pixels, integer nearest-neighbor display',fill='black',font=heading)
        blocks = []
        for offset, case in enumerate(subset):
            y = 55+offset*row_height
            title = (f"{start+offset+1}. {case['category']} | true={case['label']} | "
                     f"baseline={case['baseline']} / disentangle={case['disentangle']} / H={case['full_h']} "
                     f"| atoms={case['atom_count']}")
            draw.rectangle((8,y,page_width-8,y+row_height-10),outline='#999999',width=1)
            draw.text((16,y+8),title,fill='black',font=font)
            draw.text((16,y+34),f"ID: {case['id']} | omega="+', '.join(f'{v:.3f}' for v in case['omega']),fill='black',font=font)
            panels = [('Query',case['chips'],None),('True-class support',case.get('true_support_chips'),case['top_true_support']),
                      ('Confusing-class support',case.get('confusing_support_chips'),case['top_confusing_support'])]
            html_panels = []
            for column,(name,chips,support) in enumerate(panels):
                x = 16+column*panel_width
                caption = name if support is None else f"{name}: {support['label']} / E={support['evidence_contribution']:.4f}"
                draw.text((x,y+64),caption,fill='black',font=font)
                thumbnails = []
                if chips:
                    for pindex,pol in enumerate(('vv','vh')):
                        info = chips.get(pol)
                        if not info:
                            draw.text((x+pindex*(cell_width+18),y+96),f'{pol.upper()}: unavailable',fill='#666666',font=font); continue
                        with Image.open(gallery/info['path']) as native:
                            w,h = native.size
                            factor = max(1,min(4,cell_width//w,cell_height//h))
                            shown = native.convert('RGB').resize((w*factor,h*factor),Image.Resampling.NEAREST)
                        draw.text((x+pindex*(cell_width+18),y+90),f'{pol.upper()} {w}x{h} / {factor}x',fill='black',font=font)
                        page.paste(shown,(x+pindex*(cell_width+18),y+116))
                        thumbnails.append(f'<figure><figcaption>{pol.upper()} {w}×{h}</figcaption>'
                            f'<a href="{info["path"]}"><img src="{info["path"]}" width="{w*factor}" height="{h*factor}"></a></figure>')
                else:
                    draw.text((x,y+110),'No positive legal reference contribution',fill='#666666',font=font)
                if support: draw.text((x,y+cell_height+120),support['reference_id'],fill='#333333',font=font)
                html_panels.append('<section><b>'+html.escape(caption)+'</b><div class="pol">'+''.join(thumbnails)+'</div>'
                    +('<p>'+html.escape(support['reference_id'])+'</p>' if support else '')+'</section>')
            blocks.append('<article><h2>'+html.escape(title)+'</h2><p>'+html.escape(case['id'])
                          +'</p><div class="panels">'+''.join(html_panels)+'</div></article>')
        basename=f'page_{len(pages)+1:02d}'
        page.save(gallery/(basename+'.png'))
        markup='<!doctype html><meta charset="utf-8"><title>OSS2 evidence review</title><style>'\
               'body{font:15px sans-serif;margin:20px}article{border:1px solid #aaa;padding:12px;margin-bottom:20px}'\
               '.panels,.pol{display:flex;gap:16px;flex-wrap:wrap}section{flex:1;min-width:400px}'\
               'figure{margin:6px}img{image-rendering:pixelated}h2{font-size:18px}</style>'+''.join(blocks)
        (gallery/(basename+'.html')).write_text(markup,encoding='utf-8')
        pages.append(dict(png=basename+'.png',html=basename+'.html',cases=[c['id'] for c in subset]))
    index='<!doctype html><meta charset="utf-8"><h1>Private OSS2 seed17 evidence review</h1>'\
          '<p>Fixed class-round-robin / ID-order selection. Native chips are copied byte-for-byte; '
    index+='display enlarges with integer nearest-neighbor only. No positive supporting reference is fabricated.</p><ul>'
    index+=''.join(f'<li><a href="{p["html"]}">Page {i+1}</a> | <a href="{p["png"]}">PNG</a></li>' for i,p in enumerate(pages))+'</ul>'
    (gallery/'index.html').write_text(index,encoding='utf-8')
    return pages


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args(argv)
    state,ledger,code_root=readiness(ROOT)
    output_path=ROOT/'case_evidence_local.json'
    margin_path=ROOT/'aggregate_h_margin_diagnostics.json'
    gallery=ROOT/'gallery_local'/'evidence_review_seed17'
    if output_path.exists() or margin_path.exists() or gallery.exists():
        raise RuntimeError('Case review output already exists; refusing to overwrite it')
    os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',SAR_PORT_PREPARE_CACHE=str(ROOT/'prepared_cache_local'))
    sys.path.insert(0,str(code_root))
    import numpy as np  # Must precede Torch on this Windows environment.
    import torch
    from PIL import Image
    from sar_port.two_stage.cli import (load_records,load_checkpoint,metadata,prepare,restore_encoder,encode_samples)
    from sar_port.two_stage.data import InputConfig
    from sar_port.two_stage.geometry import GeometryConfig
    from sar_port.two_stage.h import HModel,HConfig
    from sar_port.protocol import legal_reference_mask
    torch.set_num_threads(4)
    rows=load_records(ROOT/'experiment_manifest_local.json'); by_id={r['id']:i for i,r in enumerate(rows)}
    chosen,selection_counts=choose_cases(predictions(ROOT,rows))
    validation_expected=validation_predictions(ROOT,state,rows)
    validation_indices=[i for i,r in enumerate(rows) if r.get('development_split') in ('val','validation')]
    base_path=ROOT/'runs'/'seed17'/'disentangle_base.pt'; full_path=ROOT/'runs'/'seed17'/'full_h.pt'
    base,full=load_checkpoint(base_path),load_checkpoint(full_path)
    frozen=verify_frozen(base,full,torch)
    if full['manifest_metadata'] != metadata(rows): raise RuntimeError('Checkpoint manifest metadata changed')
    if len(full['reference_ids']) != 32: raise RuntimeError('Expected the actual locked 32-reference bank')
    if not set(full['reference_ids']) <= set(full['source_ids']): raise RuntimeError('Reference bank is not train-only')
    refs_index=[by_id[sid] for sid in full['reference_ids']]; ref_rows=[rows[i] for i in refs_index]
    required=sorted(set(refs_index+validation_indices+[by_id[c['id']] for c in chosen]))
    del base
    samples=prepare(rows,ROOT,InputConfig(**full['input_config']),GeometryConfig(**full['geometry_config']),required)
    encoder=restore_encoder(full,args.device).freeze_for_h()
    features=encode_samples(encoder,samples,required,args.device,8)
    del encoder
    h=HModel(HConfig(**full['h_config'])).to(args.device)
    h.load_state_dict(full['h']); h.set_stage('match').eval(); h.requires_grad_(False)
    classes=full['labels']; label_map={label:i for i,label in enumerate(classes)}
    reference_labels=torch.tensor([label_map[r['label']] for r in ref_rows],device=args.device)
    refs=[features[i].to(args.device) for i in refs_index]
    cases=[]
    with torch.inference_mode():
        for case in chosen:
            index=by_id[case['id']]; row=rows[index]; feature=features[index].to(args.device)
            identity={key:row[key] for key in ('vessel_id','group_id')}
            legal=torch.as_tensor(legal_reference_mask([identity],ref_rows),device=args.device)
            # Forward receives frozen image features, source labels/ports and identity legality; no query label.
            result=h([feature],refs,reference_labels,[r['port'] for r in ref_rows],legal)
            actual=classes[int(result['logits'][0].argmax())]; h0=classes[int(result['z0'][0].argmax())]
            if actual != case['full_h'] or h0 != case['disentangle']:
                raise RuntimeError(f'Replay predictions differ from saved test results: {case["id"]}')
            contributions,reconstructed=actual_contributions(result,ref_rows,classes,h.config,torch)
            if actual != case['label']:
                confusion,reason=actual,'H predicted this wrong class'
            elif h0 != case['label']:
                confusion,reason=h0,'Shared h0 predicted this wrong class before H rescued it'
            else:
                other=[i for i,c in enumerate(classes) if c != case['label']]
                confusion=classes[max(other,key=lambda i:float(result['logits'][0,i]))]
                reason='Highest final H logit among non-true classes'
            top_true=supporting_reference(case['label'],contributions,result['details'][0]['pairs'])
            top_confusing=supporting_reference(confusion,contributions,result['details'][0]['pairs'])
            item=dict(case,atom_count=feature.geometry.atom_count,
                intervals=feature.geometry.intervals.cpu().tolist(),placement=samples[index].placement,
                omega=result['weights'][0].cpu().tolist(),z0=result['z0'][0].cpu().tolist(),
                logits=result['logits'][0].cpu().tolist(),evidence=result['evidence'][0].cpu().tolist(),
                support=result['support'][0].cpu().tolist(),class_evidence_reconstructed=reconstructed,
                confusing_class=confusion,confusing_class_reason=reason,
                reference_contributions=contributions,top_true_support=top_true,top_confusing_support=top_confusing)
            cases.append(item)
        margins=margin_diagnostics(h,features,validation_indices,rows,refs,ref_rows,reference_labels,
            validation_expected,classes,args.device,torch,np,legal_reference_mask)
    for item in cases:
        item['chips']=export_chips(rows[by_id[item['id']]],ROOT,gallery,Image)
        for name,key in (('true_support_chips','top_true_support'),('confusing_support_chips','top_confusing_support')):
            support=item[key]
            item[name]=export_chips(rows[by_id[support['reference_id']]],ROOT,gallery,Image) if support else None
    pages=render_gallery(cases,gallery,Image)
    payload=dict(created_at=datetime.now(timezone.utc).isoformat(),scope='private seed17 post-test case review',
        seed=17,classes=classes,manifest_sha256=ledger['inputs']['manifest']['sha256'],
        checkpoint_sha256=dict(disentangle_base=digest(base_path),full_h=digest(full_path)),frozen_verification=frozen,
        reference_ids=full['reference_ids'],selection_rule='category order; rotating sorted true classes; ID ascending; no duplicate cases',
        selection_counts=selection_counts,contribution_formula='sum_n omega[n]*pi[r,n]*pair_evidence[r,n]/active_ports[class,n]',
        query_labels_used_in_matching=False,cases=cases,gallery_directory=str(gallery),gallery_pages=pages,
        validation_margin_diagnostics=str(margin_path))
    temporary=output_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(temporary,output_path)
    margins.update(created_at=payload['created_at'],checkpoint_sha256=payload['checkpoint_sha256'],
                   manifest_sha256=payload['manifest_sha256'],frozen_verification=frozen)
    temporary=margin_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(margins,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(temporary,margin_path)
    print(json.dumps(dict(cases=len(cases),evidence=str(output_path),gallery=str(gallery/'index.html'),
                         validation_margin_diagnostics=str(margin_path)),ensure_ascii=False))


if __name__=='__main__':
    try: main()
    except (OSError,ValueError,RuntimeError,KeyError) as exc:
        print(f'STOP: {exc}',file=sys.stderr); sys.exit(1)
