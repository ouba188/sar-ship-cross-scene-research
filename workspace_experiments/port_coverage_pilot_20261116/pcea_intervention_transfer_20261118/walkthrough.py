"""Trace one cached source example through the unified model, without training."""
import inspect
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matched_run as runner

base, torch, np = runner.base, runner.torch, runner.np
OUT = runner.output_dir('unified')/'walkthrough_local'


def cpu(x):
    return x.detach().float().cpu().numpy()


def main():
    OUT.mkdir(exist_ok=True)
    bank = torch.load(runner.output_dir('unified')/'verification_bank_local.pt',
                      map_location='cpu', weights_only=True)
    assert bank['model_variant'] == 'unified'
    index = int(bank['ids'][0])
    inputs, _, _ = base.runtime('unified')
    row = inputs.rows[index]
    data = inputs.data(np.array([index]), 'small')
    refs = inputs.groups['references']
    rd = inputs.data(refs, 'small')
    net = base.model('unified').eval()
    capture, handles = {}, []
    with torch.no_grad():
        reference = net.reference(rd)
        def feature(module, args, result):
            capture['layer1'] = cpu(result)
        handles.append(net.net.base.backbone.model.layer1.register_forward_hook(feature))
        for p in range(2):
            def response(module, args, result, p=p):
                capture[f'm1_{p}'] = cpu(result[1]['response'])
            handles.append(net.net.m1[p].register_forward_hook(response))
        def matcher_pre(module, args, kwargs):
            kwargs = dict(kwargs, details=True)
            capture['matcher_shapes'] = [list(x.shape) for x in args if torch.is_tensor(x)]
            return args, kwargs
        def matcher_post(module, args, result):
            capture['m3_update'] = cpu(result[0])
            capture['match'] = result[3]
        handles.append(net.net.matcher.register_forward_pre_hook(matcher_pre, with_kwargs=True))
        handles.append(net.net.matcher.register_forward_hook(matcher_post))
        def exchange(module, args, result):
            capture['pcea_update'] = cpu(result-args[0])
        handles.append(net.exchange.register_forward_hook(exchange))
        z, _ = net(data, [row], reference, inputs.reference_rows)
        scale = {k:cpu(v) for k,v in net.scale_trace.items()}
        for h in handles:
            h.remove()
        off, _ = net(data, [row], reference, inputs.reference_rows, m3_on=False)
    actual = cpu(z)[0]
    expected = bank['targets']['clean'][0].numpy()
    replay_error = float(abs(actual-expected).max())
    assert replay_error < 1e-4, replay_error
    label = int(inputs.y[index])
    rival = int(np.where(np.arange(8)==label, -np.inf, actual).argmax())
    same = int(np.flatnonzero(inputs.y[refs]==label)[0])
    other = int(np.flatnonzero(inputs.y[refs]==rival)[0])
    images = cpu(data['images'])[0,:,0]
    raw, task = base.m.old.t.raw, base.m.old.t.e.task
    views = task.data.prepare(row)
    regions, slots = [], []
    for v in views:
        regions.append(raw.tokens.soft_regions(v['geometry'])*v['footprint'][None])
        slots.append(raw.tokens.slots(v['geometry'],v['footprint']))
    regions, slots = np.asarray(regions), np.asarray(slots)
    fig, axes = plt.subplots(2,4,figsize=(12,6.1))
    colors = ['#e45c40','#197a72','#dcae24']
    for p in range(2):
        for col in range(3):
            axes[p,col].imshow(images[p],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        for r in range(3):
            axes[p,1].contour(regions[p,r],levels=[.5],colors=[colors[r]],linewidths=1)
        for s in range(20):
            if slots[p,s].max() <= 0:
                continue
            color = colors[0 if s<8 else 2 if s<16 else 1]
            axes[p,2].contour(slots[p,s],levels=[.5],colors=[color],linewidths=.7)
        heat = np.linalg.norm(capture['layer1'][p],axis=0)
        im = axes[p,3].imshow(heat,cmap='magma',interpolation='nearest')
        fig.colorbar(im,ax=axes[p,3],shrink=.65)
        for c,title in enumerate(['Input chip (160 x 160)','OBB-relative regions','20 fine-slot boundaries','Layer1 norm (40 x 40)']):
            axes[p,c].set_title(('VV','VH')[p]+' | '+title,fontsize=10)
            axes[p,c].axis('off')
    fig.legend(handles=[Patch(color=c,label=t) for c,t in zip(colors,['Body','Axial ends','Sides'])],
               loc='lower center',ncol=3)
    fig.suptitle('Actual source chip: '+row['id']+' | bulk carrier | no parameter updates')
    fig.tight_layout(rect=(0,.05,1,.95))
    fig.savefig(OUT/'01_input_regions.png',dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2,4,figsize=(12,6.1))
    for p in range(2):
        maps = [capture[f'm1_{p}'][0,label],
                np.linalg.norm(capture['m3_update'][0,p],axis=0),
                np.linalg.norm(capture['pcea_update'][0,p],axis=0)]
        for c,arr in enumerate(maps):
            im=axes[p,c].imshow(arr,cmap='magma',interpolation='nearest')
            fig.colorbar(im,ax=axes[p,c],shrink=.65)
        axes[p,3].imshow(cpu(rd['images'])[same,p,0],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        for c,title in enumerate(['Bulk-class response (20 x 20)','M3 update norm (10 x 10)',
                                 'PCEA update norm (10 x 10)','Actual same-class reference']):
            axes[p,c].set_title(('VV','VH')[p]+' | '+title,fontsize=10)
            axes[p,c].axis('off')
    fig.suptitle('Measured activations and residual updates; NOT contribution or physical-cause maps')
    fig.tight_layout()
    fig.savefig(OUT/'02_response_updates.png',dpi=160)
    plt.close(fig)
    class_names={}
    for r in inputs.rows:
        v=r.get('vv') or r.get('vh')
        if v and 'image' in v:
            class_names[int(r['label'])]=Path(v['image']).parent.name
    report=dict(source=row['id'],index=index,label=label,classes=class_names,
        original_paths={p:row[p]['image'] for p in ('vv','vh')},
        shapes={k:list(v.shape) for k,v in data.items() if torch.is_tensor(v)},
        raw_image_sizes={}, reference_count=len(refs),
        reference_examples=[inputs.rows[int(refs[same])]['id'],inputs.rows[int(refs[other])]['id']],
        logits=actual.tolist(),probabilities=cpu(z.softmax(-1))[0].tolist(),
        m3_off_logits=cpu(off)[0].tolist(),replay_error=replay_error,
        fine_selection=scale['choice'][0].tolist(),matcher_shapes=capture['matcher_shapes'],
        class_response_shapes=[list(capture[f'm1_{p}'].shape) for p in range(2)],
        m3_update_shape=list(capture['m3_update'].shape),pcea_update_shape=list(capture['pcea_update'].shape),
        same_reference_axis_weights=[cpu(capture['match'][p]['regions'][0]['orientation'])[0,same].tolist() for p in range(2)],
        limitations=['Source example, not held-out evaluation','No parameter updates',
                     'Reference displayed is a bank example, not proven most influential',
                     'Feature/update magnitude is not class contribution'])
    from PIL import Image
    for p in ('vv','vh'):
        with Image.open(row[p]['image']) as im:
            report['raw_image_sizes'][p]=list(im.size)
    (OUT/'trace.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'model_forward.txt').write_text(inspect.getsource(type(net.net).forward),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
