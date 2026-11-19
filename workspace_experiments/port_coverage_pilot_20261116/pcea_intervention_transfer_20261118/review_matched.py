"""Review cached source measurements without running another network forward."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matched_run as run

torch, np, base = run.torch, run.np, run.base


def main():
    bank = torch.load(run.OUT/'verification_bank_local.pt', map_location='cpu', weights_only=True)
    inputs, _, _ = base.runtime()
    ids = bank['ids'].numpy()
    t = bank['targets']
    clean = t['clean'].numpy()
    y = bank['labels'].numpy()
    correct = clean.argmax(1) == y
    picks = list(dict.fromkeys(list(np.flatnonzero(correct)[:2])+list(np.flatnonzero(~correct)[:2])+
                              list(np.flatnonzero(t['matched'].numpy())[:2])))
    refs = inputs.groups['references']
    ref_images = inputs.data(refs, 'small')['images'].cpu().numpy()
    records = []
    for j in picks:
        i = int(ids[j])
        code = int(bank['codes'][j//run.BATCH])
        data = inputs.data(np.array([i]), 'small', code)
        original = data['images'].cpu().numpy()
        masks = base.native_masks(inputs, [i], code)
        proposals, controls, _, _ = base.make_interventions(original,
            data['valid'].cpu().numpy(), masks, data['available'].cpu().numpy())
        k = int(t['selected'][j])
        rival = int(np.where(np.arange(8) == y[j], -np.inf, clean[j]).argmax())
        same = int(np.flatnonzero(inputs.y[refs] == y[j])[0])
        other = int(np.flatnonzero(inputs.y[refs] == rival)[0])
        columns = [original[0], proposals[k,0], controls[k,0],
                   proposals[k,0]-original[0], ref_images[same], ref_images[other]]
        names = ['Original', 'Intervention', 'Level control', 'Difference', 'Same-class ref', 'Rival-class ref']
        fig, axes = plt.subplots(2,6,figsize=(16,5.6))
        for p in range(2):
            for c, pixels in enumerate(columns):
                img = pixels[p,0]
                scale = max(float(abs(img).max()), 1e-6)
                axes[p,c].imshow(img, cmap='RdBu_r' if c == 3 else 'gray',
                    vmin=-scale if c == 3 else 0, vmax=scale if c == 3 else 1,
                    interpolation='nearest')
                axes[p,c].set_title(('VV','VH')[p]+' '+names[c])
                axes[p,c].axis('off')
        def margin(z):
            return float(z[y[j]]-np.max(np.delete(z,y[j])))
        changed = bank['intervention_scores'][k,j].numpy()
        ctrl = bank['control_scores'][k,j].numpy()
        record = dict(source=inputs.rows[i]['id'], index=i, label=int(y[j]),
            augmentation=code, condition=list(base.CONDITIONS[k]),
            original_margin=margin(clean[j]), intervention_margin=margin(changed),
            control_margin=margin(ctrl), original_correct=bool(correct[j]),
            intervention_correct=bool(changed.argmax()==y[j]), accepted=bool(t['use'][j]),
            matched=bool(t['matched'][j]), references=[int(refs[same]),int(refs[other])],
            source_paths={p:inputs.rows[i].get(p,{}).get('image') if inputs.rows[i].get(p) else None for p in ('vv','vh')},
            reference_note='Actual bank examples, not identified top contributors; not independently rescaled')
        fig.suptitle(f"{record['source']} | {record['condition']} | margin {record['original_margin']:.3f} -> {record['intervention_margin']:.3f}")
        fig.tight_layout()
        path = run.OUT/f'review_{j:02d}_local.png'
        fig.savefig(path,dpi=120)
        plt.close(fig)
        record['panel'] = str(path)
        records.append(record)
    (run.OUT/'review_local.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(records,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
