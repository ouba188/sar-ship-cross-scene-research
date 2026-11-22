"""Check saved artifacts independently, including deliberate corruption controls."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'oss2_q_pilot'


def pixels(folder, prefix):
    return np.stack([np.asarray(Image.open(folder/f'{prefix}_{p}.png')) for p in ('vv','vh')])


def assert_intervention(original, altered, q, donor, ids):
    assert np.array_equal(original[:,~q], altered[:,~q]), 'Changed protected/non-Q pixels'
    for p in range(2):
        assert np.array_equal(altered[p,q], donor[p].ravel()[ids]), 'Broken paired donor mapping'


def main():
    records = {r['id']:r for r in json.loads((ROOT/'q_review_selection_local.json').read_text(encoding='utf8'))}
    reviews = json.loads((OUT/'q_qualification_local.json').read_text(encoding='utf8'))
    actions = {r['sample_id']:r for r in json.loads((OUT/'interventions_local.json').read_text(encoding='utf8'))}
    regions = json.loads((OUT/'review_regions_local.json').read_text(encoding='utf8'))
    checks = negatives = 0
    for review in reviews:
        sid = review['sample_id']; folder=OUT/sid; r=records[sid]
        q=np.asarray(Image.open(folder/'q_mask.png')) > 0
        before, after = pixels(folder, 'original'), pixels(folder, 'intervened')
        exclusion = unary_union([Polygon(d['corners']).buffer(8) for p in ('vv','vh') for d in r[p]['detections']]+
                                [box(*b).buffer(8) for b in regions[sid]['neighbor_boxes_xyxy']])
        assert all(not exclusion.intersects(box(x-.5,y-.5,x+.5,y+.5)) for y,x in zip(*np.nonzero(q)))
        assert np.array_equal(before, np.stack([np.asarray(Image.open(r[p]['image'])) for p in ('vv','vh')]))
        assert review['training_qualified'] is False and review['leakage_controlled'] is False
        if actions[sid]['donor_id']:
            donor_folder=OUT/actions[sid]['donor_id']; donor=pixels(donor_folder,'original')
            donor_q=np.asarray(Image.open(donor_folder/'q_mask.png')) > 0
            with np.load(folder/'intervention_indices_local.npz') as z:
                ids=z['donor_flat_indices']; targets=z['recipient_flat_indices']
            assert np.array_equal(targets,np.flatnonzero(q)) and donor_q.ravel()[ids].all()
            assert_intervention(before,after,q,donor,ids); checks+=1
            # A changed non-Q pixel and a one-channel mapping error must fail.
            for bad in ('outside','one_channel'):
                changed=after.copy()
                flat=int(np.flatnonzero(~q if bad=='outside' else q)[0])
                channel=0 if bad=='outside' else 1
                changed[channel].ravel()[flat] ^= 1
                try:
                    assert_intervention(before,changed,q,donor,ids)
                except AssertionError:
                    negatives+=1
                else:
                    raise AssertionError('Corruption control was not detected')
        else:
            assert not review['background_validated'] and np.array_equal(before,after)
    assert checks==3 and negatives==6
    result=dict(reviewed_pairs_verified=checks, corruption_controls_rejected=negatives,
                all_four_masks_avoid_full_protection=True, original_pixels_unchanged=True,
                controlled_qualification_false=True, training_started=False)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
