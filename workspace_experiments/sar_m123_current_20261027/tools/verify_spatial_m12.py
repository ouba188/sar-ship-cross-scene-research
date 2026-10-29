"""Real-chip engineering smoke check, not a classification experiment."""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sar_m123.current.data import read_manifest
from sar_m123.current.model import load_model, predict_records, setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--references', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    setup('cpu')
    old, new = load_model(args.legacy), load_model(args.model)
    old_state, new_state = old.state_dict(), new.state_dict()
    assert all(torch.equal(old_state[k], v) for k, v in new_state.items())
    removed = sorted(set(old_state) - set(new_state))
    assert removed and not any(k.startswith('m3.') for k in removed)
    assert not any('channel_selectors' in k for k in new_state)
    assert not any(k.startswith(('m2.', 'deep_pool.')) for k in new_state)
    assert not any(k.startswith('internal_m2.') and '.pool.' in k for k in new_state)
    records = read_manifest(args.manifest)[:2]
    refs = read_manifest(args.references)
    with torch.no_grad():
        scores = predict_records(new, records, refs, device='cpu', batch_size=2)
        legacy_scores = predict_records(old, records, refs, device='cpu', batch_size=2)
    assert scores.shape == (len(records), 8) and torch.isfinite(scores).all()
    result = dict(kind='engineering_smoke_not_accuracy_validation',
        ids=[r['id'] for r in records], references=len(refs),
        architecture=new.architecture, retained_weights_exact=True, m3_weights_unchanged=True,
        old_parameters=sum(p.numel() for p in old.parameters()),
        new_parameters=sum(p.numel() for p in new.parameters()), removed_keys=removed,
        logits=scores.tolist(), legacy_logits=legacy_scores.tolist(),
        max_logit_change=(scores-legacy_scores).abs().max().item(),
        message='Transferred weights need paired evaluation; no gain is claimed.')
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'removed_keys'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
