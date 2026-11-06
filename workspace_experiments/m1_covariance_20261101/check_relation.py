"""A same-mean/different-cooccurrence example checks the operator's intended distinction."""
import json
from pathlib import Path
import torch
from operators import RelationFeedback

torch.manual_seed(17)
op = RelationFeedback(2, 'class', hidden=2)
with torch.no_grad():
    op.value.weight.copy_(torch.eye(2).reshape(2, 2, 1, 1))
    op.reader.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]).reshape(4, 2, 1, 1))
    together = torch.tensor([[[[1., 0.]], [[1., 0.]]]])
    opposite = torch.tensor([[[[1., 0.]], [[0., 1.]]]])
    mask = torch.ones(1, 1, 1, 2)
    _, a = op(together, mask, details=True)
    _, b = op(opposite, mask, details=True)
    same_mean = float(abs(a['mean']-b['mean']).max())
    covariance_difference = float(abs(a['covariance']-b['covariance']).max())
    assert same_mean == 0 and covariance_difference > .5
    result = dict(mean_error=same_mean, covariance_difference=covariance_difference,
                  interpretation='Synthetic implementation property, NOT evidence of classification improvement')
    Path(__file__).with_name('relation_check.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result)
