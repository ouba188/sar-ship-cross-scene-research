"""Render the exact residual-only harm from saved paired predictions, without refitting."""
import analysis
import model

root = analysis.BASE_ROOT/'reader_lr1e5'
np, torch, full = model.np, model.torch, model.full
with np.load(root/'analysis_joint_fine_last_scores.npz') as data:
    ids, y = data['ids'], data['label']
    scores = {k: data[k] for k in data.files if k not in ('ids', 'label')}
bad = np.flatnonzero((scores['joint_fine'].argmax(1) != y) & (scores['fine_off'].argmax(1) == y))
assert len(bad) == 1
inputs = full.Inputs()
for split in ('small', 'validation'):
    inputs.prepare_geometry(split)
net = model.Model(17, 'joint_fine')
net.load_state_dict(torch.load(root/'joint_fine_s17/last.pt', map_location='cpu', weights_only=True, mmap=True))
net.eval().requires_grad_(False)
with torch.inference_mode():
    reference = net.reference(inputs.data(inputs.groups['references'], 'small'))
    analysis.prior.make_gallery(model, inputs, net, reference, ids, y, scores, 'joint_fine', 'last',
        [('harmed_by_fine', int(bad[0]))], {'harmed_by_fine': 1}, root/'gallery_residual_harm')
print('RESIDUAL_HARM_ID', int(ids[bad[0]]))
