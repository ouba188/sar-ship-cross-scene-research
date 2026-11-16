"""Selected-checkpoint replay, common-update controls and native-chip cases."""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('prior_joint_analysis', ROOT.parent/'ccrel_pcea_joint_20261116/analysis.py')
helper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helper
spec.loader.exec_module(helper)
prior, np = helper.prior, helper.np


def analyze(model, inputs, arm, receipts, endpoint):
    full, torch = model.full, model.torch
    ids = inputs.groups['validation']; y = inputs.y[ids]
    receipt = receipts[arm]
    epoch = receipt['epochs'] if endpoint == 'last' else receipt['best']['epoch']
    filename = 'last_validation.npz' if endpoint == 'last' else 'validation.npz'
    scores = dict(initial_full=prior.load_score(full.ROOT/'selected_s17/validation.npz', ids, y))
    for name in model.ARMS:
        scores[name] = prior.load_score(ROOT/f'{name}_s17'/filename, ids, y)
    net = model.Model(17, arm)
    net.load_state_dict(torch.load(ROOT/f'{arm}_s17'/(endpoint+'.pt'), map_location='cpu', weights_only=True, mmap=True))
    net.eval().requires_grad_(False)
    try:
        with torch.inference_mode():
            ref = net.reference(inputs.data(inputs.groups['references'], 'small'))
            actual, arrays = helper.collect_trace(model, inputs, net, ids, ref)
            replay = helper.replay_check(actual, scores[arm], arm)
            scores[arm] = actual
            predict = lambda: full.predict(net, inputs, ids, 'validation')
            scores.update(helper.withdraw(net, predict, False))
            helper.replay_check(scores['branch_off'], scores['initial_full'], 'branch off')
            scores['common_replacement'] = helper.switched_prediction(net.exchange, 'force_common', predict)
            factual_centered = net.exchange.mode == 'centered'
            raw = net.ccrel.project_raw(net.source_raw)
            standardized, _ = net.exchange.standardized(raw)
            weight = net.source_weight[:, None, :, None, None]
            source_mean = (standardized*weight).sum(0)/weight.sum(0)
            local, _, _ = net.exchange.contents(standardized)
            source_output = torch.stack(local, 1)
            sw = net.source_weight[:, None, :, None]
            source_output_mean = (source_output*sw).sum(0)/sw.sum(0)
            centered_mean = ((source_output-source_output_mean)*sw).sum(0)/sw.sum(0)
            selected, counts = prior.choose_cases(y, scores, arm, ids)
            # Include a sample whose correct prediction needs query-specific content, when present.
            needed = (actual.argmax(1) == y) & (scores['common_replacement'].argmax(1) != y)
            candidates = np.flatnonzero(needed)
            if len(candidates):
                j = int(candidates[len(candidates)//2])
                if j not in [index for _, index in selected]:
                    selected.append(('query_content_needed', j))
            suffix = f'{arm}_{endpoint}'
            folder = ROOT/f'gallery_{suffix}'
            cases = prior.make_gallery(model, inputs, net, ref, ids, y, scores, arm, endpoint, selected, counts, folder)
            active = arrays['active'].any(1)
            report = dict(arm=arm, endpoint=endpoint, epoch=epoch, selected_best=receipt['best']['epoch'],
                split='reused OSS2 development validation, seed17; not independent confirmation',
                replay=replay, metrics={k: prior.metrics(y, v) for k, v in scores.items()},
                versus_initial={k: prior.paired(y, v, scores['initial_full']) for k, v in scores.items()},
                versus_controls={k: prior.paired(y, actual, scores[k]) for k in
                                 ('coordinated', 'centered', 'common', 'common_replacement', 'no_product')},
                score_changes={k: helper.score_difference(actual, scores[k]) for k in ('common_replacement', 'no_product')},
                constant_shift=prior.constant_shift(actual, scores['branch_off'], y, active),
                trace=prior.trace_diagnostics(arrays, inputs.available[ids].astype(bool)),
                source_standardized_mean_max=float(source_mean.abs().max()),
                source_centered_content_mean_max=float(centered_mean.abs().max()),
                subtracting_source_mean=factual_centered,
                state_counts=counts, cases_count=len(cases), gallery=str(folder/'index.html'),
                notes=['Common-only paired training retains query geometry and RMS; it is not a pure class-logit bias.',
                       'Fixed-weight common_replacement changes content only and is not a retrained ablation.',
                       'All statistics/anchors use frozen source raw tokens transformed by current reader weights.',
                       'Centered source content is zero-mean, but target logits need not be zero-mean.',
                       'Validation-fitted mean class shifts are descriptive only.'])
            np.savez_compressed(ROOT/f'analysis_{suffix}_scores.npz', ids=ids, label=y, **scores)
            np.savez_compressed(ROOT/f'analysis_{suffix}_traces.npz', ids=ids, label=y, **arrays)
            prior.write(ROOT/f'analysis_{suffix}.json', report)
            print(json.dumps(dict(arm=arm, endpoint=endpoint, epoch=epoch, metrics=report['metrics'][arm],
                paired=report['versus_initial'][arm], common=report['versus_controls']['common_replacement'],
                mean_energy=report['constant_shift']['fraction_squared_change']), ensure_ascii=False), flush=True)
    finally:
        del net
        gc.collect(); torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arms', nargs='+')
    parser.add_argument('--endpoint', choices=('best', 'last'), default='best')
    args = parser.parse_args()
    import model
    receipts = {a: prior.read(ROOT/f'{a}_s17/receipt.json') for a in model.ARMS}
    markdown, table = helper.summary_contents(model.ARMS, receipts)
    markdown = markdown.replace('Joint reader / PCEA', 'Source-coordinate repair / PCEA')
    markdown = markdown.split('Detailed outputs:')[0]
    markdown += ('Current projections of frozen source raw tokens define coordinates and anchors; '
                 'no validation statistics are fitted for inference.\n'
                 'Best and last diagnostics are explicitly named; no automatic checkpoint substitution.\n')
    (ROOT/'RESULTS.md').write_text(markdown, encoding='utf-8')
    (ROOT/'RESULTS.csv').write_text(table, encoding='utf-8')
    inputs = model.full.Inputs()
    for split in ('small', 'validation'):
        inputs.prepare_geometry(split)
    for arm in args.arms or model.ARMS:
        analyze(model, inputs, arm, receipts, args.endpoint)


if __name__ == '__main__':
    main()
