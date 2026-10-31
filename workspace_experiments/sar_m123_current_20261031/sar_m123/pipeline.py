"""Public raw-chip to class-score API with only local exported assets."""
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax
import torch

from .input import prepare
from .backbone import Backbone
from .baseline import Baseline, extract as extract_baseline
from .m12 import M12, SummaryReader
from .m3 import M3


class ShipClassifier:
    def __init__(self, assets, device='cpu'):
        self.assets = Path(assets)
        self.manifest = json.loads((self.assets/'manifest.json').read_text(encoding='utf-8'))
        if not self.manifest['complete']:
            raise ValueError('Asset export is incomplete; export all M1/M2/M3 assets first.')
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.backbone = Backbone(self.assets/'backbone/resnet50.pth', device)
        self.summary_reader = SummaryReader(self.assets)
        self.m3 = M3(self.assets, device=device)
        self.baselines, self.m12 = {}, {}

    @torch.inference_mode()
    def extract(self, record, directory=Path('.')):
        prepared = prepare(record, directory)
        maps = self.backbone(prepared)
        baseline = extract_baseline(prepared, maps, self.backbone)
        # A polarization without a target may still inform the historical
        # baseline background path; it must not invent M1-M3 target evidence.
        target = [v if v is not None and v['valid_target'] else None for v in prepared]
        target_maps = [m if v is not None else None for v, m in zip(target, maps)]
        summaries = self.summary_reader.summarize(target, target_maps)
        local = self.m3.extract(target, target_maps)
        return dict(baseline=baseline, summaries=summaries, local=local,
                    placement=[v['placement'] if v else None for v in prepared])

    @torch.inference_mode()
    def score(self, features, seed=17):
        if seed not in self.manifest['seeds']:
            raise ValueError(f'Exported seeds are {self.manifest["seeds"]}.')
        if seed not in self.baselines:
            self.baselines[seed] = Baseline(self.assets, seed)
            self.m12[seed] = M12(self.assets, seed)
        base, base_details = self.baselines[seed](features['baseline'])
        s12, stage_details = self.m12[seed](base, features['summaries'])
        scores, match_details = self.m3(features['local'], s12, seed)
        return scores, dict(base=base, s12=s12, base_details=base_details,
                            stages=stage_details, matching=match_details)

    def predict(self, record, directory=Path('.'), seeds=(17,), trace=False):
        features = self.extract(record, directory)
        outputs = []
        for seed in seeds:
            scores, details = self.score(features, seed)
            pred = int(scores.argmax())
            output = dict(seed=seed, class_id=pred, class_name=self.manifest['class_names'][pred],
                          class_scores=scores.tolist(), softmax_scores=softmax(scores).tolist(),
                          polar_weights=details['matching']['polar_weights'].tolist(),
                          selected_steps=details['stages']['selected_steps'])
            if trace:
                output['trace'] = to_json(details)
            outputs.append(output)
        return dict(id=record.get('id'), predictions=outputs,
                    class_order=self.manifest['class_names'],
                    target_evidence_available=features['summaries']['valid'].tolist(),
                    placement=features['placement'],
                    softmax_is_calibrated_probability=False)


def to_json(value):
    if isinstance(value, dict):
        return {key: to_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
