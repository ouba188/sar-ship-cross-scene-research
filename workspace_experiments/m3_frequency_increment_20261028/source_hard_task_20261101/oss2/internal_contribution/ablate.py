"""Isolate class-conditioned metric and cost competition in aggregate M3."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import train_task as training
from model_task import Model, MaskedCostLocalMatch


class IsolatedMatch(MaskedCostLocalMatch):
    def __init__(self, seed=17, classes=4, variant='full'):
        super().__init__(seed=seed, classes=classes)
        self.variant = variant

    def metric_condition(self, mu, other, all_mu, arm):
        mode = 'shared' if self.variant == 'shared_metric' else arm
        return super().metric_condition(mu, other, all_mu, mode)

    def cost_competition(self, competition, arm):
        mode = 'shared' if self.variant == 'no_competition' else arm
        return super().cost_competition(competition, mode)


class AblationModel(Model):
    def __init__(self, arm, seed=17, variant='full'):
        assert arm == 'aggregate'
        super().__init__(arm, seed=seed)
        replacement = IsolatedMatch(seed=seed, classes=4, variant=variant).to('cuda')
        replacement.load_state_dict(self.matcher.state_dict())
        self.matcher = replacement


if __name__ == '__main__':
    variant = sys.argv.pop(1)
    assert variant in ('shared_metric', 'no_competition')
    training.Model = lambda arm, seed=17: AblationModel(arm, seed, variant)
    training.main()
