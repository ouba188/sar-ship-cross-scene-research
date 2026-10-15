"""One diagnostic repair: only alter support influence by the conditional increment."""

import run_condition_support_matching as experiment


if __name__ == '__main__':
    experiment.OUT = experiment.BASE/'condition_support_increment_20261011'
    experiment.INCREMENT_ONLY = True
    experiment.main()
