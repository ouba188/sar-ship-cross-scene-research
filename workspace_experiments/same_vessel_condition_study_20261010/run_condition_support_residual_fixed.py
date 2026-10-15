"""Rerun both bridges with the signed-target fix in new, exclusive directories."""

import run_condition_support_matching as experiment


if __name__ == '__main__':
    for stem, increment in [('condition_support_matching_20261011', False),
                            ('condition_support_increment_20261011', True)]:
        experiment.OUT = experiment.BASE/(stem+'_residual_fixed')
        experiment.INCREMENT_ONLY = increment
        experiment.main()
