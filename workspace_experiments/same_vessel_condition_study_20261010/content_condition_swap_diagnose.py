"""Frozen decoder diagnostics; no fitting, checkpoint loading, or file writes."""

import numpy as np
import pandas as pd
import torch

from relative_factor_separation import relation_pools, ship_average


def _mse(prediction, target):
    return np.square(prediction.astype(np.float64)-target).mean(axis=1)


def _skill(error, zero):
    return float(100*(1-error/zero)) if zero > 0 else float('nan')


def _encode(model, x, ids, device):
    stable, condition = [], []
    for lo in range(0, len(ids), 256):
        batch = torch.from_numpy(x[ids[lo:lo+256]]).to(device)
        stable.append(model.stable(batch).cpu().numpy())
        condition.append(model.condition(batch).cpu().numpy())
    return np.concatenate(stable), np.concatenate(condition)


def _decode(model, stable, condition, device):
    rows = []
    for lo in range(0, len(stable), 256):
        batch = np.concatenate([stable[lo:lo+256], condition[lo:lo+256]], axis=1)
        rows.append(model.decoder(torch.from_numpy(batch).to(device)).cpu().numpy())
    return np.concatenate(rows)


def _other_ship_donors(meta):
    """One fixed donor per target: uniform other ship, then uniform observation."""
    rng = np.random.default_rng(41033)
    donors = np.full(len(meta), -1, dtype=np.int64)
    for _, same_class in meta.groupby('class_id', sort=False):
        ships = [g.index.to_numpy() for _, g in same_class.groupby('mmsi', sort=False)]
        if len(ships) < 2:
            continue
        for k, ids in enumerate(ships):
            choices = rng.integers(len(ships)-1, size=len(ids))
            choices += choices >= k
            for target, choice in zip(ids, choices):
                donors[target] = rng.choice(ships[choice])
    return donors


@torch.no_grad()
def reconstruction_diagnostics(model, x, meta, fit, test) -> dict:
    """Diagnose a FactorNet-compatible decoder on standardized float32 features.

    ``fit`` and ``test`` are disjoint global positional indices into x/meta;
    DataFrame index labels are ignored. Only fit supplies latent means (ordinary
    observation means, matching the existing decoder diagnosis). Nothing is fit.

    All MSEs average feature coordinates, then pairs/directions within each ship,
    then ships equally. Self and mean replacements use the SAME paired endpoint
    population as swap, not all test observations. Every cross-product pair in
    test is evaluated both ways. For a -> b, condition_delta predicts x_b-x_a
    using D(s_a,c_b)-D(s_a,c_a); the reverse is evaluated independently.
    Skills are percentages: 100*(1-ship_mean_error/ship_mean_zero_error).

    Controls replace c_b with c_j from test, of the same class but a different
    ship, retaining s_a and the original target x_b. Donors are sampled once per
    target, with a fixed local seed, uniformly by vessel then observation; their
    product is unrestricted. This is a class-matched donor control, NOT a
    product-matched or physically causal intervention. Covered directions have
    their own zero baseline and actual-pair reference for a fair comparison.

    Missing pairs/controls or zero delta energy yield NaN metrics, with explicit
    coverage counts. Mean-replacement output changes reveal decoder sensitivity
    even when reconstruction MSE alone is inconclusive; zero change on this
    cohort does not prove global independence of a branch. Parameters, gradients,
    requires_grad flags, and module training modes are preserved.
    """
    fit, test = [np.asarray(ids, dtype=np.int64) for ids in (fit, test)]
    if not len(fit):
        raise ValueError('fit must be nonempty to define training latent means')
    if np.intersect1d(fit, test).size:
        raise ValueError('fit and test must be disjoint global observation indices')
    vm = meta.iloc[test].reset_index(drop=True)
    pairs, _ = relation_pools(vm)
    paired_ships = int(vm.iloc[pairs[:, 0]].mmsi.nunique())
    paired_observations = int(np.unique(pairs).size)
    test_ships = int(vm.mmsi.nunique())
    names = (
        'self_mse', 'swap_mse', 'swap_forward_mse', 'swap_reverse_mse',
        'condition_delta_mse', 'condition_delta_zero_mse', 'condition_delta_skill',
        'mean_condition_mse', 'mean_stable_mse',
        'mean_condition_output_change_mse', 'mean_stable_output_change_mse',
        'same_class_other_ship_swap_mse', 'same_class_other_ship_delta_mse',
        'same_class_other_ship_delta_zero_mse', 'same_class_other_ship_delta_skill',
        'control_matched_swap_mse', 'control_matched_condition_delta_mse',
        'control_matched_condition_delta_skill',
    )
    result = dict.fromkeys(names, float('nan'))
    result.update(
        fit_observations=len(fit), test_observations=len(test), test_ships=test_ships,
        same_ship_pairs=len(pairs), directed_pairs=2*len(pairs),
        paired_observations=paired_observations, paired_ships=paired_ships,
        pair_observation_coverage_fraction=paired_observations/len(test) if len(test) else 0.,
        pair_ship_coverage_fraction=paired_ships/test_ships if test_ships else 0.,
        control_directed_pairs=0, control_ships=0,
        control_direction_coverage_fraction=0., control_ship_coverage_fraction=0.,
        control_policy='test_only_same_class_other_ship_uniform_ship_then_observation_seed41033',
    )
    if not len(pairs):
        return result

    modes = [(module, module.training) for module in model.modules()]
    device = next(model.parameters(), torch.empty(0)).device
    try:
        model.eval()
        train_stable, train_condition = _encode(model, x, fit, device)
        stable, condition = _encode(model, x, test, device)
        own = _decode(model, stable, condition, device)
        mean_condition = _decode(model, stable,
                                 np.repeat(train_condition.mean(0, keepdims=True), len(test), axis=0), device)
        mean_stable = _decode(model,
                              np.repeat(train_stable.mean(0, keepdims=True), len(test), axis=0),
                              condition, device)

        # Alternating directions preserve a/b alignment even for shuffled global IDs.
        source, target = pairs.ravel(), pairs[:, ::-1].ravel()
        ships = vm.iloc[source].mmsi.to_numpy()
        pair_ships = vm.iloc[pairs[:, 0]].mmsi.to_numpy()
        truth = x[test]
        swapped = _decode(model, stable[source], condition[target], device)
        swap_error = _mse(swapped, truth[target])
        delta = truth[target]-truth[source]
        delta_error = _mse(swapped-own[source], delta)
        zero_error = _mse(np.zeros_like(delta), delta)
        result.update(
            self_mse=ship_average(_mse(own, truth)[source], ships),
            swap_mse=ship_average(swap_error, ships),
            swap_forward_mse=ship_average(swap_error[::2], pair_ships),
            swap_reverse_mse=ship_average(swap_error[1::2], pair_ships),
            condition_delta_mse=ship_average(delta_error, ships),
            condition_delta_zero_mse=ship_average(zero_error, ships),
            mean_condition_mse=ship_average(_mse(mean_condition, truth)[source], ships),
            mean_stable_mse=ship_average(_mse(mean_stable, truth)[source], ships),
            mean_condition_output_change_mse=ship_average(_mse(mean_condition, own)[source], ships),
            mean_stable_output_change_mse=ship_average(_mse(mean_stable, own)[source], ships),
        )
        result['condition_delta_skill'] = _skill(result['condition_delta_mse'],
                                                result['condition_delta_zero_mse'])

        donors = _other_ship_donors(vm)[target]
        covered = donors >= 0
        count = int(covered.sum())
        if count:
            control_ships = ships[covered]
            control = _decode(model, stable[source[covered]], condition[donors[covered]], device)
            control_delta = control-own[source[covered]]
            error = ship_average(_mse(control_delta, delta[covered]), control_ships)
            zero = ship_average(zero_error[covered], control_ships)
            actual_error = ship_average(delta_error[covered], control_ships)
            n_ships = int(pd.Series(control_ships).nunique())
            result.update(
                control_directed_pairs=count, control_ships=n_ships,
                control_direction_coverage_fraction=count/len(source),
                control_ship_coverage_fraction=n_ships/paired_ships,
                same_class_other_ship_swap_mse=ship_average(_mse(control, truth[target[covered]]), control_ships),
                same_class_other_ship_delta_mse=error,
                same_class_other_ship_delta_zero_mse=zero,
                same_class_other_ship_delta_skill=_skill(error, zero),
                control_matched_swap_mse=ship_average(swap_error[covered], control_ships),
                control_matched_condition_delta_mse=actual_error,
                control_matched_condition_delta_skill=_skill(actual_error, zero),
            )
        return result
    finally:
        for module, training in modes:
            module.training = training
