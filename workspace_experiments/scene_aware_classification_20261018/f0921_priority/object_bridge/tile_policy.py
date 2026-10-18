"""Apply an explicit, complete land-mask plan without changing unplanned runs."""
import csv


def select_tiles(tiles, policy_path=None):
    if policy_path is None:
        return tiles, []
    with open(policy_path, encoding='utf-8-sig', newline='') as f:
        policies = {r['tile_id']:r['skip_land_interior'] for r in csv.DictReader(f)}
    kept, skipped = [], []
    for tile in tiles:
        decision = policies[tile['tile_id']]
        if decision not in ('True', 'False'):
            raise ValueError('Expected boolean skip_land_interior in mask plan')
        (skipped if decision == 'True' else kept).append(tile)
    return kept, skipped
