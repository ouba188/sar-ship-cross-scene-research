"""Build candidate geometry from native pixels, not upsampled coarse masks."""
from collections import OrderedDict
import numpy as np
import torch


def can_split(reads, groups):
    result = np.zeros(len(groups), bool)
    for read in reads:
        normalized = read.reshape(20, -1)
        mass = normalized.sum(-1)
        normalized = normalized/np.maximum(mass[:, None], 1e-12)
        for g, children in enumerate(groups):
            live = [i for i in children if mass[i] > 0]
            result[g] |= len(live) > 1 and np.max(
                np.abs(normalized[live, None]-normalized[None, live]).sum(-1)) > 1e-4
    return result


class MixedReads:
    def __init__(self, native, groups):
        self.native, self.groups, self.cache = native, groups, OrderedDict()

    def one(self, row, code=0):
        # The caller's stable row id plus transform is enough for this local run.
        key = (row['id'], int(code))
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        m = self.native
        result = dict(read40=np.zeros((2, 20, 40, 40), np.float32),
                      read20=np.zeros((2, 20, 20, 20), np.float32),
                      fine_write=np.zeros((2, 20, 10, 10), np.float32),
                      slots=np.zeros((2, 20, 10, 10), np.float32),
                      local_slots=np.zeros((2, 20, 20, 20), np.float32),
                      eligible=np.zeros((2, 8), bool))
        for p, view in enumerate(m.old.old.t.e.task.data.prepare(row)):
            if view is None or not view['valid_target']:
                continue
            masks = m.raw.tokens.slots(view['geometry'], view['footprint'])
            if code:
                masks = m.old.old.t.ft.transform(masks, code)
            for name, size in (('read40', 40), ('read20', 20), ('fine_write', 10)):
                result[name][p] = m.raw.tokens.cell_mass(masks, 160//size)/m.geometry._cell_area(size)
            result['fine_write'][p] /= np.maximum(result['fine_write'][p].sum(0, keepdims=True), 1)
            # The two center-origin grids are not nested: inspect BOTH actual inputs.
            result['eligible'][p] = can_split([result['read40'][p], result['read20'][p]], self.groups)
            mask_tensor = torch.from_numpy(np.ascontiguousarray(masks)).float()[None]
            for name, size in (('slots', 10), ('local_slots', 20)):
                result[name][p] = m.raw.grid_occupancy(mask_tensor, (size, size))[0].numpy()
        self.cache[key] = result
        if len(self.cache) > 256:
            self.cache.popitem(last=False)
        return result

    def batch(self, rows, device, code=0):
        records = [self.one(row, code) for row in rows]
        return {k: torch.as_tensor(np.stack([r[k] for r in records]), device=device) for k in records[0]}


def pool_native(feature, read):
    weights = read.flatten(-2).to(feature)
    weights = weights/weights.sum(-1, keepdim=True).clamp_min(1e-12)
    return weights @ feature.flatten(-2).transpose(-1, -2)
