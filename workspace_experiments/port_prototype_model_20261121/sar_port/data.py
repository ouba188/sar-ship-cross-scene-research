"""Native chip/OBB input. Labels and identities stay outside model tensors."""
import json
from pathlib import Path
import numpy as np
import torch
from . import geometry
from .selection import GROUPS


def load_records(path):
    path = Path(path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    rows = payload['records'] if isinstance(payload, dict) else payload
    result = []
    for i, source in enumerate(rows):
        row = dict(source)
        row.setdefault('id', str(i))
        for pol in ('vv', 'vh'):
            if row.get(pol):
                view = dict(row[pol])
                image = Path(view['image'])
                view['image'] = str(image if image.is_absolute() else path.parent/image)
                row[pol] = view
        result.append(row)
    return result


def occupancy(native, count):
    edges = np.r_[-.5, (np.arange(count-1)+.5)*(160/count), 159.5]
    area = np.diff(edges)[:, None]*np.diff(edges)[None, :]
    return (geometry.cell_mass(native, 160//count)/area).astype(np.float32)


def eligible_parents(reads):
    eligible = np.zeros(8, dtype=bool)
    for read in reads:
        flat = read.reshape(20, -1)
        mass = flat.sum(-1)
        norm = flat/np.maximum(mass[:, None], 1e-12)
        for g, children in enumerate(GROUPS):
            live = [i for i in children if mass[i] > 0]
            eligible[g] |= len(live) > 1 and np.abs(norm[live, None]-norm[None, live]).sum(-1).max() > 1e-4
    return eligible


def prepare_record(row, code=0):
    """D4 augment images and all native masks before native-grid integration."""
    result = dict(images=np.zeros((2,1,160,160),np.float32), valid=np.zeros((2,1,160,160),np.float32),
                  available=np.zeros(2,np.float32), read40=np.zeros((2,20,40,40),np.float32),
                  read20=np.zeros((2,20,20,20),np.float32), slots=np.zeros((2,20,10,10),np.float32),
                  local_slots=np.zeros((2,20,20,20),np.float32), fine_write=np.zeros((2,20,10,10),np.float32),
                  eligible=np.zeros((2,8),bool), regions_native=np.zeros((2,3,160,160),np.float32),
                  size_condition=np.zeros((2,8,5),np.float32))
    def transform(x):
        value = np.rot90(x, code % 4, axes=(-2,-1))
        return np.ascontiguousarray(value[..., ::-1] if code >= 4 else value)
    for p, view in enumerate(geometry.prepare(row)):
        if view is None or not view['valid_target']:
            continue
        masks = transform(geometry.slots(view['geometry'], view['footprint']))
        result['images'][p,0] = transform(view['pixels']).astype(np.float32)/255.
        result['valid'][p,0] = transform(view['footprint'])
        result['available'][p] = 1
        result['regions_native'][p] = np.stack((masks[:8].sum(0),masks[16:].sum(0),masks[8:16].sum(0)))
        for name, count in (('read40',40),('read20',20),('slots',10),('local_slots',20)):
            result[name][p] = occupancy(masks,count)
        writer = result['slots'][p].copy()
        result['fine_write'][p] = writer/np.maximum(writer.sum(0,keepdims=True),1)
        result['eligible'][p] = eligible_parents((result['read40'][p],result['read20'][p]))
        result['size_condition'][p,:,:2] = np.log1p(view['geometry'][2:4])/np.log1p(160)
        for level, name in enumerate(('read40','read20','slots'), start=2):
            read = result[name][p]
            coverage = np.asarray([read[list(ids)].sum() for ids in GROUPS])
            result['size_condition'][p,:,level] = np.log1p(coverage)/np.log1p(read.shape[-1]**2)
    return result


def prepare_batch(rows, device='cpu', code=0):
    if not rows:
        raise ValueError('A batch must contain at least one chip.')
    batch = [prepare_record(row,code) for row in rows]
    result = {key:torch.as_tensor(np.stack([b[key] for b in batch]),device=device) for key in batch[0]}
    ports = sorted({str(r.get('port','unknown')) for r in rows})
    result['port_index'] = torch.tensor([ports.index(str(r.get('port','unknown'))) for r in rows],device=device)
    return result


def pool_native(feature, read):
    weight = read.flatten(-2).to(feature)
    weight = weight/weight.sum(-1,keepdim=True).clamp_min(1e-12)
    return weight @ feature.flatten(-2).transpose(-1,-2)
