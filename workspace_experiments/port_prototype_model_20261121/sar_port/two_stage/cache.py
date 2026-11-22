"""Opt-in, content-addressed cache of deterministic native-pixel preparation.

Set SAR_PORT_PREPARE_CACHE to a directory to enable it. Cache payloads contain
only tensors and Python containers and are always read with weights_only=True.
"""
from dataclasses import asdict, fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from . import data
from .contracts import LevelGeometry, ShipGeometry
from .data import InputConfig, PreparedShip
from .geometry import GeometryConfig


_VERSION = 1
_IMPLEMENTATION_FILES = tuple(Path(__file__).with_name(name) for name in
                              ('data.py', 'geometry.py', 'contracts.py', 'cache.py'))
_CLASSES = {cls.__name__: cls for cls in (PreparedShip, ShipGeometry, LevelGeometry)}


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unsupported cache-key value: {type(value).__name__}')


def cache_key(record, directory=Path('.'), input_config=None, geometry_config=None):
    """Hash every input byte on every call; timestamps never establish identity."""
    directory = Path(directory).resolve()
    inputs = []
    for polarization in ('vv', 'vh'):
        view = record.get(polarization)
        if not view:
            continue
        for field in ('image', 'valid_mask', 'background_mask'):
            if view.get(field):
                path = (directory / view[field]).resolve()
                # Inactive branches/unvalidated Q may legitimately reference absent
                # files that prepare_record does not read. Keep that state in the
                # key without imposing additional qualification requirements.
                inputs.append((polarization, field, str(path),
                               _digest(path) if path.is_file() else None))
    specification = dict(
        version=_VERSION, record=record, directory=str(directory), inputs=inputs,
        input_config=asdict(input_config or InputConfig()),
        geometry_config=asdict(geometry_config or GeometryConfig()),
        implementation=[(p.name, _digest(p)) for p in _IMPLEMENTATION_FILES])
    encoded = json.dumps(specification, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False, default=_json_default)
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def _root():
    value = os.environ.get('SAR_PORT_PREPARE_CACHE', '').strip()
    return Path(value).expanduser().resolve() if value else None


def _pack(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if is_dataclass(value):
        name = type(value).__name__
        if name not in _CLASSES or type(value) is not _CLASSES[name]:
            raise TypeError(f'Unsupported prepared dataclass: {name}')
        return {'_prepared_type': name,
                'fields': {f.name: _pack(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, dict):
        return {key: _pack(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_pack(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unsupported prepared value: {type(value).__name__}')


def _unpack(value):
    if isinstance(value, dict):
        if '_prepared_type' in value:
            cls = _CLASSES[value['_prepared_type']]
            if set(value) != {'_prepared_type', 'fields'}:
                raise ValueError('Malformed prepared-cache dataclass')
            if set(value['fields']) != {f.name for f in fields(cls)}:
                raise ValueError('Prepared-cache dataclass fields changed')
            return cls(**{key: _unpack(item) for key, item in value['fields'].items()})
        return {key: _unpack(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_unpack(item) for item in value)
    return value


def _store(root, key, prepared):
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f'{key}.pt'
    descriptor, temporary = tempfile.mkstemp(prefix=key + '.', suffix='.tmp', dir=root)
    os.close(descriptor)
    try:
        torch.save(dict(version=_VERSION, key=key, prepared=_pack(prepared)), temporary)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination


def save_prepared(record, directory, input_config, geometry_config, prepared):
    """Store an already prepared sample, without recomputation; disabled is a no-op."""
    root = _root()
    if root is None:
        return None
    return _store(root, cache_key(record, directory, input_config, geometry_config), prepared)


def prepare_record_cached(record, directory=Path('.'), input_config=None, geometry_config=None):
    root = _root()
    if root is None:
        return data.prepare_record(record, directory, input_config, geometry_config)
    key = cache_key(record, directory, input_config, geometry_config)
    destination = root / f'{key}.pt'
    if destination.exists():
        payload = torch.load(destination, map_location='cpu', weights_only=True)
        if payload.get('version') != _VERSION or payload.get('key') != key:
            raise ValueError(f'Prepared-cache metadata mismatch: {destination}')
        prepared = _unpack(payload['prepared'])
        if not isinstance(prepared, PreparedShip):
            raise ValueError(f'Prepared-cache payload is not PreparedShip: {destination}')
        return prepared
    prepared = data.prepare_record(record, directory, input_config, geometry_config)
    _store(root, key, prepared)
    return prepared
