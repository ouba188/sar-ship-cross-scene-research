"""OBB error design in native pixels; intentionally no tensor-library imports."""
import math


def conditions():
    values = [dict(name='zero', family='zero', value=0.)]
    for family, levels in [('along', (.05, .1, .2)), ('across', (.05, .1, .2)),
                           ('scale', (.1, .2)), ('angle', (5., 10., 20.)),
                           ('length', (.1, .2)), ('width', (.1, .2))]:
        for level in levels:
            for sign in (-1, 1):
                value = sign * level
                tag = int(round(abs(value) * (1 if family == 'angle' else 100)))
                values.append(dict(name=f'{family}_{"m" if sign < 0 else "p"}{tag:02d}',
                                   family=family, value=value))
    return values


def pipeline_conditions():
    return [v for v in conditions() if v['family'] == 'zero' or
            (v['family'] in ('along', 'across', 'scale', 'angle') and
             abs(v['value']) == (20 if v['family'] == 'angle' else .2))]


def perturb(geometry, condition):
    g = [float(v) for v in geometry]
    value, family = condition['value'], condition['family']
    co, si = math.cos(g[4]), math.sin(g[4])
    if family == 'along':
        g[0] += value * g[2] * co
        g[1] += value * g[2] * si
    elif family == 'across':
        g[0] -= value * g[3] * si
        g[1] += value * g[3] * co
    elif family == 'scale':
        g[2] *= 1 + value
        g[3] *= 1 + value
    elif family == 'length':
        g[2] *= 1 + value
    elif family == 'width':
        g[3] *= 1 + value
    elif family == 'angle':
        g[4] += math.radians(value)
    return g
