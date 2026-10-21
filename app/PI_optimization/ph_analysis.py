import numpy as np
import phbin
from config import DIM

def determine_mesh_range_from_pd(pd, margin=5.0):
    d = pd[0]
    b = np.asarray(d.births, float)
    de = np.asarray(d.deaths, float)
    v = np.concatenate([b[np.isfinite(b)], de[np.isfinite(de)]])
    if not v.size:
        raise ValueError('No finite birth/death values.')
    return float(v.min()-margin), float(v.max()+margin)


def image_to_pd(binary_particle):
    return phbin.make_pd([~binary_particle.astype(bool)], mode='sublevel', dim=DIM)


def pd_to_pi(pd, mesh_range, bins=30, sigma=1.0, weight=('atan', .0001, 3), superlevel=False):
    p = phbin.make_pi(pd, mesh_range=mesh_range, bins=bins,
                      sigma=sigma, weight=weight, superlevel=superlevel)
    return np.asarray(p['pi'][0])


def image_to_pd_pi(binary_particle, mesh_range=None, **kwargs):
    pd = image_to_pd(binary_particle)
    if mesh_range is None:
        mesh_range = determine_mesh_range_from_pd(pd)
    pi = pd_to_pi(pd, mesh_range, **kwargs)
    return pd, pi, mesh_range
