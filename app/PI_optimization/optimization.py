import numpy as np
import scipy.optimize as opt
from generator_porespy import generate_porespy_image
from ph_analysis import image_to_pd_pi
from config import (BOUNDS, GAP_THRESHOLD, GAP_SCALE_FACTOR, LOC_SMALL, PI_BINS,
                     PI_SIGMA, PI_WEIGHT, PI_SUPERLEVEL)


PARAMETER_NAMES = [
    'loc_small', 'scale_small', 'loc_large',
    'scale_large', 'porosity', 'r_min'
]


def make_target_large_bounds(sem, loc_margin=1.0, scale_factor_low=0.5,
                             scale_factor_high=2.0):
    large = np.asarray(sem.get('large_radii', []), dtype=float)
    if large.size < 2:
        raise ValueError('At least two large-particle radii are required.')

    loc_small = float(LOC_SMALL)
    rmax_large = float(np.max(large))
    large_scale_std = max(float(np.std(large, ddof=1)), 1e-3)
    peak_gap = float(sem.get('peak_gap', 0.0))
    delta_r = 5.0

    if peak_gap >= GAP_THRESHOLD:
        scale_hi = max(1e-3, GAP_SCALE_FACTOR * peak_gap)
        return (rmax_large-delta_r, rmax_large+delta_r), (1e-3, scale_hi)

    margin = max(loc_margin, 0.1 * large_scale_std)
    loc_lo = max(1e-3, float(np.min(large)) - margin)
    loc_hi = rmax_large + margin
    scale_lo = max(1e-3, scale_factor_low * large_scale_std)
    scale_hi = max(scale_lo * 1.01, scale_factor_high * large_scale_std)
    return (loc_lo, loc_hi), (scale_lo, scale_hi)


def make_bounds(sem):
    loc_large_bounds, scale_large_bounds = make_target_large_bounds(sem)
    return [BOUNDS[0], BOUNDS[1], loc_large_bounds, scale_large_bounds,
            BOUNDS[4], BOUNDS[5]]


def _pi_grid(mesh_range, bins):
    lo, hi = mesh_range
    edges = np.linspace(lo, hi, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return np.meshgrid(centers, centers, indexing='ij')


def _normalize_pi(pi):
    p = np.asarray(pi, dtype=float)
    s = p.sum()
    return p / s if s > 0 else p


def _weights_for_loss(mesh_range, bins, loss_name,
                      large_birth_cut=None, diag_width=None,
                      large_weight=5.0, diag_weight=0.15):
    b, d = _pi_grid(mesh_range, bins)
    persistence = np.maximum(d - b, 0.0)
    w = np.ones_like(b)

    if loss_name == 'global_hybrid':
        return None

    if loss_name == 'birth_weighted':
        if large_birth_cut is None:
            large_birth_cut = float(mesh_range[0] + 0.30 * (mesh_range[1] - mesh_range[0]))
        # Smoothly emphasize the left-side birth region rather than a hard mask.
        width = max(1.0, 0.05 * (mesh_range[1] - mesh_range[0]))
        w += (large_weight - 1.0) / (1.0 + np.exp((b - large_birth_cut) / width))
        return w

    if loss_name == 'large_region':
        if large_birth_cut is None:
            large_birth_cut = float(mesh_range[0] + 0.30 * (mesh_range[1] - mesh_range[0]))
        if diag_width is None:
            diag_width = 0.15 * (mesh_range[1] - mesh_range[0])
        mask = (b <= large_birth_cut) & (persistence <= diag_width)
        w[:] = 1.0
        w[mask] = large_weight
        return w

    if loss_name == 'diag_suppressed':
        if diag_width is None:
            diag_width = 0.12 * (mesh_range[1] - mesh_range[0])
        # Suppress the diagonal strip where short-lived generator noise is expected.
        w = np.where(persistence <= diag_width, diag_weight, 1.0)
        return w

    if loss_name == 'large_plus_diag':
        if large_birth_cut is None:
            large_birth_cut = float(mesh_range[0] + 0.30 * (mesh_range[1] - mesh_range[0]))
        if diag_width is None:
            diag_width = 0.15 * (mesh_range[1] - mesh_range[0])
        mask_large = (b <= large_birth_cut) & (persistence <= diag_width)
        w = np.where(persistence <= diag_width, diag_weight, 1.0)
        w[mask_large] = large_weight
        return w

    raise ValueError(f'Unknown loss_name: {loss_name}')


def pi_loss(pi_real, pi_gen, mesh_range, loss_name='global_hybrid',
             L_lambda=.8, large_birth_cut=None, diag_width=None,
             large_weight=5.0, diag_weight=0.15):
    a = np.asarray(pi_real, dtype=float)
    b = np.asarray(pi_gen, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f'PI shapes differ: {a.shape} vs {b.shape}')

    if loss_name == 'global_hybrid':
        return float(
            L_lambda * np.mean(np.abs(np.sqrt(np.maximum(a, 0)) - np.sqrt(np.maximum(b, 0))))
            + (1 - L_lambda) * np.mean(np.abs(a - b))
        )

    w = _weights_for_loss(mesh_range, a.shape[0], loss_name,
                          large_birth_cut, diag_width,
                          large_weight, diag_weight)
    diff = a - b
    # Normalize weights so changing the mask does not trivially change the scale.
    w = w / np.mean(w)
    return float(np.sum(w * np.abs(diff)) / diff.size)


def make_cost(pi_real, image_shape, mesh_range, dist_weight, bounds,
              L_lambda=.8, sampling_mode='separate', loss_name='global_hybrid',
              large_birth_cut=None, diag_width=None,
              large_weight=5.0, diag_weight=0.15):
    def cost(x):
        alpha = dict(zip(PARAMETER_NAMES, x))
        pore = generate_porespy_image(
            image_shape, alpha, dist_weight, sampling_mode=sampling_mode
        )
        _, pi, _ = image_to_pd_pi(
            pore, mesh_range, bins=PI_BINS, sigma=PI_SIGMA,
            weight=PI_WEIGHT, superlevel=PI_SUPERLEVEL
        )
        return pi_loss(
            pi_real, pi, mesh_range, loss_name=loss_name,
            L_lambda=L_lambda, large_birth_cut=large_birth_cut,
            diag_width=diag_width, large_weight=large_weight,
            diag_weight=diag_weight
        )
    return cost


def optimize_parameters(pi_real, image_shape, mesh_range, dist_weight,
                         x_ini, bounds, L_lambda=.8,
                         sampling_mode='separate', loss_name='global_hybrid',
                         large_birth_cut=None, diag_width=None,
                         large_weight=5.0, diag_weight=0.15):
    return opt.minimize(
        make_cost(
            pi_real, image_shape, mesh_range, dist_weight, bounds,
            L_lambda, sampling_mode, loss_name, large_birth_cut,
            diag_width, large_weight, diag_weight
        ),
        x_ini,
        method='Powell',
        bounds=bounds,
        options={'maxiter': 60, 'xtol': 1e-2, 'ftol': 1e-3, 'disp': True},
    )
