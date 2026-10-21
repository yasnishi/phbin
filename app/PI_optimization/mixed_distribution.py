import numpy as np
import scipy.stats as spst

from config import GAP_THRESHOLD, GAP_SCALE_FACTOR


class MixedExponentialNormal(spst.rv_continuous):
    """Two-component particle-radius distribution."""

    def __init__(self, weight, loc_small, scale_small, loc_large,
                 scale_large, r_min):
        super().__init__(a=min(loc_small, r_min),
                         name='mixed_exponential_normal')

        self.weight = float(weight)
        self.loc_small = float(loc_small)
        self.scale_small = max(float(scale_small), 1e-8)
        self.loc_large = float(loc_large)
        self.scale_large = max(float(scale_large), 1e-8)
        self.r_min = float(r_min)

        gap = self.loc_large - self.loc_small
        self.has_gap = gap >= GAP_THRESHOLD
        self.effective_scale_large = self.scale_large

        self.a_large = (
            self.r_min - self.loc_large
        ) / self.effective_scale_large

    def _cdf(self, x):
        small = spst.expon.cdf(
            x, loc=self.loc_small, scale=self.scale_small
        )

        a = self.a_large
        pa = spst.norm.cdf(a)
        px = spst.norm.cdf(
            (x - self.loc_large) / self.effective_scale_large
        )

        large = np.where(
            x < self.r_min,
            0.0,
            (px - pa) / (1.0 - pa),
        )

        return self.weight * small + (1.0 - self.weight) * large

    def small_ppf(self, q):
        return spst.expon.ppf(
            q, loc=self.loc_small, scale=self.scale_small
        )

    def large_ppf(self, q):
        return spst.truncnorm.ppf(
            q,
            a=self.a_large,
            b=np.inf,
            loc=self.loc_large,
            scale=self.effective_scale_large,
        )


def make_distribution(alpha, dist_weight):
    return MixedExponentialNormal(
        dist_weight,
        alpha['loc_small'],
        alpha['scale_small'],
        alpha['loc_large'],
        alpha['scale_large'],
        alpha['r_min'],
    )


# -------------------------------------------------------------------------
# code11 mode
# -------------------------------------------------------------------------

def make_bimodal_generator_radii(alpha, dist_weight, nbins,
                                 q_low=0.05, q_high=0.95):
    """Component-aware discrete radii used by code11.

    The fixed number of bins is allocated to the small/large components
    according to the fixed mixture weight.
    """
    nbins = int(nbins)
    if nbins < 2:
        raise ValueError("nbins must be >= 2 for a bimodal distribution")

    w_small = float(dist_weight)
    if not 0.0 < w_small < 1.0:
        raise ValueError("dist_weight must be between 0 and 1")

    dist = make_distribution(alpha, dist_weight)

    n_small = int(round(nbins * w_small))
    n_small = min(max(n_small, 1), nbins - 1)
    n_large = nbins - n_small

    q_small = q_low + (np.arange(n_small) + 0.5) / n_small * (q_high - q_low)
    q_large = q_low + (np.arange(n_large) + 0.5) / n_large * (q_high - q_low)

    r_small = np.asarray(dist.small_ppf(q_small), dtype=float)
    r_large = np.asarray(dist.large_ppf(q_large), dtype=float)

    radii = np.concatenate([r_small, r_large])
    component = np.concatenate([
        np.full(n_small, 'small', dtype=object),
        np.full(n_large, 'large', dtype=object),
    ])

    order = np.argsort(radii)
    return radii[order], component[order], dist


class BimodalGeneratorDistribution:
    """PoreSpy adapter for code11 discrete radii."""

    def __init__(self, base_dist, radii, q_low=0.05, q_high=0.95):
        self.base_dist = base_dist
        self.radii = np.asarray(radii, dtype=float)
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.q_grid = np.linspace(self.q_low, self.q_high, len(self.radii))

    def ppf(self, q):
        q_arr = np.asarray(q, dtype=float)
        q_clip = np.clip(q_arr, self.q_low, self.q_high)
        out = np.interp(q_clip, self.q_grid, self.radii)
        return float(out) if np.ndim(q) == 0 else out

    def interval(self, *args, **kwargs):
        return self.base_dist.interval(*args, **kwargs)

    def rvs(self, *args, **kwargs):
        return self.base_dist.rvs(*args, **kwargs)

    def cdf(self, *args, **kwargs):
        return self.base_dist.cdf(*args, **kwargs)

    def pdf(self, *args, **kwargs):
        return self.base_dist.pdf(*args, **kwargs)


# -------------------------------------------------------------------------
# code12 mode
# -------------------------------------------------------------------------

class ComponentAwarePPF:
    """Continuous component-aware PPF used by code12.

    The cumulative probability axis is partitioned by the fixed mixture
    weight. PoreSpy performs the final discretization through nbins.
    """

    def __init__(self, base_dist, q_low=0.05, q_high=0.95):
        self.base_dist = base_dist
        self.weight = float(base_dist.weight)
        self.q_low = float(q_low)
        self.q_high = float(q_high)

        if not 0.0 < self.weight < 1.0:
            raise ValueError('base_dist.weight must be between 0 and 1')
        if not 0.0 <= self.q_low < self.q_high <= 1.0:
            raise ValueError('Require 0 <= q_low < q_high <= 1')

    def ppf(self, q):
        q_arr = np.asarray(q, dtype=float)
        scalar = np.ndim(q) == 0
        q_arr = np.clip(q_arr, 0.0, 1.0)
        out = np.empty_like(q_arr, dtype=float)

        small_mask = q_arr < self.weight
        large_mask = ~small_mask

        if np.any(small_mask):
            u = q_arr[small_mask] / self.weight
            u = self.q_low + u * (self.q_high - self.q_low)
            out[small_mask] = self.base_dist.small_ppf(u)

        if np.any(large_mask):
            u = (q_arr[large_mask] - self.weight) / (1.0 - self.weight)
            u = self.q_low + u * (self.q_high - self.q_low)
            out[large_mask] = self.base_dist.large_ppf(u)

        return float(out) if scalar else out

    def interval(self, *args, **kwargs):
        return self.base_dist.interval(*args, **kwargs)

    def rvs(self, *args, **kwargs):
        return self.base_dist.rvs(*args, **kwargs)

    def cdf(self, *args, **kwargs):
        return self.base_dist.cdf(*args, **kwargs)

    def pdf(self, *args, **kwargs):
        return self.base_dist.pdf(*args, **kwargs)


def make_generator_distribution(alpha, dist_weight,
                                q_low=0.05, q_high=0.95):
    base_dist = make_distribution(alpha, dist_weight)
    return ComponentAwarePPF(base_dist, q_low=q_low, q_high=q_high)
