"""Visualization utilities for the particle-radius distribution."""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spst

from config import NBINS
from mixed_distribution import (
    make_distribution,
    make_bimodal_generator_radii,
    make_generator_distribution,
)


def plot_mixed_distribution(
    alpha,
    dist_weight,
    r_max=None,
    n_points=1000,
    show_components=True,
):
    dist = make_distribution(alpha, dist_weight)

    r_lo = max(0.0, float(alpha["r_min"]))
    if r_max is None:
        q999 = float(dist.ppf(0.999))
        r_max = max(
            q999,
            float(alpha["loc_large"]) + 3.0 * float(alpha["scale_large"])
        )

    r = np.linspace(r_lo, r_max, n_points)
    pdf_mixed = dist.pdf(r)

    plt.figure(figsize=(8, 5))
    plt.plot(r, pdf_mixed, linewidth=2, label="Mixed distribution")

    if show_components:
        w = float(dist_weight)
        small_pdf = w * spst.expon.pdf(
            r, loc=alpha["loc_small"], scale=alpha["scale_small"]
        )

        large_scale = dist.effective_scale_large
        a = (alpha["r_min"] - alpha["loc_large"]) / large_scale
        Phi_a = spst.norm.cdf(a)
        large_pdf = spst.norm.pdf(
            (r - alpha["loc_large"]) / large_scale
        ) / large_scale
        large_pdf = np.where(
            r < alpha["r_min"], 0.0, large_pdf / (1.0 - Phi_a)
        )
        large_pdf *= (1.0 - w)

        plt.plot(r, small_pdf, "--", label="Small component")
        plt.plot(r, large_pdf, "--", label="Large component")

    plt.axvline(
        alpha["loc_small"], linestyle=":",
        label=f"loc_small={alpha['loc_small']:.2f}"
    )
    plt.axvline(
        alpha["loc_large"], linestyle=":",
        label=f"loc_large={alpha['loc_large']:.2f}"
    )

    gap = float(alpha["loc_large"]) - float(alpha["loc_small"])
    if dist.has_gap:
        plt.axvspan(
            alpha["loc_small"], alpha["loc_large"],
            alpha=0.12, label=f"gap={gap:.2f}"
        )

    plt.xlabel("Particle radius r [pixel]")
    plt.ylabel("Probability density")
    plt.title("MixedDistribution")
    plt.xlim(r_lo, r_max)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return dist


def plot_generator_histogram(
    alpha,
    dist_weight,
    nbins=NBINS,
    n_samples=100000,
    seed=0,
    show_probability=True,
    sampling_mode="separate",
):
    """Show the actual radius representation used by code11 or code12."""

    if int(nbins) < 2:
        raise ValueError("nbins must be >= 2")

    nbins = int(nbins)
    dist = make_distribution(alpha, dist_weight)

    if sampling_mode == "separate":
        radii, component, _ = make_bimodal_generator_radii(
            alpha=alpha,
            dist_weight=dist_weight,
            nbins=nbins,
        )
        title = "code11: separate component discretization"
        ylabel = "Probability density"
    elif sampling_mode == "mixture":
        generator_dist = make_generator_distribution(
            alpha=alpha,
            dist_weight=dist_weight,
        )
        q_grid = np.linspace(0.05, 0.95, nbins)
        radii = np.asarray(generator_dist.ppf(q_grid))
        component = np.where(
            q_grid < float(dist_weight), "small", "large"
        )
        title = "code12: continuous component-aware PPF"
        ylabel = "Probability density"
    else:
        raise ValueError(
            "sampling_mode must be 'separate' or 'mixture'"
        )

    rng = np.random.default_rng(seed)
    sample_radii = np.asarray(
        dist.rvs(size=n_samples, random_state=rng)
    )

    sample_edges = np.linspace(
        max(0.0, float(alpha["r_min"])),
        float(np.max(sample_radii)),
        80,
    )
    counts, edges = np.histogram(
        sample_radii, bins=sample_edges, density=True
    )
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        centers, counts, width=np.diff(edges),
        alpha=0.25, align="center",
        label="Continuous distribution (sampled)",
    )

    small_mask = component == "small"
    large_mask = component == "large"

    y_small = max(counts.max() * 0.035, 1e-6)
    y_large = max(counts.max() * 0.07, 1e-6)

    ax.scatter(
        radii[small_mask],
        np.full(np.sum(small_mask), y_small),
        marker="o", s=22,
        label=f"Small q-points ({np.sum(small_mask)})",
    )
    ax.scatter(
        radii[large_mask],
        np.full(np.sum(large_mask), y_large),
        marker="^", s=28,
        label=f"Large q-points ({np.sum(large_mask)})",
    )

    ax.axvline(
        alpha["loc_large"], linestyle=":",
        label="loc_large"
    )
    ax.set_xlabel("Particle radius r [pixel]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    plt.show()

    print("\n=== Generator radius representation ===")
    print(f"Mode = {sampling_mode}")
    print(f"NBINS = {nbins}")
    print(f"Small points = {np.sum(small_mask)}")
    print(f"Large points = {np.sum(large_mask)}")
    print("Generator radii:")
    print(np.array2string(radii, precision=3))
    print(f"Maximum generator radius = {np.max(radii):.3f}")

    return {
        "dist": dist,
        "generator_radii": radii,
        "component": component,
        "sample_radii": sample_radii,
        "sampling_mode": sampling_mode,
    }
