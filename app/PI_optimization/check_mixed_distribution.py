# %% [1] Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spst

from config import GAP_THRESHOLD, LOC_SMALL
from mixed_distribution import make_distribution

# %% [2] Control parameters
# Edit these values to inspect the mixed distribution.
alpha = {
    "loc_small": LOC_SMALL,
    "scale_small": 3.0,
    "loc_large": 42.0,
    "scale_large": 3.0,
    "r_min": 0.5,
}

# Fraction of small particles.
# Replace by the value estimated from a target SEM when desired.
dist_weight = 0.90

R_MAX_PLOT = 55.0
N_POINTS = 1000

# %% [3] Create MixedDistribution
dist = make_distribution(alpha, dist_weight)

gap = alpha["loc_large"] - alpha["loc_small"]
print("=== Mixed distribution ===")
for name in ["loc_small", "scale_small", "loc_large", "scale_large", "r_min"]:
    print(f"{name:12s} = {alpha[name]:.4f}")
print(f"dist_weight     = {dist_weight:.4f}")
print(f"gap             = {gap:.4f}")
print(f"GAP_THRESHOLD   = {GAP_THRESHOLD:.4f}")
print(f"has_gap         = {dist.has_gap}")
print(f"effective_scale_large = {dist.effective_scale_large:.4f}")

# %% [4] Evaluate the distribution actually used by the generator
r = np.linspace(max(0.0, alpha["r_min"]), R_MAX_PLOT, N_POINTS)
pdf_mixed = dist.pdf(r)

# Component PDFs, for visual interpretation.
small_pdf = spst.expon.pdf(
    r, loc=alpha["loc_small"], scale=alpha["scale_small"]
) * dist_weight

large_scale = dist.effective_scale_large
a = (alpha["r_min"] - alpha["loc_large"]) / large_scale
Phi_a = spst.norm.cdf(a)
large_pdf = spst.norm.pdf(
    (r - alpha["loc_large"]) / large_scale
) / large_scale
large_pdf = np.where(
    r < alpha["r_min"], 0.0, large_pdf / (1.0 - Phi_a)
)
large_pdf *= (1.0 - dist_weight)

# %% [5] Plot
plt.figure(figsize=(8, 5))
plt.plot(r, pdf_mixed, label="Mixed distribution", linewidth=2)
plt.plot(r, small_pdf, "--", label="Small component")
plt.plot(r, large_pdf, "--", label="Large component")

plt.axvline(alpha["loc_small"], linestyle=":",
            label=f"loc_small={alpha['loc_small']:.2f}")
plt.axvline(alpha["loc_large"], linestyle=":",
            label=f"loc_large={alpha['loc_large']:.2f}")

if dist.has_gap:
    plt.axvspan(alpha["loc_small"], alpha["loc_large"], alpha=0.12,
                label=f"gap={gap:.2f}")

plt.xlabel("Particle radius r [pixel]")
plt.ylabel("Probability density")
plt.title(
    "MixedDistribution | "
    + ("separated bimodal" if dist.has_gap else "continuous bimodal")
)
plt.xlim(0, R_MAX_PLOT)
plt.grid(alpha=0.2)
plt.legend()
plt.tight_layout()
plt.show()

# %% [6] Optional: obtain controls directly from a target SEM
# Uncomment and edit TARGET_SEM when desired.
#
# from particle_analysis import estimate_bimodal_parameters
#
# TARGET_SEM = "./64.jpg"
# sem = estimate_bimodal_parameters(
#     TARGET_SEM,
#     ratio=0.3,
#     threshold=128,
#     min_area=20,
#     show_result=True,
# )
#
# alpha["loc_small"] = LOC_SMALL
# alpha["loc_large"] = sem["large_loc"]
# alpha["scale_large"] = sem["large_scale"]
# dist_weight = sem["dist_weight"]
#
# print(sem)
