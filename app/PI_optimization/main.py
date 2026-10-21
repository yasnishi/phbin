import numpy as np

from config import BOUNDS, RATIO, LOC_SMALL, NBINS, GAP_THRESHOLD
from optimization import make_bounds
from particle_analysis import estimate_bimodal_parameters
from real_sem import prepare_real_sem
from optimization import optimize_parameters
from optimized_comparison import compare_optimized_to_sem
from distribution_visualization import (
    plot_mixed_distribution,
    plot_generator_histogram,
)
from inverse_optimal_volume import reverse_map_pi_region


# ============================================================
# Settings
# ============================================================

TARGET_SEM = "./64.jpg"

RATIO = 0.3
L_LAMBDA = 0.8

# ============================================================
# Optional PH optimal-volume reverse analysis
# ============================================================
# Set True to run the reverse analysis after the normal code15 workflow.
# The ranges below are Persistent Diagram birth/death coordinates, not
# PI pixel indices.  They follow the pair_nodes_in_rectangle() convention
# used in the reference large_radi_check.ipynb.
RUN_OPTIMAL_VOLUME_REVERSE = False
REVERSE_B_RANGE = (-60.0, -40.0)
REVERSE_D_RANGE = (-60.0, -40.0)
REVERSE_HOMOLOGY_DIM = 0

# ============================================================
# Main
# ============================================================


def main():

    # --------------------------------------------------------
    # 1. Estimate particle-size distribution from real SEM
    # --------------------------------------------------------

    print("\n======================================")
    print("1. SEM particle-size analysis")
    print("======================================")

    sem = estimate_bimodal_parameters(
        TARGET_SEM,
        ratio=RATIO,
        threshold=128,
        min_area=20,
        show_result=False,
    )

    print("\nEstimated parameters:")
    print(f"  dist_weight = {sem['dist_weight']:.6f}")
    print(f"  split_radius = {sem['split_radius']:.6f}")
    print(f"  large_loc = {sem['large_loc']:.6f}")
    print(f"  large_scale = {sem['large_scale']:.6f}")
    print(f"  r_max = {sem['r_max']:.6f}")
    print(f"  porosity = {sem['porosity']:.6f}")
    print(f"  peak_gap = {sem['peak_gap']:.6f}")
    print(f"  GAP_THRESHOLD = {GAP_THRESHOLD:.6f}")

    # --------------------------------------------------------
    # 1a. Adaptive code11/code12 selection
    # --------------------------------------------------------
    # The decision is made only from the target SEM and remains fixed
    # during optimization. This avoids a discontinuous objective.
    if sem['peak_gap'] >= GAP_THRESHOLD:
        sampling_mode = 'separate'
    else:
        sampling_mode = 'mixture'

    print(f"  sampling_mode = {sampling_mode}")

    # --------------------------------------------------------
    # 1b. Check the particle-radius distribution used by the generator
    # --------------------------------------------------------

    # Use the same control parameters as the current initial generator
    # setup.  This lets us inspect both the continuous MixedDistribution
    # and the discrete radius sizes implied by PoreSpy's NBINS setting
    # before starting the expensive PH optimization.
    alpha_ini = {
        "loc_small": float(LOC_SMALL),
        "scale_small": float((BOUNDS[1][0] + BOUNDS[1][1]) / 2),
        "loc_large": float(sem["large_loc"]),
        "scale_large": float(sem["large_scale"]),
        "r_min": float((BOUNDS[5][0] + BOUNDS[5][1]) / 2),
    }

    print("\n======================================")
    print("1b. Generator radius-distribution check")
    print("======================================")
    print("Control parameters:")
    for key, value in alpha_ini.items():
        print(f"  {key:12s} = {value:.6f}")
    print(f"  dist_weight  = {sem['dist_weight']:.6f}")
    print(f"  NBINS        = {NBINS}")

    # plot_mixed_distribution(
    #    alpha=alpha_ini,
    #    dist_weight=sem["dist_weight"],
    # )
    #
    # plot_generator_histogram(
    #    alpha=alpha_ini,
    #    dist_weight=sem["dist_weight"],
    #    nbins=NBINS,
    #    n_samples=1000, #100000,
    #    seed=0,
    #    sampling_mode=sampling_mode,
    # )

    # --------------------------------------------------------
    # 2. Real SEM -> PD -> PI
    # --------------------------------------------------------

    print("\n======================================")
    print("2. Real SEM -> PD -> PI")
    print("======================================")

    sem_binary, pd_real, pi_real, mesh_range = \
        prepare_real_sem(
            TARGET_SEM,
            RATIO,
        )

    print("Real image shape :", sem_binary.shape)
    print("PI mesh range    :", mesh_range)

    # --------------------------------------------------------
    # 3. Initial parameters for optimization
    # --------------------------------------------------------

    # Target-derived bounds for the large-particle distribution.
    # The number of optimized parameters remains exactly the same as jeni_code6.
    bounds = make_bounds(sem)

    x_ini = np.array([
        # loc_small
        (bounds[0][0] + bounds[0][1]) / 2,

        # scale_small
        (bounds[1][0] + bounds[1][1]) / 2,

        # loc_large: target SEM estimate
        sem["large_loc"],

        # scale_large: target SEM estimate
        sem["large_scale"],

        # porosity
        (bounds[4][0] + bounds[4][1]) / 2,

        # r_min
        (bounds[5][0] + bounds[5][1]) / 2,
    ])

    print("\n======================================")
    print("3. Initial parameters")
    print("======================================")

    print("x_ini =", x_ini)
    print("\nTarget-derived bounds:")
    for name, b in zip(["loc_small", "scale_small", "loc_large", "scale_large", "porosity", "r_min"], bounds):
        print(f"  {name:12s} = {b}")

    # --------------------------------------------------------
    # 4. Powell optimization
    # --------------------------------------------------------

    print("\n======================================")
    print("4. Powell optimization")
    print("======================================")

    result = optimize_parameters(
        pi_real=pi_real,
        image_shape=sem_binary.shape,
        mesh_range=mesh_range,
        dist_weight=sem["dist_weight"],
        x_ini=x_ini,
        bounds=bounds,
        L_lambda=L_LAMBDA,
        sampling_mode=sampling_mode,
        #loss_name='global_hybrid'
    )

    # --------------------------------------------------------
    # 5. Optimization result
    # --------------------------------------------------------

    print("\n======================================")
    print("5. Optimization result")
    print("======================================")

    parameter_names = [
        "loc_small",
        "scale_small",
        "loc_large",
        "scale_large",
        "porosity",
        "r_min",
    ]

    for name, value in zip(parameter_names, result.x):
        print(f"  {name:12s} = {value:.6f}")
    print("optimized parameters", result.x)
    print(f"\n  PI loss = {result.fun:.8f}")
    print(f"  success = {result.success}")
    print(f"  nfev    = {result.nfev}")

    # --------------------------------------------------------
    # 6. Optimized parameters -> image -> PD / PI comparison
    # --------------------------------------------------------

    print("\n======================================")
    print("6. Optimized image vs real SEM")
    print("======================================")

    comparison = compare_optimized_to_sem(
        sem_binary=sem_binary,
        mesh_range=mesh_range,
        optimized_params=result.x,
        dist_weight=sem["dist_weight"],
        sampling_mode=sampling_mode,
        show=False,
    )

    # --------------------------------------------------------
    # 7. Final comparison metrics
    # --------------------------------------------------------

    print("\n======================================")
    print("7. Final comparison")
    print("======================================")

    print(
        f"PI MAE  = {comparison['pi_mae']:.8f}"
    )

    print(
        f"PI RMSE = {comparison['pi_rmse']:.8f}"
    )

    # --------------------------------------------------------
    # 8. PH optimal-volume reverse analysis (optional)
    # --------------------------------------------------------
    # This follows the reference notebook:
    #   make_phtrees -> pair_nodes_in_rectangle -> reverse_map
    # The analysis is performed on sem['binary'], which is already resized
    # by RATIO in estimate_bimodal_parameters().
    if RUN_OPTIMAL_VOLUME_REVERSE:
        print("\n======================================")
        print("8. PH optimal-volume reverse analysis")
        print("======================================")
        print(f"birth range = {REVERSE_B_RANGE}")
        print(f"death range = {REVERSE_D_RANGE}")
        print(f"homology dim = {REVERSE_HOMOLOGY_DIM}")
        print("image shape =", sem["binary"].shape)
        print("Using RATIO-scaled sem['binary'] and phbin.reverse_map().")

        nodes, reverse_result = reverse_map_pi_region(
            binary_image=sem["binary"],
            b_range=REVERSE_B_RANGE,
            d_range=REVERSE_D_RANGE,
            dim=REVERSE_HOMOLOGY_DIM,
        )

        print(f"selected PH pair nodes = {len(nodes)}")
        print("Optimal-volume reverse mapping completed.")
        print("The reverse visualization/output is generated by phbin.reverse_map().")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
