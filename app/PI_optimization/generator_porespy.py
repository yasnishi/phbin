import porespy as ps

from config import NBINS
from mixed_distribution import (
    make_distribution,
    make_bimodal_generator_radii,
    BimodalGeneratorDistribution,
    make_generator_distribution,
)


def generate_porespy_image(shape, alpha, dist_weight, sampling_mode="separate"):
    """Generate a PoreSpy image using code11 or code12 sampling.

    sampling_mode='separate' -> code11
    sampling_mode='mixture'  -> code12

    The mode is selected once from the target SEM and is kept fixed during
    Powell optimization. This prevents the objective function from changing
    discontinuously as optimization parameters move.
    """
    if sampling_mode not in ("separate", "mixture"):
        raise ValueError("sampling_mode must be 'separate' or 'mixture'")

    if sampling_mode == "separate":
        base_dist = make_distribution(alpha, dist_weight)
        radii, component, _ = make_bimodal_generator_radii(
            alpha=alpha,
            dist_weight=dist_weight,
            nbins=NBINS,
        )
        generator_dist = BimodalGeneratorDistribution(base_dist, radii)
    else:
        generator_dist = make_generator_distribution(
            alpha=alpha,
            dist_weight=dist_weight,
        )

    return ps.generators.polydisperse_spheres(
        shape=shape,
        porosity=alpha['porosity'],
        dist=generator_dist,
        nbins=NBINS,
        r_min=alpha['r_min'],
    )
