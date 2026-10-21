import numpy as np
import matplotlib.pyplot as plt

import phbin
from generator_porespy import generate_porespy_image
from ph_analysis import image_to_pd_pi
from config import PI_BINS, PI_SIGMA, PI_WEIGHT, PI_SUPERLEVEL


def _make_pi(pd, mesh_range):
    """Create a Persistent Image using the same settings as PI optimization."""
    return phbin.make_pi(
        pd,
        mesh_range=mesh_range,
        bins=PI_BINS,
        sigma=PI_SIGMA,
        weight=PI_WEIGHT,
        superlevel=PI_SUPERLEVEL,
    )


def _get_pi_array(piimage):
    """Extract the PI vector from a phbin PI dictionary."""
    if not isinstance(piimage, dict):
        raise TypeError(
            "Expected a dictionary returned by phbin.make_pi()."
        )

    if "pi" not in piimage:
        raise KeyError(
            "The PI dictionary does not contain the 'pi' key."
        )

    arr = np.asarray(piimage["pi"])

    # A single PD may be returned as a 2D array with one row.
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 1:
        raise ValueError(
            f"Expected a 1D PI vector, but got shape {arr.shape}."
        )

    return arr

def compare_optimized_to_sem(
    sem_binary,
    mesh_range,
    optimized_params,
    dist_weight,
    sampling_mode="separate",
    show=True,
):
    """Compare a target SEM image with an optimized generated image.

    PI construction uses PI_BINS, PI_SIGMA, PI_WEIGHT, and PI_SUPERLEVEL
    from config.py, matching the settings used during optimization.
    """
    names = [
        "loc_small", "scale_small", "loc_large",
        "scale_large", "porosity", "r_min",
    ]
    alpha = dict(zip(names, optimized_params))

    generated = generate_porespy_image(
        sem_binary.shape,
        alpha,
        dist_weight,
        sampling_mode=sampling_mode,
    )

    # Keep the existing PI arrays used for numerical comparison.
    pd_sem, pi_sem, _ = image_to_pd_pi(
        sem_binary,
        mesh_range,
        bins=PI_BINS,
        sigma=PI_SIGMA,
        weight=PI_WEIGHT,
        superlevel=PI_SUPERLEVEL,
    )
    pd_gen, pi_gen, _ = image_to_pd_pi(
        generated,
        mesh_range,
        bins=PI_BINS,
        sigma=PI_SIGMA,
        weight=PI_WEIGHT,
        superlevel=PI_SUPERLEVEL,
    )

    mae = float(np.mean(np.abs(pi_sem - pi_gen)))
    rmse = float(np.sqrt(np.mean((pi_sem - pi_gen) ** 2)))

    # Build PI images directly from the PDs, with the same settings as
    # the optimization. pd_sem/pd_gen are expected to contain the H0 PD at [0].
    piimage_sem = _make_pi(pd_sem, mesh_range)
    piimage_gen = _make_pi(pd_gen, mesh_range)
    pi_array_sem = _get_pi_array(piimage_sem)
    pi_array_gen = _get_pi_array(piimage_gen)

    if show:
        fig, ax = plt.subplots(2, 3, figsize=(15, 9))

        ax[0, 0].imshow(sem_binary, cmap="gray")
        ax[0, 0].set_title("Real SEM")
        ax[0, 0].axis("off")

        ax[0, 1].imshow(generated, cmap="gray")
        ax[0, 1].set_title("Generated")
        ax[0, 1].axis("off")

        ax[0, 2].scatter(
            pd_sem[0].births, pd_sem[0].deaths,
            s=8, alpha=0.5, label="SEM",
        )
        ax[0, 2].scatter(
            pd_gen[0].births, pd_gen[0].deaths,
            s=8, alpha=0.5, label="Generated",
        )
        ax[0, 2].set_xlabel("Birth")
        ax[0, 2].set_ylabel("Death")
        ax[0, 2].legend()
        ax[0, 2].set_title("Persistence Diagram")

        # Do not reshape the packed PI vector: its length may encode a
        # triangular mesh (e.g. 465 = 30*31/2), not a rectangular image.
        # Show the PI using phbin's native plotting helper and its own mesh.
        # This preserves the library's PI coordinate convention.
        plt.tight_layout()
        plt.show()

        for label, piimage in (
            ("Real SEM: Persistent Image", piimage_sem),
            ("Generated: Persistent Image", piimage_gen),
        ):
            if not isinstance(piimage, dict) or "mesh" not in piimage:
                raise KeyError(
                    "phbin.make_pi() output must include 'mesh' for native PI plotting."
                )
            # phbin's drawing helper handles the packed PI vector and mesh.
            phbin.draw_pi_color(
                np.asarray(piimage["pi"]).reshape(-1),
                piimage["mesh"],
                cmap="viridis",
                save_to=None,
            )

        plt.tight_layout()

    return dict(
        pd_sem=pd_sem,
        pd_generated=pd_gen,
        pi_sem=pi_sem,
        pi_generated=pi_gen,
        piimage_sem=piimage_sem,
        piimage_generated=piimage_gen,
        #pi_image_array_sem=pi_array_sem,
        #pi_image_array_generated=pi_array_gen,
        pi_mae=mae,
        pi_rmse=rmse,
        generated_binary=generated,
        optimized_params=optimized_params,
    )
