"""Optimal-volume reverse analysis for Persistent Homology.

This module follows the reference workflow in ``large_radi_check.ipynb``:

    make_phtrees -> pair_nodes_in_rectangle -> reverse_map

The selected rectangle is specified directly in Persistent Diagram
birth/death coordinates.  The reverse map is performed on the RATIO-scaled
binary SEM image supplied by ``particle_analysis.estimate_bimodal_parameters``.
"""

import phbin
from config import DIM

def reverse_map_pi_region(binary_image, b_range, d_range, dim=DIM):
    """Reverse-map a birth/death rectangle to its optimal volume.

    Parameters
    ----------
    binary_image : numpy.ndarray
        RATIO-scaled binary SEM image, e.g. ``sem['binary']``.
        This is deliberately used exactly as in the reference notebook.
    b_range : tuple(float, float)
        Persistent Diagram birth-coordinate interval ``(bmin, bmax)``.
    d_range : tuple(float, float)
        Persistent Diagram death-coordinate interval ``(dmin, dmax)``.
    dim : int, optional
        Homology dimension passed to ``phbin.make_phtrees``.

    Returns
    -------
    nodes
        The PH tree pair nodes selected by the requested rectangle.

    Notes
    -----
    ``phbin.reverse_map`` is responsible for the optimal-volume reverse
    mapping/visualization.  No independent particle-occlusion attribution is
    performed here.
    """
    if len(b_range) != 2 or len(d_range) != 2:
        raise ValueError("b_range and d_range must each contain two values.")

    bmin, bmax = map(float, b_range)
    dmin, dmax = map(float, d_range)

    if bmin > bmax:
        raise ValueError("b_range must satisfy bmin <= bmax.")
    if dmin > dmax:
        raise ValueError("d_range must satisfy dmin <= dmax.")

    phtrees = phbin.make_phtrees(
        [binary_image],
        mode="sublevel",
        dim=dim,
    )

    nodes = phtrees[0].pair_nodes_in_rectangle(
        bmin, bmax, dmin, dmax
    )

    if len(nodes) == 0:
        raise ValueError(
            "No PH pair nodes were found in the selected birth/death rectangle: "
            f"b_range={tuple(b_range)}, d_range={tuple(d_range)}"
        )

    # Follow the reference notebook exactly: reverse_map receives the
    # complement of the RATIO-scaled binary image.
    reverse_result = phbin.reverse_map(
        nodes,
        ~binary_image.astype(bool),
    )

    return nodes, reverse_result
