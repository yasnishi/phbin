PI_BINS = 30
PI_SIGMA = 1.0
PI_WEIGHT = ('atan', 0.0001, 3)
PI_SUPERLEVEL = False
MESH_MARGIN = 5.0
NBINS = 10
SEED = 0
RATIO = 0.3
DIM = 1

# Fixed location parameter for the small-particle component.
LOC_SMALL = 0.5

# Target-SEM criterion for selecting the sampling strategy.
# If the distance between the two detected particle-size histogram peaks
# is >= GAP_THRESHOLD, use code11 (separate component discretization).
# Otherwise use code12 (continuous component-aware PPF).
GAP_THRESHOLD = 20.0

# In the separated-bimodal regime, the large-particle peak is centered
# at the maximum observed large-particle radius, and its optimized width
# is constrained to at most half of the detected gap.
GAP_SCALE_FACTOR = 0.1

# Bounds for parameters that are not target-derived.
# loc_large / scale_large are determined dynamically from the target SEM.
BOUNDS = [(0.01, 2.0), (1.0, 5.0), None,
          None, (0.4, 0.9), (0.1, 2.0)]
