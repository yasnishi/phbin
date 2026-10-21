import numpy as np
import phbin
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
import homcloud.interface as hc


# Estimation of porosity from generated images
def estimate_porosity(binary_data):
    area = 0
    white_region = 0
    for i in range(len(binary_data)):
        for j in range(len(binary_data[i])):
            if binary_data[i][j]:
                white_region = white_region + 1
            area = area + 1

    print('porosity =', white_region/area)


import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh

# customization of plotting PI
def draw_pi_color(pi, mesh, cmap='magma', save_to='pi.png', dpi=300):
    hist = mesh.histogram_from_vector(pi)
    hist.plot()

    # カラーマップを指定
    #cmap_name = "magma"  # 例: "viridis", "plasma", "Greens", "cividis", "magma"

    # 図中のAxesを走査し、PI本体の色を変更
    fig = plt.gcf()

    for ax in fig.axes:
        for obj in ax.collections:
            if isinstance(obj, QuadMesh):
                obj.set_cmap(cmap)
                obj.changed()
                
    plt.savefig(save_to, dpi=dpi)

# customization of plotting PD
def draw_pd_color(pd,
                x_range=None,
                x_bins=None,
                y_range=None,
                y_bins=None,
                dpi=300,
                colorbar='log',
                title='title',
                save_to='pd.png',
                cmap='magma'):

    # extract pairs
    pairs = pd.pairs()
    pairs_birth_list = []
    pairs_death_list = []

    for i in range(len(pairs)):
        pairs_birth_list.append(float(pairs[i].birth))
        pairs_death_list.append(float(pairs[i].death))

    x_max = np.max(pairs_birth_list)
    x_min = np.min(pairs_birth_list)
    y_max = np.max(pairs_death_list)
    y_min = np.min(pairs_death_list)

    if x_range is None:
        if abs(x_max - x_min) < 0.1:
            if abs(y_max - y_min) < 0.1:
                x_range = (x_min - 1.0, x_max + 1.0)
                y_range = (y_min - 1.0, y_max + 1.0)
            else:
                x_range = (x_min - 1.0, x_max + 1.0)
                y_range = (y_min, y_max)
        elif abs(y_max - y_min) < 0.1:
            x_range = (
                x_min - 0.5 * abs(x_min),
                x_max + 0.5 * abs(x_max)
            )
            y_range = (y_min - 1.0, y_max + 1.0)
        else:
            x_range = (
                x_min - 0.5 * abs(x_min),
                x_max + 0.5 * abs(x_max)
            )
            y_range = (
                y_min - 0.5 * abs(y_min),
                y_max + 0.5 * abs(y_max)
            )

    if x_bins is None:
        bins = np.max([x_max, y_max]) - np.min([x_min, y_min])
        delta_bins = bins / 50
        x_bins = int(bins / delta_bins)
        y_bins = int(bins / delta_bins)

    histogram = hc.HistoSpec(
        x_range, x_bins, y_range, y_bins
    ).pd_histogram(pd)

    histogram.plot(
        colorbar={"type": colorbar, "midpoint": 0},
        title=title
    )

    # Change the colormap of the PD plot
    from matplotlib.collections import QuadMesh

    fig = plt.gcf()
    for ax in fig.axes:
        for obj in ax.collections:
            if isinstance(obj, QuadMesh):
                obj.set_cmap(cmap)
                obj.changed()

    plt.savefig(save_to, dpi=dpi)