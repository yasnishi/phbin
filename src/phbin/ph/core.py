"""Core image and persistent-homology utilities for phbin.

The functions in this module wrap selected HomCloud operations and provide
helpers for image preprocessing, persistence-diagram calculation, PI
vectorization, and visualization.
"""

# -------------------------------------------------------------
import cv2
import os
import shutil
import sys
import numpy as np
import homcloud.interface as hc
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix


def make_input_imgobj(img_file, th=128, focus_region=False):
    """Load a grayscale image and return it with a thresholded binary mask.

    Args:
        img_file: Path to the input image.
        th: Grayscale threshold in the range 0–255.
        focus_region: If True, select pixels greater than the threshold; otherwise select pixels below it.

    Returns:
        A tuple of (grayscale image, binary mask)."""
    img = load_image(img_file)
    if focus_region:
        bin_img = img > th
    else:
        bin_img = img < th
    return img, bin_img


def resize_img(img, ratio):
    """Resize an image by a multiplicative ratio.

    Args:
        img: Input image array.
        ratio: Width and height scaling factor.

    Returns:
        Resized image array."""
    width = int(img.shape[1] * ratio)
    height = int(img.shape[0] * ratio)
    return cv2.resize(img, (width, height))


def pd_liftime_filter(pd, phtree, th=10):
    """Return persistence-tree nodes associated with pairs above a lifetime threshold.

    Note: The historical public name contains the spelling “liftime”.
    Args:
        pd: HomCloud persistence diagram.
        phtree: Corresponding bitmap persistence-tree object.
        th: Strict minimum lifetime for selected pairs.
    """
    pairs = pd.pairs()
    pairs_birth_list = []
    pairs_death_list = []
    for i in range(len(pairs)):
        pairs_birth_list.append(float(pairs[i].birth))
        pairs_death_list.append(float(pairs[i].death))

    x2 = np.array(pairs_birth_list)
    y2 = np.array(pairs_death_list)
    z2 = y2 - x2

    long_lifitime_nodes = []
    for i in range(len(z2)):
        if z2[i] > th:
            long_lifitime_nodes = long_lifitime_nodes + phtree.pair_nodes_in_rectangle(
                x2[i], x2[i], y2[i], y2[i])

    return long_lifitime_nodes


def pd_filter(pd, phtree, area):
    """Map persistence-diagram pairs selected by an area object to tree nodes.

    Args:
        pd: HomCloud persistence diagram.
        phtree: Corresponding bitmap persistence-tree object.
        area: HomCloud area/filter object exposing ``filter_pairs``.
    """
    pairs_in_mask = area.filter_pairs(pd.pairs())
    pairs_list = []
    for i in range(len(pairs_in_mask)):
        pairs_list.append(
            (pairs_in_mask[i].birth_time(), pairs_in_mask[i].death_time()))

    nodes = []
    for i in range(len(pairs_list)):
        nodes = nodes + phtree.pair_nodes_in_rectangle(
            pairs_list[i][0], pairs_list[i][0], pairs_list[i][1],
            pairs_list[i][1])
        # temp_nodes = phtree.nearest_pair_node(pairs_list[i][0],
        #                                      pairs_list[i][1])

        # nodes.append(temp_nodes)

    return nodes


def reverse_map(nodes,
                sem_bin,
                color=(255, 0, 0),
                alpha=0.6,
                birth_poition=(255, 0, 0),
                marker_size=2,
                save_to='reverse.png'):
    """Draw selected persistence-tree nodes back onto a binary image and save it.

    Args:
        nodes: Nodes returned by the filtering helpers.
        sem_bin: Binary image array.
        color: Requested overlay color (legacy implementation currently uses a fixed red color).
        alpha: Requested overlay opacity (legacy implementation currently uses a fixed value).
        birth_poition: Legacy misspelled argument retained for compatibility.
        marker_size: Requested marker size (legacy implementation currently uses a fixed value).
        save_to: Output image path.
    """
    mapping = hc.draw_volumes_on_2d_image(nodes,
                                          (sem_bin * 255).astype(np.uint8),
                                          color=(255, 0, 0),
                                          alpha=0.6,
                                          birth_position=(255, 0, 0),
                                          marker_size=2)
    mapping.save(save_to)


def draw_pi(pi, mesh, save_to='pi.png', dpi=300):
    """Render a persistence image vector using its HomCloud mesh and save the figure.

    Args:
        pi: One PI vector, typically ``make_pi(...)["pi"][0]``.
        mesh: HomCloud PIVectorizerMesh used to create the vector.
        save_to: Output image path.
        dpi: Output resolution in dots per inch.
    """
    mesh.histogram_from_vector(pi).plot()
    plt.savefig(save_to, dpi=dpi)


def make_pi(pds,
            mesh_range=(-2, 3),
            bins=32,
            sigma=0.01,
            weight=("atan", 0.01, 3),
            save_to='desc.dill',
            superlevel=True):

    """Vectorize one or more persistence diagrams as persistence images.

    Args:
        pds: Sequence of HomCloud persistence diagrams.
        mesh_range: Lower and upper bounds for the PI mesh.
        bins: Number of mesh bins.
        sigma: Gaussian smoothing parameter passed to HomCloud.
        weight: Weight-function specification passed to HomCloud.
        save_to: Retained for API compatibility; this function returns data rather than saving it.
        superlevel: Whether to use superlevel-set convention.

    Returns:
        Dictionary with ``pi`` (a 2-D array, one vector per diagram) and ``mesh``.
    """
    mesh_range = mesh_range
    bins = bins
    sigma = sigma
    weight = weight
    save_to = save_to

    mesh = hc.PIVectorizerMesh(mesh_range,
                               bins,
                               sigma=sigma,
                               weight=weight,
                               superlevel=superlevel)

    # pdvects = np.vstack(
    #    [mesh.vectorize(pds[i]) for i in tqdm(range(len(pds)))])

    pdvects = np.vstack(
        [mesh.vectorize(pds[i]) for i in range(len(pds))])

    desc = pdvects / pdvects.max()

    return {'pi': desc, 'mesh': mesh}


# --- Persistence diagrams from point clouds ---
def make_pd_pc_alpha(pointcloud, dim=0):
    """Compute alpha-filtration persistence diagrams for a list of point clouds.

    Args:
        pointcloud: List of point-cloud arrays.
        dim: Homology dimension to return.

    Returns:
        List of persistence diagrams. Intermediate diagram files are written to ``./pds``.
    """
    if type(pointcloud) is list:
        pds = []

        dirpath = './pds'
        if os.path.exists(dirpath):
            shutil.rmtree('./pds')
        else:
            pass
        os.mkdir('./pds')

        # for i in tqdm(range(len(pointcloud))):
        for i in range(len(pointcloud)):
            pd = hc.PDList.from_alpha_filtration(pointcloud[i],
                                                 save_to='./pds/%s.pdgm' %
                                                 str(i),
                                                 save_boundary_map=True)
            pds.append(pd.dth_diagram(dim))
    else:
        print('pointcloud must be given list type')
        sys.exit(0)

    return pds


def make_pd_pc_rips(pointcloud, maxdim=2, dim=0):
    """Compute Rips-filtration persistence diagrams for a list of point clouds.

    Args:
        pointcloud: List of point-cloud arrays.
        maxdim: Maximum homology dimension computed by HomCloud.
        dim: Homology dimension to return.

    Returns:
        List of persistence diagrams. Intermediate diagram files are written to ``./pds``.
    """
    if type(pointcloud) is list:

        pds = []

        dirpath = './pds'
        if os.path.exists(dirpath):
            shutil.rmtree('./pds')
        else:
            pass
        os.mkdir('./pds')

        # for i in tqdm(range(len(pointcloud))):
        for i in range(len(pointcloud)):
            dmatrix = distance_matrix(pointcloud[i], pointcloud[i])
            pd = hc.PDList.from_rips_filtration(dmatrix,
                                                maxdim=maxdim,
                                                save_to='./pds/%s.pdgm' %
                                                str(i))
            pds.append(pd.dth_diagram(dim))

    else:
        print('pointcloud must be given list type')
        sys.exit(0)
    return pds


# --- Persistence diagrams from image data ---
def make_pd(np_img, mode='sublevel', dim=0):
    """Compute persistence diagrams from binary images using signed distance transforms.

    Args:
        np_img: List of image arrays.
        mode: HomCloud bitmap filtration mode (for example, ``sublevel``).
        dim: Homology dimension to return.

    Returns:
        List of persistence diagrams. Intermediate diagram files are written to ``./pds``.
    """
    if type(np_img) is list:
        pds = []

        dirpath = './pds'
        if os.path.exists(dirpath):
            shutil.rmtree('./pds')
        else:
            pass
        os.mkdir('./pds')

        # for i in tqdm(range(len(np_img))):
        for i in range(len(np_img)):
            pd = hc.PDList.from_bitmap_levelset(
                hc.distance_transform(np_img[i], signed=True),
                mode=mode,
                save_to='./pds/%s.pdgm' % str(i))
            pds.append(pd.dth_diagram(dim))

    else:
        print('pointcloud must be given list type')
        sys.exit(0)

    return pds


def make_pd_gray(np_img, mode='sublevel', dim=0):
    """Compute persistence diagrams directly from grayscale images.

    Args:
        np_img: List of grayscale image arrays.
        mode: HomCloud bitmap filtration mode.
        dim: Homology dimension to return.

    Returns:
        List of persistence diagrams. Intermediate diagram files are written to ``./pds``.
    """
    if type(np_img) is list:
        pds = []

        dirpath = './pds'
        if os.path.exists(dirpath):
            shutil.rmtree('./pds')
        else:
            pass
        os.mkdir('./pds')

        # for i in tqdm(range(len(np_img))):
        for i in range(len(np_img)):
            pd = hc.PDList.from_bitmap_levelset(np_img[i],
                                                mode=mode,
                                                save_to='./pds/%s.pdgm' %
                                                str(i))
            pds.append(pd.dth_diagram(dim))

    else:
        print('pointcloud must be given list type')
        sys.exit(0)

    return pds


def make_phtrees(np_img, mode='sublevel', dim=0):
    """Compute bitmap persistence trees for a list of binary images.

    Args:
        np_img: List of image arrays.
        mode: HomCloud bitmap filtration mode.
        dim: Homology dimension used to construct the returned trees.

    Returns:
        List of persistence-tree objects. Intermediate diagram files are written to ``./pds``.
    """
    if type(np_img) is list:
        phtrees = []

        dirpath = './pds'
        if os.path.exists(dirpath):
            shutil.rmtree('./pds')
        else:
            pass
        os.mkdir('./pds')

        # for i in tqdm(range(len(np_img))):
        for i in range(len(np_img)):
            pd = hc.BitmapPHTrees.for_bitmap_levelset(
                hc.distance_transform(np_img[i], signed=True),
                mode=mode,
                save_to='./pds/%s.pdgm' % str(i))
            phtrees.append(pd.bitmap_phtrees(dim))

    else:
        print('pointcloud must be given list type')
        sys.exit(0)

    return phtrees


def show_image(img_file):
    """Display an image file as a grayscale image."""
    img = cv2.imread(img_file, 0)
    plt.imshow(img, cmap='gray')
    plt.show()


def show_pixel_dist(img_file):
    """Plot the grayscale pixel-value histogram for an image file."""
    img = cv2.imread(img_file, 0)
    img_sizex = img.shape[1]
    img_sizey = img.shape[0]
    img_data = []
    for j in range(img_sizey):
        img_data.append(img[j][0:img_sizex])
        img_np = np.array(img_data)
    plt.hist(np.ravel(img_np), range=(0, 256), bins=256)
    None


def binarization(img, th=128):
    # img = cv2.imread(img_file, 0)

    """Threshold an image array into a boolean mask.

    Pixels strictly greater than ``th`` are True.
    """
    binary_img = np.array(img) > th
    # binary_img = binary_img.astype(np.int) * 255

    return binary_img


def load_image(img_file):
    """Read an image file as an 8-bit grayscale NumPy array.

    Returns None if OpenCV cannot read the given path.
    """
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    return img


def write_image(img, save_to):
    """Save an image array to a file using Matplotlib, without axes."""
    plt.imshow(img, cmap='gray')
    plt.axis("off")
    plt.savefig(save_to, bbox_inches='tight', pad_inches=0)


# --- Visualization helpers ---
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
    """Draw a persistence-diagram density histogram and save it.

    Args:
        pd: HomCloud persistence diagram.
        x_range: Optional birth-axis range.
        x_bins: Number of bins on the birth axis.
        y_range: Optional death-axis range.
        y_bins: Number of bins on the death axis.
        dpi: Output resolution.
        colorbar: HomCloud histogram colorbar scaling (for example, ``log``).
        title: Plot title.
        save_to: Output image path.
        cmap: Matplotlib colormap name for the histogram (for example, ``magma``).
    """
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


def draw_pd(pd,
            x_range=None,
            x_bins=None,
            y_range=None,
            y_bins=None,
            dpi=300,
            colorbar='log',
            title='title',
            save_to='pd.png'):

    # extract pairs
    """Draw a persistence-diagram density histogram and save it."""
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

    if x_range == None:
        # x_range = (np.min([x_min, y_min]) * 1.5, np.max([x_max, y_max]) * 1.5)
        # y_range = (np.min([x_min, y_min]) * 1.5, np.max([x_max, y_max]) * 1.5)
        if abs(x_max - x_min) < 0.1:
            if abs(y_max - y_min) < 0.1:
                x_range = (x_min - 1.0, x_max + 1.0)
                y_range = (y_min - 1.0, y_max + 1.0)
            else:
                x_range = (x_min - 1.0, x_max + 1.0)
                y_range = (y_min, y_max)
        elif abs(y_max - y_min) < 0.1:
            x_range = (x_min - 0.5 * abs(x_min), x_max + 0.5 * abs(x_max))
            y_range = (y_min - 1.0, y_max + 1.0)
        else:
            x_range = (x_min - 0.5 * abs(x_min), x_max + 0.5 * abs(x_max))
            y_range = (y_min - 0.5 * abs(y_min), y_max + 0.5 * abs(y_max))
    else:
        x_range = x_range
        y_range = y_range

    if x_bins == None:
        bins = np.max([x_max, y_max]) - np.min([x_min, y_min])
        delta_bins = bins / 50
        x_bins = int(bins / delta_bins)
        y_bins = int(bins / delta_bins)
    else:
        x_bins = x_bins
        y_bins = y_bins

    histogram = hc.HistoSpec(x_range, x_bins, y_range, y_bins).pd_histogram(pd)
    histogram.plot(colorbar={"type": colorbar, "midpoint": 0}, title=title)
    plt.savefig(save_to, dpi=dpi)  # , bbox_inches='tight', pad_inches=0)


def draw_pd_lifetime(pd,
                     x_range=None,
                     x_bins=None,
                     y_range=None,
                     y_bins=None,
                     dpi=300,
                     colorbar='log',
                     title='title',
                     save_to='pd_lifetime.png'):

    # extract pairs
    """Plot birth, death, and lifetime in a 3-D scatter plot.

    The plot is displayed and also saved to ``save_to``.
    """
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

    x2 = np.array(pairs_birth_list)
    y2 = np.array(pairs_death_list)
    z2 = y2 - x2

    if x_range == None:
        # x_range = (np.min([x_min, y_min]) * 1.5, np.max([x_max, y_max]) * 1.5)
        # y_range = (np.min([x_min, y_min]) * 1.5, np.max([x_max, y_max]) * 1.5)
        if abs(x_max - x_min) < 0.1:
            if abs(y_max - y_min) < 0.1:
                x_range = (x_min - 1.0, x_max + 1.0)
                y_range = (y_min - 1.0, y_max + 1.0)
            else:
                x_range = (x_min - 1.0, x_max + 1.0)
                y_range = (y_min, y_max)
        elif abs(y_max - y_min) < 0.1:
            x_range = (x_min - 0.5 * abs(x_min), x_max + 0.5 * abs(x_max))
            y_range = (y_min - 1.0, y_max + 1.0)
        else:
            x_range = (x_min - 0.5 * abs(x_min), x_max + 0.5 * abs(x_max))
            y_range = (y_min - 0.5 * abs(y_min), y_max + 0.5 * abs(y_max))
    else:
        x_range = x_range
        y_range = y_range

    if x_bins == None:
        bins = np.max([x_max, y_max]) - np.min([x_min, y_min])
        delta_bins = bins / 50
        x_bins = int(bins / delta_bins)
        y_bins = int(bins / delta_bins)
    else:
        x_bins = x_bins
        y_bins = y_bins

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('birth', size=14)
    ax.set_ylabel('death', size=14)
    ax.set_zlabel('lifetime', size=14)
    cm = plt.cm.get_cmap('RdYlBu_r')
    ax.scatter(x2, y2, z2, c=z2, cmap=cm, s=40)

    max_range = np.array(
        [x2.max() - x2.min(),
         y2.max() - y2.min(),
         z2.max() - z2.min()]).max() * 0.5

    mid_x = (x2.max() + x2.min()) * 0.5
    mid_y = (y2.max() + y2.min()) * 0.5
    mid_z = (z2.max() + z2.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()
    plt.savefig(save_to, dpi=dpi)  # , bbox_inches='tight', pad_inches=0)
