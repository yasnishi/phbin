import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage


def estimate_bimodal_parameters(target_sem, ratio=0.3, threshold=128,
                                min_area=20, show_result=False):
    """Estimate particle statistics and the target distribution gap.

    ``peak_gap`` is the distance between the two detected histogram peaks.
    If two peaks are not detected, ``peak_gap`` is set to zero, which sends
    the target to the code12 (continuous-mixture) branch. The sampling mode
    is then fixed before optimization and is not changed during optimization.
    """
    img = plt.imread(target_sem)
    if img.ndim == 3:
        img = np.mean(img[..., :3], axis=2)
    if img.max() <= 1.0:
        img = img * 255.0
    if ratio != 1.0:
        h, w = img.shape
        img = cv2.resize(img, (int(w * ratio), int(h * ratio)),
                         interpolation=cv2.INTER_AREA)

    binary = img < threshold
    binary = ndimage.binary_closing(ndimage.binary_opening(binary))
    labels, n = ndimage.label(binary)
    areas = np.bincount(labels.ravel())[1:]
    areas = areas[areas >= min_area]

    if len(areas) < 5:
        raise ValueError(f"Too few particles detected: {len(areas)}")

    radii = np.sqrt(areas / np.pi)

    hist, edges = np.histogram(radii, bins='auto')
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth = ndimage.gaussian_filter1d(hist.astype(float), 1.0)

    peaks = [
        i for i in range(1, len(smooth) - 1)
        if smooth[i] > smooth[i - 1]
        and smooth[i] >= smooth[i + 1]
    ]
    peaks.append(len(smooth)-1)

    peak_gap = 0.0
    peak_centers = []

    if len(peaks) >= 2:
        #peaks = sorted(peaks, key=lambda i: smooth[i], reverse=True)[:2]
        #p1, p2 = sorted(peaks)
        p1, p2 = np.min(peaks), np.max(peaks)
        peak_centers = [float(centers[p1]), float(centers[p2])]
        peak_gap = float(centers[p2] - centers[p1])
        split = float(centers[p1 + np.argmin(smooth[p1:p2 + 1])])
        two_peak_detected = True
    else:
        split = float(np.median(radii))
        two_peak_detected = False
        print('[Warning] Two clear peaks were not detected; '
              'peak_gap=0 -> code12 branch (sampling mode fixed).')

    small = radii <= split
    large = radii[~small]

    weight = float(np.sum(small) / len(radii))
    large_radii = radii[~small]

    if len(large_radii) == 0:
        raise ValueError("No large-particle component detected.")

    loc = float(np.mean(large_radii))
    scale = max(
        float(np.std(large_radii, ddof=1))
        if len(large_radii) > 1 else 1e-3,
        1e-3
    )
    rmax = float(np.max(radii))
    porosity = float(1 - binary.mean())

    if show_result:
        plt.figure(figsize=(7, 4))
        plt.hist(radii, bins='auto', alpha=.7)
        plt.axvline(split, ls='--', label=f'split={split:.2f}')
        plt.axvline(loc, ls='-', label=f'large mean={loc:.2f}')
        if peak_centers:
            for j, pc in enumerate(peak_centers):
                plt.axvline(
                    pc, ls=':',
                    label=f'peak {j+1}={pc:.2f}'
                )
        plt.xlabel('Equivalent radius [pixel]')
        plt.ylabel('Number')
        plt.legend()
        plt.tight_layout()
        plt.show()

    print(
        f'dist_weight={weight:.6f}, split={split:.6f}, '
        f'large_loc={loc:.6f}, large_scale={scale:.6f}, '
        f'r_max={rmax:.6f}, porosity={porosity:.6f}'
    )
    print(
        f'peak_gap={peak_gap:.6f}, '
        f'two_peak_detected={two_peak_detected}'
    )

    return dict(
        image=img,
        binary=binary,
        areas=areas,
        radii=radii,
        n_particles=len(radii),
        r_min=float(radii.min()),
        r_max=rmax,
        porosity=porosity,
        dist_weight=weight,
        split_radius=split,
        large_radii=large_radii,
        large_loc=loc,
        large_scale=scale,
        peak_gap=peak_gap,
        peak_centers=peak_centers,
        two_peak_detected=two_peak_detected,
    )
