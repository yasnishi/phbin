# phbin

`phbin` is a small Python utility library for image preprocessing and persistent-homology workflows. It wraps selected [HomCloud](https://homcloud.dev/) operations for persistence-diagram (PD) calculation, persistence-image (PI) vectorization, and visualization.

> **Status:** early-stage research software. APIs and dependency compatibility may change. Please validate results for your application.

## Features

- Load, resize, threshold, and display grayscale images.
- Compute persistence diagrams from image arrays and point clouds.
- Build bitmap persistence trees and map selected nodes back to images.
- Vectorize persistence diagrams into persistence images.
- Draw persistence diagrams, lifetime plots, and persistence images.

## Requirements

Python 3.9 or later. The package depends on HomCloud and scientific Python packages; installation of all dependencies can take time and may require platform-specific system libraries. Consult the [HomCloud installation documentation](https://homcloud.dev/) if its installation fails.

## Install from a local checkout

```bash
git clone https://github.com/pearcandy/phbin.git
cd phbin
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
```

For development, including test tooling:

```bash
python -m pip install -e ".[dev]"
```

## Quick start

```python
import phbin

# Load a grayscale image and create a binary mask.
gray, binary = phbin.make_input_imgobj("samples/sem0.png", th=128)

# Compute H0 persistence diagrams from a list of binary images.
pds = phbin.make_pd([binary], mode="sublevel", dim=0)

# Create a persistence-image vector. The returned `pi` is a 2-D array
# with one row per input persistence diagram.
pi_data = phbin.make_pi(
    pds, mesh_range=(-2, 3), bins=32, sigma=0.01,
    weight=("atan", 0.01, 3), superlevel=True,
)
pi_vector = pi_data["pi"][0]
mesh = pi_data["mesh"]
phbin.draw_pi(pi_vector, mesh, save_to="pi.png", dpi=300)
```

The filtration mode, threshold direction, homology dimension, and PI parameters affect the result. Choose them according to the image semantics and keep PI mesh parameters consistent when comparing samples.

## Tutorials and examples

The `samples/` directory contains example images and Jupyter notebooks. Some notebooks use optional packages or datasets beyond the core library dependencies; run them in a separate research environment.

## Notes and current limitations

- Several functions expect a **list** of arrays/point clouds, even for one sample.
- Diagram/tree generation writes intermediate files to a `./pds` directory in the current working directory; repeated calls replace that directory. Run these functions from a dedicated working directory and back up any existing `pds` data first.
- The historical function name `pd_liftime_filter` is retained for compatibility.
- This package is a research utility, not a validated clinical or safety-critical tool.

## License

BSD 3-Clause License. See [LICENSE](LICENSE).

## Citation

If you use `phbin` in research, please cite the repository and identify the version/commit used.
