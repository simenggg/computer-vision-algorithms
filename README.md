# Computer Vision Algorithm Implementations

A collection of computer vision assignments implemented in Python. The repository explores fundamental image-processing, feature-detection, image-segmentation, and geometric-analysis techniques using NumPy and OpenCV.

## Projects

### [Assignment 1: Template Matching and Image Processing](assignment-01-template-matching/)

Implemented image preprocessing, Sobel-style gradient computation, normalized cross-correlation, non-maximum suppression, and template-matching experiments. Compared direct and optimized correlation implementations and analyzed limitations caused by scale and perspective changes.

### [Assignment 2: Edge Detection and Hough Transforms](assignment-02-hough-transform/)

Implemented a Canny-style edge-detection pipeline, linear and circular Hough transforms, gradient-guided voting, and parameter experiments for detecting lines and circles in real images.

### [Assignment 3: Image Segmentation and Texture Description](assignment-03-image-segmentation/)

Implemented mean-shift segmentation, Gabor and Laplacian of Gaussian filters, filter-bank feature extraction, texton dictionary learning, and texture-based image segmentation.

### [Assignment 4: Feature Matching and Computer Vision Applications](assignment-04-feature-matching/)

Implemented Harris corner detection, local feature descriptors, descriptor matching, homography estimation, RANSAC-based image stitching, and mirror-symmetry detection using Hough voting.

## Skills Demonstrated

- Image preprocessing and convolution
- Gradient and edge analysis
- Template matching and correlation
- Non-maximum suppression
- Hough-space voting
- Line and circle detection
- Image segmentation and clustering
- Texture analysis with Gabor and LoG filters
- Keypoint detection and local feature description
- Feature matching and ratio-test filtering
- Homography estimation and image warping
- RANSAC-based outlier rejection
- Mirror-symmetry detection
- Algorithm visualization and runtime comparison

## Repository Structure

```text
computer-vision-algorithms/
├── README.md
├── assignment-01-template-matching/
├── assignment-02-hough-transform/
├── assignment-03-image-segmentation/
└── assignment-04-feature-matching/
```

Each assignment folder contains its own README, implementation files, Jupyter notebook, results, and any permitted input data.

## Technologies

- Python
- NumPy
- OpenCV
- SciPy
- scikit-image
- scikit-learn
- Matplotlib
- Jupyter Notebook

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies listed in the relevant assignment folder or in the repository-level requirements file, if provided:

```bash
pip install -r requirements.txt
```

Launch a notebook from its assignment directory:

```bash
jupyter notebook assignment-01-template-matching/template_matching.ipynb
```

## Notes

Some notebooks use course-provided sample images. These files are included only where redistribution is permitted. To reproduce the results, run each notebook from its assignment directory and ensure that its expected `data/` or `inputs/` folder is available.

This repository is organized as a portfolio of learning projects and focuses on algorithm implementation, experimentation, and analysis.
