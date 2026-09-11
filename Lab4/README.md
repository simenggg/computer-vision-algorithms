# Assignment 04: Feature Matching and Computer Vision Applications

Implemented a feature-matching pipeline for keypoint detection, local description, image stitching, homography estimation, and mirror-symmetry detection using Python, NumPy, OpenCV, scikit-image, and Matplotlib.

## Overview

This project demonstrates how local image features can be used to establish correspondences between images and recover higher-level geometric relationships. The assignment progresses from detecting and describing keypoints to robust image alignment and symmetry analysis.

## Implemented Methods

### Keypoint Detection, Description, and Matching

- Implemented Harris corner detection
- Applied non-maximum suppression to obtain keypoint coordinates
- Implemented a normalized patch-based descriptor
- Implemented a simplified SIFT-style descriptor using local gradient orientations
- Matched descriptors using nearest-neighbor distances and the ratio test
- Investigated the effect of descriptor type, match threshold, and image rotation

### Image Stitching and Homography

- Matched local features between overlapping image panels
- Implemented homography estimation from corresponding keypoints
- Warped and combined image panels into a stitched image
- Implemented RANSAC-based homography estimation to reject outlier matches
- Compared direct homography estimation with robust RANSAC estimation
- Tested matching under rotation and challenging image transformations

### Mirror-Symmetry Detection

- Created virtual mirror descriptors by transforming SIFT-like orientation histograms
- Matched keypoints with their reflected counterparts
- Converted matched keypoint pairs into candidate symmetry-line parameters
- Used Hough voting to identify dominant lines of mirror symmetry
- Detected one or multiple symmetry axes in moth and architectural facade images

## Key Observations

- Harris corners provide useful keypoints in images containing strong intensity changes in multiple directions.
- A simple SIFT-style descriptor is generally more robust than a raw patch descriptor, although its performance depends on the quality of the orientation representation.
- Increasing the ratio-test threshold accepts more matches but also increases the risk of false correspondences.
- Direct homography estimation is sensitive to incorrect matches, while RANSAC improves robustness by selecting a consensus set of inliers.
- Homography-based stitching works well when images depict approximately planar scenes and share sufficient visual overlap.
- Feature matching becomes more difficult when images contain large rotations, scale changes, or weakly distinctive regions.
- Mirror symmetry can be detected by matching reflected feature descriptors and accumulating votes in a line-parameter space.

## Technologies

- Python
- NumPy
- OpenCV
- scikit-image
- SciPy
- Matplotlib
- Jupyter Notebook

## Files

- `lab4.ipynb` — experiments, visualizations, and analysis
- `lab4.py` — feature detection, description, matching, homography, RANSAC, and symmetry implementations
- `utils.py` — supporting visualization and image-processing utilities, if included
- `data/input/` — test images, if redistribution is permitted
- `data/output/` — generated or reference results, if included

## Running the Notebook

Install the required packages:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook lab4.ipynb
```

Run the notebook from top to bottom to reproduce keypoint visualizations, descriptor matches, stitched images, RANSAC results, and detected symmetry lines.

## Data and Reproducibility

The notebook expects test images inside `data/input/`. These images should only be uploaded if redistribution is permitted. Otherwise, add a `data/input/README.md` explaining how to obtain the data or replace the images with your own practice images.
