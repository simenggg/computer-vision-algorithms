# Assignment 01: Template Matching and Image Processing

Implemented fundamental image-processing and template-matching algorithms from scratch using Python, NumPy, OpenCV, and Matplotlib.

## Overview

This project explores how images can be transformed, compared, and analyzed using low-level computer vision operations. The assignment includes both straightforward and optimized implementations of normalized cross-correlation, followed by experiments that examine the strengths and limitations of template matching.

## Implemented Methods

### Image Preprocessing

- Converted RGB images to grayscale using weighted channel combinations
- Computed horizontal, vertical, and diagonal gradient maps using Sobel-style filters
- Implemented zero-padding for images and feature maps

### Normalized Cross-Correlation

Implemented three versions of template matching:

- A direct nested-loop implementation
- A faster implementation using element-wise array operations
- A matrix/vectorized implementation for improved computational efficiency

The implementations were compared in terms of detection response and runtime.

### Non-Maximum Suppression

Implemented non-maximum suppression to identify strong local response peaks and reduce duplicate detections in correlation maps.

### Template-Matching Experiments

Investigated how template matching behaves under different conditions:

- Different image and template inputs
- Different template sizes and spatial arrangements
- Mean-subtracted normalized cross-correlation
- Auto-correlation, where the template is the image itself
- Detection failures caused by scale changes and perspective distortion

## Key Observations

- Normalized cross-correlation measures the similarity between a template and local image regions while reducing the effect of overall intensity differences.
- Mean subtraction makes the comparison less sensitive to brightness and helps emphasize structural similarity.
- Non-maximum suppression allows multiple detections to be represented by distinct local peaks instead of many overlapping responses.
- Template size and content strongly affect the locations and quality of detected matches.
- Basic template matching is sensitive to scale, rotation, viewpoint, and perspective changes. A template of a fixed size may fail when the same object appears at a different distance or angle.
- Replacing inner loops with NumPy matrix operations can improve runtime while preserving the same matching concept.

## Technologies

- Python
- NumPy
- OpenCV
- Matplotlib
- Jupyter Notebook

## Files

- `lab1.ipynb` — experiments, visualizations, runtime comparisons, and analysis
- `lab1.py` — algorithm implementations
- `inputs/` — sample images used for testing, if redistribution is permitted

## Running the Notebook

Install the required packages, if a requirements file is included:

```bash
pip install -r requirements.txt
```

Then launch Jupyter Notebook:

```bash
jupyter notebook lab1.ipynb
```

Run the notebook from top to bottom to reproduce the preprocessing, correlation, non-maximum suppression, and template-matching experiments.

## Data and Reproducibility

The notebook expects sample images such as `tiles.png`, `lattice.png`, `holes.png`, and `chairs.png` inside an `inputs/` directory. These images should only be included if redistribution is permitted. Otherwise, add an `inputs/README.md` describing how to obtain the data or replace the paths with your own practice images.
