# Assignment 03: Image Segmentation and Texture Description

Implemented classical computer vision methods for image segmentation and texture analysis using Python, NumPy, OpenCV, SciPy, scikit-image, scikit-learn, and Matplotlib.

## Overview

This assignment explores two complementary approaches:

1. **Mean-shift segmentation** using joint spatial and RGB feature representations.
2. **Texture-based segmentation** using Gabor and Laplacian of Gaussian (LoG) filter responses, texton dictionaries, and local texton histograms.

## Implemented Methods

### Mean-Shift Segmentation

- Constructed a 5-dimensional feature space for each pixel:
  `(spatial row, spatial column, R, G, B)`
- Implemented a Gaussian-kernel mean-shift update step
- Implemented iterative mean-shift segmentation and mode merging
- Investigated the effect of bandwidth on the number of discovered segments
- Compared mean-shift segmentation with K-means clustering

### Texture Description and Segmentation

- Generated Gabor filters at multiple orientations and scales
- Generated Laplacian of Gaussian filters for spot-like texture patterns
- Built a 12-dimensional filter bank containing:
  - 8 Gabor responses
  - 4 LoG responses
- Learned a texton dictionary using MiniBatch K-means
- Assigned texton labels to image pixels
- Computed local texton histograms using sliding windows
- Compared segmentation using smoothed filter responses and texton histograms

## Key Observations

- Increasing the mean-shift bandwidth generally produces fewer, larger segments, while a smaller bandwidth can lead to over-segmentation.
- Mean-shift does not require the number of clusters to be specified in advance, but it is more computationally expensive and sensitive to bandwidth selection than K-means.
- Gabor filters respond to texture patterns with particular orientations and scales.
- Larger texton-histogram windows produce smoother segmentation but may reduce boundary precision.
- Smoothed filter responses can preserve local boundaries more accurately than regional texton histograms.

## Technologies

- Python
- NumPy
- OpenCV
- SciPy
- scikit-image
- scikit-learn
- Matplotlib
- Jupyter Notebook

## Files

- `lab3.ipynb` — experiments, visualizations, and analysis
- `lab3.py` — algorithm implementations
- `utils.py` — supporting utility functions, if included
- `MSRC_v2/` — optional image data used for texton dictionary learning, if available and permitted for redistribution

## Running the Notebook

Install the required packages:

```bash
pip install -r requirements.txt
```

Then launch Jupyter Notebook:

```bash
jupyter notebook lab3.ipynb
```

Run the notebook from top to bottom to reproduce the segmentation and texture-analysis results.

## Note on Data

The MSRC image dataset is not included in this repository unless redistribution is permitted. If the dataset is unavailable, the mean-shift and filter-bank sections can still be run using the built-in sample image; the texton-dictionary section requires the training images listed in `Train.txt`.
