# Assignment 02: Edge Detection and Hough Transforms

Implemented classical edge-detection and geometric-shape detection algorithms from scratch using Python, NumPy, OpenCV, and Matplotlib.

## Overview

This project develops a complete image-processing pipeline that begins with Gaussian smoothing and Canny-style edge detection, then uses Hough-space voting to detect lines and circles in real images.

## Implemented Methods

### Canny-Style Edge Detection

- Constructed Gaussian kernels with configurable kernel size and standard deviation
- Applied Gaussian blurring to reduce image noise
- Computed image gradients, magnitudes, and orientations using Sobel filters
- Implemented non-maximum suppression, including an interpolated version
- Applied double thresholding to classify weak and strong edge pixels
- Linked connected edge pixels to produce final edge maps

### Linear Hough Transform

- Built an accumulator space for line parameters
- Implemented voting using edge pixels
- Located local maxima in Hough space
- Detected and visualized lines in checkerboard and Rubik’s Cube images

### Circular Hough Transform

- Implemented three-dimensional voting for circle center coordinates and radius
- Detected complete and partial circles
- Visualized circles in images containing golf balls, Olympic rings, coins, cells, and flowers
- Implemented gradient-guided circle voting to improve detection efficiency and accuracy

## Experiments

The notebook investigates how different parameters affect detection quality:

- Voting weights for circles of different sizes
- Gradient-guided voting
- Hough-space interval and bin size
- Gaussian smoothing parameter, `sigma`
- Local-maximum window size and threshold

## Key Observations

- Gaussian smoothing reduces noise but excessive smoothing can weaken or remove fine edges.
- Non-maximum suppression narrows broad gradient responses into thin edge structures.
- Hough transforms can detect geometric shapes even when only partial contours are visible.
- Smaller Hough bins provide more precise parameter estimates but require more computation and memory.
- Gradient-guided voting reduces unnecessary votes by using edge orientation information.
- Circle detection can be biased toward larger or stronger circles; weighted voting can help balance detections across different radii.
- Hough-transform results depend strongly on smoothing, voting, radius range, accumulator resolution, and peak-selection parameters.

## Technologies

- Python
- NumPy
- OpenCV
- Matplotlib
- Jupyter Notebook

## Files

- `lab2.ipynb` — experiments, visualizations, and parameter analysis
- `lab2.py` — algorithm implementations
- `data/input/` — test images, if redistribution is permitted
- `data/output/` — reference or generated results, if included

## Running the Notebook

Install the required packages:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook lab2.ipynb
```

Run the notebook from top to bottom to reproduce the edge maps, Hough accumulators, line detections, and circle detections.

## Data and Reproducibility

The notebook expects test images inside `data/input/`. These images should only be uploaded if redistribution is permitted. Otherwise, add a `data/input/README.md` explaining how to obtain the data or replace the images with your own practice images.
