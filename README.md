# Clustering-Algorithms-Comparison

This repository contains implementations of various clustering algorithms, including K-Means, K-Means++, and Bisecting K-Means, along with a synthetic dataset generator. The project evaluates clustering performance using silhouette scores and visualizes the results.

## Algorithms Included

1. **Standard K-Means**: Traditional implementation of the K-Means clustering algorithm.
2. **K-Means++**: Improved version of K-Means with better initial centroid selection.
3. **Bisecting K-Means**: Hierarchical approach that recursively splits clusters using K-Means.
4. **Synthetic Dataset Generator**: Creates synthetic datasets for testing clustering algorithms.

## Features

- **Silhouette Score Calculation**: Measures the quality of clustering for different values of *k*.
- **Visualization**: Plots silhouette scores across varying numbers of clusters.
- **Modular Code**: Each algorithm is implemented in a separate, well-documented Python file.
- **Reproducibility**: Fixed random seeds ensure consistent results across runs.

## Files

- `KMeans.py`: Standard K-Means implementation.
- `KMeansplusplus.py`: K-Means++ implementation with optimized centroid initialization.
- `BisectingKMeans.py`: Bisecting K-Means algorithm.
- `KMeansSynthetic.py`: Generates synthetic data and applies K-Means clustering.
- `Bisect.py`: Alternative implementation of Bisecting K-Means (for reference).
