"""Synthetic dataset generators.

These create self-contained, reproducible classification problems so the
network can be trained without downloading any external data.
"""

from __future__ import annotations

import numpy as np


def make_moons(n_samples: int = 1000, noise: float = 0.20,
               seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaving half-circles -- a classic non-linear benchmark."""
    rng = np.random.default_rng(seed)
    n_out = n_samples // 2
    n_in = n_samples - n_out

    outer_theta = np.linspace(0, np.pi, n_out)
    inner_theta = np.linspace(0, np.pi, n_in)

    outer_x = np.c_[np.cos(outer_theta), np.sin(outer_theta)]
    inner_x = np.c_[1 - np.cos(inner_theta), 1 - np.sin(inner_theta) - 0.5]

    X = np.vstack([outer_x, inner_x])
    y = np.hstack([np.zeros(n_out, dtype=int), np.ones(n_in, dtype=int)])

    X += rng.normal(0.0, noise, size=X.shape)
    return X, y


def make_spiral(n_samples: int = 1000, n_classes: int = 3, noise: float = 0.20,
                seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Interleaved spiral arms -- a harder multi-class problem."""
    rng = np.random.default_rng(seed)
    per_class = n_samples // n_classes
    X = np.zeros((per_class * n_classes, 2))
    y = np.zeros(per_class * n_classes, dtype=int)

    for c in range(n_classes):
        idx = range(per_class * c, per_class * (c + 1))
        r = np.linspace(0.0, 1.0, per_class)
        t = np.linspace(c * 4, (c + 1) * 4, per_class) + rng.normal(0, noise, per_class)
        X[idx] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[idx] = c
    return X, y


def train_test_split(X: np.ndarray, y: np.ndarray, test_frac: float = 0.2,
                     seed: int = 0):
    """Shuffle and split into train / test sets."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    n_test = int(len(X) * test_frac)
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize(X_train: np.ndarray, X_test: np.ndarray):
    """Zero-mean, unit-variance scaling fit on the training set only."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std
