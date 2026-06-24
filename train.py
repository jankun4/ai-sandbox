"""Train the from-scratch MLP on a synthetic dataset and report results.

Usage:
    python train.py                 # two-moons, default settings
    python train.py --dataset spiral --epochs 400
    python train.py --no-plot       # skip the decision-boundary image
"""

from __future__ import annotations

import argparse

import numpy as np

import data
from neural_network import MLP


def build_dataset(name: str, seed: int):
    if name == "moons":
        X, y = data.make_moons(n_samples=1500, noise=0.20, seed=seed)
        layer_sizes = [2, 32, 32, 2]
    elif name == "spiral":
        X, y = data.make_spiral(n_samples=1500, n_classes=3, noise=0.20, seed=seed)
        layer_sizes = [2, 64, 64, 3]
    else:
        raise ValueError(f"unknown dataset: {name}")
    return X, y, layer_sizes


def maybe_plot(model: MLP, X, y, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[plot skipped: matplotlib unavailable: {exc}]")
        return

    pad = 0.5
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, zz, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, s=12, cmap="coolwarm",
                edgecolors="k", linewidths=0.3)
    plt.title("Learned decision boundary")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    print(f"Saved decision-boundary plot to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["moons", "spiral"], default="moons")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--activation", choices=["relu", "tanh"], default="relu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    X, y, layer_sizes = build_dataset(args.dataset, args.seed)
    X_train, X_test, y_train, y_test = data.train_test_split(X, y, 0.2, args.seed)
    X_train, X_test = data.standardize(X_train, X_test)

    print(f"Dataset: {args.dataset}  |  train={len(X_train)}  test={len(X_test)}  "
          f"|  architecture={layer_sizes}  activation={args.activation}")
    print("-" * 70)

    model = MLP(layer_sizes, activation=args.activation, seed=args.seed)
    model.fit(X_train, y_train,
              epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
              X_val=X_test, y_val=y_test, verbose=True)

    print("-" * 70)
    train_acc = (model.predict(X_train) == y_train).mean()
    test_acc = (model.predict(X_test) == y_test).mean()
    print(f"Final train accuracy: {train_acc:.4f}")
    print(f"Final test  accuracy: {test_acc:.4f}")

    if not args.no_plot:
        maybe_plot(model, np.vstack([X_train, X_test]),
                   np.hstack([y_train, y_test]),
                   f"decision_boundary_{args.dataset}.png")


if __name__ == "__main__":
    main()
