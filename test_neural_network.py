"""Sanity checks for the from-scratch neural network.

Run with:  python test_neural_network.py
These verify the maths (gradient checking) and that training actually learns.
"""

import numpy as np

import data
from neural_network import MLP, Dense, SoftmaxCrossEntropy


def test_gradient_check() -> None:
    """Numerically verify backprop against finite differences."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 4))
    y = rng.integers(0, 3, size=8)

    model = MLP([4, 6, 3], seed=1)
    # Analytic gradient for the first dense layer.
    model.loss_fn.forward(model.forward(X), y)
    model.backward()
    layer = model.layers[0]
    analytic = layer.dW.copy()

    # Numerical gradient via central differences.
    eps = 1e-5
    numeric = np.zeros_like(layer.W)
    for i in range(layer.W.shape[0]):
        for j in range(layer.W.shape[1]):
            orig = layer.W[i, j]
            layer.W[i, j] = orig + eps
            loss_plus = model.loss_fn.forward(model.forward(X), y)
            layer.W[i, j] = orig - eps
            loss_minus = model.loss_fn.forward(model.forward(X), y)
            layer.W[i, j] = orig
            numeric[i, j] = (loss_plus - loss_minus) / (2 * eps)

    rel_err = np.abs(analytic - numeric).max() / (np.abs(numeric).max() + 1e-12)
    assert rel_err < 1e-4, f"gradient check failed: rel_err={rel_err:.2e}"
    print(f"  gradient check passed (max relative error {rel_err:.2e})")


def test_softmax_rows_sum_to_one() -> None:
    ce = SoftmaxCrossEntropy()
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    ce.forward(logits, np.array([0, 1]))
    sums = ce.probs.sum(axis=1)
    assert np.allclose(sums, 1.0), sums
    print("  softmax probabilities sum to 1")


def test_learns_moons() -> None:
    """The network should comfortably solve the two-moons problem."""
    X, y = data.make_moons(n_samples=1000, noise=0.15, seed=3)
    Xtr, Xte, ytr, yte = data.train_test_split(X, y, 0.2, seed=3)
    Xtr, Xte = data.standardize(Xtr, Xte)

    model = MLP([2, 32, 32, 2], seed=3)
    model.fit(Xtr, ytr, epochs=120, lr=0.01, batch_size=32, verbose=False)

    acc = (model.predict(Xte) == yte).mean()
    assert acc > 0.95, f"test accuracy too low: {acc:.3f}"
    print(f"  two-moons test accuracy {acc:.3f} (> 0.95 required)")


if __name__ == "__main__":
    print("Running tests:")
    test_softmax_rows_sum_to_one()
    test_gradient_check()
    test_learns_moons()
    print("All tests passed.")
