"""A small neural network library implemented from scratch in NumPy.

No deep-learning frameworks are used -- forward propagation, backpropagation
and the optimizers are all written by hand. The goal is a readable reference
implementation that nonetheless trains to high accuracy on real problems.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Activation functions
# --------------------------------------------------------------------------- #
class ReLU:
    """Rectified linear unit: f(x) = max(0, x)."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self._mask


class Tanh:
    """Hyperbolic tangent activation."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._out = np.tanh(x)
        return self._out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * (1.0 - self._out ** 2)


# --------------------------------------------------------------------------- #
# Dense (fully connected) layer
# --------------------------------------------------------------------------- #
class Dense:
    """A fully connected layer: y = x @ W + b.

    Weights use He initialization, which works well with ReLU activations.
    """

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator):
        scale = np.sqrt(2.0 / n_in)
        self.W = rng.normal(0.0, scale, size=(n_in, n_out))
        self.b = np.zeros(n_out)
        # Gradient and Adam optimizer state buffers.
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._mW = np.zeros_like(self.W)
        self._vW = np.zeros_like(self.W)
        self._mb = np.zeros_like(self.b)
        self._vb = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # Average gradients over the batch.
        batch = self._x.shape[0]
        self.dW = self._x.T @ grad / batch
        self.db = grad.mean(axis=0)
        return grad @ self.W.T

    def step_adam(self, lr: float, t: int,
                  beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        """Apply one Adam update to this layer's parameters."""
        for param, grad, m, v in (
            (self.W, self.dW, self._mW, self._vW),
            (self.b, self.db, self._mb, self._vb),
        ):
            m[...] = beta1 * m + (1 - beta1) * grad
            v[...] = beta2 * v + (1 - beta2) * grad ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)


# --------------------------------------------------------------------------- #
# Loss: softmax + cross-entropy (fused for numerical stability)
# --------------------------------------------------------------------------- #
class SoftmaxCrossEntropy:
    """Combines softmax and cross-entropy into one numerically stable block."""

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        self._probs = exp / exp.sum(axis=1, keepdims=True)
        self._y = y
        n = logits.shape[0]
        log_likelihood = -np.log(self._probs[np.arange(n), y] + 1e-12)
        return float(log_likelihood.mean())

    def backward(self) -> np.ndarray:
        n = self._y.shape[0]
        grad = self._probs.copy()
        grad[np.arange(n), self._y] -= 1.0
        return grad

    @property
    def probs(self) -> np.ndarray:
        return self._probs


# --------------------------------------------------------------------------- #
# The multi-layer perceptron
# --------------------------------------------------------------------------- #
class MLP:
    """A configurable multi-layer perceptron classifier.

    Parameters
    ----------
    layer_sizes : list[int]
        e.g. ``[2, 64, 64, 2]`` for a 2-input, 2-class network with two
        hidden layers of 64 units each.
    activation : {"relu", "tanh"}
        Hidden-layer activation function.
    seed : int
        Seed for reproducible weight initialization.
    """

    def __init__(self, layer_sizes: list[int], activation: str = "relu",
                 seed: int = 0):
        self.rng = np.random.default_rng(seed)
        act_cls = {"relu": ReLU, "tanh": Tanh}[activation]

        self.layers: list = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Dense(layer_sizes[i], layer_sizes[i + 1], self.rng))
            # Activation on every layer except the output (which feeds softmax).
            if i < len(layer_sizes) - 2:
                self.layers.append(act_cls())

        self.loss_fn = SoftmaxCrossEntropy()
        self._t = 0  # Adam timestep.

    # -- core passes ------------------------------------------------------- #
    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self) -> None:
        grad = self.loss_fn.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def _step(self, lr: float) -> None:
        self._t += 1
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.step_adam(lr, self._t)

    # -- public API -------------------------------------------------------- #
    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x).argmax(axis=1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = self.forward(x)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray, *,
            epochs: int = 200, lr: float = 0.01, batch_size: int = 32,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
            verbose: bool = True) -> dict:
        """Train with mini-batch Adam. Returns a history dict of metrics."""
        n = X.shape[0]
        history = {"loss": [], "acc": [], "val_acc": []}

        for epoch in range(1, epochs + 1):
            perm = self.rng.permutation(n)
            X_shuf, y_shuf = X[perm], y[perm]

            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                xb = X_shuf[start:start + batch_size]
                yb = y_shuf[start:start + batch_size]
                logits = self.forward(xb)
                epoch_loss += self.loss_fn.forward(logits, yb) * len(xb)
                self.backward()
                self._step(lr)

            epoch_loss /= n
            train_acc = float((self.predict(X) == y).mean())
            history["loss"].append(epoch_loss)
            history["acc"].append(train_acc)

            val_acc = None
            if X_val is not None:
                val_acc = float((self.predict(X_val) == y_val).mean())
                history["val_acc"].append(val_acc)

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
                msg = f"epoch {epoch:4d}/{epochs}  loss={epoch_loss:.4f}  acc={train_acc:.3f}"
                if val_acc is not None:
                    msg += f"  val_acc={val_acc:.3f}"
                print(msg)

        return history
