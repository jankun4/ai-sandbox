"""ResonanceLayer -- gleboka "warstwa mozgu" oparta o liczby naturalne.

Rozniczkowalna warstwa, ktora wpina sie MIEDZY bloki modelu (np. jezykowego).
Cale wnioskowanie -- kazdy ukryty stan H -- jest przepuszczane przez K trybow
liczbowych (0..K-1):

    bramka:    g(H) = softmax(H Wg + bg)        in R^K   (sygnatura rezonansu)
    eksperci:  o_k(H) = relu(H We_k) Wo_k                (transformacja trybu k)
    wyjscie:   H' = H + sum_k g_k(H) * o_k(H)            (mieszanka + residual)

Wektor ``g(H)`` to "sygnatura numeryczna" -- jak silnie dany token rezonuje
z kazda liczba. Specjalizacja liczb WYLANIA sie z zadania (np. przewidywania
nastepnego znaku) plus nacisku na rownomierne uzycie trybow (load balancing),
bez podawania zadnych instrukcji w jezyku naturalnym.

Czysty NumPy, reczna propagacja wsteczna. Klasa trzyma wlasne parametry
i gradienty; ``params()``/``grads()`` pozwalaja podpiac wspolny optymalizator.
"""

from __future__ import annotations

import numpy as np


def _he(shape, rng):
    return rng.normal(0.0, np.sqrt(2.0 / shape[0]), size=shape)


def _softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


class ResonanceLayer:
    def __init__(self, d_model: int, n_numbers: int = 7, expert_hidden=None,
                 seed: int = 0, prefix: str = "res"):
        self.d = d_model
        self.K = n_numbers
        h = expert_hidden or max(8, d_model // 2)
        self.h = h
        self.prefix = prefix
        rng = np.random.default_rng(seed)
        p = {}
        p["Wg"] = _he((d_model, n_numbers), rng) * 0.5
        p["bg"] = np.zeros(n_numbers)
        for k in range(n_numbers):
            p[f"We{k}"] = _he((d_model, h), rng)
            p[f"be{k}"] = np.zeros(h)
            p[f"Wo{k}"] = _he((h, d_model), rng) * 0.5
            p[f"bo{k}"] = np.zeros(d_model)
        self.P = {f"{prefix}_{k}": v for k, v in p.items()}
        self.g = {k: np.zeros_like(v) for k, v in self.P.items()}
        self._last_sig = None

    def _w(self, name):
        return self.P[f"{self.prefix}_{name}"]

    # --------------------------------------------------------------- forward
    def forward(self, H):
        B = H.shape[0]
        logits = H @ self._w("Wg") + self._w("bg")
        gate = _softmax(logits)                          # (B, K)
        self._last_sig = gate

        experts_out, experts_a = [], []
        out = H.copy()                                   # residual
        for k in range(self.K):
            a = np.maximum(0.0, H @ self._w(f"We{k}") + self._w(f"be{k}"))
            o = a @ self._w(f"Wo{k}") + self._w(f"bo{k}")
            experts_a.append(a)
            experts_out.append(o)
            out += gate[:, k:k + 1] * o
        self._cache = (H, gate, experts_a, experts_out)
        return out, gate

    # --------------------------------------------------------------- backward
    def backward(self, dout, lb_weight=0.0):
        """Zwraca dH. Akumuluje gradienty parametrow w self.g.
        ``lb_weight`` > 0 dodaje gradient straty rownowazenia obciazenia."""
        H, gate, experts_a, experts_out = self._cache
        B = H.shape[0]
        for v in self.g.values():
            v[...] = 0.0

        dH = dout.copy()                                 # przez residual
        dgate = np.zeros_like(gate)                       # (B, K)

        for k in range(self.K):
            a, o = experts_a[k], experts_out[k]
            gk = gate[:, k:k + 1]
            # wyjscie miesza sie jako gate_k * o_k
            do = dout * gk
            dgate[:, k] += (dout * o).sum(axis=1)
            # ekspert: o = a @ Wo + bo
            self.g[f"{self.prefix}_Wo{k}"] += a.T @ do
            self.g[f"{self.prefix}_bo{k}"] += do.sum(axis=0)
            da = do @ self._w(f"Wo{k}").T
            da_pre = da * (a > 0)
            self.g[f"{self.prefix}_We{k}"] += H.T @ da_pre
            self.g[f"{self.prefix}_be{k}"] += da_pre.sum(axis=0)
            dH += da_pre @ self._w(f"We{k}").T

        # load balancing: L = K * sum_k (mean_b gate_k)^2  (min przy rownym)
        if lb_weight:
            importance = gate.mean(axis=0)               # (K,)
            dgate += lb_weight * 2.0 * self.K * importance[None, :] / B

        # softmax wstecz
        dlogits = gate * (dgate - (gate * dgate).sum(axis=1, keepdims=True))
        self.g[f"{self.prefix}_Wg"] += H.T @ dlogits
        self.g[f"{self.prefix}_bg"] += dlogits.sum(axis=0)
        dH += dlogits @ self._w("Wg").T
        return dH

    @staticmethod
    def balance_loss(gate, K):
        importance = gate.mean(axis=0)
        return float(K * (importance ** 2).sum())

    def params(self):
        return self.P

    def grads(self):
        return self.g
