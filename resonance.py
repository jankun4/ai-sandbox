"""N(X) -- uniwersalna siec rezonansu liczbowego (Deep Clustering Network).

Jedna siec sparametryzowana liczba naturalna X. Podajesz liczbe (jako liczbe,
nie tekst), a siec mowi, jak silnie dany wzorzec "rezonuje" z ta liczba.
Rozne X ucza sie -- bez nadzoru -- rozpoznawac rozne typy struktury.

Mechanizm (DCN -- Deep Clustering Network):

    w --[enkoder]--> z --[dekoder]--> w_hat        (autoenkoder: uczy z)
                     |
                     +-- prototypy C[X] = N(X) w przestrzeni z
                         rezonans_X(w) = softmax(-||z - C[X]||^2 / tau)

Trening laczy:
  * rekonstrukcje  ||w - w_hat||^2     -> z niesie sens, brak kolapsu
  * klastrowanie   ||z - C[X*]||^2     -> 7 prototypow dzieli przestrzen
  * bias uzycia                        -> kazda liczba zawlaszcza inny wzorzec

Etykiety nie sa NIGDY uzywane w treningu. Specjalizacja liczb wylania sie
z rywalizacji prototypow. ``archetype(X)`` dekoduje prototyp C[X] -- pokazuje
kanoniczny wzorzec, ktory dana liczba reprezentuje.

Implementacja: czysty NumPy, reczna propagacja wsteczna, optymalizator Adam.
"""

from __future__ import annotations

import numpy as np


def _he(shape, rng):
    return rng.normal(0.0, np.sqrt(2.0 / shape[0]), size=shape)


class _Adam:
    def __init__(self, params: dict, lr=2e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: dict):
        self.t += 1
        for k in self.p:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


class ResonanceNet:
    """Uniwersalna siec N(X) oparta o Deep Clustering Network."""

    def __init__(self, window: int, n_numbers: int = 7, *,
                 hidden=64, code=8, seed=0, lam_cluster=0.5):
        self.L = window
        self.K = n_numbers
        self.code = code
        self.lam = lam_cluster
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.P = {
            "We1": _he((window, hidden), rng), "be1": np.zeros(hidden),
            "We2": _he((hidden, code), rng),   "be2": np.zeros(code),
            "Wd1": _he((code, hidden), rng),   "bd1": np.zeros(hidden),
            "Wd2": _he((hidden, window), rng), "bd2": np.zeros(window),
            # prototypy liczb: C[X] = N(X) w przestrzeni ukrytej
            "C": rng.normal(0.0, 0.3, size=(n_numbers, code)),
        }
        self.usage = np.full(n_numbers, 1.0 / n_numbers)

    # --------------------------------------------------------------- forward
    def _encode(self, w):
        a1 = np.maximum(0.0, w @ self.P["We1"] + self.P["be1"])
        z = a1 @ self.P["We2"] + self.P["be2"]
        return a1, z

    def _decode(self, z):
        hd = np.maximum(0.0, z @ self.P["Wd1"] + self.P["bd1"])
        wh = hd @ self.P["Wd2"] + self.P["bd2"]
        return hd, wh

    def _dists(self, z):
        # ||z - C[X]||^2 dla kazdego X -> (B, K)
        diff = z[:, None, :] - self.P["C"][None, :, :]
        return (diff ** 2).sum(axis=2)

    def _assign(self, dists, use_bias=True):
        d = dists.copy()
        if use_bias:
            d = d + 3.0 * (self.usage - self.usage.mean())  # kara za naduzycie
        return d.argmin(axis=1)

    # --------------------------------------------------------------- gradient
    def _grads(self, w, assign_fixed=None):
        B = w.shape[0]
        a1, z = self._encode(w)
        hd, wh = self._decode(z)

        # rekonstrukcja
        diff = wh - w
        L_rec = float((diff ** 2).sum(axis=1).mean())

        # klastrowanie (twardy przydzial)
        dists = self._dists(z)
        assign = self._assign(dists) if assign_fixed is None else assign_fixed
        Cz = self.P["C"][assign]                       # (B, code)
        L_clu = float(((z - Cz) ** 2).sum(axis=1).mean())
        loss = L_rec + self.lam * L_clu

        g = {k: np.zeros_like(v) for k, v in self.P.items()}

        # --- gradient rekonstrukcji ---
        g_wh = 2.0 * diff / B                           # (B, L)
        g["Wd2"] += hd.T @ g_wh
        g["bd2"] += g_wh.sum(axis=0)
        g_hd = (g_wh @ self.P["Wd2"].T) * (hd > 0)
        g["Wd1"] += z.T @ g_hd
        g["bd1"] += g_hd.sum(axis=0)
        g_z = g_hd @ self.P["Wd1"].T                    # (B, code)

        # --- gradient klastrowania ---
        g_z = g_z + self.lam * 2.0 * (z - Cz) / B
        for X in range(self.K):
            mask = assign == X
            if mask.any():
                g["C"][X] += self.lam * 2.0 * (self.P["C"][X] - z[mask]).sum(0) / B

        # --- enkoder ---
        g["We2"] += a1.T @ g_z
        g["be2"] += g_z.sum(axis=0)
        g_a1 = (g_z @ self.P["We2"].T) * (a1 > 0)
        g["We1"] += w.T @ g_a1
        g["be1"] += g_a1.sum(axis=0)

        return loss, g, assign

    # ------------------------------------------------------------------- fit
    def fit(self, X, *, epochs=120, batch=128, lr=2e-3, y_eval=None, verbose=True):
        opt = _Adam(self.P, lr=lr)
        n = X.shape[0]
        hist = {"loss": [], "purity": []}
        for ep in range(1, epochs + 1):
            perm = self.rng.permutation(n)
            ep_loss = 0.0
            for s in range(0, n, batch):
                xb = X[perm[s:s + batch]]
                loss, g, assign = self._grads(xb)
                opt.step(g)
                counts = np.bincount(assign, minlength=self.K) / len(xb)
                self.usage = 0.95 * self.usage + 0.05 * counts
                ep_loss += loss * len(xb)
            ep_loss /= n
            hist["loss"].append(ep_loss)
            purity = self.purity(X, y_eval) if y_eval is not None else None
            if purity is not None:
                hist["purity"].append(purity)
            if verbose and (ep % max(1, epochs // 10) == 0 or ep == 1):
                msg = f"epoka {ep:3d}/{epochs}  loss={ep_loss:.4f}"
                if purity is not None:
                    msg += f"  czystosc={purity:.3f}"
                print(msg)
        return hist

    # --------------------------------------------------------------- API N(X)
    def resonance(self, w, tau=0.3):
        """Macierz rezonansu (B, K): jak silnie kazdy przyklad rezonuje z X."""
        w = np.atleast_2d(w)
        _, z = self._encode(w)
        d = self._dists(z)
        logits = -d / tau
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    def detect(self, w):
        """Liczba X, z ktora wzorzec rezonuje najsilniej (bez biasu uzycia)."""
        w = np.atleast_2d(w)
        _, z = self._encode(w)
        return self._assign(self._dists(z), use_bias=False)

    def archetype(self, X):
        """Kanoniczny wzorzec liczby X: dekodowany prototyp C[X]."""
        _, wh = self._decode(self.P["C"][X][None, :])
        return wh.ravel()

    # ------------------------------------------------------------ ewaluacja
    def purity(self, X, y):
        assign = self.detect(X)
        total = 0
        for mode in range(self.K):
            mask = assign == mode
            if mask.any():
                total += np.bincount(y[mask], minlength=self.K).max()
        return total / len(y)

    def confusion(self, X, y):
        assign = self.detect(X)
        M = np.zeros((self.K, self.K), dtype=int)
        for fam, mode in zip(y, assign):
            M[fam, mode] += 1
        return M

    # ----------------------------------------------------------- zapis/odczyt
    def save(self, path):
        np.savez(path, usage=self.usage, code=self.code, L=self.L, K=self.K,
                 lam=self.lam, **self.P)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        net = cls(int(d["L"]), int(d["K"]), code=int(d["code"]),
                  lam_cluster=float(d["lam"]))
        net.P = {k: d[k] for k in net.P}
        net.usage = d["usage"]
        return net
