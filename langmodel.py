"""Maly model jezykowy (char-level) z ResonanceLayer jako "warstwa mozgu".

Architektura (kazdy krok wnioskowania przechodzi przez liczby 0..6):

    kontekst C znakow --[embedding]--> x
    x --[Dense+ReLU]--> h  (ukryty stan, d-wymiarowy)
    h --[ResonanceLayer]--> h'   + sygnatura numeryczna g in R^7
    h' --[Dense]--> logity nad slownikiem --> softmax --> nastepny znak

To jest dowod koncepcji: model uczy sie statystyki znakow danego korpusu,
a CALE wnioskowanie jest przepuszczane przez warstwe liczbowa. Przy generacji
mozemy odczytac, z ktora liczba rezonuje kazdy krok (patrz chat.py).

Czysty NumPy, reczna propagacja wsteczna, optymalizator Adam.
"""

from __future__ import annotations

import numpy as np

from resonance_layer import ResonanceLayer


def _he(shape, rng):
    return rng.normal(0.0, np.sqrt(2.0 / shape[0]), size=shape)


def _softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


_FOLD = str.maketrans({
    "a": "a", "ą": "a", "ć": "c", "ę": "e", "ł": "l", "ń": "n", "ó": "o",
    "ś": "s", "ź": "z", "ż": "z",
})


def normalize(text: str) -> str:
    """Sprowadza tekst do ASCII-lowercase zgodnego ze slownikiem korpusu
    (polskie znaki diakrytyczne -> ich odpowiedniki, male litery)."""
    return text.lower().translate(_FOLD)


class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for k in self.p:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


class ResonanceLM:
    def __init__(self, vocab, *, context=8, emb=16, d=64, n_numbers=7, seed=0):
        self.vocab = list(vocab)
        self.V = len(self.vocab)
        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        self.itos = {i: c for i, c in enumerate(self.vocab)}
        self.C = context
        self.emb = emb
        self.d = d
        self.K = n_numbers
        rng = np.random.default_rng(seed)

        self.P = {
            "Emb": rng.normal(0, 0.1, size=(self.V, emb)),
            "W1": _he((context * emb, d), rng), "b1": np.zeros(d),
            "W2": _he((d, self.V), rng),        "b2": np.zeros(self.V),
        }
        self.res = ResonanceLayer(d, n_numbers=n_numbers, seed=seed, prefix="res")
        # wspolny slownik parametrow (model + warstwa rezonansu)
        self.P.update(self.res.params())

    # ----------------------------------------------------------- dane
    def encode_text(self, text):
        text = normalize(text)
        return np.array([self.stoi[c] for c in text if c in self.stoi])

    def make_batches(self, ids, batch, rng):
        # konteksty z paddingiem na poczatku (indeks 0 jako pad)
        xs, ys = [], []
        for i in range(1, len(ids)):
            ctx = ids[max(0, i - self.C):i]
            if len(ctx) < self.C:
                ctx = np.concatenate([np.zeros(self.C - len(ctx), int), ctx])
            xs.append(ctx)
            ys.append(ids[i])
        xs, ys = np.array(xs), np.array(ys)
        idx = rng.permutation(len(xs))
        for s in range(0, len(xs), batch):
            b = idx[s:s + batch]
            yield xs[b], ys[b]

    # ----------------------------------------------------------- forward
    def forward(self, ctx):
        B = ctx.shape[0]
        e = self.P["Emb"][ctx]                            # (B, C, emb)
        x = e.reshape(B, self.C * self.emb)
        hpre = x @ self.P["W1"] + self.P["b1"]
        h = np.maximum(0.0, hpre)
        hres, sig = self.res.forward(h)
        logits = hres @ self.P["W2"] + self.P["b2"]
        cache = (ctx, x, hpre, h, hres)
        return logits, sig, cache

    def loss_and_grad(self, ctx, y, lb_weight=0.02):
        B = ctx.shape[0]
        logits, sig, cache = self.forward(ctx)
        probs = _softmax(logits)
        ce = -np.log(probs[np.arange(B), y] + 1e-12).mean()
        lb = ResonanceLayer.balance_loss(sig, self.K)
        loss = ce + lb_weight * lb

        g = {k: np.zeros_like(v) for k, v in self.P.items()}
        ctx_, x, hpre, h, hres = cache

        dlogits = probs
        dlogits[np.arange(B), y] -= 1.0
        dlogits /= B
        g["W2"] += hres.T @ dlogits
        g["b2"] += dlogits.sum(axis=0)
        dhres = dlogits @ self.P["W2"].T

        dh = self.res.backward(dhres, lb_weight=lb_weight)
        for k, v in self.res.grads().items():
            g[k] += v

        dhpre = dh * (hpre > 0)
        g["W1"] += x.T @ dhpre
        g["b1"] += dhpre.sum(axis=0)
        dx = dhpre @ self.P["W1"].T                       # (B, C*emb)
        de = dx.reshape(B, self.C, self.emb)
        np.add.at(g["Emb"], ctx_, de)
        return loss, g, ce, lb

    # ----------------------------------------------------------- trening
    def fit(self, text, *, epochs=8, batch=64, lr=3e-3, lb_weight=0.02,
            seed=0, verbose=True):
        ids = self.encode_text(text)
        opt = Adam(self.P, lr=lr)
        rng = np.random.default_rng(seed)
        for ep in range(1, epochs + 1):
            tot, n, ce_s, lb_s = 0.0, 0, 0.0, 0.0
            for xb, yb in self.make_batches(ids, batch, rng):
                loss, g, ce, lb = self.loss_and_grad(xb, yb, lb_weight)
                opt.step(g)
                tot += loss * len(xb); ce_s += ce * len(xb)
                lb_s += lb * len(xb); n += len(xb)
            if verbose:
                print(f"epoka {ep:2d}/{epochs}  loss={tot/n:.3f}  "
                      f"ce={ce_s/n:.3f}  balans={lb_s/n:.3f}  "
                      f"perplexity={np.exp(ce_s/n):.2f}")
        return self

    # ----------------------------------------------------------- generacja
    def generate(self, prompt, length=200, temperature=0.8, seed=0):
        rng = np.random.default_rng(seed)
        prompt = normalize(prompt)
        ids = list(self.encode_text(prompt))
        if not ids:
            ids = [0]
        out = list(prompt)
        sig_acc = np.zeros(self.K)
        dominance = np.zeros(self.K)
        steps = 0
        for _ in range(length):
            ctx = ids[-self.C:]
            if len(ctx) < self.C:
                ctx = [0] * (self.C - len(ctx)) + ctx
            logits, sig, _ = self.forward(np.array([ctx]))
            sig_acc += sig[0]
            dominance[int(sig[0].argmax())] += 1
            steps += 1
            p = _softmax(logits / temperature)[0]
            nxt = rng.choice(self.V, p=p)
            ids.append(nxt)
            out.append(self.itos[nxt])
        return "".join(out), sig_acc / max(1, steps), dominance / max(1, steps)

    def signature(self, text):
        """Srednia sygnatura numeryczna (rezonans 0..K-1) dla danego tekstu."""
        ids = self.encode_text(text)
        if len(ids) < 1:
            return np.ones(self.K) / self.K
        sigs = []
        for i in range(1, len(ids) + 1):
            ctx = ids[max(0, i - self.C):i]
            if len(ctx) < self.C:
                ctx = np.concatenate([np.zeros(self.C - len(ctx), int), ctx])
            _, sig, _ = self.forward(np.array([ctx]))
            sigs.append(sig[0])
        return np.mean(sigs, axis=0)

    # ----------------------------------------------------------- zapis
    def save(self, path):
        meta = dict(vocab="".join(self.vocab), C=self.C, emb=self.emb,
                    d=self.d, K=self.K)
        np.savez(path, meta=np.array([repr(meta)], dtype=object), **self.P)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        meta = eval(d["meta"][0])
        m = cls(meta["vocab"], context=meta["C"], emb=meta["emb"],
                d=meta["d"], n_numbers=meta["K"])
        for k in m.P:
            m.P[k] = d[k]
        # ponowne podpiecie parametrow warstwy rezonansu
        for k in m.res.P:
            m.res.P[k] = m.P[k]
        return m
