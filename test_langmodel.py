"""Testy warstwy rezonansu i modelu jezykowego.

Uruchom:  python test_langmodel.py
"""

import numpy as np

from resonance_layer import ResonanceLayer
from langmodel import ResonanceLM


def test_layer_gradient():
    """Gradient check dla ResonanceLayer (strata = suma kwadratow wyjscia)."""
    rng = np.random.default_rng(0)
    H = rng.normal(size=(4, 6))
    layer = ResonanceLayer(6, n_numbers=3, expert_hidden=5, seed=1)
    target = rng.normal(size=(4, 6))

    def loss_of():
        out, _ = layer.forward(H)
        return 0.5 * ((out - target) ** 2).sum()

    out, _ = layer.forward(H)
    layer.backward(out - target)  # dL/dout = out - target
    ganalytic = {k: v.copy() for k, v in layer.grads().items()}

    eps = 1e-5
    worst = 0.0
    for name in list(layer.P.keys()):
        W = layer.P[name]
        flat = W.ravel()
        gflat = ganalytic[name].ravel()
        for i in rng.choice(flat.size, size=min(5, flat.size), replace=False):
            orig = flat[i]
            flat[i] = orig + eps; lp = loss_of()
            flat[i] = orig - eps; lm = loss_of()
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            denom = abs(num) + abs(gflat[i])
            if denom < 1e-9:
                continue
            worst = max(worst, abs(num - gflat[i]) / denom)
    assert worst < 1e-4, f"gradient warstwy zle: {worst:.2e}"
    print(f"  ResonanceLayer gradient check OK (max blad {worst:.2e})")


def test_lm_gradient():
    """Gradient check calej sciezki LM wzgledem wybranych parametrow."""
    rng = np.random.default_rng(0)
    vocab = "abcd "
    lm = ResonanceLM(vocab, context=4, emb=5, d=8, n_numbers=3, seed=2)
    ctx = rng.integers(0, len(vocab), size=(3, 4))
    y = rng.integers(0, len(vocab), size=3)

    _, g, _, _ = lm.loss_and_grad(ctx, y, lb_weight=0.01)

    eps = 1e-5
    worst = 0.0
    for name in ["W2", "W1", "Emb", "res_Wg", "res_Wo0"]:
        W = lm.P[name]; flat = W.ravel(); gflat = g[name].ravel()
        for i in rng.choice(flat.size, size=min(5, flat.size), replace=False):
            orig = flat[i]
            flat[i] = orig + eps; lp, _, _, _ = lm.loss_and_grad(ctx, y, 0.01)
            flat[i] = orig - eps; lm_, _, _, _ = lm.loss_and_grad(ctx, y, 0.01)
            flat[i] = orig
            num = (lp - lm_) / (2 * eps)
            denom = abs(num) + abs(gflat[i])
            if denom < 1e-9:
                continue
            worst = max(worst, abs(num - gflat[i]) / denom)
    assert worst < 1e-3, f"gradient LM zle: {worst:.2e}"
    print(f"  ResonanceLM gradient check OK (max blad {worst:.2e})")


def test_lm_learns():
    """Model powinien obnizyc perplexity na malym powtarzalnym tekscie."""
    text = "abcabcabcabc " * 20
    vocab = sorted(set(text))
    lm = ResonanceLM(vocab, context=4, emb=8, d=24, n_numbers=7, seed=0)
    ids = lm.encode_text(text)
    # perplexity startowa ~ |V|; po treningu wyraznie nizsza
    lm.fit(text, epochs=15, batch=32, verbose=False)
    rng = np.random.default_rng(0)
    ce = 0.0; n = 0
    for xb, yb in lm.make_batches(ids, 32, rng):
        _, _, c, _ = lm.loss_and_grad(xb, yb)
        ce += c * len(xb); n += len(xb)
    ppl = np.exp(ce / n)
    assert ppl < len(vocab) * 0.6, f"model sie nie nauczyl: ppl={ppl:.2f}"
    print(f"  ResonanceLM uczy sie OK (perplexity {ppl:.2f} < {len(vocab)})")


if __name__ == "__main__":
    print("Testy warstwy rezonansu i LM:")
    test_layer_gradient()
    test_lm_gradient()
    test_lm_learns()
    print("Wszystkie testy przeszly.")
