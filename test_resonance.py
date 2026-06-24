"""Testy uniwersalnej sieci N(X).

Uruchom:  python test_resonance.py
Sprawdzaja: (1) poprawnosc backpropu metoda roznic skonczonych,
            (2) ze tryby SAME wylaniaja specjalizacje (wysoka czystosc).
"""

import numpy as np

import motifs
from resonance import ResonanceNet


def test_gradient_check():
    """Analityczny gradient straty rekonstrukcji vs. numeryczny."""
    rng = np.random.default_rng(0)
    w = rng.normal(size=(5, 8))
    net = ResonanceNet(window=8, n_numbers=4, hidden=10, code=6, seed=1)

    # Zamrazamy przydzial klastrow, by strata byla gladka wzgledem wag.
    _, z = net._encode(w)
    assign = net._assign(net._dists(z))

    _, g, _ = net._grads(w, assign_fixed=assign)

    eps = 1e-5
    for name in ["Wd2", "We1", "C"]:
        W = net.P[name]
        flat = W.ravel()
        gflat = g[name].ravel()
        idxs = rng.choice(flat.size, size=min(8, flat.size), replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            lp, _, _ = net._grads(w, assign_fixed=assign)
            flat[i] = orig - eps
            lm, _, _ = net._grads(w, assign_fixed=assign)
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            denom = abs(num) + abs(gflat[i])
            if denom < 1e-9:
                continue  # oba gradienty ~0 -> metryka wzgledna niestabilna
            rel = abs(num - gflat[i]) / denom
            assert rel < 1e-3, f"{name}[{i}] rel={rel:.2e} (num={num}, an={gflat[i]})"
    print("  gradient check OK (backprop zgadza sie z roznicami skonczonymi)")


def test_emergence():
    """Bez etykiet tryby powinny zawlaszczyc rozne rodziny wzorcow."""
    X, y = motifs.make_dataset(n_per_number=400, seed=7)
    net = ResonanceNet(window=motifs.L, n_numbers=motifs.N_NUMBERS, seed=7)
    net.fit(X, epochs=60, verbose=False)
    purity = net.purity(X, y)
    # Los dla 7 klas = 0.14; prog 0.58 = ~4x powyzej, solidna emergencja.
    assert purity > 0.58, f"za niska czystosc emergencji: {purity:.3f}"
    print(f"  emergencja OK (czystosc {purity:.3f} >> los 0.14)")


def test_resonance_is_a_distribution():
    net = ResonanceNet(window=motifs.L, seed=0)
    r = net.resonance(np.zeros(motifs.L))
    assert np.allclose(r.sum(), 1.0), r
    print("  rezonans jest poprawnym rozkladem (sumuje sie do 1)")


if __name__ == "__main__":
    print("Testy N(X):")
    test_resonance_is_a_distribution()
    test_gradient_check()
    test_emergence()
    print("Wszystkie testy przeszly.")
