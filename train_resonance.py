"""Trening uniwersalnej sieci N(X) i analiza wylonionej specjalizacji.

Uzycie:
    python train_resonance.py                  # domyslny trening + raport
    python train_resonance.py --epochs 150
    python train_resonance.py --no-plot
"""

from __future__ import annotations

import argparse

import numpy as np

import motifs
from resonance import ResonanceNet


def report(net: ResonanceNet, X, y) -> dict:
    """Drukuje macierz pomylek (rodzina x tryb) i mapowanie liczba->wzorzec."""
    M = net.confusion(X, y)
    print("\nMacierz emergencji (wiersz = ukryta rodzina, kolumna = tryb X sieci):")
    header = "rodzina \\ X    " + "".join(f"{x:>6}" for x in range(net.K))
    print(header)
    for fam in range(net.K):
        name = motifs.NAMES[fam]
        row = "".join(f"{M[fam, x]:>6}" for x in range(net.K))
        print(f"{name:<13}{row}")

    # Dla kazdego trybu X: ktora rodzine zawlaszczyl (wartosc dominujaca).
    print("\nCo odkryla kazda liczba X (bez nadzoru):")
    mapping = {}
    assign = net.detect(X)
    for x in range(net.K):
        mask = assign == x
        if not mask.any():
            print(f"  N({x}) -> (tryb nieuzywany)")
            continue
        counts = np.bincount(y[mask], minlength=net.K)
        fam = int(counts.argmax())
        share = counts.max() / counts.sum()
        mapping[x] = fam
        print(f"  N({x}) -> '{motifs.NAMES[fam]:<12}'  (czystosc trybu {share:.2f}, "
              f"{mask.sum()} przykladow)")
    return mapping


def maybe_plot(net: ResonanceNet, X, y, path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[wykres pominiety: {exc}]")
        return

    M = net.confusion(X, y)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    im = ax1.imshow(M, cmap="magma", aspect="auto")
    ax1.set_xlabel("tryb X sieci")
    ax1.set_ylabel("ukryta rodzina wzorca")
    ax1.set_yticks(range(net.K))
    ax1.set_yticklabels([motifs.NAMES[i] for i in range(net.K)])
    ax1.set_xticks(range(net.K))
    ax1.set_title("Emergencja: rodzina -> tryb")
    fig.colorbar(im, ax=ax1, fraction=0.046)

    for x in range(net.K):
        ax2.plot(net.archetype(x) + x * 1.3, label=f"N({x})")
    ax2.set_title("Archetypy: czego szuka kazda liczba X")
    ax2.set_xlabel("pozycja w oknie")
    ax2.set_yticks([])
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=110)
    print(f"\nZapisano wykres: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--n-per", type=int, default=600)
    ap.add_argument("--code", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--save", default="resonance_net.npz")
    args = ap.parse_args()

    X, y = motifs.make_dataset(n_per_number=args.n_per, seed=args.seed)
    print(f"Dane: {len(X)} wzorcow, okno={motifs.L}, "
          f"liczb naturalnych={motifs.N_NUMBERS}, kod z={args.code}")
    print("Trening BEZ etykiet (uczenie kompetycyjne)...")
    print("-" * 70)

    net = ResonanceNet(window=motifs.L, n_numbers=motifs.N_NUMBERS,
                       code=args.code, seed=args.seed)
    net.fit(X, epochs=args.epochs, y_eval=y, verbose=True)

    print("-" * 70)
    print(f"Koncowa czystosc emergencji: {net.purity(X, y):.3f}")
    report(net, X, y)
    net.save(args.save)
    print(f"\nZapisano siec: {args.save}")
    if not args.no_plot:
        maybe_plot(net, X, y, "resonance_emergence.png")


if __name__ == "__main__":
    main()
