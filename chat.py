"""Czat oparty o model jezykowy z gleboka warstwa liczbowa (ResonanceLayer).

Cale wnioskowanie modelu przechodzi przez warstwe liczb 0..6. Po kazdej
wymianie czat pokazuje "sygnature numeryczna" -- z ktora liczba (i jej
magicznym znaczeniem) rezonowala odpowiedz.

Uzycie:
    python chat.py --train       # wytrenuj model i zapisz
    python chat.py               # wczytaj model i rozmawiaj (REPL)
    python chat.py --once "tekst" # jednorazowa odpowiedz + sygnatura

To proof-of-concept char-level (model uczy sie statystyki znakow, nie
prowadzi swiadomej rozmowy). Gwiazda jest warstwa liczbowa i jej sygnatura.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import corpus
import meanings
from langmodel import ResonanceLM

MODEL_PATH = "chat_model.npz"


def train_and_save(epochs=40, path=MODEL_PATH):
    text = corpus.corpus(repeat=6)
    vocab = corpus.vocabulary(text)
    print(f"Korpus: {len(text)} znakow, slownik: {len(vocab)} znakow")
    print("Trening modelu z gleboka warstwa liczbowa...")
    lm = ResonanceLM(vocab, context=10, emb=20, d=80, n_numbers=7, seed=0)
    lm.fit(text, epochs=epochs, batch=64, lr=3e-3, lb_weight=0.02, verbose=True)
    lm.save(path)
    print(f"Zapisano model: {path}")
    return lm


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        print("Brak wytrenowanego modelu -- trenuje teraz (jednorazowo)...")
        return train_and_save(path=path)
    return ResonanceLM.load(path)


def respond(lm, prompt, length=160, temperature=0.7, seed=0):
    return lm.generate(prompt, length=length, temperature=temperature, seed=seed)


def show_exchange(lm, prompt, seed=0):
    text, sig, dominance = respond(lm, prompt, seed=seed)
    print("\n--- odpowiedz modelu (kontynuacja) ---")
    print(text)
    print("\n" + meanings.signature_report(sig, dominance=dominance))


def repl(lm):
    print("\n=== CZAT REZONANSU LICZBOWEGO ===")
    print("Pisz wiadomosci. Po kazdej zobaczysz sygnature liczb 0..6.")
    print("Komendy: /quit aby wyjsc, /znaczenia aby zobaczyc sens liczb.\n")
    seed = 0
    while True:
        try:
            msg = input("ty> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbywaj.")
            break
        if not msg:
            continue
        if msg == "/quit":
            print("bywaj.")
            break
        if msg == "/znaczenia":
            for x in range(7):
                print("  " + meanings.describe(x))
            continue
        seed += 1
        show_exchange(lm, msg, seed=seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", action="store_true", help="wytrenuj i zapisz model")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--once", type=str, default=None, help="jednorazowa odpowiedz")
    args = ap.parse_args()

    if args.train:
        train_and_save(epochs=args.epochs)
        return
    lm = load_model()
    if args.once is not None:
        show_exchange(lm, args.once, seed=1)
    else:
        repl(lm)


if __name__ == "__main__":
    main()
