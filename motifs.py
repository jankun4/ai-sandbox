"""Generator surowych wzorcow liczbowych (motywow) -- testbed odkrywania.

Kazdy motyw to wektor liczb w oknie dlugosci ``L``. Motywy roznia sie
STRUKTURA, nie etykieta -- siec NIGDY nie dostaje informacji "to jest
granica" czy "to jest struktura". Etykiety ``y`` zwracamy wylacznie po to,
by PO treningu sprawdzic, ktory tryb N(X) samodzielnie przyjal dany wzorzec.

Kazda liczba ucielesnia swoja esencje wg szkoly magicznej, w sposob mozliwie
odrebny strukturalnie (by emergencja byla weryfikowalna):

    0  pustka       -- brak sygnalu (zero energii)
    1  byt          -- pojedynczy, wyrozniony element
    2  rozroznienie -- granica: dwa poziomy, jedno przejscie
    3  scalenie     -- trzy elementy zwiazane w calosc
    4  struktura    -- regularna, powtarzalna krata
    5  opor         -- napiecie: dwie przeciwstawne biegunowosci (+/-)
    6  doskonalosc  -- pelny, symetryczny, harmonijny ksztalt

Motywy sa losowo przesuniete / sparametryzowane, wiec nie da sie ich
sklastrowac trywialnie po surowych pikselach -- siec musi nauczyc sie
cech niezmienniczych.
"""

from __future__ import annotations

import numpy as np

L = 16  # dlugosc okna
N_NUMBERS = 7  # liczby naturalne 0..6

NAMES = {
    0: "pustka",
    1: "byt",
    2: "rozroznienie",
    3: "scalenie",
    4: "struktura",
    5: "opor",
    6: "doskonalosc",
}


def _spikes(rng, k):
    v = np.zeros(L)
    v[rng.choice(L, size=k, replace=False)] = 1.0
    return v


def _void(rng):
    return np.zeros(L)


def _being(rng):
    return _spikes(rng, 1)


def _distinction(rng):
    cut = rng.integers(4, L - 3)
    v = np.full(L, 0.15)
    v[cut:] = 0.9
    return v


def _merging(rng):
    return _spikes(rng, 3)


def _structure(rng):
    period = int(rng.choice([2, 3]))
    phase = int(rng.integers(0, period))
    idx = np.arange(L)
    return ((idx % period) == phase).astype(float)


def _resistance(rng):
    # Napiecie dwoch przeciwstawnych biegunow (jedyna rodzina z wartosciami
    # ujemnymi) -- "opor" jako polaryzacja, sila przeciwna sile.
    cut = rng.integers(5, L - 4)
    v = np.empty(L)
    v[:cut] = 0.8
    v[cut:] = -0.8
    return v


def _perfection(rng):
    center = (L - 1) / 2.0
    width = rng.uniform(L * 0.16, L * 0.26)
    idx = np.arange(L)
    g = np.exp(-0.5 * ((idx - center) / width) ** 2)
    return g / g.max()


_GENERATORS = [_void, _being, _distinction, _merging,
               _structure, _resistance, _perfection]


def make_dataset(n_per_number: int = 600, noise: float = 0.03, seed: int = 0):
    """Zwraca (X, y). ``X`` to dane treningowe, ``y`` to ukryte etykiety
    rodzin -- uzywane WYLACZNIE do ewaluacji po treningu."""
    rng = np.random.default_rng(seed)
    samples, labels = [], []
    for number, gen in enumerate(_GENERATORS):
        for _ in range(n_per_number):
            v = gen(rng) + rng.normal(0.0, noise, size=L)
            samples.append(v)
            labels.append(number)
    X = np.array(samples)
    y = np.array(labels)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]
