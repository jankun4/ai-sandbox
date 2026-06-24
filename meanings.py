"""Znaczenia liczb naturalnych wg szkoly magicznej.

Warstwa interpretacyjna: sama siec operuje wylacznie na liczbach 0..6
(bez zadnych slow). Te opisy sluza WYLACZNIE czlowiekowi do odczytania
"sygnatury numerycznej", ktora siec wygenerowala.
"""

NUMBER_MEANINGS = {
    0: {
        "nazwa": "pustka",
        "haslo": "nic, nieswiadomosc, nieoznaczonosc",
        "opis": "pierwotny paradoks -- czy nic istnieje? Brak, potencjal, cisza.",
    },
    1: {
        "nazwa": "byt",
        "haslo": "istnienie, swiadomosc, bog, prawda",
        "opis": "czysta obecnosc, jedno, zrodlo, afirmacja istnienia.",
    },
    2: {
        "nazwa": "rozroznienie",
        "haslo": "dualizm, podzial",
        "opis": "granica, kontrast, ja i nie-ja, narodziny roznicy.",
    },
    3: {
        "nazwa": "synteza",
        "haslo": "scalenie, holizm",
        "opis": "zwiazanie przeciwienstw w calosc, pojednanie, trojca.",
    },
    4: {
        "nazwa": "struktura",
        "haslo": "zlozenie, rama",
        "opis": "trwala konstrukcja, porzadek, regularnosc, fundament.",
    },
    5: {
        "nazwa": "opor",
        "haslo": "bunt, zycie",
        "opis": "napiecie, sila przeciwna, ruch, to co zywe i nieujarzmione.",
    },
    6: {
        "nazwa": "doskonalosc",
        "haslo": "piekno, harmonia",
        "opis": "pelnia, rownowaga, zwienczenie, lad ktory zachwyca.",
    },
}


def describe(x: int) -> str:
    m = NUMBER_MEANINGS[x]
    return f"{x} {m['nazwa']} ({m['haslo']})"


def signature_report(sig, dominance=None) -> str:
    """Tekstowy opis sygnatury numerycznej (wektor rezonansu po 0..6).

    ``dominance`` (opcjonalnie): udzial krokow, w ktorych dana liczba byla
    najsilniejsza -- pokazuje, jak wnioskowanie rozkladalo sie na liczby.
    """
    src = dominance if dominance is not None else sig
    order = sorted(range(len(src)), key=lambda i: -src[i])
    title = ("Sygnatura numeryczna wnioskowania (dominacja per-krok):"
             if dominance is not None else "Sygnatura numeryczna wnioskowania:")
    lines = [title]
    for i in range(len(src)):
        bar = "#" * int(round(src[i] * 30))
        m = NUMBER_MEANINGS[i]
        lines.append(f"  {i} {m['nazwa']:<13} {src[i]*100:5.1f}% {bar}")
    dom = order[0]
    md = NUMBER_MEANINGS[dom]
    lines.append(f"-> dominuje {dom} ({md['nazwa']}): {md['haslo']}")
    return "\n".join(lines)
