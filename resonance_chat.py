"""Czat A: prawdziwy LLM jako silnik rozumujacy + warstwa liczb 0-6.

Cale wnioskowanie jest przepuszczane przez warstwe liczbowa na poziomie
ORKIESTRACJI -- ta sama architektura, co ResonanceLayer (mieszanka ekspertow),
ale podniesiona na poziom zdolnego modelu jezykowego:

    1. BRAMKA   -> model ocenia rezonans pytania z kazda liczba 0-6 (sygnatura)
    2. EKSPERCI -> dla dominujacych liczb model rozumuje PRZEZ dana zasade
    3. SYNTEZA  -> laczy perspektywy (wazone rezonansem) w jedna odpowiedz

To realizacja opcji A: odpowiedz jest AUTENTYCZNIE rozumowana przez model,
nie wklejona. Liczby strukturyzuja obliczenie, nie podsuwaja wyniku.

Backend LLM: Anthropic SDK (wymaga ANTHROPIC_API_KEY w srodowisku). Sygnature
mozna tez policzyc lokalnie, wytrenowana siec ResonanceLM (bez API).

Uzycie:
    python resonance_chat.py --question "..."         # pelny tryb (z kluczem)
    python resonance_chat.py --question "..." --local-signature   # sama sygnatura sieci
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import meanings
from meanings import NUMBER_MEANINGS

MODEL_DEFAULT = "claude-opus-4-8"


# --------------------------------------------------------------------------- #
# Definicja warstwy liczbowej dla modelu (z meanings.py -- wspolne zrodlo prawdy)
# --------------------------------------------------------------------------- #
def principles_block() -> str:
    lines = []
    for x in range(7):
        m = NUMBER_MEANINGS[x]
        lines.append(f"  {x} = {m['nazwa']} ({m['haslo']}): {m['opis']}")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""Jestes silnikiem rozumujacym, ktorego CALE wnioskowanie przechodzi
przez warstwe siedmiu zasad-liczb naturalnych (szkola magiczna uzytkownika):

{principles_block()}

Masz odpowiadac AUTENTYCZNIE rozumujac, a nie recytujac. Liczby maja
strukturyzowac Twoje myslenie (jak warstwa ekspertow w sieci), nie narzucac
gotowej tezy. Sam rozstrzygasz tresc -- w tym gdzie sie zatrzymac."""


def gate_and_reason_prompt(question: str) -> str:
    return f"""Pytanie uzytkownika:
\"{question}\"

Wykonaj DOKLADNIE trzy kroki i zwroc WYLACZNIE poprawny JSON:

1. BRAMKA: oszacuj rezonans pytania z kazda z 7 liczb (0..6) jako rozklad
   sumujacy sie do 1.0. To miara, jak silnie dana zasada dotyczy pytania.
2. EKSPERCI: dla 2-3 liczb o najwyzszym rezonansie napisz krotkie (1-2 zdania)
   rozumowanie PRZEZ ta zasade.
3. SYNTEZA: jedna spojna odpowiedz na pytanie, wyrastajaca z powyzszych
   perspektyw. Rozstrzygnij merytorycznie.

Format JSON:
{{
  "rezonans": [r0, r1, r2, r3, r4, r5, r6],
  "eksperci": {{"<liczba>": "<rozumowanie>", ...}},
  "odpowiedz": "<finalna odpowiedz>"
}}"""


# --------------------------------------------------------------------------- #
# Backend LLM (Anthropic)
# --------------------------------------------------------------------------- #
class AnthropicBackend:
    def __init__(self, model: str = MODEL_DEFAULT):
        import anthropic  # import lokalny -- wymagany tylko w tym trybie
        if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
            raise RuntimeError(
                "Brak ANTHROPIC_API_KEY -- ustaw klucz, aby uruchomic tryb LLM.")
        self.client = anthropic.Anthropic()
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 1500) -> str:
        r = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}])
        return r.content[0].text


def ask_llm(question: str, model: str = MODEL_DEFAULT) -> dict:
    """Pelny przebieg A: bramka -> eksperci -> synteza, przez prawdziwy LLM."""
    backend = AnthropicBackend(model=model)
    raw = backend.complete(SYSTEM_PROMPT, gate_and_reason_prompt(question))
    start, end = raw.find("{"), raw.rfind("}")
    data = json.loads(raw[start:end + 1])
    sig = np.array(data["rezonans"], dtype=float)
    sig = sig / sig.sum()
    data["rezonans"] = sig
    return data


# --------------------------------------------------------------------------- #
# Sygnatura lokalna (bez API): wytrenowana siec ResonanceLM
# --------------------------------------------------------------------------- #
def local_signature(question: str, model_path: str = "chat_model.npz"):
    from langmodel import ResonanceLM
    if not os.path.exists(model_path):
        raise RuntimeError(f"Brak modelu {model_path} -- uruchom: python chat.py --train")
    lm = ResonanceLM.load(model_path)
    return lm.signature(question)


# --------------------------------------------------------------------------- #
def render(question: str, sig, answer: str | None = None, experts: dict | None = None):
    print(f"PYTANIE: {question}\n")
    if experts:
        print("Rozumowanie przez dominujace liczby:")
        for k in sorted(experts, key=lambda i: -sig[int(i)]):
            m = NUMBER_MEANINGS[int(k)]
            print(f"  [{k} {m['nazwa']}] {experts[k]}")
        print()
    if answer:
        print("ODPOWIEDZ:\n" + answer + "\n")
    print(meanings.signature_report(sig))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--question", required=True)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--local-signature", action="store_true",
                    help="policz tylko sygnature wytrenowana siecia (bez API)")
    args = ap.parse_args()

    if args.local_signature:
        sig = local_signature(args.question)
        render(args.question, sig)
        return
    data = ask_llm(args.question, model=args.model)
    render(args.question, data["rezonans"], data["odpowiedz"], data.get("eksperci"))


if __name__ == "__main__":
    main()
