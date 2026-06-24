# ai-sandbox — sieć neuronowa od zera (NumPy)

Implementacja w pełni działającej sieci neuronowej (MLP) napisanej **od podstaw
w NumPy** — bez PyTorcha, TensorFlow czy innych frameworków. Forward, wsteczna
propagacja (backprop) i optymalizator Adam są napisane ręcznie.

## Co jest w środku

| Plik | Opis |
|------|------|
| `neural_network.py` | Biblioteka sieci: warstwy `Dense`, aktywacje `ReLU`/`Tanh`, fuzja `Softmax + Cross-Entropy`, optymalizator **Adam**, klasa `MLP` z `fit/predict`. |
| `data.py` | Generatory danych syntetycznych (`make_moons`, `make_spiral`), podział train/test, standaryzacja. |
| `train.py` | Skrypt treningowy z CLI — trenuje, raportuje dokładność i zapisuje wykres granicy decyzyjnej. |
| `test_neural_network.py` | Testy: **gradient checking** (weryfikacja backpropu metodą różnic skończonych), poprawność softmaxu, test uczenia. |

## Szybki start

```bash
pip install numpy matplotlib

# Trening na zbiorze "two moons" (klasyfikacja binarna, nieliniowa)
python train.py --dataset moons --epochs 200

# Trudniejszy, 3-klasowy problem "spiral"
python train.py --dataset spiral --epochs 400

# Testy poprawności (w tym numeryczna weryfikacja gradientów)
python test_neural_network.py
```

## Wyniki

| Zbiór | Architektura | Dokładność (test) |
|-------|--------------|-------------------|
| two-moons | `[2, 32, 32, 2]` | ~97% |
| spiral (3 klasy) | `[2, 64, 64, 3]` | ~99% |

Po treningu zapisywany jest wykres nauczonej granicy decyzyjnej
(`decision_boundary_*.png`).

## Jak to działa

1. **Forward** — dane przechodzą przez warstwy `Dense` przeplatane aktywacjami.
2. **Strata** — `Softmax + Cross-Entropy` połączone w jeden, numerycznie
   stabilny blok.
3. **Backward** — gradient propagowany wstecz przez wszystkie warstwy
   (reguła łańcuchowa, ręcznie).
4. **Aktualizacja** — optymalizator **Adam** (momenty 1. i 2. rzędu z korekcją
   biasu) aktualizuje wagi.

Poprawność backpropu jest gwarantowana przez **gradient checking** —
analityczny gradient zgadza się z numerycznym z błędem rzędu `1e-11`.

## Konfiguracja CLI

```
--dataset {moons,spiral}   wybór zbioru danych
--epochs N                 liczba epok
--lr FLOAT                 learning rate (Adam)
--batch-size N             rozmiar mini-batcha
--activation {relu,tanh}   funkcja aktywacji
--seed N                   ziarno losowości (powtarzalność)
--no-plot                  pominięcie generowania wykresu
```
