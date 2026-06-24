# ai-sandbox — model językowy z liczbową „warstwą mózgu" (NumPy)

Wszystko napisane **od zera w NumPy** — bez PyTorcha, TensorFlow czy innych
frameworków. Forward, wsteczna propagacja i optymalizator Adam są ręczne.

Projekt realizuje koncepcję: **model językowy, którego całe wnioskowanie
przechodzi przez specjalną warstwę opartą o liczby naturalne 0–6**, gdzie każda
liczba sama (bez nadzoru i bez instrukcji słownych) uczy się rezonować z innym
typem wzorca.

## Filozofia liczb (szkoła magiczna)

| Liczba | Znaczenie |
|--------|-----------|
| **0** pustka | nic, nieświadomość, nieoznaczoność — pierwotny paradoks |
| **1** byt | istnienie, świadomość, bóg, prawda |
| **2** rozróżnienie | dualizm, podział |
| **3** synteza | scalenie, holizm |
| **4** struktura | złożenie, rama |
| **5** opór | bunt, życie |
| **6** doskonałość | piękno, harmonia |

Siec operuje **wyłącznie na liczbach** — opisy słowne to tylko warstwa
interpretacyjna dla człowieka (`meanings.py`).

## Trzy warstwy projektu

### 1. `N(X)` — uniwersalna sieć rezonansu (dowód emergencji)
Jedna sieć sparametryzowana liczbą `X`. Uczy się **bez etykiet** (Deep
Clustering Network: autoenkoder + rywalizujące prototypy `C[X]=N(X)`)
rozpoznawać odrębne typy struktury. Na testbedzie 7 rodzin wzorców osiąga
czystość ~0.69 (los = 0.14, ~5×), a część liczb izoluje swój wzorzec idealnie
(np. `N` przypisane polaryzacji/oporowi — 100%). To dowód, że liczby potrafią
**samodzielnie** zawłaszczyć różne wzorce.

| Plik | Opis |
|------|------|
| `motifs.py` | Nieoznaczone rodziny wzorców (pustka…doskonałość); etykiety tylko do ewaluacji. |
| `resonance.py` | `ResonanceNet` — enkoder/dekoder + prototypy `C[X]`, `resonance()`/`detect()`/`archetype()`. |
| `train_resonance.py` | Trening bez nadzoru, macierz emergencji, mapowanie liczba→wzorzec, wykres. |

### 2. `ResonanceLayer` — głęboka „warstwa mózgu"
Różniczkowalna warstwa wpinana **między bloki modelu**. Każdy ukryty stan jest
przepuszczany przez 7 trybów-liczb (mieszanka ekspertów + bramka rezonansu):

```
g(H) = softmax(H·Wg)              # sygnatura: rezonans z każdą liczbą
H' = H + Σ_k g_k(H) · ekspert_k(H)  # całe wnioskowanie idzie przez liczby
```

Specjalizacja liczb wyłania się z zadania + nacisku na równe użycie trybów
(load balancing) — **bez instrukcji słownych**. Plik: `resonance_layer.py`.

### 3. `ResonanceLM` + czat
Mały char-level model językowy z `ResonanceLayer` w środku ścieżki wnioskowania.
Po każdej odpowiedzi czat pokazuje **sygnaturę numeryczną** — z którą liczbą
(i jej znaczeniem) rezonowało wnioskowanie.

| Plik | Opis |
|------|------|
| `langmodel.py` | `ResonanceLM`: embedding → Dense → `ResonanceLayer` → wyjście; trening, generacja. |
| `corpus.py` | Samowystarczalny korpus PL (tematyka ścieżki liczb). |
| `meanings.py` | Warstwa interpretacyjna — znaczenia liczb, raport sygnatury. |
| `chat.py` | REPL czatu z podglądem sygnatury numerycznej. |

## Szybki start

```bash
pip install numpy matplotlib

# Dowód emergencji N(X): liczby same odkrywają wzorce
python train_resonance.py

# Trening modelu czatu i rozmowa
python chat.py --train
python chat.py                       # REPL
python chat.py --once "czym jest doskonalosc"

# Testy (gradient checking + emergencja + uczenie)
python test_resonance.py
python test_langmodel.py
python test_neural_network.py
```

## Przykład

```
ty> czym jest doskonalosc
--- odpowiedz modelu ---
czym jest doskonalosc, szosta i ostatnia, piekno i harmonia, pelnia...

Sygnatura numeryczna wnioskowania (dominacja per-krok):
  6 doskonalosc    20.6% ######
-> dominuje 6 (doskonalosc): piekno, harmonia
```

Pytanie o doskonałość → wnioskowanie najsilniej rezonuje z liczbą **6**.
To zachowanie **nie jest zakodowane** — wyłania się z treningu.

## Uczciwie o ograniczeniach

- Char-level model na małym korpusie to **proof-of-concept** — uczy się
  statystyki znaków, nie prowadzi świadomej rozmowy jak duży LLM.
- Mapowanie liczba→znaczenie częściowo się wyłania, ale nie jest gwarantowane;
  „magiczne" sensy to nakładka interpretacyjna na odkryte, odrębne tryby.
- Wszystko jest jednak **realne i weryfikowalne**: backprop sprawdzony numerycznie
  (`gradient checking`, błąd ~1e-8), emergencja mierzona względem ukrytej prawdy.

## Bonus: `neural_network.py`
Samodzielna biblioteka MLP od zera (He init, ReLU/Tanh, Softmax-CE, Adam) +
dane syntetyczne (`data.py`, `train.py`). ~97% na two-moons, ~99% na spirali.
```bash
python train.py --dataset moons
python train.py --dataset spiral
```
