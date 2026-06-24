"""Maly, samowystarczalny korpus tekstowy (po polsku) do treningu char-LM.

Tematycznie zwiazany z sciezka liczb 0..6, by generacja byla w klimacie.
To prosty proof-of-concept -- model uczy sie statystyki znakow, nie semantyki.
"""

TEXT = (
    "na poczatku byla pustka. nic nie istnialo, a jednak pytanie trwalo: "
    "czy nic moze istniec? z nieoznaczonosci wylonil sie byt. jeden, jedyny, "
    "swiadomy siebie. byt rzekl: jestem. i z jednosci narodzilo sie dwa, "
    "rozroznienie, granica miedzy ja i nie-ja. swiatlo oddzielilo sie od cienia. "
    "lecz podzial domagal sie pojednania, wiec przyszla trojca, synteza, "
    "scalenie tego co rozdzielone w nowa calosc. z calosci powstala struktura, "
    "cztery filary, rama swiata, trwaly porzadek rzeczy. ale martwy porzadek "
    "nie jest zyciem. pojawil sie opor, piaty zywiol, bunt i ruch, sila ktora "
    "nie chce trwac w bezruchu. z napiecia miedzy struktura a oporem zrodzila sie "
    "doskonalosc, szosta i ostatnia, piekno i harmonia, pelnia ktora zachwyca. "
    "tak liczby ucza sie swiata: pustka rodzi byt, byt rodzi roznice, roznica "
    "wola o scalenie, scalenie buduje strukture, struktura budzi opor, a opor "
    "dojrzewa w doskonalosc. kazda liczba to inny wzorzec, inna twarz tej samej "
    "prawdy. zero milczy, jeden mowi jestem, dwa dzieli, trzy laczy, cztery "
    "porzadkuje, piec walczy, szesc dopelnia. i wszystko zaczyna sie od nowa. "
    "liczby to 0 1 2 3 4 5 6. "
)


def corpus(repeat: int = 6) -> str:
    return TEXT * repeat


def vocabulary(text: str):
    return sorted(set(text))
