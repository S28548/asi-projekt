## O projekcie

Projekt realizowany w ramach przedmiotu **ASI** i dotyczy budowy rozwiązania ML w podejściu **MLOps**. Celem jest stopniowe przejście od prostego eksperymentu w notebooku do bardziej uporządkowanego i wdrażalnego systemu, obejmującego pipeline danych, trening modelu, ewaluację oraz późniejsze udostępnienie modelu przez API.

Aktualny etap prac odpowiada **Sprintowi 1**, którego celem jest przygotowanie środowiska, zapis danych do bazy **SQLite**, przeprowadzenie **eksploracyjnej analizy danych (EDA)** oraz wytrenowanie **baseline modelu**. W kolejnych etapach projekt będzie rozwijany o pipeline w Kedro, śledzenie eksperymentów, warstwę REST API oraz dashboard.  

## Dane

W projekcie wykorzystujemy tabelaryczny zbiór danych dotyczący **nieruchomości mieszkalnych** i ich cen sprzedaży. Dane zawierają **1460 obserwacji** oraz **81 kolumn**, w tym zmienną docelową **`SalePrice`**. Zbiór obejmuje zarówno cechy liczbowe, jak i kategoryczne, opisujące m.in.:

- parametry działki,
- lokalizację i typ zabudowy,
- powierzchnię użytkową,
- jakość wykonania i stan techniczny,
- cechy garażu, piwnicy i części zewnętrznych budynku.

W zbiorze występują również braki danych, szczególnie w atrybutach opcjonalnych, takich jak np. `PoolQC`, `MiscFeature`, `Alley` czy `Fence`, co zostało uwzględnione na etapie analizy danych i przygotowania modelu.

## Aktualny etap

Na obecnym etapie repozytorium obejmuje:

- przygotowanie środowiska projektowego i zależności,
- zapis danych do bazy **SQLite**,
- notebook do analizy danych i treningu modelu,
- wstępną analizę statystyczną i wizualizacje,
- trenowanie modelu bazowego do predykcji cen nieruchomości,
- zapis metryk ewaluacyjnych do pliku JSON.

## Wyniki wstępne

Aktualnie zapisanym wynikiem modelu bazowego jest:

- **RMSE:** `24904.71`

Oznacza to, że średni błąd predykcji modelu względem rzeczywistej ceny nieruchomości wynosi około **24,9 tys.** jednostek ceny. Wynik ten traktujemy jako **punkt odniesienia** dla kolejnych iteracji projektu i dalszego ulepszania pipeline’u oraz modeli.
