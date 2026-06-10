# Auto Answers Finder For Tests

Program do automatycznego sprawdzania testów jednokrotnego i wielokrotnego wyboru
na podstawie zeskanowanych kartek odpowiedzi (A/B/C/D). Wykrywa zaznaczenia, rozpoznaje
indeks studenta i liczy punkty.

## Wymagania

- Python 3
- Biblioteki OpenCV i NumPy:
  ```
  pip install opencv-python numpy
  ```
- (Opcjonalnie) PyInstaller — do zbudowania pliku `.exe`:
  ```
  pip install pyinstaller
  ```

## Jak obsługiwać program?

1. Wstaw prace do folderu `dane/PraceDoSprawdzenia`. Obsługiwane formaty: `.jpg` i `.png`.

2. Otwórz plik `config/PoprawneOdpowiedzi.txt` i wpisz klucz odpowiedzi:
   - numer linijki = numer pytania (np. trzecia linijka to trzecie pytanie),
   - każda linijka ma 4 kolumny: kolejno A, B, C, D,
   - `1` oznacza odpowiedź prawidłową, `0` — nieprawidłową
     (np. `1010` oznacza, że poprawne odpowiedzi to A i C),
   - liczba linijek = liczba sprawdzanych pytań (maksymalnie 48).

3. Uruchom `build/Python Launcher.bat`.
   Alternatywnie `build/PythonToExeCompiler.bat` zbuduje plik `.exe` (wymaga PyInstallera).

4. Finalna punktacja pojawi się w konsoli oraz w pliku `dane/WynikiTestu.txt`
   w formacie:
   ```
   {indeks}: {liczba punktów}, {% punktów}%
   ```

5. Dodatkowo, do celów weryfikacji/debugowania:
   - w folderze `dane/PraceZorientowane` pojawią się kopie prac po wstępnej obróbce
     (obrót i skalowanie),
   - w folderze `dane/PracePrzeanalizowane` pojawią się kopie prac z oznaczonymi polami,
     tak jak odczytał je program:
     - **zielone** — pole zaznaczone,
     - **czerwone** — pole zaznaczone i wzięte w kółko (czyli anulowane),
     - **szare** — pole niezaznaczone.

## Struktura projektu

```
main.py        Główny program (runner)
config/        Konfiguracja: config.json, Template.jpg, PoprawneOdpowiedzi.txt
dane/          Dane wejściowe i wyniki (PraceDoSprawdzenia, PracePrzeanalizowane, WynikiTestu.txt)
assets/        Pliki źródłowe szablonu i zbudowany .exe
build/         Skrypty buildu (main.spec, .bat)
examples/      Przykładowe prace i skrypty pomocnicze
docs/          Notatki i lista zadań
```

## Wykonane testy

- zmniejszona kartka — OK,
- obrócona kartka — przy pochyleniu większym niż kilka stopni pojawiają się drobne problemy,
- szum na kartce (symulacja jakości papieru) — OK,
- długopis czarny i niebieski — OK,
- przezroczystość pisma — przy półprzezroczystym program przestaje wykrywać zaznaczenie.

## Autor

Michał „Eight" Zając
