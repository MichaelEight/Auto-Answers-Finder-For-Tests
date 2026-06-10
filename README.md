# Auto Answers Finder For Tests

Program do automatycznego sprawdzania testów jednokrotnego i wielokrotnego wyboru
na podstawie zeskanowanych kartek odpowiedzi (A/B/C/D). Wykrywa zaznaczenia, rozpoznaje
indeks studenta i liczy punkty.

## Wymagania

- Python 3
- Biblioteki:
  ```
  pip install -r requirements.txt
  ```
  (OpenCV i NumPy — wymagane; opcjonalnie: tkinterdnd2 — przeciąganie plików
  do okna, pymupdf — skany PDF, pillow-heif — zdjęcia HEIC z telefonu)
- (Opcjonalnie) PyInstaller — do zbudowania pliku `.exe`:
  ```
  pip install pyinstaller
  ```

## Jak obsługiwać program?

Uruchom `build/Python Launcher.bat` (lub `python main.py`). Otworzy się aplikacja
okienkowa, która prowadzi przez 4 kroki:

1. **Prace** — przeciągnij zeskanowane prace do okna albo dodaj je przyciskami.
   Obsługiwane formaty: JPG, PNG, PDF, TIFF, BMP, WEBP, GIF, HEIC (zdjęcia
   z telefonu). Wielostronicowy PDF lub TIFF rozwija się automatycznie — każda
   strona to osobna praca (np. `skan.pdf [str. 2]`). Prace z folderu
   `dane/PraceDoSprawdzenia` wczytują się automatycznie.

2. **Klucz odpowiedzi** — kliknij litery A–D przy każdym pytaniu, aby oznaczyć
   poprawne odpowiedzi, albo wczytaj gotowy plik klucza (format jak
   `config/PoprawneOdpowiedzi.txt` — wczytywany automatycznie, jeśli istnieje).
   Klucz można też zapisać do pliku na później.

3. **Sprawdzanie** — kliknij „Rozpocznij sprawdzanie" i poczekaj na pasek postępu.

4. **Wyniki** — tabela z indeksami, punktami i procentami. Kliknięcie pracy pokazuje
   podgląd, a „Weryfikuj zaznaczenia" otwiera pełny widok skanu:
   - **zielone** pole — zaznaczone (zaliczone),
   - **czerwone** pole — zaznaczone i wzięte w kółko (anulowane),
   - **szare** pole — niezaznaczone,
   - **niebieska ramka** — poprawna odpowiedź według klucza.

   Kliknięcie pola na skanie **zmienia jego stan** (puste → zaliczone → anulowane),
   a punkty przeliczają się od razu — tak poprawisz błędne odczyty. Nieodczytany
   indeks można wpisać ręcznie. Na koniec „Zapisz wyniki" tworzy plik z punktacją
   (format jak `dane/WynikiTestu.txt`), a „Kopiuj do schowka" pozwala wkleić tabelę
   np. do Excela.

Skróty: `Enter` — następny krok / główna akcja · `Esc` — powrót · `←`/`→` — zmiana
pracy w weryfikacji · `⌘/Ctrl + kółko myszy` — zoom · przeciąganie — przesuwanie skanu.

### Tryb konsolowy (dawne zachowanie)

```
python main.py --cli
```

Czyta prace z `dane/PraceDoSprawdzenia`, klucz z `config/PoprawneOdpowiedzi.txt`
i zapisuje punktację do `dane/WynikiTestu.txt` w formacie:
```
{indeks}: {liczba punktów}, {% punktów}%
```

Do celów weryfikacji w `dane/PraceZorientowane` zapisują się kopie prac po obróbce
(obrót i skalowanie), a w `dane/PracePrzeanalizowane` — kopie z oznaczonymi polami.

Plik `build/PythonToExeCompiler.bat` zbuduje wersję `.exe` (wymaga PyInstallera).

## Struktura projektu

```
main.py        Główny program (bez argumentów: okno, --cli: konsola)
app/           Kod aplikacji: core.py (analiza skanów), gui.py (okno), cli.py (konsola)
config/        Konfiguracja: config.json, Template.jpg, PoprawneOdpowiedzi.txt
dane/          Dane wejściowe i wyniki (PraceDoSprawdzenia, PracePrzeanalizowane, WynikiTestu.txt)
assets/        Pliki źródłowe szablonu i zbudowany .exe
build/         Skrypty buildu (main.spec, .bat)
examples/      Przykładowe prace i skrypty pomocnicze
tests/         Testy (test_core.py, smoke_gui.py, screenshot_tour.py)
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
