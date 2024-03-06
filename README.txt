======================WYMAGANIA======================
- Python 3
- Biblioteki CV2 oraz NumPy:
	pip install opencv-python numpy
- (Opcjonalnie) Biblioteka PyInstaller:
	pip install pyinstaller

===============JAK OBSŁUGIWAĆ PROGRAM?===============

1. Wstaw prace do folderu "PraceDoSprawdzenia". Obsługuje .jpg i .png

2. Otwórz plik PoprawneOdpowiedzi.txt:
- numer linijki to numer pytania np. trzecia linijka to trzecie pytanie
- każda linijka składa się z 4 kolumn, odpowiednio A B C D
- "1" oznacza odpowiedzi prawidłowe, a "0" nieprawidłowe
np. 1010 oznacza, że poprawne odpoweidzi to A i C
- tyle, ile jest linijek, tyle będzie sprawdzanych pytań

3. Uruchom "Python Launcher.bat"
Alternatywnie plik "PythonToExeCompiler.bat" utworzy plik .exe (wymagany PyInstaller!).

4. W konsoli (jeśli jest otworzona) oraz w pliku "WynikiTestu.txt" pojawi się finalna punktacja w formacie:
{indeks}: {liczba punktów}, {% punktów}%

5. Dodatkowo, do celów weryfikacji/debugowania:
= W folderze "PraceZorientowane" pojawią się kopie prac po wstępnej obróbce (obrót i skalowanie)
= W folderze "PracePrzeanalizowane" pojawią się kopie prac, na których zostaną oznaczone pola, tak jak je odczytał program:
- ZIELONE: pole zostało zaznaczone
- CZERWONE: pole zostało zaznaczone i wzięte w kółko (czyli anulowane)
- SZARE: pole to nie było zaznaczone


================INFORMACJE DODATKOWE================
Autor: Michał 'Eight' Zając

Wykonane testy:
- zmniejszona kartka (OK)
- obrócona kartka (przy pochyleniu większym niż kilka stopni program zaczyna mieć drobne problemy)
- szum nałożony na kartkę, symulacja jakości papieru i jego niedoskonałości (OK)
- długopis czarny i niebieski (OK)
- przezroczystość pisma (przy półprzezroczystym przestaje wykrywać zaznaczenie)
