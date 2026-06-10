@echo off
rem Buduje main.exe z katalogu glownego repozytorium (wynik w dist\)
cd /d "%~dp0.."
pyinstaller build/main.spec
pause
