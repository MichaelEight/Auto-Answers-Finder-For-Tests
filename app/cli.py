"""Console mode — reproduces the original batch behavior of main.py."""

from __future__ import annotations

from . import core


def run_cli() -> int:
    try:
        key = core.load_answer_key(core.ANSWER_KEY_FILE)
    except FileNotFoundError:
        print(f"Brak pliku klucza: {core.ANSWER_KEY_FILE}")
        return 1
    except ValueError as e:
        print("Error:", e)
        return 1

    print("Odpowiedzi załadowane! Przygotowywanie prac...")

    sources = core.find_scan_sources(core.INPUT_DIR)
    if not sources:
        print(f"Brak prac w folderze {core.INPUT_DIR}")
        return 1

    print("Prace załadowane! Przygotowywanie do korekty orientacji i skalowania...")

    def progress(i, total, name):
        print(f"Analizowanie pracy {i + 1} z {total}")

    works = core.process(sources, len(key), progress_cb=progress)

    for i, work in enumerate(works):
        if not work.ok:
            print(f"Błąd przetwarzania pracy nr {i + 1}: {work.error}")
        elif not work.id_ok:
            print(f"Błąd w odczycie indeksu dla pracy nr {i + 1}")

    print("Ukończono analizowanie! Wyświetlam odpowiedzi...\n")

    maximum = core.max_points(key)
    for work in works:
        if not work.ok:
            continue
        score = core.score_marks(key, work.marks)
        pct = (score / maximum * 100) if maximum else 0.0
        print(f"{work.student_id}: {score}, {pct:.2f}%")

    core.write_results_file(core.RESULTS_FILE, works, key)
    return 0
