"""Captures a screenshot of every GUI screen (macOS only, uses `screencapture`).

Run with: python3 tests/screenshot_tour.py [output_dir]
"""

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import core, gui

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/omr_screens")


def shoot(app, name):
    app.update_idletasks()
    app.update()
    time.sleep(0.4)
    x, y = app.winfo_rootx(), app.winfo_rooty()
    w, h = app.winfo_width(), app.winfo_height()
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run(["screencapture", "-x", "-R", f"{x},{y},{w},{h}",
                    str(OUT / f"{name}.png")], check=True)
    print(f"captured {name}.png")


def main():
    app = gui.App()
    app.update()
    time.sleep(0.8)  # let macOS finish mapping the window

    shoot(app, "step1_prace")

    app.show_step(2)
    shoot(app, "step2_klucz")

    app.show_step(3)
    shoot(app, "step3_przed")

    works = core.process(app.files, len(app.key_vars), cfg=app.cfg)
    app._finalize_processing(works)
    app.show_step(3)
    shoot(app, "step3_po")

    app.show_step(4)
    app._focus_results_table()
    app.update()
    time.sleep(0.3)
    shoot(app, "step4_wyniki")

    app.open_verify(0)
    shoot(app, "step5_weryfikacja")

    app.destroy()
    print(f"\nScreens saved to {OUT}")


if __name__ == "__main__":
    main()
