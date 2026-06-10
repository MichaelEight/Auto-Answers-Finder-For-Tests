"""Auto Answers Finder For Tests — main runner.

Uruchomienie bez argumentów otwiera aplikację okienkową.
Tryb konsolowy (dawne zachowanie skryptu): python main.py --cli
"""

import sys


def main() -> int:
    if "--cli" in sys.argv[1:]:
        from app.cli import run_cli
        return run_cli()
    from app.gui import run_gui
    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
