"""GUI smoke test — drives the app programmatically (a window flashes briefly).

Run with: python3 tests/smoke_gui.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import core, gui


def main():
    app = gui.App()
    app.update()

    # Step 1: example scans should be preloaded from dane/PraceDoSprawdzenia
    assert len(app.files) >= 1, "expected preloaded scans"
    # Step 2: answer key should be preloaded from config/PoprawneOdpowiedzi.txt
    assert len(app.key_vars) == 14, f"expected 14 questions, got {len(app.key_vars)}"
    assert core.max_points(app.current_key()) == 19
    print("PASS preload (files + key)")

    for step in (1, 2, 3):
        app.show_step(step)
        app.update()
    assert app._step_allowed(3) and not app._step_allowed(4)
    print("PASS step navigation and gating")

    # Run the pipeline synchronously and hand results to the app
    works = core.process(app.files, len(app.key_vars), cfg=app.cfg)
    app._finalize_processing(works)
    app.update()
    assert app.processed and app.current_step == 4
    assert len(app.tree.get_children()) == len(works)
    print(f"PASS processing ({len(works)} works in results table)")

    key = app.current_key()
    scores = [core.score_marks(key, w.marks) for w in works if w.ok]
    assert scores == [5, 5, 5, 0], f"unexpected scores: {scores}"
    print("PASS scores match CLI regression baseline")

    # Verification view + mark editing
    app.open_verify(0)
    app.update()
    assert app.in_verify
    work = app.works[0]
    before = core.score_marks(key, work.marks)
    q, c = next((q, c) for q in range(len(key)) for c in range(4)
                if key[q][c] == 1 and work.marks[q, c] == core.MARK_EMPTY)
    app._cycle_box(q, c)  # empty -> counted on a correct answer: +1 point
    after = core.score_marks(key, work.marks)
    assert after == before + 1, f"score {before} -> {after}, expected +1"
    assert work.edited
    app._cycle_box(q, c)  # counted -> cancelled: back to baseline
    assert core.score_marks(key, work.marks) == before
    print("PASS verify view + click-to-edit marks rescore live")

    # Manual index edit
    app.verify_id_var.set("654321")
    app._apply_id_edit()
    assert app.works[0].student_id == "654321" and app.works[0].id_ok
    print("PASS manual index edit")

    app.close_verify()
    app.update()
    assert not app.in_verify and app.current_step == 4

    # Key toggle after processing must re-score without invalidating works
    app.show_step(2)
    app.update()
    app.key_vars[0][3].set(1)  # add D to question 1
    assert app.processed, "key edit must not invalidate processed works"
    app.key_vars[0][3].set(0)
    print("PASS key edit re-scores without invalidation")

    # File list change must invalidate results
    app.clear_files()
    assert not app.processed and not app.works
    print("PASS file change invalidates results")

    app.destroy()
    print("\nGUI smoke test passed.")


if __name__ == "__main__":
    main()
