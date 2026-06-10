"""Unit tests for app.core — plain asserts, run with: python3 tests/test_core.py"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app import core


def test_load_answer_key():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("1010\n0001\n\n")
        path = f.name
    assert core.load_answer_key(path) == [[1, 0, 1, 0], [0, 0, 0, 1]]

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("101\n")
        path = f.name
    try:
        core.load_answer_key(path)
        assert False, "expected ValueError for wrong length"
    except ValueError:
        pass

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("10a0\n")
        path = f.name
    try:
        core.load_answer_key(path)
        assert False, "expected ValueError for bad char"
    except ValueError:
        pass


def test_max_points():
    assert core.max_points([[1, 0, 1, 0], [0, 1, 1, 1]]) == 5
    assert core.max_points([]) == 0


def test_score_marks():
    C, X, E = core.MARK_COUNTED, core.MARK_CANCELLED, core.MARK_EMPTY
    key = [[1, 0, 0, 0]]
    assert core.score_marks(key, np.array([[C, E, E, E]])) == 1   # correct
    assert core.score_marks(key, np.array([[E, C, E, E]])) == 0   # wrong, clamped
    assert core.score_marks(key, np.array([[C, C, E, E]])) == 0   # +1 -1
    assert core.score_marks(key, np.array([[X, E, E, E]])) == 0   # cancelled ignored
    # Running clamp happens after every row (original behavior)
    key2 = [[0, 0, 0, 0], [1, 0, 0, 0]]
    marks2 = np.array([[C, E, E, E], [C, E, E, E]])
    assert core.score_marks(key2, marks2) == 1  # row1: -1 -> 0, row2: +1


def test_is_marked():
    # In the inverted threshold image: ink = 255, paper = 0
    paper = np.zeros((54, 54), dtype=np.uint8)
    ink = np.full((54, 54), 255, dtype=np.uint8)
    assert not core.is_marked(paper, 54, 54, 0.9)
    assert core.is_marked(ink, 54, 54, 0.9)


def test_ring_and_detection():
    cfg = {
        "image_processing": {
            "box_size": [54, 54],
            "marking_threshold_factor": 0.9,
            "circle_proximity_range": 30,
        },
        "questions": [{
            "number": 1,
            "choices": [
                {"label": "A", "center": [100, 100]},
                {"label": "B", "center": [300, 100]},
                {"label": "C", "center": [100, 300]},
                {"label": "D", "center": [300, 300]},
            ],
        }],
    }
    thresh = np.zeros((400, 400), dtype=np.uint8)
    # A: inked box, clean ring -> counted
    thresh[73:127, 73:127] = 255
    # B: inked box AND inked ring (circled) -> cancelled
    thresh[43:157, 243:357] = 255
    # C, D: empty

    assert core.ring_is_clean(thresh, 100, 100, 54, 30)
    assert not core.ring_is_clean(thresh, 300, 100, 54, 30)

    marks = core.detect_marks(thresh, cfg, 1)
    assert marks[0, 0] == core.MARK_COUNTED
    assert marks[0, 1] == core.MARK_CANCELLED
    assert marks[0, 2] == core.MARK_EMPTY
    assert marks[0, 3] == core.MARK_EMPTY


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    main()
