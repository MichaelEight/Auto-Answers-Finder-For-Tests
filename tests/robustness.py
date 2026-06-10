"""Robustness suite — stress-tests the OMR pipeline against scan distortions.

Generates distorted variants of a known-good scan (rotation, scale, noise,
lighting, perspective, paper damage), runs alignment + detection on each and
compares the detected mark matrix and student id against the pristine
baseline. Inputs and `__detected` overlays land in tests/robustness_cases/,
together with report.txt.

Run with: python3 tests/robustness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from app import core

BASE_SCAN = core.INPUT_DIR / "EXAMPLE (1).jpg"
OUT_DIR = Path(__file__).resolve().parent / "robustness_cases"
SEED = 42


## DISTORTIONS

def rotate(img, deg, border=255):
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - center[0]
    m[1, 2] += nh / 2 - center[1]
    return cv2.warpAffine(img, m, (nw, nh), borderValue=border)


def scale(img, factor):
    interp = cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=interp)


def gauss_noise(img, sigma):
    rng = np.random.default_rng(SEED)
    noisy = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def salt_pepper(img, fraction):
    rng = np.random.default_rng(SEED)
    out = img.copy()
    mask = rng.random(img.shape)
    out[mask < fraction / 2] = 0
    out[mask > 1 - fraction / 2] = 255
    return out


def blur(img, k):
    return cv2.GaussianBlur(img, (k, k), 0)


def brightness(img, delta):
    return np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)


def gray_paper(img, factor):
    # Multiplicative darkening: white paper turns gray, ink stays dark
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def low_contrast(img, factor):
    return np.clip((img.astype(np.float32) - 128) * factor + 128, 0, 255).astype(np.uint8)


def jpeg_artifacts(img, quality):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def perspective(img, strength):
    # Keystone: top edge pinched, like photographing the sheet at an angle
    h, w = img.shape
    dx = w * strength
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[dx, 0], [w - dx, 0], [w, h], [0, h]])
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (w, h), borderValue=255)


def shadow(img, strength):
    # Horizontal lighting falloff across the page
    h, w = img.shape
    grad = np.linspace(1.0, 1.0 - strength, w, dtype=np.float32)[None, :]
    return np.clip(img * grad, 0, 255).astype(np.uint8)


def on_desk(img, margin=0.15, bg=60):
    # Photographed on a dark desk: dark border around the sheet
    h, w = img.shape
    mh, mw = int(h * margin), int(w * margin)
    canvas = np.full((h + 2 * mh, w + 2 * mw), bg, dtype=np.uint8)
    canvas[mh:mh + h, mw:mw + w] = img
    return canvas


def creases(img, count=3):
    # Dark fold lines across the page
    rng = np.random.default_rng(SEED)
    out = img.copy()
    h, w = img.shape
    for _ in range(count):
        x0, x1 = rng.integers(0, w, 2)
        cv2.line(out, (int(x0), 0), (int(x1), h - 1), 90, 3)
    return out


def torn_corner(img, frac=0.12):
    # Missing corner shows the dark scanner background
    out = img.copy()
    h, w = img.shape
    pts = np.array([[w, 0], [w - int(w * frac), 0], [w, int(h * frac)]])
    cv2.fillPoly(out, [pts], 30)
    return out


CASES = [
    # rotation
    ("rot_+2", lambda i: rotate(i, 2)),
    ("rot_-5", lambda i: rotate(i, -5)),
    ("rot_+10", lambda i: rotate(i, 10)),
    ("rot_-15", lambda i: rotate(i, -15)),
    ("rot_+30", lambda i: rotate(i, 30)),
    ("rot_+45", lambda i: rotate(i, 45)),
    ("rot_90", lambda i: rotate(i, 90)),
    ("rot_180", lambda i: rotate(i, 180)),
    # scale
    ("scale_0.30", lambda i: scale(i, 0.30)),
    ("scale_0.50", lambda i: scale(i, 0.50)),
    ("scale_0.75", lambda i: scale(i, 0.75)),
    ("scale_1.50", lambda i: scale(i, 1.50)),
    ("scale_2.00", lambda i: scale(i, 2.00)),
    # noise
    ("noise_g10", lambda i: gauss_noise(i, 10)),
    ("noise_g25", lambda i: gauss_noise(i, 25)),
    ("noise_g40", lambda i: gauss_noise(i, 40)),
    ("saltpepper_1%", lambda i: salt_pepper(i, 0.01)),
    ("saltpepper_5%", lambda i: salt_pepper(i, 0.05)),
    # focus / compression
    ("blur_k5", lambda i: blur(i, 5)),
    ("blur_k9", lambda i: blur(i, 9)),
    ("jpeg_q15", lambda i: jpeg_artifacts(i, 15)),
    # lighting / paper tone
    ("dark_-80", lambda i: brightness(i, -80)),
    ("dark_-130", lambda i: brightness(i, -130)),
    ("bright_+70", lambda i: brightness(i, 70)),
    ("bright_+110", lambda i: brightness(i, 110)),
    ("graypaper_0.70", lambda i: gray_paper(i, 0.70)),
    ("graypaper_0.45", lambda i: gray_paper(i, 0.45)),
    ("lowcontrast_0.5", lambda i: low_contrast(i, 0.5)),
    ("shadow_30%", lambda i: shadow(i, 0.30)),
    ("shadow_60%", lambda i: shadow(i, 0.60)),
    # geometry / surroundings / damage
    ("perspective_3%", lambda i: perspective(i, 0.03)),
    ("perspective_6%", lambda i: perspective(i, 0.06)),
    ("on_dark_desk", lambda i: on_desk(i)),
    ("creases", lambda i: creases(i)),
    ("torn_corner", lambda i: torn_corner(i)),
    # realistic combos
    ("combo_rot10+noise25", lambda i: gauss_noise(rotate(i, 10), 25)),
    ("combo_scale0.5+blur5", lambda i: blur(scale(i, 0.5), 5)),
    ("combo_phone_photo", lambda i: gauss_noise(
        shadow(perspective(on_desk(i, 0.08), 0.04), 0.35), 12)),
    ("combo_rot180+dark", lambda i: brightness(rotate(i, 180), -80)),
]


## EVALUATION

def detect_on(img, aligner, cfg, n_questions):
    aligned = aligner.align(img)
    thresh = core.threshold_inv(aligned)
    marks = core.detect_marks(thresh, cfg, n_questions)
    student_id, id_ok, id_boxes = core.detect_student_id(thresh, cfg)
    return aligned, marks, student_id, id_boxes


def main() -> int:
    cfg = core.load_config()
    key = core.load_answer_key(core.ANSWER_KEY_FILE)
    n = len(key)
    aligner = core.Aligner(core.load_template())
    base = core.imread_gray(BASE_SCAN)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Baseline on the pristine scan defines the expected detection
    _, expected_marks, expected_id, _ = detect_on(base, aligner, cfg, n)
    expected_score = core.score_marks(key, expected_marks)
    assert expected_id == "123456" and expected_score == 5, \
        f"baseline drifted: id={expected_id} score={expected_score}"
    print(f"Baseline: {BASE_SCAN.name} -> id={expected_id}, "
          f"{expected_score} pkt (marks zapisane jako wzorzec)\n")

    rows = []
    failures = 0
    for idx, (name, distort) in enumerate(CASES, 1):
        img = distort(base)
        case_path = OUT_DIR / f"{idx:02d}_{name}.jpg"
        cv2.imwrite(str(case_path), img)

        try:
            aligned, marks, student_id, id_boxes = detect_on(img, aligner, cfg, n)
        except core.PipelineError as e:
            rows.append((name, "ALIGN FAIL", "—", "—", "—", str(e)))
            failures += 1
            continue

        overlay = core.annotate_work(aligned, marks, cfg, n, id_boxes)
        cv2.imwrite(str(OUT_DIR / f"{idx:02d}_{name}__detected.jpg"), overlay)

        mismatches = int(np.sum(marks != expected_marks))
        score = core.score_marks(key, marks)
        id_match = student_id == expected_id
        if mismatches == 0 and id_match:
            rows.append((name, "PASS", student_id, score, 0, ""))
        else:
            failures += 1
            problems = []
            if not id_match:
                problems.append(f"indeks {student_id!r}")
            if mismatches:
                problems.append(f"{mismatches} pól różni się")
            rows.append((name, "FAIL", student_id, score, mismatches,
                         ", ".join(problems)))

    header = f"{'przypadek':<24} {'wynik':<11} {'indeks':<9} {'pkt':>4} {'≠pól':>5}  uwagi"
    lines = [f"Baza: {BASE_SCAN.name} (id={expected_id}, {expected_score} pkt), "
             f"przypadków: {len(CASES)}", "", header, "-" * len(header)]
    for name, status, student_id, score, mism, note in rows:
        lines.append(f"{name:<24} {status:<11} {str(student_id):<9} "
                     f"{str(score):>4} {str(mism):>5}  {note}")
    lines += ["", f"PASS: {len(CASES) - failures} / {len(CASES)}   "
                  f"FAIL: {failures} / {len(CASES)}"]

    report = "\n".join(lines)
    print(report)
    (OUT_DIR / "report.txt").write_text(report + "\n", encoding="utf-8")
    print(f"\nPrzypadki i nakładki __detected: {OUT_DIR}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
