"""Core OMR pipeline: answer-key loading, scan alignment, mark detection, scoring.

Detection works on a THRESH_BINARY_INV image, so ink pixels are 255 and clean
paper is 0. A box counts as a final answer when it is marked AND the ring
around it is clean; a marked box with an inked ring was circled by the
student, which cancels the answer.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

MAX_QUESTIONS = 48
NUM_CHOICES = 4

# States of a single answer box
MARK_EMPTY = 0
MARK_COUNTED = 1    # marked by the student, counts towards the score
MARK_CANCELLED = 2  # marked but circled (withdrawn by the student)

# The ring around a box must contain at least this many clean-paper pixels,
# otherwise the surrounding ink is interpreted as a cancellation circle.
RING_CLEAN_PIXEL_THRESHOLD = 8000
RING_EXCLUSION_BORDER = 7

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Overlay colors (BGR)
LIGHT_GREEN = (0, 255, 0)
LIGHT_RED = (0, 0, 255)
GRAY = (128, 128, 128)


def project_root() -> Path:
    if getattr(sys, "frozen", False):  # PyInstaller bundle
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


ROOT = project_root()
CONFIG_JSON = ROOT / "config" / "config.json"
TEMPLATE_IMAGE = ROOT / "config" / "Template.jpg"
ANSWER_KEY_FILE = ROOT / "config" / "PoprawneOdpowiedzi.txt"
INPUT_DIR = ROOT / "dane" / "PraceDoSprawdzenia"
ALIGNED_DIR = ROOT / "dane" / "PraceZorientowane"
ANALYZED_DIR = ROOT / "dane" / "PracePrzeanalizowane"
RESULTS_FILE = ROOT / "dane" / "WynikiTestu.txt"


class PipelineError(Exception):
    """Recoverable error while processing a single scan."""


class AlignmentError(PipelineError):
    """The scan could not be matched against the template."""


## ANSWER KEY

def load_answer_key(path) -> list[list[int]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    matrix = []
    for line_number, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        if len(line) != NUM_CHOICES:
            raise ValueError(
                f"Linia {line_number} ma nieprawidłową długość (wymagane 4 znaki)."
            )
        if not all(char in "01" for char in line):
            raise ValueError(
                f"Linia {line_number} zawiera niedozwolone znaki (dozwolone tylko 0 i 1)."
            )
        matrix.append([int(char) for char in line])
    if not matrix:
        raise ValueError("Plik klucza nie zawiera żadnych odpowiedzi.")
    if len(matrix) > MAX_QUESTIONS:
        raise ValueError(f"Za dużo pytań ({len(matrix)}). Maksymalnie {MAX_QUESTIONS}.")
    return matrix


def save_answer_key(path, key: list[list[int]]) -> None:
    text = "\n".join("".join(str(v) for v in row) for row in key)
    Path(path).write_text(text + "\n", encoding="utf-8")


def max_points(key: list[list[int]]) -> int:
    return sum(sum(row) for row in key)


## CONFIG AND IMAGE LOADING

def load_config(path=CONFIG_JSON) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def imread_gray(path) -> np.ndarray:
    # np.fromfile + imdecode handles non-ASCII paths, unlike cv2.imread
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise PipelineError("Nie można wczytać obrazu.")
    return img


def load_template() -> np.ndarray:
    return imread_gray(TEMPLATE_IMAGE)


def find_image_files(folder) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    files = [
        Path(root) / name
        for root, _, names in os.walk(folder)
        for name in names
        if Path(name).suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files, key=lambda p: str(p).lower())


def threshold_inv(gray: np.ndarray) -> np.ndarray:
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    return thresh


## ALIGNMENT

class Aligner:
    """Matches scans against the template with ORB features and warps them
    into template space, so config box coordinates apply directly."""

    def __init__(self, template_gray: np.ndarray):
        self.template = template_gray
        self.orb = cv2.ORB_create()
        self.keypoints, self.descriptors = self.orb.detectAndCompute(template_gray, None)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def align(self, img_gray: np.ndarray) -> np.ndarray:
        keypoints, descriptors = self.orb.detectAndCompute(img_gray, None)
        if descriptors is None or len(keypoints) == 0:
            raise AlignmentError("Nie znaleziono punktów charakterystycznych na skanie.")
        matches = self.matcher.match(self.descriptors, descriptors)
        if len(matches) < 4:
            raise AlignmentError("Za mało dopasowań do szablonu.")
        points_template = np.float32([self.keypoints[m.queryIdx].pt for m in matches])
        points_scan = np.float32([keypoints[m.trainIdx].pt for m in matches])
        homography, _ = cv2.findHomography(points_scan, points_template, cv2.RANSAC)
        if homography is None:
            raise AlignmentError("Nie udało się dopasować skanu do szablonu.")
        height, width = self.template.shape
        return cv2.warpPerspective(img_gray, homography, (width, height))


## DETECTION

def is_marked(box: np.ndarray, box_w: int, box_h: int, factor: float) -> bool:
    # In the inverted threshold image ink is 255; counting pixels below 255
    # counts paper. Less paper than factor*area means enough ink to be a mark.
    threshold = factor * box_w * box_h
    paper_pixels = np.sum(box < 255)
    return paper_pixels < threshold


def ring_is_clean(thresh: np.ndarray, center_x: int, center_y: int,
                  box_size: int, proximity: int,
                  clean_threshold: int = RING_CLEAN_PIXEL_THRESHOLD) -> bool:
    x_start = max(center_x - (box_size // 2 + proximity), 0)
    y_start = max(center_y - (box_size // 2 + proximity), 0)
    x_end = min(center_x + (box_size // 2 + proximity), thresh.shape[1])
    y_end = min(center_y + (box_size // 2 + proximity), thresh.shape[0])
    region = thresh[y_start:y_end, x_start:x_end]

    # Mask out the answer box itself (with a safety border) so only the ring counts
    mask = np.zeros(region.shape, dtype=np.uint8)
    ex_start = proximity - RING_EXCLUSION_BORDER
    ey_start = proximity - RING_EXCLUSION_BORDER
    ex_end = ex_start + box_size + RING_EXCLUSION_BORDER * 2
    ey_end = ey_start + box_size + RING_EXCLUSION_BORDER * 2
    cv2.rectangle(mask, (ex_start, ey_start), (ex_end, ey_end), 255, -1)

    paper_pixels_in_ring = np.sum(region[mask == 0] < 128)
    return paper_pixels_in_ring > clean_threshold


def detect_marks(thresh: np.ndarray, cfg: dict, n_questions: int) -> np.ndarray:
    """Returns an (n_questions, 4) matrix of MARK_* states."""
    ip = cfg["image_processing"]
    box_w, box_h = ip["box_size"]
    factor = ip["marking_threshold_factor"]
    proximity = ip["circle_proximity_range"]

    marks = np.zeros((n_questions, NUM_CHOICES), dtype=np.int8)
    for question in cfg["questions"]:
        q = question["number"] - 1
        if q >= n_questions:
            continue
        for choice in question["choices"]:
            c = ord(choice["label"]) - ord("A")
            center_x, center_y = choice["center"]
            x_start = center_x - box_w // 2
            y_start = center_y - box_h // 2
            box = thresh[y_start:y_start + box_h, x_start:x_start + box_w]
            if not is_marked(box, box_w, box_h, factor):
                continue
            if ring_is_clean(thresh, center_x, center_y, box_w, proximity):
                marks[q, c] = MARK_COUNTED
            else:
                marks[q, c] = MARK_CANCELLED
    return marks


def detect_student_id(thresh: np.ndarray, cfg: dict):
    """Reads the 7-column index grid. Returns (id_string, ok, detected_boxes)."""
    idx = cfg["index"]
    grid_x, grid_y = idx["starting_box_position"]
    box_w, box_h = idx["box_size"]
    offset_x, offset_y = idx["offset"]
    factor = cfg["image_processing"]["marking_threshold_factor"]

    student_id = ""
    boxes = []
    for col in range(7):
        for row in range(10):
            x = grid_x + col * offset_x - box_w // 2
            y = grid_y + row * offset_y - box_h // 2
            box = thresh[y:y + box_h, x:x + box_w]
            if is_marked(box, box_w, box_h, factor):
                digit = row + 1
                if digit == 10:
                    digit = 0
                student_id += str(digit)
                boxes.append((x, y, box_w, box_h))
                break
        else:
            student_id += " "

    student_id = student_id.strip()
    ok = " " not in student_id and len(student_id) >= 6
    return (student_id if ok else "------"), ok, boxes


## SCORING

def score_marks(key: list[list[int]], marks: np.ndarray) -> int:
    # +1 for each correctly marked answer, -1 for each wrongly marked one;
    # the running score is clamped at 0 after every question (original behavior).
    score = 0
    for key_row, marks_row in zip(key, marks):
        for key_cell, mark in zip(key_row, marks_row):
            marked = 1 if mark == MARK_COUNTED else 0
            if marked == 1 and key_cell == 0:
                score -= 1
            elif key_cell == 1 and marked == 1:
                score += 1
        score = max(0, score)
    return score


## ANNOTATION

def overlay_rectangle(img, top_left, bottom_right, color, opacity=0.7):
    overlay = img.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)


def annotate_work(aligned_gray: np.ndarray, marks: np.ndarray, cfg: dict,
                  n_questions: int, id_boxes=()) -> np.ndarray:
    """Renders the verification image: green = counted, red = cancelled, gray = empty."""
    box_w, box_h = cfg["image_processing"]["box_size"]
    color_img = cv2.cvtColor(aligned_gray, cv2.COLOR_GRAY2BGR)

    for x, y, w, h in id_boxes:
        overlay_rectangle(color_img, (x, y), (x + w, y + h), LIGHT_GREEN)

    for question in cfg["questions"]:
        q = question["number"] - 1
        if q >= n_questions:
            continue
        for choice in question["choices"]:
            c = ord(choice["label"]) - ord("A")
            center_x, center_y = choice["center"]
            x_start = center_x - box_w // 2
            y_start = center_y - box_h // 2
            x_end = x_start + box_w
            y_end = y_start + box_h
            state = marks[q, c]
            if state == MARK_CANCELLED:
                overlay_rectangle(color_img, (x_start, y_start), (x_end, y_end), LIGHT_RED)
            elif state == MARK_COUNTED:
                overlay_rectangle(color_img, (x_start, y_start), (x_end, y_end), LIGHT_GREEN)
            else:
                overlay_rectangle(color_img, (x_start, y_start), (x_end, y_end), GRAY)
    return color_img


## BATCH PROCESSING

@dataclass
class WorkResult:
    path: Path
    error: str | None = None
    marks: np.ndarray | None = None
    student_id: str = "------"
    id_ok: bool = False
    id_boxes: list = field(default_factory=list)
    aligned_path: Path | None = None
    edited: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None


def process(files, n_questions: int, cfg: dict | None = None,
            progress_cb=None, cancel_cb=None) -> list[WorkResult]:
    """Aligns, detects and annotates every scan. Per-file errors are recorded
    in the WorkResult instead of aborting the whole batch."""
    if cfg is None:
        cfg = load_config()
    aligner = Aligner(load_template())
    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    ANALYZED_DIR.mkdir(parents=True, exist_ok=True)

    works = []
    total = len(files)
    for i, path in enumerate(files):
        if cancel_cb is not None and cancel_cb():
            break
        if progress_cb is not None:
            progress_cb(i, total, Path(path).name)

        work = WorkResult(path=Path(path))
        try:
            img = imread_gray(path)
            aligned = aligner.align(img)
            thresh = threshold_inv(aligned)
            work.marks = detect_marks(thresh, cfg, n_questions)
            work.student_id, work.id_ok, work.id_boxes = detect_student_id(thresh, cfg)

            work.aligned_path = ALIGNED_DIR / f"aligned_image_{i}.jpg"
            cv2.imwrite(str(work.aligned_path), aligned)
            annotated = annotate_work(aligned, work.marks, cfg, n_questions, work.id_boxes)
            cv2.imwrite(str(ANALYZED_DIR / f"corrected_and_detected_{i}.jpg"), annotated)
        except PipelineError as e:
            work.error = str(e)
        except Exception as e:  # malformed scan must not kill the batch
            work.error = f"{type(e).__name__}: {e}"
        works.append(work)
    return works


## RESULTS OUTPUT

def write_results_file(path, works: list[WorkResult], key: list[list[int]]) -> None:
    maximum = max_points(key)
    lines = [
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        f"Liczba pytań: {len(key)}",
        f"Max punktów: {maximum}",
        "Wyniki:",
    ]
    for work in works:
        if work.ok:
            score = score_marks(key, work.marks)
            pct = (score / maximum * 100) if maximum else 0.0
            lines.append(f"{work.student_id}: {score}, {pct:.2f}%")
        else:
            lines.append(f"# BŁĄD: {work.path.name} — {work.error}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
