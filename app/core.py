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

# Optional format support: PDF via PyMuPDF, HEIC/GIF/multi-page TIFF via Pillow.
try:
    import pymupdf
except ImportError:
    try:
        import fitz as pymupdf  # older PyMuPDF releases
    except ImportError:
        pymupdf = None

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

HEIF_SUPPORTED = False
if PILImage is not None:
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
        HEIF_SUPPORTED = True
    except ImportError:
        pass

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

# Formats cv2.imdecode reads natively
CV2_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jp2",
                  ".pbm", ".pgm", ".ppm"}
TIFF_EXTENSIONS = {".tif", ".tiff"}
PIL_EXTRA_EXTENSIONS = {".gif"}
HEIF_EXTENSIONS = {".heic", ".heif"}
PDF_EXTENSIONS = {".pdf"}

# Render PDF pages to the template's pixel width (300 dpi A4 scan) regardless
# of the page's nominal size, so detection sees template-scale ink.
PDF_TARGET_WIDTH_PX = 2480


def supported_extensions() -> set[str]:
    exts = CV2_EXTENSIONS | TIFF_EXTENSIONS
    if pymupdf is not None:
        exts |= PDF_EXTENSIONS
    if PILImage is not None:
        exts |= PIL_EXTRA_EXTENSIONS
    if HEIF_SUPPORTED:
        exts |= HEIF_EXTENSIONS
    return exts

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


## SCAN SOURCES — one gradable sheet: an image file or a page of a multi-page file

@dataclass(frozen=True)
class ScanSource:
    path: Path
    page: int = 0   # 0-based page within the file
    pages: int = 1  # total pages in the file

    @property
    def display_name(self) -> str:
        if self.pages > 1:
            return f"{self.path.name} [str. {self.page + 1}]"
        return self.path.name


def page_count(path: Path) -> int:
    ext = path.suffix.lower()
    if ext in PDF_EXTENSIONS:
        if pymupdf is None:
            raise PipelineError("Obsługa PDF wymaga: pip install pymupdf")
        with pymupdf.open(path) as doc:
            if doc.needs_pass:
                raise PipelineError("PDF jest zaszyfrowany.")
            return doc.page_count
    if PILImage is not None and ext in (TIFF_EXTENSIONS | PIL_EXTRA_EXTENSIONS
                                        | HEIF_EXTENSIONS):
        try:
            with PILImage.open(path) as img:
                return getattr(img, "n_frames", 1)
        except Exception:
            return 1
    return 1


def expand_sources(paths) -> tuple[list[ScanSource], list[str]]:
    """Expands file paths into per-page scan sources.
    Returns (sources, skipped_descriptions)."""
    exts = supported_extensions()
    sources, skipped = [], []
    for path in paths:
        path = Path(path).resolve()
        ext = path.suffix.lower()
        if ext not in exts:
            if ext in PDF_EXTENSIONS:
                reason = "PDF wymaga: pip install pymupdf"
            elif ext in HEIF_EXTENSIONS:
                reason = "HEIC wymaga: pip install pillow-heif"
            else:
                reason = "nieobsługiwany format"
            skipped.append(f"{path.name} ({reason})")
            continue
        try:
            pages = page_count(path)
        except (PipelineError, OSError) as e:
            skipped.append(f"{path.name} ({e})")
            continue
        for page in range(pages):
            sources.append(ScanSource(path=path, page=page, pages=pages))
    return sources, skipped


def find_scan_sources(folder) -> list[ScanSource]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    exts = supported_extensions()
    paths = sorted(
        (Path(root) / name
         for root, _, names in os.walk(folder)
         for name in names
         if Path(name).suffix.lower() in exts),
        key=lambda p: str(p).lower())
    sources, _ = expand_sources(paths)
    return sources


def load_source_gray(source) -> np.ndarray:
    if isinstance(source, (str, Path)):
        source = ScanSource(path=Path(source))
    ext = source.path.suffix.lower()
    if ext in PDF_EXTENSIONS:
        return _load_pdf_page_gray(source.path, source.page)
    if ext in TIFF_EXTENSIONS | PIL_EXTRA_EXTENSIONS | HEIF_EXTENSIONS:
        if PILImage is not None:
            return _load_pil_gray(source.path, source.page)
        if ext in TIFF_EXTENSIONS:
            return imread_gray(source.path)  # first page only without Pillow
        raise PipelineError("Ten format wymaga: pip install Pillow")
    if ext in CV2_EXTENSIONS:
        return imread_gray(source.path)
    if PILImage is not None:
        return _load_pil_gray(source.path, source.page)
    raise PipelineError(f"Nieobsługiwany format: {ext}")


def _load_pdf_page_gray(path, page_index: int) -> np.ndarray:
    if pymupdf is None:
        raise PipelineError("Obsługa PDF wymaga: pip install pymupdf")
    with pymupdf.open(path) as doc:
        if doc.needs_pass:
            raise PipelineError("PDF jest zaszyfrowany.")
        page = doc[page_index]
        zoom = PDF_TARGET_WIDTH_PX / max(page.rect.width, 1)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom),
                              colorspace=pymupdf.csGRAY, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8)
        # Pixmap rows may be padded to the stride width
        return arr.reshape(pix.height, pix.stride)[:, :pix.width].copy()


def _load_pil_gray(path, page_index: int) -> np.ndarray:
    try:
        with PILImage.open(path) as img:
            if page_index and getattr(img, "n_frames", 1) > page_index:
                img.seek(page_index)
            return np.array(img.convert("L"))
    except Exception as e:
        raise PipelineError(f"Nie można wczytać obrazu: {e}")


def threshold_inv(gray: np.ndarray) -> np.ndarray:
    """Binarizes a scan into ink=255 / paper=0, robust to lighting.

    A fixed threshold fails on dark, washed-out or shadowed scans, so the
    paper background is estimated (morphological close removes ink, blur
    smooths it) and divided out, flattening gray paper and shadow gradients
    to uniform white. Otsu then separates ink adaptively.
    """
    denoised = cv2.medianBlur(gray, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    background = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (0, 0), sigmaX=15)
    background = np.maximum(background, 1)
    normalized = np.clip(denoised.astype(np.float32) / background * 255,
                         0, 255).astype(np.uint8)
    # Normalization pins paper at ~255, so a cut close to white picks up even
    # faint, washed-out pen strokes that a global Otsu split would miss. The
    # cut backs off below the paper-noise tail on grainy scans, where a fixed
    # high cut would read noise speckle as ink.
    paper = normalized[normalized > 200]
    sigma = float(paper.std()) if paper.size else 10.0
    cut = min(230.0, 255.0 - 4.0 * max(sigma, 3.0))
    _, thresh = cv2.threshold(normalized, int(cut), 255, cv2.THRESH_BINARY_INV)
    return thresh


## ALIGNMENT

class Aligner:
    """Matches scans against the template with ORB features and warps them
    into template space, so config box coordinates apply directly."""

    MIN_INLIERS = 12
    # Hamming distance below which an ORB match is considered reliable
    GOOD_MATCH_DISTANCE = 64

    def __init__(self, template_gray: np.ndarray):
        self.template = template_gray
        self.orb = cv2.ORB_create(nfeatures=3000)
        self.keypoints, self.descriptors = self.orb.detectAndCompute(template_gray, None)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def align(self, img_gray: np.ndarray) -> np.ndarray:
        # Resize the scan to template width first: feature matching then works
        # in a single scale regime regardless of the scan resolution.
        factor = self.template.shape[1] / img_gray.shape[1]
        if abs(factor - 1) > 0.05:
            interp = cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC
            img_gray = cv2.resize(img_gray, None, fx=factor, fy=factor,
                                  interpolation=interp)

        keypoints, descriptors = self.orb.detectAndCompute(img_gray, None)
        if descriptors is None or len(keypoints) < 4:
            raise AlignmentError("Nie znaleziono punktów charakterystycznych na skanie.")
        matches = self.matcher.match(self.descriptors, descriptors)
        if len(matches) < self.MIN_INLIERS:
            raise AlignmentError("Za mało dopasowań do szablonu.")

        # Garbage matches drag the homography off target — keep reliable ones
        matches = sorted(matches, key=lambda m: m.distance)
        good = [m for m in matches if m.distance <= self.GOOD_MATCH_DISTANCE]
        if len(good) < 30:
            good = matches[:max(30, len(matches) // 3)]

        points_template = np.float32([self.keypoints[m.queryIdx].pt for m in good])
        points_scan = np.float32([keypoints[m.trainIdx].pt for m in good])
        homography, mask = cv2.findHomography(points_scan, points_template,
                                              cv2.RANSAC, 5.0)
        inliers = int(mask.sum()) if mask is not None else 0
        if homography is None or inliers < self.MIN_INLIERS:
            raise AlignmentError("Za mało wiarygodnych dopasowań do szablonu.")

        # Reject degenerate fits (mirroring, wild scale) before they produce
        # a garbage warp that would be silently graded.
        det = (homography[0, 0] * homography[1, 1]
               - homography[0, 1] * homography[1, 0])
        if not 0.25 <= det <= 4.0:
            raise AlignmentError("Dopasowanie do szablonu odrzucone (zniekształcenie).")

        height, width = self.template.shape
        return cv2.warpPerspective(img_gray, homography, (width, height),
                                   borderValue=255)


## DETECTION

def is_marked(box: np.ndarray, box_w: int, box_h: int, factor: float) -> bool:
    # Sample only the inner core of the box: the printed outline sits at the
    # edge and, smeared by blur or rescaling, would otherwise read as a mark.
    margin = max(2, int(min(box_w, box_h) * 0.15))
    inner = box[margin:-margin, margin:-margin]
    if inner.size == 0:
        inner = box
    # In the inverted threshold image ink is 255; counting pixels below 255
    # counts paper. Less paper than factor*area means enough ink to be a mark.
    threshold = factor * inner.size
    paper_pixels = np.sum(inner < 255)
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
    source: ScanSource
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

    @property
    def name(self) -> str:
        return self.source.display_name


def process(sources, n_questions: int, cfg: dict | None = None,
            progress_cb=None, cancel_cb=None) -> list[WorkResult]:
    """Aligns, detects and annotates every scan. Per-scan errors are recorded
    in the WorkResult instead of aborting the whole batch."""
    if cfg is None:
        cfg = load_config()
    aligner = Aligner(load_template())
    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    ANALYZED_DIR.mkdir(parents=True, exist_ok=True)

    works = []
    total = len(sources)
    for i, source in enumerate(sources):
        if cancel_cb is not None and cancel_cb():
            break
        if isinstance(source, (str, Path)):
            source = ScanSource(path=Path(source))
        if progress_cb is not None:
            progress_cb(i, total, source.display_name)

        work = WorkResult(source=source)
        try:
            img = load_source_gray(source)
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


def process_paths(paths, n_questions: int, **kwargs) -> list[WorkResult]:
    sources, _ = expand_sources(paths)
    return process(sources, n_questions, **kwargs)


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
            lines.append(f"# BŁĄD: {work.name} — {work.error}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
