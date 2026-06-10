"""Desktop GUI — a guided 4-step grading workflow for teachers.

Steps: ① add scans → ② set the answer key → ③ run the check → ④ review
results, with a full-screen verification view where detected marks can be
corrected by clicking the boxes on the scan.
"""

from __future__ import annotations

import base64
import queue
import threading
import tkinter as tk
import sys
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from . import core

# tkinterdnd2 may import fine yet fail to load its tkdnd binary against the
# running Tk build (e.g. conda Tk), so probe with a throwaway root window.
_TK_BASE = tk.Tk
HAS_DND = False
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except Exception:
    TkinterDnD = None
if TkinterDnD is not None:
    try:
        _probe = TkinterDnD.Tk()
        _probe.withdraw()
        _probe.destroy()
        _TK_BASE = TkinterDnD.Tk
        HAS_DND = True
    except Exception:
        # A failed probe can leave a half-built window registered as the
        # default root; clean it up, otherwise variables created without an
        # explicit master would bind to a dead interpreter and widgets would
        # render desynchronized from their values.
        try:
            if tk._default_root is not None:
                tk._default_root.destroy()
        except Exception:
            pass
        tk._default_root = None

ACCENT = "#1565c0"
ACCENT_DARK = "#0d47a1"
OK_GREEN = "#2e7d32"
WARN_ORANGE = "#e65100"
ERR_RED = "#c62828"
BG = "#f5f6fa"
CARD_BG = "#ffffff"
MUTED = "#5f6368"

STEP_TITLES = ["1. Prace", "2. Klucz odpowiedzi", "3. Sprawdzanie", "4. Wyniki"]

# Mark-state cycle on click: empty -> counted -> cancelled -> empty
NEXT_STATE = {
    core.MARK_EMPTY: core.MARK_COUNTED,
    core.MARK_COUNTED: core.MARK_CANCELLED,
    core.MARK_CANCELLED: core.MARK_EMPTY,
}

STATE_COLORS = {
    core.MARK_COUNTED: OK_GREEN,
    core.MARK_CANCELLED: ERR_RED,
    core.MARK_EMPTY: "#9e9e9e",
}

ZOOM_MIN, ZOOM_MAX, ZOOM_STEP = 0.1, 1.0, 1.25


class App(_TK_BASE):
    def __init__(self):
        super().__init__()
        self.title("Sprawdzarka testów — Auto Answers Finder")
        self.geometry("1280x840")
        self.minsize(1020, 700)
        self.configure(bg=BG)

        try:
            self.cfg = core.load_config()
        except Exception as e:
            messagebox.showerror(
                "Błąd konfiguracji",
                f"Nie można wczytać {core.CONFIG_JSON}:\n{e}", parent=self)
            raise SystemExit(1)

        # ---- state
        self.files: list[Path] = []
        self.works: list[core.WorkResult] = []
        self.processed = False
        self.current_step = 1
        self.in_verify = False
        self.verify_index = 0
        self.zoom = 0.25
        self._verify_gray: np.ndarray | None = None
        self._verify_photo = None
        self._preview_photo = None
        self._preview_cache_idx = -1
        self._loading_key = False
        self._drag_moved = False
        self._queue: queue.Queue = queue.Queue()
        self._cancel = threading.Event()
        self._poll_id = None
        self.key_vars: list[list[tk.IntVar]] = []
        self.show_key_var = tk.BooleanVar(master=self, value=True)

        self._build_style()
        self._build_header()
        self._build_content()
        self._build_footer()
        self._bind_shortcuts()
        self._setup_dnd()

        self._preload_inputs()
        self.show_step(1)

        # Bring the window to front when launched from a terminal/bat file
        self.lift()
        self.attributes("-topmost", True)
        self.after(600, lambda: self.attributes("-topmost", False))
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------- style/layout

    def _build_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background=BG, font=("TkDefaultFont", 12))
        style.configure("Card.TFrame", background=CARD_BG)
        style.configure("TLabel", background=BG)
        style.configure("Card.TLabel", background=CARD_BG)
        style.configure("Muted.TLabel", foreground=MUTED, background=BG)
        style.configure("CardMuted.TLabel", foreground=MUTED, background=CARD_BG)
        style.configure("H1.TLabel", font=("TkDefaultFont", 17, "bold"), background=BG)
        style.configure("Step.TLabel", font=("TkDefaultFont", 13), foreground=MUTED,
                        background=BG, padding=(10, 8))
        style.configure("StepActive.TLabel", font=("TkDefaultFont", 13, "bold"),
                        foreground=ACCENT, background=BG, padding=(10, 8))
        style.configure("StepDone.TLabel", font=("TkDefaultFont", 13),
                        foreground=OK_GREEN, background=BG, padding=(10, 8))
        style.configure("TButton", padding=(12, 7))
        style.configure("Big.TButton", font=("TkDefaultFont", 14, "bold"), padding=(22, 12))
        style.configure("Accent.TButton", font=("TkDefaultFont", 13, "bold"),
                        foreground="#ffffff", background=ACCENT, padding=(16, 9))
        style.map("Accent.TButton",
                  background=[("active", ACCENT_DARK), ("disabled", "#b0bec5")])
        style.configure("Key.TCheckbutton", font=("TkDefaultFont", 12, "bold"),
                        background=CARD_BG, padding=6)
        style.configure("Treeview", rowheight=30, font=("TkDefaultFont", 12))
        style.configure("Treeview.Heading", font=("TkDefaultFont", 12, "bold"))

    def _build_header(self):
        bar = ttk.Frame(self, padding=(16, 12, 16, 4))
        bar.pack(fill="x")
        self.step_labels = []
        for i, title in enumerate(STEP_TITLES, 1):
            if i > 1:
                ttk.Label(bar, text="▸", style="Step.TLabel").pack(side="left")
            lbl = ttk.Label(bar, text=title, style="Step.TLabel", cursor="hand2")
            lbl.pack(side="left")
            lbl.bind("<Button-1>", lambda _e, step=i: self.try_goto(step))
            self.step_labels.append(lbl)

    def _build_content(self):
        self.content = ttk.Frame(self, padding=(16, 4))
        self.content.pack(fill="both", expand=True)
        self.frames = {}
        for builder, key in ((self._build_step1, 1), (self._build_step2, 2),
                             (self._build_step3, 3), (self._build_step4, 4),
                             (self._build_verify, "verify")):
            frame = ttk.Frame(self.content)
            frame.grid(row=0, column=0, sticky="nsew")
            builder(frame)
            self.frames[key] = frame
        self.content.rowconfigure(0, weight=1)
        self.content.columnconfigure(0, weight=1)

    def _build_footer(self):
        bar = ttk.Frame(self, padding=(16, 8, 16, 14))
        bar.pack(fill="x", side="bottom")
        self.hint_label = ttk.Label(bar, text="", style="Muted.TLabel")
        self.hint_label.pack(side="left")
        self.next_btn = ttk.Button(bar, text="Dalej ▶", style="Accent.TButton",
                                   command=self.go_next)
        self.next_btn.pack(side="right")
        self.back_btn = ttk.Button(bar, text="◀ Wstecz", command=self.go_back)
        self.back_btn.pack(side="right", padx=(0, 10))

    def _bind_shortcuts(self):
        self.bind("<Return>", self._on_enter_key)
        self.bind("<Escape>", lambda _e: self._on_escape())
        self.bind("<Left>", lambda _e: self._on_arrow(-1))
        self.bind("<Right>", lambda _e: self._on_arrow(+1))

    def _setup_dnd(self):
        if not HAS_DND:
            return
        self.drop_target_register(DND_FILES)
        self.dnd_bind("<<Drop>>", self._on_drop)

    # ------------------------------------------------------------- navigation

    def _step_allowed(self, step) -> bool:
        if step in (1, 2):
            return True
        if step == 3:
            return bool(self.files) and bool(self.key_vars)
        return self.processed  # step 4

    def try_goto(self, step):
        if self._step_allowed(step):
            self.show_step(step)
        else:
            self.bell()
            if step == 3:
                self.set_hint("Najpierw dodaj prace (krok 1) i ustaw klucz (krok 2).")
            elif step == 4:
                self.set_hint("Najpierw uruchom sprawdzanie (krok 3).")

    def show_step(self, step):
        self.in_verify = False
        self.current_step = step
        self.frames[step].tkraise()
        for i, lbl in enumerate(self.step_labels, 1):
            if i == step:
                lbl.configure(style="StepActive.TLabel")
            elif self._step_allowed(i):
                lbl.configure(style="StepDone.TLabel")
            else:
                lbl.configure(style="Step.TLabel")

        hints = {
            1: ("Przeciągnij pliki na okno lub użyj przycisków. "
                "Delete usuwa zaznaczone. Enter = dalej.") if HAS_DND else
               ("Dodaj pliki przyciskami. Delete usuwa zaznaczone. Enter = dalej."),
            2: "Kliknij litery A–D, aby oznaczyć poprawne odpowiedzi. Enter = dalej.",
            3: "Enter uruchamia sprawdzanie.",
            4: "Podwójne kliknięcie lub Enter na pracy otwiera weryfikację. ↑/↓ wybiera pracę.",
        }
        self.set_hint(hints[step])

        if step == 3:
            self._refresh_summary()
        if step == 4:
            self._focus_results_table()
        self._update_nav()

    def _update_nav(self):
        if self.in_verify:
            self.back_btn.pack_forget()
            self.next_btn.pack_forget()
            return
        self.back_btn.pack(side="right", padx=(0, 10))
        self.next_btn.pack(side="right")
        self.back_btn.state(["!disabled"] if self.current_step > 1 else ["disabled"])
        if self.current_step == 4:
            self.next_btn.state(["disabled"])
        else:
            allowed = self._step_allowed(self.current_step + 1)
            self.next_btn.state(["!disabled"] if allowed else ["disabled"])

    def go_next(self):
        if self.current_step < 4:
            self.try_goto(self.current_step + 1)

    def go_back(self):
        if self.in_verify:
            self.close_verify()
        elif self.current_step > 1:
            self.show_step(self.current_step - 1)

    def set_hint(self, text):
        self.hint_label.configure(text=text)

    def _on_enter_key(self, event):
        widget = event.widget
        if isinstance(widget, (tk.Entry, ttk.Entry, tk.Text, ttk.Spinbox)):
            return
        if self.in_verify:
            return
        if self.current_step == 3:
            if self.start_btn.instate(["!disabled"]):
                self.start_processing()
        elif self.current_step == 4:
            self.open_selected_verify()
        else:
            self.go_next()

    def _on_escape(self):
        if self.in_verify:
            self.close_verify()

    def _on_arrow(self, direction):
        if self.in_verify:
            self.verify_show(self.verify_index + direction)

    def _on_close(self):
        self._cancel.set()
        if self._poll_id is not None:
            self.after_cancel(self._poll_id)
        self.destroy()

    # ------------------------------------------------------------- drag & drop

    def _on_drop(self, event):
        paths = [Path(p) for p in self.tk.splitlist(event.data)]
        if self.in_verify:
            return
        if self.current_step == 2:
            txts = [p for p in paths if p.suffix.lower() == ".txt"]
            if txts:
                self._load_key_file(txts[0])
                return
        images = []
        for p in paths:
            if p.is_dir():
                images.extend(core.find_image_files(p))
            elif p.suffix.lower() in core.IMAGE_EXTENSIONS:
                images.append(p)
        if images:
            self.add_files(images)
            if self.current_step != 1:
                self.show_step(1)

    # ------------------------------------------------------------- step 1: scans

    def _build_step1(self, frame):
        ttk.Label(frame, text="Krok 1 · Dodaj zeskanowane prace",
                  style="H1.TLabel").pack(anchor="w", pady=(4, 10))

        drop = tk.Frame(frame, bg="#eef3fb", highlightthickness=2,
                        highlightbackground="#9bb3d4")
        drop.pack(fill="x", pady=(0, 12), ipady=18)
        drop_text = ("⬇  Przeciągnij tutaj pliki .jpg / .png  ⬇" if HAS_DND
                     else "Dodaj pliki .jpg / .png przyciskami poniżej")
        tk.Label(drop, text=drop_text, bg="#eef3fb", fg=ACCENT_DARK,
                 font=("TkDefaultFont", 15, "bold")).pack(pady=(10, 6))
        btns = tk.Frame(drop, bg="#eef3fb")
        btns.pack(pady=(0, 8))
        ttk.Button(btns, text="📂  Dodaj pliki…", command=self.pick_files).pack(
            side="left", padx=6)
        ttk.Button(btns, text="📁  Dodaj folder…", command=self.pick_folder).pack(
            side="left", padx=6)

        head = ttk.Frame(frame)
        head.pack(fill="x")
        self.files_count_label = ttk.Label(head, text="Dodane prace: 0",
                                           font=("TkDefaultFont", 13, "bold"))
        self.files_count_label.pack(side="left")
        ttk.Button(head, text="Wyczyść wszystko", command=self.clear_files).pack(side="right")
        ttk.Button(head, text="Usuń zaznaczone", command=self.remove_selected).pack(
            side="right", padx=(0, 8))

        list_wrap = ttk.Frame(frame)
        list_wrap.pack(fill="both", expand=True, pady=(8, 4))
        self.files_list = tk.Listbox(list_wrap, selectmode="extended",
                                     font=("TkDefaultFont", 12), activestyle="none")
        scroll = ttk.Scrollbar(list_wrap, orient="vertical", command=self.files_list.yview)
        self.files_list.configure(yscrollcommand=scroll.set)
        self.files_list.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.files_list.bind("<Delete>", lambda _e: self.remove_selected())
        self.files_list.bind("<BackSpace>", lambda _e: self.remove_selected())

        self.files_info_label = ttk.Label(frame, text="", style="Muted.TLabel")
        self.files_info_label.pack(anchor="w")

    def _preload_inputs(self):
        preloaded = core.find_image_files(core.INPUT_DIR)
        if preloaded:
            self.add_files(preloaded)
            self.files_info_label.configure(
                text=f"Wczytano automatycznie {len(preloaded)} prac(e) z folderu "
                     f"{core.INPUT_DIR.relative_to(core.ROOT)}.")
        if core.ANSWER_KEY_FILE.exists():
            self._load_key_file(core.ANSWER_KEY_FILE, silent=True)
        if not self.key_vars:
            self._set_question_count(1)

    def pick_files(self):
        paths = filedialog.askopenfilenames(
            parent=self, title="Wybierz zeskanowane prace",
            filetypes=[("Obrazy", "*.jpg *.jpeg *.png"), ("Wszystkie pliki", "*")])
        if paths:
            self.add_files([Path(p) for p in paths])

    def pick_folder(self):
        folder = filedialog.askdirectory(parent=self, title="Wybierz folder z pracami")
        if folder:
            found = core.find_image_files(folder)
            if found:
                self.add_files(found)
            else:
                messagebox.showinfo("Brak prac",
                                    "W wybranym folderze nie ma plików .jpg ani .png.",
                                    parent=self)

    def add_files(self, paths):
        existing = {p.resolve() for p in self.files}
        added = 0
        for p in paths:
            rp = Path(p).resolve()
            if rp not in existing:
                self.files.append(rp)
                existing.add(rp)
                added += 1
        if added:
            self._refresh_file_list()
            self.invalidate_results()

    def remove_selected(self):
        selected = sorted(self.files_list.curselection(), reverse=True)
        if not selected:
            return
        for i in selected:
            del self.files[i]
        self._refresh_file_list()
        self.invalidate_results()

    def clear_files(self):
        if self.files:
            self.files.clear()
            self._refresh_file_list()
            self.invalidate_results()

    def _refresh_file_list(self):
        self.files_list.delete(0, "end")
        for p in self.files:
            self.files_list.insert("end", f"  {p.name}    —    {p.parent}")
        self.files_count_label.configure(text=f"Dodane prace: {len(self.files)}")
        self._update_nav()

    # ------------------------------------------------------------- step 2: answer key

    def _build_step2(self, frame):
        ttk.Label(frame, text="Krok 2 · Ustaw klucz odpowiedzi",
                  style="H1.TLabel").pack(anchor="w", pady=(4, 10))

        bar = ttk.Frame(frame)
        bar.pack(fill="x", pady=(0, 10))
        ttk.Button(bar, text="📂  Wczytaj z pliku…", command=self.load_key_dialog).pack(
            side="left")
        ttk.Button(bar, text="💾  Zapisz do pliku…", command=self.save_key_dialog).pack(
            side="left", padx=(8, 24))
        ttk.Label(bar, text="Liczba pytań:").pack(side="left")
        self.n_var = tk.IntVar(master=self, value=1)
        self.n_spin = ttk.Spinbox(bar, from_=1, to=core.MAX_QUESTIONS, width=4,
                                  textvariable=self.n_var, command=self._on_n_spin)
        self.n_spin.pack(side="left", padx=(6, 24))
        self.n_spin.bind("<Return>", lambda _e: self._on_n_spin())
        self.n_spin.bind("<FocusOut>", lambda _e: self._on_n_spin())
        self.maxpoints_label = ttk.Label(bar, text="Maks. punktów: 0",
                                         font=("TkDefaultFont", 13, "bold"))
        self.maxpoints_label.pack(side="left")
        self.key_source_label = ttk.Label(bar, text="", style="Muted.TLabel")
        self.key_source_label.pack(side="right")

        grid_wrap = ttk.Frame(frame, style="Card.TFrame")
        grid_wrap.pack(fill="both", expand=True)
        self.key_canvas = tk.Canvas(grid_wrap, bg=CARD_BG, highlightthickness=0)
        scroll = ttk.Scrollbar(grid_wrap, orient="vertical", command=self.key_canvas.yview)
        self.key_canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.key_canvas.pack(side="left", fill="both", expand=True)
        self.key_inner = ttk.Frame(self.key_canvas, style="Card.TFrame")
        self._key_window = self.key_canvas.create_window((0, 0), window=self.key_inner,
                                                         anchor="nw")
        self.key_inner.bind("<Configure>", lambda _e: self.key_canvas.configure(
            scrollregion=self.key_canvas.bbox("all")))
        self.key_canvas.bind("<MouseWheel>", lambda e: self.key_canvas.yview_scroll(
            -int(e.delta), "units"))

    def _set_question_count(self, n, preserve=True):
        n = max(1, min(core.MAX_QUESTIONS, n))
        old = self.key_vars if preserve else []
        self._loading_key = True
        self.key_vars = []
        for q in range(n):
            row = []
            for c in range(core.NUM_CHOICES):
                value = old[q][c].get() if q < len(old) else 0
                var = tk.IntVar(master=self, value=value)
                var.trace_add("write", self._on_key_toggle)
                row.append(var)
            self.key_vars.append(row)
        self._loading_key = False
        self.n_var.set(n)
        self._rebuild_key_grid()
        self._refresh_maxpoints()

    def _rebuild_key_grid(self):
        for child in self.key_inner.winfo_children():
            child.destroy()
        # Two columns of questions to limit scrolling
        per_col = (len(self.key_vars) + 1) // 2
        for q, row_vars in enumerate(self.key_vars):
            col_block = 0 if q < per_col else 5
            grid_row = q if q < per_col else q - per_col
            ttk.Label(self.key_inner, text=f"{q + 1}.", style="Card.TLabel",
                      font=("TkDefaultFont", 13, "bold"), width=4, anchor="e").grid(
                row=grid_row, column=col_block, padx=(18, 8), pady=3)
            for c, var in enumerate(row_vars):
                ttk.Checkbutton(self.key_inner, text="ABCD"[c], variable=var,
                                style="Key.TCheckbutton").grid(
                    row=grid_row, column=col_block + 1 + c, padx=3, pady=3)
        self.key_inner.columnconfigure(10, weight=1)

    def _on_n_spin(self):
        try:
            n = int(self.n_var.get())
        except (tk.TclError, ValueError):
            return
        if n != len(self.key_vars):
            self._set_question_count(n)
            self.invalidate_results()

    def _on_key_toggle(self, *_args):
        if self._loading_key:
            return
        self._refresh_maxpoints()
        self.key_source_label.configure(text="")
        if self.processed:
            # Marks stay valid when only the key changes — just re-score
            self._refresh_all_scores()

    def current_key(self) -> list[list[int]]:
        return [[var.get() for var in row] for row in self.key_vars]

    def _refresh_maxpoints(self):
        self.maxpoints_label.configure(
            text=f"Maks. punktów: {core.max_points(self.current_key())}")
        self._update_nav()

    def load_key_dialog(self):
        path = filedialog.askopenfilename(
            parent=self, title="Wczytaj klucz odpowiedzi",
            initialdir=core.ANSWER_KEY_FILE.parent,
            filetypes=[("Pliki tekstowe", "*.txt"), ("Wszystkie pliki", "*")])
        if path:
            self._load_key_file(Path(path))

    def _load_key_file(self, path, silent=False):
        try:
            key = core.load_answer_key(path)
        except (OSError, ValueError) as e:
            if not silent:
                messagebox.showerror("Błąd klucza", str(e), parent=self)
            return
        self._set_question_count(len(key), preserve=False)
        self._loading_key = True
        for row_vars, key_row in zip(self.key_vars, key):
            for var, value in zip(row_vars, key_row):
                var.set(value)
        self._loading_key = False
        self._refresh_maxpoints()
        try:
            shown = Path(path).relative_to(core.ROOT)
        except ValueError:
            shown = Path(path).name
        self.key_source_label.configure(text=f"Wczytano: {shown}")
        self.invalidate_results()

    def save_key_dialog(self):
        path = filedialog.asksaveasfilename(
            parent=self, title="Zapisz klucz odpowiedzi",
            initialdir=core.ANSWER_KEY_FILE.parent,
            initialfile=core.ANSWER_KEY_FILE.name,
            defaultextension=".txt", filetypes=[("Pliki tekstowe", "*.txt")])
        if path:
            core.save_answer_key(path, self.current_key())
            self.key_source_label.configure(text=f"Zapisano: {Path(path).name}")

    # ------------------------------------------------------------- step 3: processing

    def _build_step3(self, frame):
        ttk.Label(frame, text="Krok 3 · Sprawdzanie",
                  style="H1.TLabel").pack(anchor="w", pady=(4, 10))
        self.summary_label = ttk.Label(frame, text="", font=("TkDefaultFont", 14))
        self.summary_label.pack(anchor="w", pady=(0, 14))

        btns = ttk.Frame(frame)
        btns.pack(anchor="w", pady=(0, 12))
        self.start_btn = ttk.Button(btns, text="▶  Rozpocznij sprawdzanie",
                                    style="Big.TButton", command=self.start_processing)
        self.start_btn.pack(side="left")
        self.cancel_btn = ttk.Button(btns, text="Anuluj", command=self.cancel_processing)
        self.cancel_btn.pack(side="left", padx=10)
        self.cancel_btn.state(["disabled"])
        self.to_results_btn = ttk.Button(btns, text="Przejdź do wyników ▶",
                                         style="Accent.TButton",
                                         command=lambda: self.try_goto(4))
        self.to_results_btn.pack(side="left", padx=10)
        self.to_results_btn.state(["disabled"])

        self.progress = ttk.Progressbar(frame, mode="determinate")
        self.progress.pack(fill="x", pady=(0, 4))
        self.progress_label = ttk.Label(frame, text="", style="Muted.TLabel")
        self.progress_label.pack(anchor="w", pady=(0, 8))

        log_wrap = ttk.Frame(frame)
        log_wrap.pack(fill="both", expand=True)
        self.log = tk.Text(log_wrap, height=12, state="disabled",
                           font=("TkDefaultFont", 12), bg=CARD_BG, relief="flat")
        log_scroll = ttk.Scrollbar(log_wrap, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=log_scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")
        self.log.tag_configure("ok", foreground=OK_GREEN)
        self.log.tag_configure("warn", foreground=WARN_ORANGE)
        self.log.tag_configure("err", foreground=ERR_RED)

    def _refresh_summary(self):
        key = self.current_key()
        self.summary_label.configure(
            text=f"Prace: {len(self.files)}   ·   Pytania: {len(key)}   ·   "
                 f"Maks. punktów: {core.max_points(key)}")
        if self.processed:
            self.progress_label.configure(text="Sprawdzanie ukończone ✓")

    def _log(self, text, tag=None):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n", tag or ())
        self.log.see("end")
        self.log.configure(state="disabled")

    def start_processing(self):
        if not self.files:
            return
        self.invalidate_results(clear_log=False)
        self._cancel.clear()
        self.start_btn.state(["disabled"])
        self.cancel_btn.state(["!disabled"])
        self.to_results_btn.state(["disabled"])
        self.progress.configure(maximum=len(self.files), value=0)
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self._log(f"Rozpoczynam sprawdzanie {len(self.files)} prac…")

        files = list(self.files)
        n_questions = len(self.key_vars)

        def worker():
            def progress(i, total, name):
                self._queue.put(("progress", i, total, name))

            try:
                works = core.process(files, n_questions, cfg=self.cfg,
                                     progress_cb=progress,
                                     cancel_cb=self._cancel.is_set)
            except Exception as e:
                self._queue.put(("fatal", f"{type(e).__name__}: {e}"))
                return
            if self._cancel.is_set():
                self._queue.put(("cancelled",))
            else:
                self._queue.put(("finished", works))

        threading.Thread(target=worker, daemon=True).start()
        self._poll_id = self.after(80, self._poll_queue)

    def cancel_processing(self):
        self._cancel.set()

    def _poll_queue(self):
        try:
            while True:
                msg = self._queue.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    _, i, total, name = msg
                    self.progress.configure(value=i)
                    self.progress_label.configure(text=f"Praca {i + 1} z {total}: {name}")
                elif kind == "finished":
                    self._finalize_processing(msg[1])
                    return
                elif kind == "cancelled":
                    self._processing_done()
                    self.progress_label.configure(text="Anulowano.")
                    self._log("Anulowano sprawdzanie.", "warn")
                    return
                elif kind == "fatal":
                    self._processing_done()
                    self._log(f"Błąd krytyczny: {msg[1]}", "err")
                    messagebox.showerror("Błąd", msg[1], parent=self)
                    return
        except queue.Empty:
            pass
        self._poll_id = self.after(80, self._poll_queue)

    def _processing_done(self):
        self._poll_id = None
        self.start_btn.state(["!disabled"])
        self.cancel_btn.state(["disabled"])

    def _finalize_processing(self, works):
        self._processing_done()
        self.works = works
        self.processed = True
        self.progress.configure(value=len(self.files))

        key = self.current_key()
        ok_count = 0
        for i, work in enumerate(works):
            if not work.ok:
                self._log(f"✖ {work.path.name} — {work.error}", "err")
            elif not work.id_ok:
                self._log(f"⚠ {work.path.name} — nie odczytano indeksu, "
                          f"{core.score_marks(key, work.marks)} pkt", "warn")
                ok_count += 1
            else:
                self._log(f"✓ {work.path.name} — indeks {work.student_id}, "
                          f"{core.score_marks(key, work.marks)} pkt", "ok")
                ok_count += 1
        self._log(f"Gotowe: {ok_count} z {len(works)} prac sprawdzono poprawnie.")
        self.progress_label.configure(text="Sprawdzanie ukończone ✓")
        self.to_results_btn.state(["!disabled"])

        self._rebuild_results_table()
        self.show_step(4)

    def invalidate_results(self, clear_log=True):
        """Input data changed — previous detection results no longer apply."""
        if not self.processed and not self.works:
            self._update_nav()
            return
        self.processed = False
        self.works = []
        self._preview_cache_idx = -1
        if hasattr(self, "tree"):
            self.tree.delete(*self.tree.get_children())
        if hasattr(self, "to_results_btn"):
            self.to_results_btn.state(["disabled"])
            self.progress.configure(value=0)
            self.progress_label.configure(text="")
            if clear_log:
                self.log.configure(state="normal")
                self.log.delete("1.0", "end")
                self.log.configure(state="disabled")
        if self.in_verify or self.current_step == 4:
            self.show_step(3)
            self.set_hint("Dane się zmieniły — uruchom sprawdzanie ponownie.")
        self._update_nav()

    # ------------------------------------------------------------- step 4: results

    def _build_step4(self, frame):
        head = ttk.Frame(frame)
        head.pack(fill="x", pady=(4, 10))
        ttk.Label(head, text="Krok 4 · Wyniki", style="H1.TLabel").pack(side="left")
        ttk.Button(head, text="💾  Zapisz wyniki…", style="Accent.TButton",
                   command=self.save_results).pack(side="right")
        ttk.Button(head, text="📋  Kopiuj do schowka",
                   command=self.copy_results).pack(side="right", padx=(0, 10))
        self.save_info_label = ttk.Label(head, text="", style="Muted.TLabel")
        self.save_info_label.pack(side="right", padx=(0, 14))

        panes = ttk.Frame(frame)
        panes.pack(fill="both", expand=True)
        panes.columnconfigure(0, weight=3)
        panes.columnconfigure(1, weight=2)
        panes.rowconfigure(0, weight=1)

        table_wrap = ttk.Frame(panes)
        table_wrap.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        columns = ("lp", "id", "pkt", "proc", "status")
        self.tree = ttk.Treeview(table_wrap, columns=columns, show="headings",
                                 selectmode="browse")
        headings = {"lp": "Nr", "id": "Indeks", "pkt": "Punkty",
                    "proc": "%", "status": "Uwagi"}
        widths = {"lp": 50, "id": 110, "pkt": 80, "proc": 80, "status": 240}
        for col in columns:
            self.tree.heading(col, text=headings[col],
                              command=lambda c=col: self._sort_results(c))
            anchor = "w" if col == "status" else "center"
            self.tree.column(col, width=widths[col], anchor=anchor,
                             stretch=(col == "status"))
        tree_scroll = ttk.Scrollbar(table_wrap, orient="vertical",
                                    command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")
        self.tree.tag_configure("err", foreground=ERR_RED)
        self.tree.tag_configure("warn", foreground=WARN_ORANGE)
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._refresh_preview())
        self.tree.bind("<Double-Button-1>", lambda _e: self.open_selected_verify())
        self.tree.bind("<Return>", lambda _e: self.open_selected_verify())
        self._sort_state = ("lp", False)

        side = ttk.Frame(panes, style="Card.TFrame", padding=10)
        side.grid(row=0, column=1, sticky="nsew")
        self.preview_title = ttk.Label(side, text="Podgląd pracy", style="Card.TLabel",
                                       font=("TkDefaultFont", 13, "bold"))
        self.preview_title.pack(anchor="w")
        self.preview_label = tk.Label(side, bg=CARD_BG)
        self.preview_label.pack(fill="both", expand=True, pady=8)
        ttk.Button(side, text="🔍  Weryfikuj zaznaczenia  (Enter)",
                   style="Accent.TButton",
                   command=self.open_selected_verify).pack(pady=(0, 4))

    def _focus_results_table(self):
        children = self.tree.get_children()
        if children and not self.tree.selection():
            self.tree.selection_set(children[0])
            self.tree.focus(children[0])
        self.tree.focus_set()
        self._refresh_preview()

    def _work_row_values(self, i, work):
        key = self.current_key()
        maximum = core.max_points(key)
        if not work.ok:
            return (i + 1, "—", "—", "—", f"✖ błąd: {work.error}")
        score = core.score_marks(key, work.marks)
        pct = (score / maximum * 100) if maximum else 0.0
        notes = []
        if not work.id_ok:
            notes.append("⚠ nie odczytano indeksu")
        if work.edited:
            notes.append("✎ edytowano")
        return (i + 1, work.student_id, score, f"{pct:.1f}%", ", ".join(notes))

    def _row_tag(self, work):
        if not work.ok:
            return ("err",)
        if not work.id_ok:
            return ("warn",)
        return ()

    def _rebuild_results_table(self):
        self.tree.delete(*self.tree.get_children())
        for i, work in enumerate(self.works):
            self.tree.insert("", "end", iid=str(i),
                             values=self._work_row_values(i, work),
                             tags=self._row_tag(work))
        self._preview_cache_idx = -1

    def _refresh_result_row(self, i):
        work = self.works[i]
        self.tree.item(str(i), values=self._work_row_values(i, work),
                       tags=self._row_tag(work))

    def _refresh_all_scores(self):
        for i in range(len(self.works)):
            self._refresh_result_row(i)
        self._preview_cache_idx = -1
        if self.current_step == 4:
            self._refresh_preview()
        if self.in_verify:
            self._refresh_verify_header()

    def _sort_results(self, col):
        prev_col, reverse = self._sort_state
        reverse = not reverse if col == prev_col else False
        self._sort_state = (col, reverse)

        def sort_value(iid):
            value = self.tree.set(iid, col)
            try:
                return (0, float(str(value).rstrip("%")))
            except ValueError:
                return (1, str(value))

        order = sorted(self.tree.get_children(), key=sort_value, reverse=reverse)
        for pos, iid in enumerate(order):
            self.tree.move(iid, "", pos)

    def _selected_work_index(self):
        selection = self.tree.selection()
        return int(selection[0]) if selection else None

    def _refresh_preview(self):
        i = self._selected_work_index()
        if i is None or i >= len(self.works):
            return
        if i == self._preview_cache_idx:
            return
        work = self.works[i]
        self.preview_title.configure(text=f"Podgląd: {work.path.name}")
        img = self._render_annotated(work)
        if img is None:
            self.preview_label.configure(image="", text="(brak podglądu)")
            self._preview_photo = None
            return
        target_w = max(self.preview_label.winfo_width(), 420)
        scale = min(target_w / img.shape[1], 1.0)
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        self._preview_photo = self._photo_from_bgr(small)
        self.preview_label.configure(image=self._preview_photo, text="")
        self._preview_cache_idx = i

    def _render_annotated(self, work):
        try:
            if work.ok:
                gray = core.imread_gray(work.aligned_path)
                return core.annotate_work(gray, work.marks, self.cfg,
                                          len(self.key_vars), work.id_boxes)
            return cv2.cvtColor(core.imread_gray(work.path), cv2.COLOR_GRAY2BGR)
        except core.PipelineError:
            return None

    def save_results(self):
        if not self.works:
            return
        path = filedialog.asksaveasfilename(
            parent=self, title="Zapisz wyniki",
            initialdir=core.RESULTS_FILE.parent, initialfile=core.RESULTS_FILE.name,
            defaultextension=".txt", filetypes=[("Pliki tekstowe", "*.txt")])
        if not path:
            return
        key = self.current_key()
        core.write_results_file(path, self.works, key)
        # Refresh the annotated verification images to match any manual edits
        for i, work in enumerate(self.works):
            if not work.ok or not work.edited:
                continue
            try:
                gray = core.imread_gray(work.aligned_path)
                annotated = core.annotate_work(gray, work.marks, self.cfg,
                                               len(key), work.id_boxes)
                cv2.imwrite(str(core.ANALYZED_DIR / f"corrected_and_detected_{i}.jpg"),
                            annotated)
            except core.PipelineError:
                pass
        self.save_info_label.configure(text=f"Zapisano ✓  {Path(path).name}")

    def copy_results(self):
        if not self.works:
            return
        key = self.current_key()
        maximum = core.max_points(key)
        rows = ["Indeks\tPunkty\tProcent"]
        for work in self.works:
            if work.ok:
                score = core.score_marks(key, work.marks)
                pct = (score / maximum * 100) if maximum else 0.0
                rows.append(f"{work.student_id}\t{score}\t{pct:.2f}")
            else:
                rows.append(f"BŁĄD ({work.path.name})\t\t")
        self.clipboard_clear()
        self.clipboard_append("\n".join(rows))
        self.save_info_label.configure(text="Skopiowano do schowka ✓")

    # ------------------------------------------------------------- verification view

    def _build_verify(self, frame):
        head = ttk.Frame(frame)
        head.pack(fill="x", pady=(4, 8))
        ttk.Button(head, text="◀  Wyniki (Esc)", command=self.close_verify).pack(
            side="left")
        self.verify_title = ttk.Label(head, text="", font=("TkDefaultFont", 14, "bold"))
        self.verify_title.pack(side="left", padx=16)
        ttk.Label(head, text="Indeks:").pack(side="left", padx=(12, 4))
        self.verify_id_var = tk.StringVar(master=self)
        id_entry = ttk.Entry(head, textvariable=self.verify_id_var, width=9,
                             font=("TkDefaultFont", 13))
        id_entry.pack(side="left")
        id_entry.bind("<Return>", lambda _e: self._apply_id_edit())
        id_entry.bind("<FocusOut>", lambda _e: self._apply_id_edit())
        self.verify_score_label = ttk.Label(head, text="", font=("TkDefaultFont", 14, "bold"),
                                            foreground=ACCENT)
        self.verify_score_label.pack(side="left", padx=16)

        ttk.Button(head, text="Następna ▶", command=lambda: self.verify_show(
            self.verify_index + 1)).pack(side="right")
        ttk.Button(head, text="◀ Poprzednia", command=lambda: self.verify_show(
            self.verify_index - 1)).pack(side="right", padx=(0, 8))

        tools = ttk.Frame(frame)
        tools.pack(fill="x", pady=(0, 6))
        ttk.Button(tools, text="−", width=3, command=lambda: self._zoom_by(1 / ZOOM_STEP)).pack(
            side="left")
        ttk.Button(tools, text="+", width=3, command=lambda: self._zoom_by(ZOOM_STEP)).pack(
            side="left", padx=4)
        ttk.Button(tools, text="Dopasuj", command=self._zoom_fit).pack(side="left", padx=4)
        ttk.Button(tools, text="100%", command=lambda: self._zoom_set(1.0)).pack(
            side="left", padx=4)
        ttk.Checkbutton(tools, text="Pokaż klucz (niebieska ramka = poprawna odpowiedź)",
                        variable=self.show_key_var,
                        command=self._redraw_overlays).pack(side="left", padx=16)
        self.verify_error_label = ttk.Label(tools, text="", foreground=ERR_RED,
                                            font=("TkDefaultFont", 13, "bold"))
        self.verify_error_label.pack(side="right")

        canvas_wrap = ttk.Frame(frame)
        canvas_wrap.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(canvas_wrap, bg="#3c4043", highlightthickness=0,
                                yscrollincrement=30, xscrollincrement=30)
        vbar = ttk.Scrollbar(canvas_wrap, orient="vertical", command=self.canvas.yview)
        hbar = ttk.Scrollbar(canvas_wrap, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        canvas_wrap.rowconfigure(0, weight=1)
        canvas_wrap.columnconfigure(0, weight=1)

        self.canvas.bind("<ButtonPress-1>", self._canvas_press)
        self.canvas.bind("<B1-Motion>", self._canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._canvas_release)
        self.canvas.bind("<MouseWheel>", self._canvas_wheel)
        self.canvas.bind("<Shift-MouseWheel>", lambda e: self.canvas.xview_scroll(
            -int(e.delta), "units"))
        self.canvas.bind("<Control-MouseWheel>", self._canvas_zoom_wheel)
        self.canvas.bind("<Command-MouseWheel>", self._canvas_zoom_wheel)

    def open_selected_verify(self):
        i = self._selected_work_index()
        if i is not None:
            self.open_verify(i)

    def open_verify(self, index):
        if not self.works:
            return
        self.in_verify = True
        self.frames["verify"].tkraise()
        self._update_nav()
        self.set_hint("Kliknięcie pola zmienia stan: puste → zaliczone → anulowane. "
                      "Przeciąganie przesuwa · ⌘/Ctrl+kółko = zoom · ←/→ = zmiana pracy · Esc = powrót.")
        self.verify_show(index, fit=True)

    def close_verify(self):
        self.in_verify = False
        self.show_step(4)
        if str(self.verify_index) in self.tree.get_children():
            self.tree.selection_set(str(self.verify_index))
            self.tree.see(str(self.verify_index))

    def verify_show(self, index, fit=False):
        if not self.works:
            return
        index = max(0, min(len(self.works) - 1, index))
        self.verify_index = index
        work = self.works[index]

        self._verify_gray = None
        try:
            if work.ok:
                self._verify_gray = core.imread_gray(work.aligned_path)
                self.verify_error_label.configure(text="")
            else:
                self._verify_gray = core.imread_gray(work.path)
                self.verify_error_label.configure(
                    text=f"Nie sprawdzono: {work.error}")
        except core.PipelineError:
            self.verify_error_label.configure(text="Nie można wczytać obrazu.")

        self.verify_id_var.set(work.student_id)
        self._refresh_verify_header()
        if fit:
            self.update_idletasks()
            self._zoom_fit(redraw=False)
        self._render_verify()

    def _refresh_verify_header(self):
        work = self.works[self.verify_index]
        self.verify_title.configure(
            text=f"Praca {self.verify_index + 1} z {len(self.works)} · {work.path.name}")
        if work.ok:
            key = self.current_key()
            maximum = core.max_points(key)
            score = core.score_marks(key, work.marks)
            pct = (score / maximum * 100) if maximum else 0.0
            self.verify_score_label.configure(text=f"{score} pkt · {pct:.1f}%")
        else:
            self.verify_score_label.configure(text="—")

    def _apply_id_edit(self):
        if not self.in_verify or not self.works:
            return
        work = self.works[self.verify_index]
        new_id = self.verify_id_var.get().strip()
        if not new_id:
            new_id = "------"
        if new_id == work.student_id:
            return
        work.student_id = new_id
        work.id_ok = new_id != "------"
        work.edited = True
        self._refresh_result_row(self.verify_index)
        self._refresh_verify_header()

    # ---- rendering

    def _photo_from_bgr(self, img_bgr):
        ok, buf = cv2.imencode(".png", img_bgr)
        if not ok:
            raise RuntimeError("PNG encode failed")
        return tk.PhotoImage(master=self, data=base64.b64encode(buf).decode("ascii"))

    def _render_verify(self):
        self.canvas.delete("all")
        if self._verify_gray is None:
            return
        scaled = cv2.resize(self._verify_gray, None, fx=self.zoom, fy=self.zoom,
                            interpolation=cv2.INTER_AREA)
        self._verify_photo = self._photo_from_bgr(
            cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR))
        self.canvas.create_image(0, 0, image=self._verify_photo, anchor="nw",
                                 tags=("scan",))
        self.canvas.configure(scrollregion=(0, 0, scaled.shape[1], scaled.shape[0]))
        self._redraw_overlays()

    def _redraw_overlays(self):
        self.canvas.delete("overlay")
        work = self.works[self.verify_index] if self.works else None
        if work is None or not work.ok:
            return
        zoom = self.zoom
        box_w, box_h = self.cfg["image_processing"]["box_size"]
        key = self.current_key()
        n_questions = len(self.key_vars)

        for x, y, w, h in work.id_boxes:
            self.canvas.create_rectangle(
                x * zoom, y * zoom, (x + w) * zoom, (y + h) * zoom,
                outline=OK_GREEN, width=2, tags=("overlay",))

        for question in self.cfg["questions"]:
            q = question["number"] - 1
            if q >= n_questions:
                continue
            for choice in question["choices"]:
                c = ord(choice["label"]) - ord("A")
                center_x, center_y = choice["center"]
                x0 = (center_x - box_w // 2) * zoom
                y0 = (center_y - box_h // 2) * zoom
                x1 = x0 + box_w * zoom
                y1 = y0 + box_h * zoom
                state = int(work.marks[q, c])
                color = STATE_COLORS[state]
                stipple = "gray12" if state == core.MARK_EMPTY else "gray50"
                width = 1 if state == core.MARK_EMPTY else 3
                self.canvas.create_rectangle(
                    x0, y0, x1, y1, outline=color, width=width,
                    fill=color, stipple=stipple,
                    tags=("overlay", f"box_{q}_{c}"))
                if self.show_key_var.get() and key[q][c] == 1:
                    pad = 4 * zoom + 2
                    self.canvas.create_rectangle(
                        x0 - pad, y0 - pad, x1 + pad, y1 + pad,
                        outline=ACCENT, width=2, dash=(5, 3), tags=("overlay",))

    # ---- zoom and pan

    def _zoom_fit(self, redraw=True):
        if self._verify_gray is None:
            return
        ch = max(self.canvas.winfo_height(), 200)
        cw = max(self.canvas.winfo_width(), 200)
        ih, iw = self._verify_gray.shape
        self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, min(cw / iw, ch / ih)))
        if redraw:
            self._render_verify()

    def _zoom_set(self, value):
        self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, value))
        self._render_verify()

    def _zoom_by(self, factor):
        self._zoom_set(self.zoom * factor)

    def _canvas_wheel(self, event):
        self.canvas.yview_scroll(-int(event.delta), "units")

    def _canvas_zoom_wheel(self, event):
        self._zoom_by(ZOOM_STEP if event.delta > 0 else 1 / ZOOM_STEP)

    def _canvas_press(self, event):
        self._drag_moved = False
        self._press_xy = (event.x, event.y)
        self.canvas.scan_mark(event.x, event.y)

    def _canvas_drag(self, event):
        dx = abs(event.x - self._press_xy[0])
        dy = abs(event.y - self._press_xy[1])
        if dx + dy > 5:
            self._drag_moved = True
        if self._drag_moved:
            self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _canvas_release(self, event):
        if self._drag_moved:
            return
        work = self.works[self.verify_index] if self.works else None
        if work is None or not work.ok:
            return
        img_x = self.canvas.canvasx(event.x) / self.zoom
        img_y = self.canvas.canvasy(event.y) / self.zoom
        hit = self._find_box(img_x, img_y)
        if hit is not None:
            self._cycle_box(*hit)

    def _find_box(self, img_x, img_y):
        box_w, box_h = self.cfg["image_processing"]["box_size"]
        n_questions = len(self.key_vars)
        radius = max(box_w, box_h) * 0.75
        best = None
        best_dist = radius
        for question in self.cfg["questions"]:
            q = question["number"] - 1
            if q >= n_questions:
                continue
            for choice in question["choices"]:
                c = ord(choice["label"]) - ord("A")
                center_x, center_y = choice["center"]
                dist = max(abs(img_x - center_x), abs(img_y - center_y))
                if dist < best_dist:
                    best = (q, c)
                    best_dist = dist
        return best

    def _cycle_box(self, q, c):
        work = self.works[self.verify_index]
        work.marks[q, c] = NEXT_STATE[int(work.marks[q, c])]
        work.edited = True
        self._refresh_verify_header()
        self._refresh_result_row(self.verify_index)
        self._preview_cache_idx = -1
        self._redraw_overlays()


def run_gui() -> int:
    try:
        app = App()
    except tk.TclError as e:
        print("Nie można uruchomić trybu okienkowego:", e)
        print("Użyj wersji konsolowej: python main.py --cli")
        return 1
    except SystemExit as e:
        return int(e.code or 1)
    app.mainloop()
    return 0
