#!/usr/bin/env python3
"""check_fungen_gui.py — drag-and-drop GUI for the FunGen output checker.

Drop any of these onto the window:
  - A video file (.mp4 / .mkv / .mov / …) whose sidecars you want to check
  - Any one of its .funscript sidecars (stem is auto-derived)

The app shows an overall PASS / WARN / FAIL banner plus per-axis
detail for stroke / sway / surge / roll.

Reuses the analysis logic from check_fungen_output.py, so the
verdicts exactly match what `python3 check_fungen_output.py <path>`
would print in a terminal.
"""

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font, ttk
from typing import Optional

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    # Graceful degradation: without tkinterdnd2 the user can still
    # use the Browse button. Install with `pip install tkinterdnd2`
    # for drag-and-drop.
    TkinterDnD = None  # type: ignore
    DND_FILES = None   # type: ignore

# Reuse the analysis logic from the CLI tool so one source of truth
# owns the verdict policy.
from check_fungen_output import (
    AXIS_SUFFIXES,
    apply_cross_axis_checks,
    apply_duplicate_content_check,
    build_axis_results,
)


VERDICT_COLORS = {
    "PASS": "#1b8a3c",   # green
    "WARN": "#c98a1a",   # amber
    "FAIL": "#b02a2a",   # red
}
VERDICT_BG_SOFT = {
    "PASS": "#e2f4e8",
    "WARN": "#f9efd8",
    "FAIL": "#f7e0e0",
}


# Suffixes we treat as video files so a drop of the .mp4 auto-
# derives the stem. Kept loose — rare formats fall through and the
# caller can drop a .funscript instead.
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts"}


def stem_from_any(path: Path) -> Path:
    """Given either a video or a funscript sidecar, return the shared
    stem we'll search sidecars against."""
    sfx = path.suffix.lower()
    if sfx in VIDEO_EXTS:
        return path.with_suffix("")
    if sfx == ".funscript":
        # Strip .funscript, then also strip any axis suffix so e.g.
        # `clip.sway.funscript` → `clip`, not `clip.sway`.
        bare = path.with_suffix("")   # drops .funscript
        inner = bare.suffix.lower()
        if inner in {".sway", ".surge", ".roll", ".stroke",
                     ".heave", ".twist", ".x", ".y", ".z", ".rz"}:
            return bare.with_suffix("")
        return bare
    # Unknown suffix — treat the whole path as the stem.
    return path


def parse_drop_payload(raw: str) -> Optional[Path]:
    """tkinterdnd2 hands us a space-separated string with `{}`
    bracing around paths containing spaces. Pull out the first path."""
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("{") and "}" in s:
        s = s[1:s.index("}")]
    else:
        # Space-separated; take the first token as the dropped file.
        s = s.split(" ", 1)[0]
    p = Path(s)
    return p if p.exists() else None


class CheckerApp:

    def __init__(self) -> None:
        self._dnd_enabled = TkinterDnD is not None
        self.root = TkinterDnD.Tk() if self._dnd_enabled else tk.Tk()
        self.root.title("FunGen Output Checker")
        self.root.geometry("760x560")
        self.root.minsize(600, 420)

        self._build_ui()
        self._wire_events()

    # ---------------- UI construction ----------------

    def _build_ui(self) -> None:
        self.title_font = font.Font(family="Helvetica", size=14, weight="bold")
        self.drop_font = font.Font(family="Helvetica", size=13)
        self.banner_font = font.Font(family="Helvetica", size=18, weight="bold")
        self.report_font = font.Font(family="Menlo", size=11)

        header = tk.Frame(self.root, padx=12, pady=10)
        header.pack(fill=tk.X)
        tk.Label(
            header, text="FunGen Output Checker",
            font=self.title_font, anchor="w",
        ).pack(side=tk.LEFT)

        tk.Button(
            header, text="Browse…", command=self._on_browse,
        ).pack(side=tk.RIGHT)

        drop_note = (
            "Drop a video or any .funscript sidecar here"
            if self._dnd_enabled else
            "tkinterdnd2 not installed — use the Browse button above"
        )
        self.drop_zone = tk.Label(
            self.root,
            text=drop_note,
            font=self.drop_font,
            bg="#f2f2f2", fg="#555",
            relief=tk.GROOVE, bd=2,
            padx=20, pady=22,
        )
        self.drop_zone.pack(fill=tk.X, padx=12, pady=(0, 8))

        self.verdict_banner = tk.Label(
            self.root, text="", font=self.banner_font,
            padx=16, pady=12, anchor="w",
        )
        self.verdict_banner.pack(fill=tk.X, padx=12)
        # Hidden until we have a result to show.
        self.verdict_banner.pack_forget()

        report_frame = tk.Frame(self.root)
        report_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(8, 12))

        self.report = tk.Text(
            report_frame, wrap=tk.NONE, font=self.report_font,
            bg="#fafafa", fg="#222",
            relief=tk.FLAT, bd=0, padx=8, pady=8,
            state=tk.DISABLED,
        )
        yscroll = ttk.Scrollbar(
            report_frame, orient=tk.VERTICAL, command=self.report.yview,
        )
        self.report.configure(yscrollcommand=yscroll.set)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.report.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Colour tags for per-axis verdict prefixes.
        self.report.tag_configure("pass", foreground=VERDICT_COLORS["PASS"])
        self.report.tag_configure("warn", foreground=VERDICT_COLORS["WARN"])
        self.report.tag_configure("fail", foreground=VERDICT_COLORS["FAIL"])
        self.report.tag_configure(
            "heading", font=font.Font(family="Menlo", size=11, weight="bold"),
        )

        self.status = tk.Label(
            self.root, text="Ready.", anchor="w",
            bg="#eee", fg="#333", padx=10, pady=4,
        )
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

    def _wire_events(self) -> None:
        if not self._dnd_enabled:
            return
        # Register the whole root window so a drop anywhere on it
        # counts — small drop zones are annoying on high-DPI displays.
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind("<<Drop>>", self._on_drop)

    # ---------------- event handlers ----------------

    def _on_browse(self) -> None:
        filetypes = [
            ("Video or funscript", "*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.ts *.funscript"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            self._check_path(Path(path))

    def _on_drop(self, event: tk.Event) -> None:
        data = getattr(event, "data", "") or ""
        dropped = parse_drop_payload(data)
        if dropped is None:
            self._show_status(f"Could not parse dropped path: {data!r}")
            return
        self._check_path(dropped)

    # ---------------- checking pipeline ----------------

    def _check_path(self, path: Path) -> None:
        self._show_status(f"Checking {path.name}…")
        self.root.update_idletasks()
        stem = stem_from_any(path.expanduser())

        # Reuse the shared analysis pipeline. build_axis_results
        # does the per-file parse + per-axis verdict;
        # apply_cross_axis_checks catches the "stale file from a
        # different run" case by comparing time spans across axes;
        # apply_duplicate_content_check catches "tracker wrote the
        # same signal to two sidecars" (the Mask-Moments / writer
        # bug that produces 2-DoF dressed as 4-DoF).
        results = build_axis_results(stem)
        apply_cross_axis_checks(results)
        apply_duplicate_content_check(results)

        verdicts = [r["verdict"] for r in results]
        if any(v == "FAIL" for v in verdicts):
            overall = "FAIL"
            summary = "One or more axes unusable. Re-track before processing."
        elif any(v == "WARN" for v in verdicts):
            overall = "WARN"
            summary = "Usable — but check the flagged axes."
        else:
            overall = "PASS"
            summary = "Looks usable across all four axes."

        self._render_report(stem, results, overall, summary)
        self._show_status(f"Checked: {stem}  →  {overall}")

    # ---------------- rendering ----------------

    def _render_report(
        self,
        stem: Path,
        results: list,
        overall: str,
        summary: str,
    ) -> None:
        # Banner.
        self.verdict_banner.configure(
            text=f"  {overall}  —  {summary}",
            fg="white",
            bg=VERDICT_COLORS[overall],
        )
        self.verdict_banner.pack(fill=tk.X, padx=12, before=self.report.master)

        # Report body.
        self.report.configure(state=tk.NORMAL)
        self.report.delete("1.0", tk.END)
        self.report.insert(tk.END, f"Checking sidecars for stem:\n    {stem}\n\n", "heading")

        for r in results:
            axis_label = r["axis"]
            verdict = r["verdict"]
            notes = r["notes"]
            path = r["path"]
            stats = r["stats"]

            tag = verdict.lower()
            self.report.insert(tk.END, f"[{verdict:>4}]  ", tag)
            self.report.insert(tk.END, f"{axis_label}\n", "heading")
            if path is None:
                self.report.insert(
                    tk.END,
                    "        file: (missing — tracker did not produce this axis)\n",
                )
            else:
                self.report.insert(tk.END, f"        file: {path.name}\n")
            if stats.get("problem"):
                self.report.insert(tk.END, f"        error: {stats['problem']}\n")
            elif stats:
                self.report.insert(
                    tk.END,
                    f"        actions: {stats['count']} over {stats['t_span_s']:.1f}s "
                    f"({stats['rate_hz']:.1f} Hz)\n"
                    f"        pos: min={stats['pos_min']} max={stats['pos_max']} "
                    f"std={stats['pos_std']:.1f}\n"
                    f"        near-center ({stats['frac_near_50']*100:.0f}%)  "
                    f"active ({stats['frac_active']*100:.0f}%)\n",
                )
            for n in notes:
                self.report.insert(tk.END, f"        - {n}\n")
            self.report.insert(tk.END, "\n")

        self.report.configure(state=tk.DISABLED)

    # ---------------- helpers ----------------

    def _show_status(self, msg: str) -> None:
        self.status.configure(text=msg)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = CheckerApp()
    app.run()


if __name__ == "__main__":
    main()
