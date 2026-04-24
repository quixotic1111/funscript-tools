#!/usr/bin/env python3
"""check_fungen_output.py — sanity-check FunGen multi-axis funscripts.

Reads the 4 sidecar files FunGen writes next to a video and reports
per-axis usability for the Spatial 3D Linear pipeline. Use after
running any tracker (Quad User ROI, Mask Moments, Dense Flow, etc.)
to decide whether the output is worth processing — or whether to
re-track before wasting time on a flat / broken signal.

Usage:
    python3 check_fungen_output.py /path/to/video.mp4
    python3 check_fungen_output.py /path/to/video       # stem works too

Exit codes:
    0 — all axes PASS
    1 — at least one axis WARN
    2 — at least one axis FAIL (unusable)
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any


# Sidecar suffixes mapped to each spatial axis. Both conventions are
# checked — canonical (.x/.y/.z/.rz) and FunGen-style (.sway/.surge/
# .roll). The stroke slot also accepts a plain <stem>.funscript (no
# suffix), which is how every tracker writes its primary axis.
AXIS_SUFFIXES: Dict[str, List[str]] = {
    "stroke (T1, primary)":  ["", ".stroke", ".y", ".heave"],
    "sway   (T2, lateral)":  [".sway", ".x"],
    "surge  (T3, depth)":    [".surge", ".z"],
    "roll   (T4, twist)":    [".roll", ".rz", ".twist"],
}


# Thresholds — expressed as "anything below this is flagged".
# Tuned for typical 30-60 second clips at 30-60 fps.
MIN_ACTIONS_FAIL = 50          # Below this = unusable
MIN_ACTIONS_WARN = 200         # Below this = thin signal
FLAT_STD_FAIL    = 5.0         # pos std below this = effectively flat
NARROW_STD_WARN  = 10.0        # pos std below this = narrow dynamic range
NEAR_CENTER_FAIL = 0.90        # >90% of samples within ±2 of 50 = silent
NEAR_CENTER_WARN = 0.70        # >70% clustered = weak signal
ACTIVE_FRAC_WARN = 0.10        # <10% samples with |pos-50|>5 = inert


def discover_sidecar(stem: Path, suffixes: List[str]) -> Optional[Path]:
    """Return the first existing <stem><suffix>.funscript for any of
    the given suffixes, else None. Empty string suffix targets plain
    <stem>.funscript (the primary axis when it has no explicit name)."""
    for sfx in suffixes:
        candidate = stem.parent / f"{stem.name}{sfx}.funscript"
        if candidate.exists():
            return candidate
    return None


def analyze_actions(actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Distribution stats + validity flags for one axis's action list."""
    if not actions:
        return {"count": 0, "problem": "empty actions array"}

    ats_raw = [a.get("at") for a in actions]
    poss_raw = [a.get("pos") for a in actions]

    problems: List[str] = []
    bad_at = sum(1 for t in ats_raw if not isinstance(t, (int, float)))
    bad_pos = sum(1 for p in poss_raw if not isinstance(p, (int, float)))
    if bad_at:
        problems.append(f"{bad_at} non-numeric 'at' entries")
    if bad_pos:
        problems.append(f"{bad_pos} non-numeric 'pos' entries")

    ats = [t for t in ats_raw if isinstance(t, (int, float))]
    poss = [p for p in poss_raw if isinstance(p, (int, float))]
    if not ats:
        return {"count": 0, "problem": "no valid actions after filtering"}

    out_of_order = sum(1 for i in range(1, len(ats)) if ats[i] < ats[i - 1])
    if out_of_order:
        problems.append(f"{out_of_order} out-of-order timestamps")

    duplicates = len(ats) - len(set(ats))
    if duplicates:
        problems.append(f"{duplicates} duplicate timestamps")

    out_of_range = sum(1 for p in poss if p < 0 or p > 100)
    if out_of_range:
        problems.append(f"{out_of_range} pos values outside [0, 100]")

    pos_min = min(poss)
    pos_max = max(poss)
    try:
        pos_std = statistics.pstdev(poss)
    except statistics.StatisticsError:
        pos_std = 0.0

    # Values within ±2 of center count as "silent" (no useful motion).
    near_50 = sum(1 for p in poss if abs(p - 50) < 2)
    frac_near_50 = near_50 / len(poss)

    # Values more than 5 from center count as "actively moving".
    active = sum(1 for p in poss if abs(p - 50) > 5)
    frac_active = active / len(poss)

    t_min = min(ats)
    t_max = max(ats)
    t_span_s = (t_max - t_min) / 1000.0

    return {
        "count": len(actions),
        "t_span_s": t_span_s,
        "rate_hz": (len(actions) / t_span_s) if t_span_s > 0 else 0.0,
        "pos_min": pos_min,
        "pos_max": pos_max,
        "pos_std": pos_std,
        "frac_near_50": frac_near_50,
        "frac_active": frac_active,
        "problems": problems,
    }


def verdict_for_axis(stats: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Aggregate raw stats into a PASS / WARN / FAIL verdict + notes."""
    if stats.get("problem"):
        return "FAIL", [stats["problem"]]

    score = "PASS"
    notes: List[str] = []

    def raise_fail(msg: str) -> None:
        nonlocal score
        score = "FAIL"
        notes.append(msg)

    def raise_warn(msg: str) -> None:
        nonlocal score
        if score == "PASS":
            score = "WARN"
        notes.append(msg)

    # Action count.
    if stats["count"] < MIN_ACTIONS_FAIL:
        raise_fail(f"too few actions ({stats['count']}; need {MIN_ACTIONS_FAIL}+)")
    elif stats["count"] < MIN_ACTIONS_WARN:
        raise_warn(f"low action count ({stats['count']})")

    # Dynamic range.
    if stats["pos_std"] < FLAT_STD_FAIL:
        raise_fail(f"signal is flat (std={stats['pos_std']:.1f})")
    elif stats["pos_std"] < NARROW_STD_WARN:
        raise_warn(f"narrow variation (std={stats['pos_std']:.1f})")

    # Center clustering.
    pct_near = stats["frac_near_50"] * 100
    if stats["frac_near_50"] > NEAR_CENTER_FAIL:
        raise_fail(f"{pct_near:.0f}% of values within ±2 of center — effectively silent")
    elif stats["frac_near_50"] > NEAR_CENTER_WARN:
        raise_warn(f"{pct_near:.0f}% of values clustered near center")

    # Active fraction.
    if stats["frac_active"] < ACTIVE_FRAC_WARN:
        raise_warn(
            f"only {stats['frac_active']*100:.0f}% of samples show real motion "
            f"(|pos-50|>5)"
        )

    # Schema / timestamp problems — classify severity per problem type.
    for p in stats.get("problems", []):
        if any(
            needle in p for needle in ("out-of-order", "outside", "non-numeric")
        ):
            raise_fail(p)
        else:
            raise_warn(p)

    if not notes:
        notes.append("clean — no issues detected")
    return score, notes


def build_axis_results(stem: Path) -> List[Dict[str, Any]]:
    """Analyze all 4 axes for a given stem. Returns a list of per-
    axis result dicts keyed by 'axis' / 'path' / 'stats' / 'verdict'
    / 'notes'. Does NOT apply cross-axis consistency checks — call
    apply_cross_axis_checks() on the return value for that.

    Shared by both the CLI and the drag-and-drop GUI so one source
    of truth owns the analysis logic.
    """
    results: List[Dict[str, Any]] = []
    for axis_label, suffixes in AXIS_SUFFIXES.items():
        sc = discover_sidecar(stem, suffixes)
        if sc is None:
            results.append({
                "axis": axis_label, "path": None, "stats": {},
                "verdict": "FAIL",
                "notes": [f"no sidecar found (tried {suffixes})"],
            })
            continue
        try:
            data = json.loads(sc.read_text())
        except json.JSONDecodeError as e:
            results.append({
                "axis": axis_label, "path": sc, "stats": {},
                "verdict": "FAIL", "notes": [f"invalid JSON: {e}"],
            })
            continue
        actions = data.get("actions", [])
        if not isinstance(actions, list):
            results.append({
                "axis": axis_label, "path": sc, "stats": {},
                "verdict": "FAIL", "notes": ["'actions' is not a list"],
            })
            continue
        stats = analyze_actions(actions)
        verdict, notes = verdict_for_axis(stats)
        results.append({
            "axis": axis_label, "path": sc, "stats": stats,
            "verdict": verdict, "notes": notes,
        })
    return results


def apply_cross_axis_checks(results: List[Dict[str, Any]]) -> None:
    """Flag axes whose time-span differs significantly from the
    others — the classic 'stale file from a different tracking run'
    case, where e.g. sway/surge cover 12 minutes but stroke covers
    18 seconds because stroke is a leftover from an earlier
    truncated Quad run that never got overwritten.

    Mutates the results list in place: adds a prominent note to the
    top of each outlier axis's notes list and forces its verdict to
    FAIL so the overall summary also fails.

    Threshold: max/min span ratio > 2× triggers the check; an
    individual axis is flagged when its span is outside [0.5×,
    2.0×] of the median span across axes. Covers the dominant
    failure mode without nagging on incidental differences.
    """
    spans: List[Tuple[Dict[str, Any], float]] = []
    for r in results:
        t_span = r.get("stats", {}).get("t_span_s")
        if t_span is not None and t_span > 0:
            spans.append((r, float(t_span)))
    if len(spans) < 2:
        return
    max_span = max(s for _, s in spans)
    min_span = min(s for _, s in spans)
    if min_span <= 0 or (max_span / min_span) < 2.0:
        return

    span_values = [s for _, s in spans]
    median_span = statistics.median(span_values)
    if median_span <= 0:
        return

    for r, span in spans:
        ratio = span / median_span
        if ratio < 0.5 or ratio > 2.0:
            note = (
                f"span mismatch: this axis covers {span:.1f}s but "
                f"the median across axes is {median_span:.1f}s — "
                f"this file is likely stale from a different "
                f"tracking run. Delete the four .funscript sidecars "
                f"and re-track so all axes come from one session."
            )
            r["notes"].insert(0, note)
            r["verdict"] = "FAIL"


def _actions_fingerprint(path: Optional[Path]) -> Optional[str]:
    """SHA256 over the normalized actions list. Two axes with the
    same fingerprint are byte-identical in their action data, which
    is the tracker/file-writer bug we hit on Mask-Moments (primary
    signal got written into both the primary slot and the .x
    sidecar). Normalized form drops whitespace and orders keys so
    formatting differences don't mask true content identity."""
    if path is None:
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    actions = data.get("actions", [])
    if not isinstance(actions, list) or not actions:
        return None
    # Re-serialize as a canonical form: each action is {at: int, pos:
    # int} sorted by key, no whitespace. Ignores metadata and format
    # trivia; catches only actual signal-level duplication.
    try:
        canonical = json.dumps(
            [{"at": int(a.get("at", 0)), "pos": int(a.get("pos", 0))} for a in actions],
            separators=(",", ":"), sort_keys=True,
        )
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def apply_duplicate_content_check(results: List[Dict[str, Any]]) -> None:
    """Flag any two axes whose action arrays are byte-identical.
    Real trackers producing distinct axes never hit this by chance —
    if stroke and sway hash the same, one axis got written with the
    other axis's data and the file set is silently 2-DoF in a 4-DoF
    wrapper. Flagged axes are forced to FAIL with a pointer to the
    duplicate so the user can diagnose from the report."""
    fingerprints: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        fp = _actions_fingerprint(r.get("path"))
        if fp is None:
            continue
        fingerprints.setdefault(fp, []).append(r)

    for fp, group in fingerprints.items():
        if len(group) < 2:
            continue
        labels = [g["axis"] for g in group]
        for r in group:
            others = [label for label in labels if label != r["axis"]]
            others_str = ", ".join(others) if others else "another axis"
            note = (
                f"duplicate-content: this axis's action data is "
                f"BYTE-IDENTICAL to {others_str}. The tracker or file "
                f"writer is emitting the same signal into multiple "
                f"sidecars — downstream this is 2-DoF dressed as "
                f"3/4-DoF. File is readable but the spatial pipeline "
                f"will process redundant axes as if they were "
                f"independent."
            )
            r["notes"].insert(0, note)
            r["verdict"] = "FAIL"


def format_report(
    axis_name: str,
    path: Optional[Path],
    stats: Dict[str, Any],
    verdict: str,
    notes: List[str],
) -> str:
    lines = [f"[{verdict:>4}]  {axis_name}"]
    if path is None:
        lines.append("        file: (missing — tracker did not produce this axis)")
        for n in notes:
            lines.append(f"        - {n}")
        return "\n".join(lines)

    lines.append(f"        file: {path.name}")
    if stats.get("problem"):
        lines.append(f"        error: {stats['problem']}")
        return "\n".join(lines)

    lines.append(
        f"        actions: {stats['count']} over {stats['t_span_s']:.1f}s "
        f"({stats['rate_hz']:.1f} Hz)"
    )
    lines.append(
        f"        pos: min={stats['pos_min']} max={stats['pos_max']} "
        f"std={stats['pos_std']:.1f}"
    )
    lines.append(
        f"        near-center ({stats['frac_near_50']*100:.0f}%)  "
        f"active ({stats['frac_active']*100:.0f}%)"
    )
    for n in notes:
        lines.append(f"        - {n}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity-check FunGen multi-axis funscript output."
    )
    parser.add_argument(
        "video_path",
        help="Path to the source video (or the stem without extension)",
    )
    args = parser.parse_args()

    path = Path(args.video_path).expanduser()
    video_exts = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts"}
    stem = path.with_suffix("") if path.suffix.lower() in video_exts else path

    print(f"Checking sidecars for stem: {stem}")
    print("=" * 72)

    results = build_axis_results(stem)
    apply_cross_axis_checks(results)
    apply_duplicate_content_check(results)

    for r in results:
        print(format_report(
            r["axis"], r["path"], r["stats"],
            r["verdict"], r["notes"],
        ))

    # Overall verdict.
    verdicts = [r["verdict"] for r in results]
    if any(v == "FAIL" for v in verdicts):
        overall, summary = (
            "FAIL",
            "One or more axes unusable. Re-track before running through the pipeline.",
        )
    elif any(v == "WARN" for v in verdicts):
        overall, summary = (
            "WARN",
            "Usable, but check the flagged axes — output quality may be degraded.",
        )
    else:
        overall, summary = ("PASS", "Looks usable across all four axes.")

    print("=" * 72)
    print(f"Overall: {overall}")
    print(summary)

    if overall == "FAIL":
        raise SystemExit(2)
    if overall == "WARN":
        raise SystemExit(1)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
