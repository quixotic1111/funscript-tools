import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from funscript import Funscript


def make_volume_ramp(input_funscript, ramp_percent_per_hour=15):
    """
    Create a volume ramp with 4 key points based on input funscript timing.
    Pattern: Start (calculated) → Rise (calculated) → Peak (1.0) → End (0)

    Args:
        input_funscript: Input funscript for timing reference
        ramp_percent_per_hour: Volume ramp rate percentage per hour (0-40)
    """
    if len(input_funscript.x) < 4:
        raise ValueError("Input funscript must have at least 4 actions to create volume ramp.")

    # Set timing: Start, 10 seconds, second-to-last, and last
    start_time = input_funscript.x[0]
    second_time = start_time + 10.0  # Fixed at 10 seconds from start
    peak_time = input_funscript.x[-2]  # Second-to-last point (where value reaches 1.0)
    end_time = input_funscript.x[-1]   # Last point

    # Calculate file duration in hours
    file_duration_hours = (peak_time - start_time) / 3600.0

    # Calculate total ramp increase as a decimal (e.g., 15% = 0.15)
    total_ramp_increase = (ramp_percent_per_hour / 100.0) * file_duration_hours

    # Calculate starting ramp value (what percentage to start at)
    # Start high and ramp up to 1.0 (100%)
    start_ramp_value = max(0.0, 1.0 - total_ramp_increase)

    # Calculate intermediate value at 10 seconds
    # Linear interpolation from start to peak
    time_progress = (second_time - start_time) / (peak_time - start_time)
    intermediate_value = start_ramp_value + (1.0 - start_ramp_value) * time_progress

    # Set timing and positions
    x = [start_time, second_time, peak_time, end_time]
    y = [0, start_ramp_value, 1.0, 0.0]

    return Funscript(x, y)


def make_volume_ramp_per_clip(input_funscript, ramp_percent_total=40):
    """
    Create a volume ramp that rises linearly across the whole clip.

    Unlike ``make_volume_ramp`` (which is rate-calibrated in %/hour and
    applies a 10-second soft start followed by a long flat plateau),
    this variant treats the setting as the TOTAL percent rise across
    the clip's duration:

        at clip start:          pos = 1.0 - ramp_percent_total/100
        at clip end (pre-fade): pos = 1.0
        at the very last point: pos = 0.0  (safety fade-out)

    So at 40%, a 73-second clip opens at 60% power and rises linearly
    to 100% by the penultimate sample, then fades to 0 on the last
    sample. Better mental model for short clips than per-hour rate.

    Args:
        input_funscript: Input funscript for timing reference (only
            uses ``x[0]``, ``x[-2]``, ``x[-1]``).
        ramp_percent_total: Total percent rise across the clip (0-100).
            40 means the clip ends 40 percentage points higher than
            it started. Clamped to [0, 100].
    """
    if len(input_funscript.x) < 4:
        raise ValueError(
            "Input funscript must have at least 4 actions to create "
            "a volume ramp.")

    pct = max(0.0, min(100.0, float(ramp_percent_total)))
    start_time = input_funscript.x[0]
    peak_time = input_funscript.x[-2]
    end_time = input_funscript.x[-1]
    start_ramp_value = max(0.0, 1.0 - pct / 100.0)

    # Three-point envelope — no mid-point like the per-hour variant
    # because the rise is the entire clip, not a 10-second soft start.
    x = [start_time, peak_time, end_time]
    y = [start_ramp_value, 1.0, 0.0]
    return Funscript(x, y)