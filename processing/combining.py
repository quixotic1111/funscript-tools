import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from funscript import Funscript


def combine_funscripts(left_funscript, right_funscript, ratio, rest_level=0.5, ramp_up_duration=0.0):
    """
    Combine two funscripts using weighted average.

    Args:
        left_funscript: First funscript
        right_funscript: Second funscript
        ratio: Combining ratio (2 for 50/50, 4 for 75/25, etc)
        rest_level: Multiplier when either input is 0
        ramp_up_duration: Duration in seconds to ramp from rest_level back to normal after rest (0 = instant)
    """
    # Get union of all timestamps
    x = np.union1d(left_funscript.x, right_funscript.x)

    # Interpolate both signals to common timestamps
    y_left = np.interp(x, left_funscript.x, left_funscript.y)
    y_right = np.interp(x, right_funscript.x, right_funscript.y)

    # Calculate the weighted average
    y = (y_left * (ratio - 1) + y_right) / ratio

    # Determine which points are at rest (either input is 0)
    is_rest = (y_left == 0) | (y_right == 0)

    # Apply rest_level immediately when at rest
    y_with_rest = np.where(is_rest, y * rest_level, y)

    # Apply ramp-up after rest periods if duration > 0
    if ramp_up_duration > 0:
        # Find all rest->active transition points
        transition_times = []
        for i in range(1, len(is_rest)):
            if is_rest[i-1] and not is_rest[i]:
                # Transition from rest to active at index i
                transition_times.append(x[i])

        # For each point, calculate time relative to nearest rest->active transition
        # Ramp is centered on the transition: starts at -duration/2, ends at +duration/2
        time_relative_to_transition = np.full_like(x, np.inf)
        half_duration = ramp_up_duration / 2.0

        for i in range(len(x)):
            # Find the nearest transition time
            if len(transition_times) > 0:
                # Calculate time difference to each transition
                time_diffs = [x[i] - t for t in transition_times]

                # Find the transition that this point is closest to (and after or near)
                # We want transitions where the point is within the ramp window
                for t in transition_times:
                    time_diff = x[i] - t
                    # Check if point is within ramp window: from -half_duration to +half_duration
                    if -half_duration <= time_diff <= half_duration:
                        time_relative_to_transition[i] = time_diff
                        break

        # Calculate ramp progress centered on transition
        # At transition - half_duration: ramp_progress = 0 (start of ramp, at rest_level)
        # At transition: ramp_progress = 0.5 (middle of ramp)
        # At transition + half_duration: ramp_progress = 1.0 (end of ramp, at normal level)
        ramp_progress = (time_relative_to_transition + half_duration) / ramp_up_duration
        ramp_progress = np.clip(ramp_progress, 0.0, 1.0)

        # Interpolate from rest_level to 1.0
        ramp_multiplier = rest_level + (1.0 - rest_level) * ramp_progress

        # Determine which points are in a ramp window
        in_ramp_window = np.isfinite(time_relative_to_transition)

        # Apply ramp multiplier to points in ramp window, rest_level to rest points outside ramp
        y_final = np.where(in_ramp_window, y * ramp_multiplier, y_with_rest)
    else:
        # No ramp-up, use immediate rest_level application
        y_final = y_with_rest

    return Funscript(x, y_final)


def multiply_funscripts(left_funscript, right_funscript):
    """Multiply two funscripts point-wise."""
    x = np.union1d(left_funscript.x, right_funscript.x)
    y = np.interp(x, left_funscript.x, left_funscript.y) * np.interp(x, right_funscript.x, right_funscript.y)
    return Funscript(x, y)


def apply_direction_bias(carrier, source, bias, polarity="up_higher",
                         smoothing_s=0.3, grid_hz=50.0):
    """Modulate a carrier funscript by the instantaneous direction of a source.

    y_out = clip(y_carrier * (1 + bias * dir), 0, 1)

    `dir` ∈ [-1, +1] is the smoothed sign of d(source)/dt. With polarity
    "up_higher", up-strokes (dy/dt > 0) raise the carrier and down-strokes
    lower it; "down_higher" flips the sign.

    Args:
        carrier: Funscript to modulate (the carrier frequency).
        source: Funscript whose direction drives the bias (typically the
            raw input position).
        bias: Strength of the bias, typically 0.0 to 0.5. 0.0 is a no-op.
        polarity: "up_higher" or "down_higher".
        smoothing_s: Seconds of moving-average smoothing on the sign signal.
            Larger values produce gentler transitions across turnarounds;
            0 means hard ±1 switches.
        grid_hz: Internal sampling rate for velocity/sign computation.
    """
    if bias <= 0.0 or len(source.x) < 2 or len(carrier.x) < 1:
        return carrier

    src_x = np.asarray(source.x, dtype=float)
    src_y = np.asarray(source.y, dtype=float)
    t0, t1 = src_x[0], src_x[-1]
    if t1 <= t0:
        return carrier

    # Resample source onto a uniform grid, compute sign of velocity.
    dt = 1.0 / grid_hz
    uniform_t = np.arange(t0, t1 + dt, dt)
    uniform_y = np.interp(uniform_t, src_x, src_y)
    velocity = np.gradient(uniform_y, uniform_t)
    sign = np.sign(velocity)

    # Smooth the sign signal into a continuous [-1, +1] direction signal.
    if smoothing_s > 0:
        window = max(1, int(round(smoothing_s * grid_hz)))
        if window > 1:
            kernel = np.ones(window) / window
            sign = np.convolve(sign, kernel, mode="same")

    if polarity == "down_higher":
        sign = -sign

    # Interpolate the direction signal onto carrier timestamps.
    carrier_x = np.asarray(carrier.x, dtype=float)
    carrier_y = np.asarray(carrier.y, dtype=float)
    dir_signal = np.interp(carrier_x, uniform_t, sign)

    y_out = np.clip(carrier_y * (1.0 + bias * dir_signal), 0.0, 1.0)
    return Funscript(carrier_x, y_out)