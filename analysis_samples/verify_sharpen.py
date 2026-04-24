"""Verify: does the input_sharpen stage shift Mask-Moments signal
characteristics toward the target (Quad) profile?

Loads stroke from both sets, runs current through sharpener with
different param combinations, reports the metrics side-by-side.
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

import sys
sys.path.append(str(HERE.parent))
from processing.input_sharpen import sharpen_signal


def load_and_resample(path, hz=50.0):
    data = json.loads(Path(path).read_text())
    actions = data.get("actions", [])
    t = np.array([a["at"] for a in actions], dtype=float) / 1000.0
    p = np.array([a["pos"] for a in actions], dtype=float) / 100.0
    t_u = np.linspace(t[0], t[-1], int((t[-1] - t[0]) * hz) + 1)
    p_u = np.interp(t_u, t, p)
    return t_u, p_u


def metrics(p_u, fs=50.0):
    dp = np.abs(np.diff(p_u) * fs)
    ddp = np.abs(np.diff(dp) * fs)
    x = p_u - p_u.mean()
    n = 1 << (len(x) - 1).bit_length()
    spec = np.abs(np.fft.rfft(x, n))
    freqs = np.fft.rfftfreq(n, 1 / fs)
    band = (freqs >= 0.05) & (freqs <= 25.0)
    hi = (freqs > 3.0) & (freqs <= 25.0)
    return {
        "std": p_u.std(),
        "peak_v": np.quantile(dp, 0.99),
        "peak_a": np.quantile(ddp, 0.99),
        "hi_frac": (spec[hi].sum() / spec[band].sum()) * 100.0,
    }


def main():
    _, target = load_and_resample(HERE / "target/123.y.funscript")
    _, current = load_and_resample(HERE / "current/test video.funscript")

    print("Target (Quad) metrics:")
    tm = metrics(target)
    for k, v in tm.items():
        print(f"  {k:<10} {v:.3f}")
    print()
    print("Current (Mask-Moments) metrics:")
    cm = metrics(current)
    for k, v in cm.items():
        print(f"  {k:<10} {v:.3f}")
    print()

    print("Sharpened variants — goal is to move CURRENT metrics toward TARGET:")
    print(f"{'pre':>5} {'sat':>5}  {'std':>8} {'peak_v':>8} {'peak_a':>8} {'hi_frac':>8}")
    print(f"{'---':>5} {'---':>5}  {'---':>8} {'---':>8} {'---':>8} {'---':>8}")
    print(f"{'t':>5} {'':>5}  {tm['std']:>8.3f} {tm['peak_v']:>8.3f} {tm['peak_a']:>8.1f} {tm['hi_frac']:>8.1f}")
    print(f"{'c':>5} {'':>5}  {cm['std']:>8.3f} {cm['peak_v']:>8.3f} {cm['peak_a']:>8.1f} {cm['hi_frac']:>8.1f}")
    for pre in [0.0, 1.0, 1.5, 2.0, 2.5]:
        for sat in [0.0, 1.0, 2.0, 3.0]:
            if pre == 0.0 and sat == 0.0:
                continue
            shaped = sharpen_signal(
                current,
                pre_emphasis=pre,
                saturation=sat,
                pre_emphasis_cutoff_hz=3.0,
                sample_rate_hz=50.0,
            )
            m = metrics(shaped)
            flags = ""
            if abs(m['std'] - tm['std']) < 0.02: flags += "std "
            if abs(m['peak_v'] - tm['peak_v']) < 0.3: flags += "peak_v "
            if abs(m['peak_a'] - tm['peak_a']) < 15: flags += "peak_a "
            if abs(m['hi_frac'] - tm['hi_frac']) < 3: flags += "hi_frac"
            print(
                f"{pre:>5.1f} {sat:>5.1f}  "
                f"{m['std']:>8.3f} {m['peak_v']:>8.3f} {m['peak_a']:>8.1f} {m['hi_frac']:>8.1f}  {flags}"
            )


if __name__ == "__main__":
    main()
