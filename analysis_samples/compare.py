"""Compare signal characteristics between target and current funscript sets.

Run: python3 compare.py
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
TARGET = HERE / "target"
CURRENT = HERE / "current"


def load_actions(path):
    data = json.loads(Path(path).read_text())
    actions = data.get("actions", [])
    t = np.array([a["at"] for a in actions], dtype=float) / 1000.0  # seconds
    p = np.array([a["pos"] for a in actions], dtype=float) / 100.0  # [0,1]
    return t, p


def resample_uniform(t, p, hz=50.0):
    t0, t1 = t[0], t[-1]
    n = int((t1 - t0) * hz) + 1
    t_uniform = np.linspace(t0, t1, n)
    p_uniform = np.interp(t_uniform, t, p)
    return t_uniform, p_uniform


def describe(name, t, p):
    dt = np.diff(t)
    hz = 1.0 / np.median(dt) if len(dt) > 0 else 0
    t_u, p_u = resample_uniform(t, p, 50.0)

    # Basic stats
    center = p_u - 0.5
    std = center.std()
    rng = p_u.max() - p_u.min()

    # Derivative (velocity): |dp/dt|
    dp = np.diff(p_u) * 50.0  # per second
    abs_dp = np.abs(dp)
    peak_v = np.quantile(abs_dp, 0.99) if len(abs_dp) else 0
    mean_v = abs_dp.mean() if len(abs_dp) else 0

    # Second derivative (acceleration / transient energy)
    ddp = np.diff(dp) * 50.0  # per second^2
    abs_ddp = np.abs(ddp)
    peak_a = np.quantile(abs_ddp, 0.99) if len(abs_ddp) else 0
    mean_a = abs_ddp.mean() if len(abs_ddp) else 0

    # Zero-crossing rate (around center = 0.5)
    zc = np.sum(np.diff(np.signbit(center)) != 0)
    zc_rate = zc / (t_u[-1] - t_u[0]) if len(t_u) > 1 else 0

    # FFT — spectral centroid + where energy concentrates
    x = center - center.mean()
    # Zero-pad to next power of 2 for FFT speed
    n_fft = 1 << (len(x) - 1).bit_length()
    spectrum = np.abs(np.fft.rfft(x, n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1/50.0)
    # Only consider [0.05, 25] Hz band
    band = (freqs >= 0.05) & (freqs <= 25.0)
    if spectrum[band].sum() > 0:
        centroid_hz = (freqs[band] * spectrum[band]).sum() / spectrum[band].sum()
        # Peak frequency
        peak_idx = band.nonzero()[0][spectrum[band].argmax()]
        peak_hz = freqs[peak_idx]
        # High-band energy fraction: >3 Hz vs all
        hi = (freqs > 3.0) & (freqs <= 25.0)
        hi_frac = spectrum[hi].sum() / spectrum[band].sum()
    else:
        centroid_hz = peak_hz = hi_frac = 0

    print(f"\n--- {name} ---")
    print(f"  samples raw:       {len(t)}  duration {t[-1]-t[0]:.1f}s  rate {hz:.1f} Hz")
    print(f"  after 50Hz resample: {len(t_u)} samples")
    print(f"  pos range:          [{p_u.min():.2f}, {p_u.max():.2f}]  std {std:.3f}  span {rng:.2f}")
    print(f"  |velocity|:  peak99 {peak_v:.2f}/s  mean {mean_v:.2f}/s")
    print(f"  |accel|:     peak99 {peak_a:.1f}/s^2 mean {mean_a:.1f}/s^2")
    print(f"  zero-crossings: {zc_rate:.2f} Hz  (rate of direction changes)")
    print(f"  spectrum:    peak {peak_hz:.2f} Hz  centroid {centroid_hz:.2f} Hz  hi-band(>3Hz) {hi_frac*100:.1f}%")


def main():
    print("=" * 78)
    print("TARGET (Quad, 'right feel')")
    print("=" * 78)
    # Quad uses .x=sway, .y=stroke, .z=surge — the canonical xyzw mapping
    for label, fname in (
        ("target stroke (y)", "123.y.funscript"),
        ("target sway   (x)", "123.x.funscript"),
        ("target surge  (z)", "123.z.funscript"),
    ):
        p = TARGET / fname
        if p.exists():
            t, y = load_actions(p)
            describe(label, t, y)

    print("\n" + "=" * 78)
    print("CURRENT (Mask-Moments, 'unfocused')")
    print("=" * 78)
    for label, fname in (
        ("current stroke",       "test video.funscript"),
        ("current sway",         "test video.sway.funscript"),
        ("current surge",        "test video.surge.funscript"),
        ("current roll",         "test video.roll.funscript"),
    ):
        p = CURRENT / fname
        if p.exists():
            t, y = load_actions(p)
            describe(label, t, y)

    print("\n" + "=" * 78)
    print("SIDE-BY-SIDE: stroke axis (target .y vs current primary)")
    print("=" * 78)
    t_tgt, y_tgt = load_actions(TARGET / "123.y.funscript")
    t_cur, y_cur = load_actions(CURRENT / "test video.funscript")

    def brief(name, t, p):
        t_u, p_u = resample_uniform(t, p, 50.0)
        dp = np.abs(np.diff(p_u) * 50.0)
        ddp = np.abs(np.diff(dp) * 50.0)
        x = (p_u - p_u.mean())
        n = 1 << (len(x) - 1).bit_length()
        spec = np.abs(np.fft.rfft(x, n))
        freqs = np.fft.rfftfreq(n, 1/50.0)
        band = (freqs >= 0.05) & (freqs <= 25.0)
        hi = (freqs > 3.0) & (freqs <= 25.0)
        return {
            "raw_hz": 1/np.median(np.diff(t)),
            "std": p_u.std(),
            "peak_v": np.quantile(dp, 0.99),
            "peak_a": np.quantile(ddp, 0.99),
            "hi_frac": spec[hi].sum() / spec[band].sum(),
        }

    a = brief("target", t_tgt, y_tgt)
    b = brief("current", t_cur, y_cur)
    print(f"{'metric':<20} {'target':>12} {'current':>12} {'ratio t/c':>12}")
    print("-" * 58)
    for k in ("raw_hz", "std", "peak_v", "peak_a", "hi_frac"):
        av, bv = a[k], b[k]
        r = av / bv if bv > 0 else float('inf')
        print(f"{k:<20} {av:>12.3f} {bv:>12.3f} {r:>12.2f}x")


if __name__ == "__main__":
    main()
