"""
Shared curve-family parameter UI builder.

Used by the Trochoid Quantization and Trochoid Spatial tabs. Both tabs
let the user edit per-family parameters (R/r/d for hypo/epi, a/k for
rose, etc.) in a small grid, with short one-line hints inline and
longer explanations on hover. Previously this was duplicated as
`_trochoid_build_param_ui` and `_ts_build_param_ui` — this module
consolidates the help dict and the build routine.
"""

import tkinter as tk
from tkinter import ttk

from ui.tooltip_helper import create_tooltip


# (short_hint, long_tooltip) per param name. Used across all curve
# families. Lissajous's 'a' gets overridden at build time below because
# it means "X frequency" rather than "amplitude" for that family.
FAMILY_PARAM_HELP = {
    'R': ('Outer (fixed) circle radius',
          'Outer (fixed) circle radius. The bounding container inside '
          'which (hypo) or outside which (epi) the rolling circle '
          'travels. Larger R = larger overall pattern. The R/r ratio '
          'sets the symmetry: integer ratios produce a closed curve '
          '(e.g. R=5, r=3 closes after 3 turns); irrational ratios '
          'trace forever without closing.'),
    'r': ('Rolling circle radius (>0; smaller r = more lobes)',
          'Rolling circle radius (must be > 0). Smaller r relative to '
          'R = more lobes/petals. For integer R/r ratios, the curve '
          'has roughly R/gcd(R,r) lobes. Try r near R/3 to R/5 for '
          'classic Spirograph-style patterns; r close to R gives soft '
          'single-loop shapes.'),
    'd': ('Pen offset; d=r=cusps, d<r=curtate, d>r=prolate',
          'Pen offset — distance from rolling-circle center to the '
          'tracing point. Three regimes: (1) d = r → cusps (true '
          'hypo/epicycloid, sharp points); (2) d < r → curtate '
          'trochoid (rounded loops, no cusps); (3) d > r → prolate '
          'trochoid (self-intersecting loops/knots).'),
    'a': ('Amplitude / petal length',
          'Amplitude / petal length. Sets the maximum radius of each '
          'petal.'),
    'k': ('Petal frequency (odd k=k petals, even k=2k petals)',
          'Petal frequency. If k is odd the rose has k petals; if k '
          'is even it has 2k petals. Non-integer k traces a '
          'non-closing pattern.'),
    'A': ('X amplitude', 'X amplitude — half-width of the figure.'),
    'B': ('Y amplitude', 'Y amplitude — half-height of the figure.'),
    'b': ('Y frequency',
          'Y frequency. The a:b ratio determines the figure structure '
          '(3:2, 5:4, etc.). Integer ratios close.'),
    'delta': ('Phase offset (radians)',
              'Phase offset (radians). delta=pi/2 gives the classic '
              'upright Lissajous; delta=0 collapses to a line for '
              'a:b=1.'),
    'scale': ('Overall scale factor',
              'Overall scale factor for the butterfly curve. Affects '
              'the bounding box only; level distribution is invariant.'),
    'm': ('Symmetry order',
          'Symmetry order. Integer m gives m-fold symmetry (m=4 → '
          'square-like, m=6 → hexagonal, etc.).'),
    'n1': ('Shape exponent 1 (>0)',
           'Shape exponent 1 (must be > 0). Lower values produce more '
           'pronounced features; higher values smooth them.'),
    'n2': ('Shape exponent 2',
           'Shape exponent 2. Together with n3, controls how rounded '
           'vs. pinched the corners are.'),
    'n3': ('Shape exponent 3',
           'Shape exponent 3. n2=n3 gives symmetric corners.'),
    'x_expr': ('x(t) — numpy math (sin, cos, exp, sqrt, pi, ...)',
               'x(t) expression. Available: sin cos tan asin acos atan '
               'atan2 sinh cosh tanh exp log log2 log10 sqrt cbrt abs '
               'sign floor ceil round pow minimum maximum, plus pi e '
               'tau. t is in [0, 2*pi). Example: sin(3*t) + 0.5*cos(7*t).'),
    'y_expr': ('y(t) — numpy math (sin, cos, exp, sqrt, pi, ...)',
               'y(t) expression. Same available functions as x(t). '
               'Example: cos(2*t) + 0.3*sin(5*t).'),
    # 3D-curve-specific params. Names that overlap with trochoid
    # families (R, r, a, b, scale) keep their existing trochoid-
    # centric hints above — callers that want 3D-flavored hints pass
    # a per-call override dict to build_family_param_grid.
    'h': ('Total height along z (helix)',
          'Total height the helix spans along the z-axis. Centered '
          'around z=0. Larger h = taller spiral; compare to r to set '
          'the aspect ratio (tall vs squat).'),
    'turns': ('Number of full revolutions (helix)',
              'How many full 360° revolutions the helix completes '
              'over the parameter range. Larger = tighter spiral; '
              'non-integer closes the curve at a non-matching angle.'),
    'p': ('p winding (torus knot — axis wrap count)',
          'Number of times the torus knot winds around the central '
          'axis of the torus. (p, q) = (2, 3) is the trefoil. p and '
          'q must be coprime for a true knot.'),
    'q': ('q winding (torus knot — tube wrap count)',
          'Number of times the torus knot winds around the tube of '
          'the torus. (p, q) must be coprime for a true knot. Swap p '
          'and q to trace the same knot from a different starting '
          'parameterization.'),
    'C': ('Z amplitude (3D Lissajous)',
          'Amplitude of the z-axis sinusoid. Sets the vertical '
          'extent of the 3D Lissajous figure.'),
    'c': ('Z frequency / longitudinal loops',
          'For 3D Lissajous: Z-axis frequency. The a:b:c ratio '
          'determines whether the curve is closed (rational) or '
          'space-filling (irrational). For spherical_spiral: how '
          'many full longitudinal loops the spiral makes per '
          'pole-to-pole pass.'),
    'phi': ('Phase offset on X (radians, 3D Lissajous)',
            'Phase offset added to the X sinusoid before evaluation. '
            'π/2 = the classic perpendicular Lissajous figure; 0 '
            'collapses x and y into a line when a = b.'),
    'psi': ('Phase offset on Z (radians, 3D Lissajous)',
            'Phase offset added to the Z sinusoid before evaluation. '
            'Changes the roll of the curve around its principal axis.'),
}


# 3D-curve-specific aliases: when a 3D curve family uses a parameter
# name that overlaps with a trochoid meaning, the 3D tab can pass
# this dict to build_family_param_grid as `help_overrides` to show
# the 3D-flavored hint instead of the (legacy trochoid) hint.
HELP_OVERRIDES_3D_CURVE = {
    'r': ('Radius (helix) / minor radius (torus knot)',
          'Radius. For helix: radius of the spiral in the xy-plane. '
          'For torus_knot: minor radius (tube thickness) — compare '
          'to R (major radius, distance from torus center to tube '
          'center). Larger = fatter tube / wider spiral.'),
    'R': ('Major radius (torus knot)',
          'Major radius of the torus the knot winds around. Distance '
          'from the torus center to the centerline of the tube. '
          'Usually several times larger than r.'),
    'a': ('X frequency (3D Lissajous)',
          'X-axis frequency of the 3D Lissajous figure. The a:b:c '
          'ratio determines the figure structure. Integer ratios '
          'give closed curves.'),
    'b': ('Y frequency (3D Lissajous)',
          'Y-axis frequency of the 3D Lissajous figure. Together '
          'with a and c, determines the figure structure.'),
}


def build_family_param_grid(parent, family, family_specs, param_vars,
                             help_overrides=None):
    """Rebuild `parent`'s children as a grid of per-param widgets.

    Args:
        parent: ttk.LabelFrame (or similar) whose children will be
            cleared and replaced with the new param widgets.
        family: key into `family_specs`, e.g. 'hypo', 'rose'.
        family_specs: dict from curve_family name to {params: {...}}.
        param_vars: nested dict {family_name: {param_name: tk.Var}}
            containing the Tk variables bound to each entry.
        help_overrides: optional dict {param_name: (short, long)} that
            replaces entries in the global FAMILY_PARAM_HELP for the
            duration of this call. Useful when the same param name
            means different things in different families (e.g., 'r' is
            "rolling circle" in trochoid but "radius" in helix).
    """
    for child in parent.winfo_children():
        child.destroy()
    spec = family_specs.get(family)
    if not spec:
        return

    # Lissajous reuses 'a' as a frequency, not amplitude; override its
    # hint locally.
    help_dict = FAMILY_PARAM_HELP
    if family == 'lissajous':
        help_dict = dict(FAMILY_PARAM_HELP)
        help_dict['a'] = ('X frequency',
                          'X frequency. The a:b ratio determines the '
                          'figure structure (3:2, 5:4, etc.). Integer '
                          'ratios produce a closed Lissajous figure.')
    if help_overrides:
        help_dict = dict(help_dict)
        help_dict.update(help_overrides)

    for i, (pname, default_v) in enumerate(spec['params'].items()):
        short, long_ = help_dict.get(pname, ('', ''))
        ttk.Label(parent, text=f"{pname}:").grid(
            row=i, column=0, sticky=tk.W, padx=4, pady=2)
        var = param_vars.get(family, {}).get(pname)
        entry_w = 32 if isinstance(default_v, str) else 10
        entry = ttk.Entry(parent, textvariable=var, width=entry_w)
        entry.grid(row=i, column=1, sticky=tk.W, padx=4, pady=2)
        hint_label = ttk.Label(parent, text=short, foreground='#555555')
        hint_label.grid(row=i, column=2, sticky=tk.W, padx=8, pady=2)
        if long_:
            create_tooltip(entry, long_)
            create_tooltip(hint_label, long_)
