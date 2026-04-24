"""
Shared tooltip helper.

Usage:
    from ui.tooltip_helper import create_tooltip
    create_tooltip(my_entry, "Explains what this setting does.")

The tooltip is lazily created after a short hover delay and destroyed
on <Leave>, so rapid mouse movement across many widgets (e.g. sweeping
the Spatial 3D tuning panel) does NOT trigger tooltip creation. Idle
cost is zero. Positioning is clamped against screen bounds so the
popup never lands off-screen.
"""

import tkinter as tk


# Hover delay before a tooltip appears. Long enough that rapid mouse
# sweeps across widgets don't trigger tooltip creation (which is
# expensive: Toplevel creation + update_idletasks + lift), short enough
# that a deliberate hover still feels responsive. 400ms matches most
# OS native tooltip timings.
_HOVER_DELAY_MS = 400


def create_tooltip(widget, text: str, wraplength: int = 420,
                   delay_ms: int = _HOVER_DELAY_MS) -> None:
    """Attach a hover tooltip to `widget`.

    Args:
        widget: Any Tk widget; the tooltip shows after `delay_ms` of
            steady hover and hides on leave.
        text: The tooltip body.
        wraplength: Pixel width at which the text wraps. 420 px fits
            most multi-sentence explanations on screen.
        delay_ms: Hover delay before the tooltip appears. Defaults to
            400 ms — rapid mouse movement across widgets produces zero
            tooltip creation, keeping the main event loop free for
            scheduled redraws (video frame ticks, etc.).
    """
    state = {'tooltip': None, 'scheduled_id': None,
             'screen_w': None, 'screen_h': None}

    def _actually_show():
        state['scheduled_id'] = None
        if state['tooltip']:
            return
        try:
            if not widget.winfo_ismapped():
                return
        except tk.TclError:
            return
        if state['screen_w'] is None:
            try:
                state['screen_w'] = widget.winfo_screenwidth()
                state['screen_h'] = widget.winfo_screenheight()
            except tk.TclError:
                state['screen_w'], state['screen_h'] = 1920, 1080
        try:
            x = widget.winfo_rootx() + widget.winfo_width() + 12
            y = widget.winfo_rooty()
        except tk.TclError:
            return
        approx_w = min(wraplength + 24, 460)
        if x + approx_w > state['screen_w']:
            x = max(4, widget.winfo_rootx() - approx_w - 4)
        if y + 60 > state['screen_h']:
            y = max(4, state['screen_h'] - 80)
        tip = tk.Toplevel(widget)
        try:
            tip.wm_overrideredirect(True)
        except tk.TclError:
            pass
        try:
            tip.wm_attributes('-topmost', True)
        except tk.TclError:
            pass
        tip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tip, text=text, background="lightyellow",
                         foreground="#000000",
                         relief=tk.SOLID, borderwidth=1,
                         font=('TkDefaultFont', 9),
                         wraplength=wraplength, justify=tk.LEFT,
                         padx=6, pady=4)
        label.pack()
        tip.update_idletasks()
        tip.lift()
        state['tooltip'] = tip

    def _cancel_scheduled():
        sid = state['scheduled_id']
        if sid is not None:
            try:
                widget.after_cancel(sid)
            except tk.TclError:
                pass
            state['scheduled_id'] = None

    def schedule_show(_event):
        # Cancel any already-scheduled show from a previous Enter that
        # didn't get a matching Leave (can happen on widget re-layout).
        _cancel_scheduled()
        if state['tooltip']:
            return
        try:
            state['scheduled_id'] = widget.after(delay_ms, _actually_show)
        except tk.TclError:
            pass

    def hide(_event):
        _cancel_scheduled()
        tip = state['tooltip']
        if tip is not None:
            try:
                tip.destroy()
            except tk.TclError:
                pass
            state['tooltip'] = None

    widget.bind('<Enter>', schedule_show, add='+')
    widget.bind('<Leave>', hide, add='+')
