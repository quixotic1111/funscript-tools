"""Non-blocking info dialog.

``tkinter.messagebox.showinfo`` maps to a native ``NSAlert`` on macOS,
which suspends the entire Tk event loop until the user dismisses it.
That freezes every ``after`` callback in the process — including the
T-code preview's playhead tick, which makes the live stream appear
stuck on the same frame while the "processing complete" dialog is up.

``show_nonblocking_info`` is a drop-in replacement for the informational
cases: it creates a plain Toplevel with an OK button, does NOT call
``grab_set``, and returns immediately. Other windows (live preview,
video playback) keep ticking. The user can still dismiss it with OK,
Enter, or Escape.

Use for post-processing success/completion notifications. Errors that
require user acknowledgement before continuing should stay on
``messagebox.showerror`` (blocking is correct there).
"""

import tkinter as tk
from tkinter import ttk


def show_nonblocking_info(parent, title, message):
    """Show a modeless info dialog that does not block the Tk event loop."""
    win = tk.Toplevel(parent)
    win.title(title)
    win.transient(parent)
    win.resizable(False, False)

    frame = ttk.Frame(win, padding=16)
    frame.pack(fill='both', expand=True)
    ttk.Label(frame, text=message, wraplength=420,
              justify='left').pack(anchor='w')
    ttk.Button(frame, text="OK", command=win.destroy,
               default='active').pack(pady=(12, 0))

    win.bind('<Return>', lambda _e: win.destroy())
    win.bind('<Escape>', lambda _e: win.destroy())

    win.update_idletasks()
    try:
        px = parent.winfo_rootx()
        py = parent.winfo_rooty()
        pw = parent.winfo_width()
        ph = parent.winfo_height()
        w = win.winfo_width()
        h = win.winfo_height()
        win.geometry(f"+{px + max(0, (pw - w) // 2)}"
                     f"+{py + max(0, (ph - h) // 3)}")
    except tk.TclError:
        pass

    win.focus_set()
    return win
