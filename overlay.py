"""Recording indicator v10: black pill, white vertical-bar audio waveform.

Per final reference image: a dark capsule containing a row of white rounded
vertical bars (a classic audio waveform). Bars dance with the live mic level
while dictating; with no speech (or while ASR/LLM processing runs) each bar
settles to its minimum height, which with round caps reads as a simple white
dot. No colors, no glow, no border accents — black and white only.

Bottom-center, above the taskbar. Tk must own the MAIN thread on Windows
(run_forever); state setters are thread-safe.
"""
import math
import random
import threading
import time
import tkinter as tk

_PANEL_W, _PANEL_H = 118, 28      # compact capsule (true pill: end radius = height/2)
_PAD = 4
_W, _H = _PANEL_W + 2 * _PAD, _PANEL_H + 2 * _PAD
_KEY = "#010203"                  # transparent color key
_TASKBAR_CLEAR = 70               # clearance from screen bottom to the pill's bottom edge

_ALPHA = 0.92                     # nearly solid, like the reference image
_FILL = "#1c1c1c"                 # black pill body
_BAR = "#f5f2e8"                  # warm white bars (matches reference)
_BAR_ERR = "#e05a4f"              # soft red bars: a failed/empty dictation flash
_BAR_TIP = "#8b8a82"              # dim halo beyond each bar tip -> soft vibrating edge
_BAR_COUNT = 15
_BAR_WIDTH = 3                    # px; with round caps a minimum-height bar is a dot
_BAR_MIN, _BAR_MAX = 0.0, _PANEL_H * 0.60  # extra extent beyond the dot


class Overlay:
    def __init__(self):
        self._state = "hidden"    # hidden | recording | processing | error
        self._level = 0.0
        self._err_until = 0.0     # error flash auto-hides after this time

    # -- thread-safe API ----------------------------------------------------
    def set_level(self, level: float):
        self._level = min(1.0, max(0.0, level))

    def recording(self):
        self._state = "recording"

    def processing(self):
        """ASR/LLM running: pill stays visible, bars settle to dots."""
        self._state = "processing"
        self._level = 0.0

    def error(self, seconds: float = 1.6):
        """Flash the pill red briefly to signal a failed or empty dictation."""
        self._state = "error"
        self._level = 0.0
        self._err_until = time.time() + seconds

    def done(self):
        self._state = "hidden"
        self._level = 0.0

    # back-compat aliases used elsewhere in the app
    show, hide, off = recording, done, done

    # -- main-thread UI -------------------------------------------------------
    def run_forever(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-transparentcolor", _KEY)
        self.root.attributes("-alpha", _ALPHA)
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{_W}x{_H}+{(sw - _W) // 2}+{sh - _TASKBAR_CLEAR - _PAD - _PANEL_H}")
        self.canvas = tk.Canvas(self.root, width=_W, height=_H, bg=_KEY, highlightthickness=0)
        self.canvas.pack()
        self.root.withdraw()
        self._heights = [0.0] * _BAR_COUNT
        self._seeds = [random.random() * math.tau for _ in range(_BAR_COUNT)]
        self._phase = 0.0
        self._visible = False
        self._tick()
        self.root.mainloop()

    def start_in_thread(self):
        threading.Thread(target=self.run_forever, daemon=True).start()

    def _tick(self):
        if self._state == "error" and time.time() >= self._err_until:
            self._state = "hidden"  # error flash expired
        want_visible = self._state != "hidden"
        if want_visible and not self._visible:
            self.root.deiconify()
            self.root.attributes("-topmost", True)
            self._visible = True
        elif not want_visible and self._visible:
            self.root.withdraw()
            self._visible = False
        if self._visible:
            self._draw()
        self.root.after(33, self._tick)  # ~30fps for smooth bar motion

    def _draw(self):
        c = self.canvas
        c.delete("all")
        cx, cy = _W / 2, _H / 2
        self._draw_pill(cx, cy)
        self._phase += 0.35
        speaking = self._state == "recording" and self._level > 0.02
        spacing = _PANEL_W * 0.72 / (_BAR_COUNT - 1)
        start_x = cx - spacing * (_BAR_COUNT - 1) / 2
        error = self._state == "error"
        bar_color = _BAR_ERR if error else _BAR
        for i in range(_BAR_COUNT):
            if speaking:
                # per-bar jitter around the live level so the row dances like
                # a real waveform instead of one synchronized pulse
                wob = 0.5 + 0.5 * math.sin(self._phase * 2.1 + self._seeds[i] + i * 0.7)
                target = self._level * (0.25 + 0.75 * wob)
            elif error:
                target = 0.30 + 0.15 * math.sin(self._phase * 2.0 + i)  # gentle red pulse
            else:
                target = 0.0  # settle to dots (min-height bar + round caps = dot)
            self._heights[i] += (target - self._heights[i]) * 0.35
            half = (_BAR_MIN + self._heights[i] * _BAR_MAX) / 2
            x = start_x + i * spacing
            # soft tips: a dimmer, thinner halo extends slightly past each end
            # so bars fade out instead of ending as hard rectangles
            if half > 1.0:
                c.create_line(x, cy - half - 2.5, x, cy + half + 2.5, fill=_BAR_TIP,
                              width=max(1, _BAR_WIDTH - 2), capstyle="round")
            c.create_line(x, cy - half, x, cy + half, fill=bar_color,
                          width=_BAR_WIDTH, capstyle="round")

    def _draw_pill(self, cx, cy):
        r = _PANEL_H / 2
        x0, x1 = cx - _PANEL_W / 2, cx + _PANEL_W / 2
        y0, y1 = cy - _PANEL_H / 2, cy + _PANEL_H / 2
        c = self.canvas
        c.create_oval(x0, y0, x0 + _PANEL_H, y1, fill=_FILL, outline=_FILL)
        c.create_oval(x1 - _PANEL_H, y0, x1, y1, fill=_FILL, outline=_FILL)
        c.create_rectangle(x0 + r, y0, x1 - r, y1, fill=_FILL, outline=_FILL)


if __name__ == "__main__":
    o = Overlay()

    def demo():
        time.sleep(0.5)
        o.recording()
        t0 = time.time()
        while time.time() - t0 < 3:          # speaking: bars dance
            o.set_level(abs(math.sin((time.time() - t0) * 4)) * 0.8)
            time.sleep(0.03)
        o.set_level(0.0)                     # silence while recording: dots
        time.sleep(1.5)
        o.processing()                       # processing: dots
        time.sleep(1.5)
        o.done()
        time.sleep(0.5)
        o.error()                            # failed dictation: red flash, auto-hides
        assert o._state == "error"
        time.sleep(2.0)
        assert o._state == "hidden", "error flash did not auto-hide"
        print("overlay v10 demo OK (incl. error flash)")
        o.root.after(0, o.root.destroy)

    threading.Thread(target=demo, daemon=True).start()
    o.run_forever()
