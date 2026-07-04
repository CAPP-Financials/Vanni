"""Vanni (वाणी, "speech") — local push-to-talk dictation.

Hold Ctrl+Win: speak, release -> transcribe -> clean (if >=10 words) -> paste at cursor.
Hold Ctrl+Alt+Win: same but always raw (no LLM cleanup).
Text always remains in the clipboard for re-pasting elsewhere.

Headless test mode: python vanni.py --simulate fixtures/fox.wav [--target notepad] [--raw]
"""
import argparse
import sys
import threading
import time
import tomllib

import numpy as np

import asr
import corrections
import formatter
import history
import injector
import smartfmt
from paths import BASE

CONFIG = tomllib.loads((BASE / "config.toml").read_text(encoding="utf-8"))

SAMPLE_RATE = 16000
MIN_SECONDS = CONFIG["audio"]["min_seconds"]


def ensure_ollama() -> bool:
    """Audit the Ollama port; spawn `ollama serve` in the background if it's down.
    Returns True once the API responds (False after ~20s of trying)."""
    import socket
    import subprocess

    host, port = "127.0.0.1", 11434
    def up() -> bool:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            return False

    if up():
        return True
    print("Ollama is down — starting it...")
    flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
    try:
        subprocess.Popen(["ollama", "serve"], creationflags=flags,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("ollama executable not found on PATH — cleanup will degrade to raw text")
        return False
    for _ in range(40):
        if up():
            print("Ollama started.")
            return True
        time.sleep(0.5)
    print("Ollama did not come up in 20s — cleanup will degrade to raw text")
    return False


class Pipeline:
    def __init__(self):
        import firstrun
        ensure_ollama()
        firstrun.ensure_models()  # first launch: pull missing Ollama models
        if not firstrun._MARKER.exists():
            print("first launch: the speech model (~1.5GB) downloads automatically below — one time only")
        print("loading ASR model...")
        self.model, desc = asr.load_model()
        asr.transcribe(self.model, np.zeros(SAMPLE_RATE // 2, dtype="float32"))  # CUDA warmup
        print(f"ASR ready: {desc}")
        print("warming up formatter...", "ok" if formatter.warm_up() else "OLLAMA DOWN (raw mode only)")

    def process(self, audio: np.ndarray, use_formatter: bool = True) -> dict:
        """audio: float32 mono 16k. Returns timing/result dict; injects on success."""
        result = {"text": "", "asr_s": 0.0, "format_s": 0.0, "total_s": 0.0,
                  "injected": False, "status": "no_speech"}
        if len(audio) < MIN_SECONDS * SAMPLE_RATE:
            return result  # too short -> no_speech
        t0 = time.perf_counter()
        raw, result["asr_s"] = asr.transcribe(self.model, audio)
        # deterministic mishear fixes, zero latency
        text, fixes = corrections.apply_verbose(raw)
        formatted = False
        fmt_status = "skipped"
        if text:
            if use_formatter and CONFIG["formatter"]["enabled"]:
                t1 = time.perf_counter()
                text, fmt_status = formatter.clean(text)
                result["format_s"] = time.perf_counter() - t1
                formatted = True
            text = smartfmt.apply(text)  # deterministic spoken→written, final authority
            result["injected"] = injector.inject(text)
            if not result["injected"]:
                result["status"] = "paste_failed"
            elif injector.is_foreground_elevated() and not injector.self_elevated():
                # inject() returned True (clipboard stuck) but an elevated window
                # silently drops the synthetic Ctrl+V — warn instead of failing silently
                result["status"] = "paste_blocked"
            elif fmt_status == "degraded":
                result["status"] = "ollama_offline_raw"
            else:
                result["status"] = "ok"
        result["text"] = text
        result["total_s"] = time.perf_counter() - t0
        history.record(raw, text, mode="formatted" if formatted else "raw",
                       corrections_applied=fixes, duration_s=result["total_s"])
        return result


def _resolve_device(v):
    """config [audio] device -> sounddevice arg: "" -> None (system default)."""
    return None if v in (None, "") else v


def _persist_device(value) -> None:
    """Rewrite the `device = ...` line in config.toml so a tray choice sticks.
    config.toml is our own controlled file with exactly one such line."""
    import re
    p = BASE / "config.toml"
    literal = f"device = {value}" if isinstance(value, int) else f'device = "{value}"'
    text = re.sub(r"(?m)^device = .*$", literal, p.read_text(encoding="utf-8"))
    p.write_text(text, encoding="utf-8")


class Recorder:
    """Captures mic audio between hotkey press and release."""

    def __init__(self, on_level=None, device=None):
        import sounddevice as sd
        self.sd = sd
        self.chunks: list[np.ndarray] = []
        self.stream = None
        self.on_level = on_level  # fed mic RMS for the recording overlay
        self.device = device  # None = system default input; int index or name otherwise

    def start(self):
        self.chunks = []

        def cb(indata, *_):
            self.chunks.append(indata.copy())
            if self.on_level:
                # RMS scaled so normal speech spans the bar range
                self.on_level(min(1.0, float(np.sqrt((indata ** 2).mean())) * 12))

        self.stream = self.sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=cb,
            device=self.device,
        )
        self.stream.start()

    def stop(self) -> np.ndarray:
        self.stream.stop()
        self.stream.close()
        return np.concatenate(self.chunks)[:, 0] if self.chunks else np.zeros(0, dtype="float32")


def run_tray(pipeline: Pipeline):
    """Tray icon + hotkey loop (the interactive app)."""
    import keyboard
    import pystray
    from PIL import Image, ImageDraw

    state = {"status": "idle", "cleanup": CONFIG["formatter"]["enabled"]}

    def make_icon(color):
        img = Image.new("RGB", (64, 64), "white")
        ImageDraw.Draw(img).ellipse([8, 8, 56, 56], fill=color)
        return img

    ICONS = {"idle": make_icon("#3b82f6"), "recording": make_icon("#ef4444"), "busy": make_icon("#f59e0b")}
    icon = pystray.Icon("vanni", ICONS["idle"], "Vanni — idle")

    def set_status(s):
        state["status"] = s
        icon.icon = ICONS[s]
        icon.title = f"Vanni — {s}"

    from overlay import Overlay
    indicator = Overlay()  # UI runs on the main thread at the end of this function
    recorder = Recorder(on_level=indicator.set_level,
                        device=_resolve_device(CONFIG["audio"].get("device")))
    busy = threading.Lock()

    def handle(use_formatter: bool):
        def on_press():
            if busy.locked() or state["status"] != "idle":
                return
            set_status("recording")
            indicator.recording()
            recorder.start()

        def on_release():
            if state["status"] != "recording":
                return
            set_status("busy")
            indicator.processing()  # shrink but stay visible while ASR/LLM run

            def work():
                with busy:
                    audio = recorder.stop()
                    r = pipeline.process(audio, use_formatter and state["cleanup"])
                    # visible feedback so failures aren't silent: red flash for
                    # hard failures, a tray notification for anything non-ok
                    if r["status"] in ("no_speech", "paste_failed", "paste_blocked"):
                        indicator.error()
                    else:
                        indicator.done()
                    msg = {
                        "no_speech": "No speech detected",
                        "paste_failed": "Paste failed — text is in your clipboard",
                        "paste_blocked": "Admin window — press Ctrl+V (text is in your clipboard)",
                        "ollama_offline_raw": "Cleanup unavailable — pasted raw",
                    }.get(r["status"])
                    if msg:
                        try:
                            icon.notify(msg, "Vanni")
                        except Exception:
                            pass  # notifications must never break dictation
                        print(f"[{time.strftime('%H:%M:%S')}] {msg}")
                    if r["text"]:
                        print(f"[{time.strftime('%H:%M:%S')}] {r['text']!r} "
                              f"(asr {r['asr_s']:.2f}s, fmt {r['format_s']:.2f}s, total {r['total_s']:.2f}s)")
                    set_status("idle")

            threading.Thread(target=work, daemon=True).start()

        return on_press, on_release

    p1, r1 = handle(use_formatter=True)
    p2, r2 = handle(use_formatter=False)
    keyboard.add_hotkey(CONFIG["hotkeys"]["dictate"], p1, suppress=False, trigger_on_release=False)
    keyboard.on_release_key("windows", lambda e: r1() if state["status"] == "recording" else None)
    keyboard.add_hotkey(CONFIG["hotkeys"]["dictate_raw"], p2, suppress=False, trigger_on_release=False)

    def toggle_cleanup(icon_, item_):
        state["cleanup"] = not state["cleanup"]

    def quit_app(i, _):
        i.stop()
        # end the Tk mainloop on its own thread
        indicator.root.after(0, indicator.root.destroy)

    def make_device_item(idx, name):
        def on_click(icon_, item_):
            recorder.device = idx
            _persist_device(idx)
        return pystray.MenuItem(name, on_click, radio=True,
                                checked=lambda item, idx=idx: recorder.device == idx)

    def default_item():
        def on_click(icon_, item_):
            recorder.device = None
            _persist_device("")
        return pystray.MenuItem("System default", on_click, radio=True,
                                checked=lambda item: recorder.device is None)

    inputs = [(i, d["name"]) for i, d in enumerate(recorder.sd.query_devices())
              if d["max_input_channels"] > 0]
    mic_menu = pystray.Menu(default_item(), *(make_device_item(i, n) for i, n in inputs))

    icon.menu = pystray.Menu(
        pystray.MenuItem(lambda i: f"LLM cleanup: {'on' if state['cleanup'] else 'off'}",
                         toggle_cleanup),
        pystray.MenuItem("Microphone", mic_menu),
        pystray.MenuItem("Quit", quit_app),
    )
    print(f"Vanni running. Hold {CONFIG['hotkeys']['dictate']} to dictate "
          f"({CONFIG['hotkeys']['dictate_raw']} = raw). Quit via tray icon.")
    # Tk must own the MAIN thread on Windows or the overlay won't render;
    # pystray is happy detached.
    icon.run_detached()
    indicator.run_forever()


def run_simulate(pipeline: Pipeline, wav: str, target: str | None, raw: bool):
    import soundfile as sf
    audio, sr = sf.read(wav, dtype="float32")
    assert sr == SAMPLE_RATE, f"fixture must be 16kHz, got {sr}"
    wrapper = None
    if target == "notepad":
        # focus a fresh notepad tab so injection lands somewhere verifiable
        import subprocess
        from pywinauto import Desktop
        import keyboard as kb
        subprocess.Popen(["notepad.exe"])
        time.sleep(2.5)
        # resolve the concrete window once; lazy re-matching can pick a
        # different Notepad window after the tab title changes
        wrapper = Desktop(backend="uia").window(
            title_re=r".* - Notepad$", found_index=0).wrapper_object()
        wrapper.set_focus()
        time.sleep(0.5)
        kb.send("ctrl+n")
        time.sleep(0.7)
    r = pipeline.process(audio, use_formatter=not raw)
    verified = ""
    if target == "notepad":
        import keyboard as kb
        time.sleep(0.8)  # let the paste land before reading back
        doc = next(c for c in wrapper.descendants() if c.element_info.control_type == "Document")
        landed = doc.iface_text.DocumentRange.GetText(-1)
        verified = f" landed={'YES' if r['text'] and r['text'] in landed else 'NO'}"
        kb.send("ctrl+w"); time.sleep(0.7)
        try:
            next(c for c in wrapper.descendants()
                 if c.element_info.name == "Don't save" and c.element_info.control_type == "Button").click_input()
        except StopIteration:
            pass
    print(f"SIMULATE {wav}: text={r['text']!r}\n"
          f"  asr={r['asr_s']:.2f}s format={r['format_s']:.2f}s total={r['total_s']:.2f}s "
          f"injected={r['injected']}{verified}")
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", metavar="WAV", help="run pipeline on a wav file instead of mic")
    ap.add_argument("--target", choices=["notepad"], help="inject into a fresh Notepad tab and verify")
    ap.add_argument("--raw", action="store_true", help="skip LLM cleanup")
    args = ap.parse_args()

    pipeline = Pipeline()
    if args.simulate:
        r = run_simulate(pipeline, args.simulate, args.target, args.raw)
        sys.exit(0 if r["text"] else 1)
    run_tray(pipeline)
