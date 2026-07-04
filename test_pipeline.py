"""Self-check for the Vanni pipeline. Run: python test_pipeline.py [test_name]"""
import sys
import time

import numpy as np
import soundfile as sf


def _tokens(s: str) -> set[str]:
    return {t.strip(".,!?").lower() for t in s.split() if t.strip(".,!?")}


def test_asr():
    import asr
    model, desc = asr.load_model()
    audio, sr = sf.read("fixtures/fox.wav", dtype="float32")
    assert sr == 16000
    asr.transcribe(model, audio)  # warmup (CUDA kernels)
    text, latency = asr.transcribe(model, audio)
    expected = _tokens("the quick brown fox jumps over the lazy dog")
    got = _tokens(text)
    overlap = len(expected & got) / len(expected)
    print(f"  asr [{desc}]: {text!r} overlap={overlap:.0%} latency={latency:.2f}s")
    assert overlap >= 0.8, f"token overlap {overlap:.0%} < 80%"
    assert latency < 1.0, f"latency {latency:.2f}s >= 1.0s"


def test_asr_silence():
    import asr
    model, _ = asr.load_model()
    audio, _ = sf.read("fixtures/silence.wav", dtype="float32")
    text, _ = asr.transcribe(model, audio)
    print(f"  silence transcript: {text!r}")
    assert text == "", f"silence produced text: {text!r}"


def test_formatter_short_skips():
    import formatter
    t0 = time.perf_counter()
    out = formatter.format_text("this is a short test phrase")  # 6 words
    ms = (time.perf_counter() - t0) * 1000
    print(f"  short input: {ms:.1f}ms")
    assert out == "this is a short test phrase"
    assert ms < 50, f"short path took {ms:.0f}ms (should not hit Ollama)"


def test_formatter_cleans():
    import formatter
    formatter.warm_up()
    raw = "um so basically the meeting is uh moved to tuesday because of the client call"
    t0 = time.perf_counter()
    out = formatter.format_text(raw)
    latency = time.perf_counter() - t0
    print(f"  cleaned: {out!r} latency={latency:.2f}s")
    low = f" {out.lower()} "
    assert "tuesday" in low, "lost content"
    assert " um " not in low and " uh " not in low, "fillers not removed"
    # phi3.5 benchmarked ~300ms warm (vs qwen3.5:4b's ~1.7s); 1.0s leaves headroom
    # for GPU contention with the whisper model resident
    assert latency < 1.0, f"warm latency {latency:.2f}s >= 1.0s"


def test_formatter_ollama_down():
    import formatter
    raw = "one two three four five six seven eight nine ten eleven"
    old = formatter.OLLAMA_URL
    formatter.OLLAMA_URL = "http://localhost:1"  # unreachable
    try:
        out = formatter.format_text(raw)
    finally:
        formatter.OLLAMA_URL = old
    print(f"  degraded output: {out!r}")
    assert out == raw, "should pass through raw text when Ollama is down"


def test_corrections():
    import corrections
    t0 = time.perf_counter()
    out = corrections.apply("I use whisper flow with olama every day")
    ms = (time.perf_counter() - t0) * 1000
    print(f"  corrected: {out!r} in {ms:.2f}ms")
    assert out == "I use Wispr Flow with Ollama every day"
    assert ms < 10, "correction pass must be effectively free"


def test_history():
    import corrections
    import history
    sentinel = f"history sentinel {int(time.time())} with olama inside"
    fixed, fixes = corrections.apply_verbose(sentinel)
    assert fixes == [("olama", "Ollama")], f"unexpected fixes: {fixes}"
    history.record(sentinel, fixed, mode="raw", corrections_applied=fixes, duration_s=0.5)
    today = history.DIR / f"{time.strftime('%Y-%m-%d')}.md"
    text = today.read_text(encoding="utf-8")
    print(f"  history file: {today.name} ({len(text)} chars)")
    assert fixed in text, "final text not recorded"
    assert "olama → Ollama" in text, "applied correction not recorded"


def test_failure_status():
    import asr
    import formatter
    import history
    import injector
    import vanni
    p = vanni.Pipeline.__new__(vanni.Pipeline)  # skip model load
    p.model = None
    audio = np.zeros(vanni.SAMPLE_RATE, dtype="float32")  # 1s, passes min-duration gate
    orig = (asr.transcribe, formatter.clean, injector.inject, history.record)
    try:
        history.record = lambda *a, **k: None
        # 1) nothing recognized -> no_speech
        asr.transcribe = lambda m, a: ("", 0.0)
        assert p.process(audio)["status"] == "no_speech"
        # 2) text produced but paste blocked -> paste_failed
        asr.transcribe = lambda m, a: ("please clean this whole sentence up for me now thanks", 0.0)
        formatter.clean = lambda t: (t, "ok")
        injector.inject = lambda t: False
        assert p.process(audio)["status"] == "paste_failed"
        # 3) pasted, but cleanup degraded to raw -> ollama_offline_raw
        injector.inject = lambda t: True
        formatter.clean = lambda t: (t, "degraded")
        assert p.process(audio)["status"] == "ollama_offline_raw"
    finally:
        asr.transcribe, formatter.clean, injector.inject, history.record = orig
    print("  statuses ok: no_speech / paste_failed / ollama_offline_raw")


def test_overlay_error():
    import overlay
    o = overlay.Overlay()
    assert hasattr(o, "error"), "Overlay has no error() state setter"
    o.recording()
    assert o._state == "recording"
    o.error()
    print(f"  overlay state after error(): {o._state!r}")
    assert o._state == "error", f"error() did not set the error state: {o._state!r}"


def test_mic_device():
    import types

    import sounddevice as sd

    import vanni
    # config exposes an [audio] device key (empty = system default)
    assert "device" in vanni.CONFIG["audio"], "config missing [audio] device"
    # Recorder forwards a chosen device to sd.InputStream
    captured = {}

    class FakeStream:
        def start(self): pass
        def stop(self): pass
        def close(self): pass

    rec = vanni.Recorder(device=7)
    rec.sd = types.SimpleNamespace(InputStream=lambda **kw: captured.update(kw) or FakeStream())
    rec.start()
    inputs = [d for d in sd.query_devices() if d["max_input_channels"] > 0]
    print(f"  input devices: {len(inputs)}; device kwarg forwarded = {captured.get('device')}")
    assert captured.get("device") == 7, f"device not forwarded to InputStream: {captured}"
    assert inputs, "no input devices enumerated"


def test_injection():
    import injector
    sentinel = f"vanni-sentinel-{int(time.time())}"
    got = injector.notepad_roundtrip(sentinel)
    import pyperclip
    print(f"  notepad got: {got!r}")
    assert sentinel in got, "text did not land in Notepad"
    assert pyperclip.paste() == sentinel, "clipboard did not retain the text"


ALL = [test_asr, test_asr_silence, test_formatter_short_skips, test_formatter_cleans,
       test_formatter_ollama_down, test_corrections, test_history, test_failure_status,
       test_overlay_error, test_mic_device, test_injection]

if __name__ == "__main__":
    wanted = sys.argv[1:] or [f.__name__ for f in ALL]
    failed = []
    for fn in ALL:
        if fn.__name__ not in wanted:
            continue
        print(f"{fn.__name__} ...")
        try:
            fn()
            print("  PASS")
        except Exception as e:
            failed.append((fn.__name__, e))
            print(f"  FAIL: {e}")
    if failed:
        sys.exit(1)
    print("all selected tests passed")
