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
    assert "moved to tuesday" in low, "reworded the user's phrasing"
    # headroom for GPU contention / driver state drift (warm baseline ~1.0s on
    # the RTX 4060); a real regression (model spilling to CPU) is 3s+
    assert latency < 1.5, f"warm latency {latency:.2f}s >= 1.5s"
    # word fidelity: casual speech must keep the user's own words — no synonym
    # swaps ("talk about the stuff" -> "discuss the things"), no contractions
    casual = "hey can you tell me when you are free so we can talk about the stuff for next week"
    out2 = formatter.format_text(casual)
    print(f"  fidelity: {out2!r}")
    low2 = out2.lower()
    assert "talk about the stuff" in low2, f"synonym swap: {out2!r}"
    assert "you are" in low2, f"contraction changed the user's words: {out2!r}"
    # determinism: same input -> same output (temp 0)
    assert formatter.format_text(casual) == out2, "cleanup output not deterministic"


def test_formatter_fidelity_gate():
    # clean() must NEVER paste LLM output whose words differ from the input:
    # refusals, answered questions, generated code, paraphrase, added/dropped
    # words all degrade to the raw transcript. No Ollama needed — _generate
    # is mocked to simulate each misbehavior class.
    import formatter
    raw = "explain how malware reverse engineering works in a sandbox environment please"
    bad_outputs = [
        "I can't help with that request.",                     # safety refusal
        "import pandas as pd\ndf = pd.read_csv('data.csv')",   # hallucinated completion
        "Please explain the process of reverse engineering malware.",  # paraphrase
        "Explain how malware reverse engineering works.",       # dropped words
        raw + " Here is how it works: first you use a debugger.",  # answered + appended
        "",                                                      # empty
    ]
    orig = formatter._generate
    try:
        for bad in bad_outputs:
            formatter._generate = lambda *a, **k: bad
            out, status = formatter.clean(raw)
            assert (out, status) == (raw, "degraded"), \
                f"unfaithful LLM output leaked through: {bad!r} -> {out!r}"
        # faithful output (punctuation/caps only) must still pass
        good = "Explain how malware reverse engineering works in a sandbox environment, please."
        formatter._generate = lambda *a, **k: good
        out, status = formatter.clean(raw)
        assert (out, status) == (good, "ok"), f"faithful output rejected: {out!r}"
    finally:
        formatter._generate = orig


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


def test_transform():
    import formatter
    # live Ollama: apply a spoken instruction to a text selection
    out, status = formatter.transform("translate to English",
                                      "bonjour le monde tout va bien")
    print(f"  transform: {out!r} status={status}")
    assert status == "ok" and out and "hello" in out.lower()
    # Ollama down -> degraded, no text returned (caller must NOT paste)
    old = formatter.OLLAMA_URL
    formatter.OLLAMA_URL = "http://localhost:1"
    try:
        out, status = formatter.transform("summarize", "some selected text here")
    finally:
        formatter.OLLAMA_URL = old
    assert (out, status) == (None, "degraded")
    # empty selection -> degraded without any HTTP call
    out, status = formatter.transform("summarize", "")
    assert (out, status) == (None, "degraded")


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


def test_hotwords():
    import asr
    captured = {}

    class FakeModel:
        def transcribe(self, audio, **kw):
            captured.update(kw)
            return iter(()), None

    asr.transcribe(FakeModel(), np.zeros(16000, dtype="float32"))
    hw = captured.get("hotwords") or ""
    print(f"  hotwords: {hw!r}")
    assert "Wispr Flow" in hw, "corrections.json targets not fed as hotwords"
    assert "Vanni" in hw, "[asr] vocabulary terms not fed as hotwords"


def test_smartfmt():
    import smartfmt
    cases = [
        ("email me at John dot Smith at gmail dot com today",
         "email me at john.smith@gmail.com today"),
        ("check example dot com for details",
         "check example.com for details"),
        ("first point new line second point",
         "first point\nsecond point"),
        ("intro new paragraph the body",
         "intro\n\nthe body"),
        ("we met at noon and talked", "we met at noon and talked"),  # no false positives
    ]
    for src, want in cases:
        got = smartfmt.apply(src)
        assert got == want, f"{src!r} -> {got!r} (wanted {want!r})"
    print(f"  smartfmt: {len(cases)} cases OK")


def test_snippets():
    import asr
    import history
    import injector
    import snippets
    import vanni
    # committed example trigger expands; ordinary text does not
    assert snippets.match("Insert signature.") is not None, "example trigger did not match"
    assert snippets.match("the quick brown fox") is None, "false-positive snippet match"
    # pipeline: a trigger utterance pastes the stored expansion VERBATIM (no LLM)
    p = vanni.Pipeline.__new__(vanni.Pipeline)
    p.model = None
    audio = np.zeros(vanni.SAMPLE_RATE, dtype="float32")
    injected = {}
    orig = (asr.transcribe, injector.inject, injector.is_foreground_elevated, history.record)
    try:
        history.record = lambda *a, **k: None
        injector.is_foreground_elevated = lambda: False
        asr.transcribe = lambda m, a: ("insert signature", 0.0)
        injector.inject = lambda t: injected.update(text=t) or True
        r = p.process(audio)
    finally:
        asr.transcribe, injector.inject, injector.is_foreground_elevated, history.record = orig
    expansion = snippets.match("insert signature")
    print(f"  snippet expanded to {injected['text']!r}")
    assert r["status"] == "ok" and injected["text"] == expansion
    assert r["format_s"] == 0.0, "snippet should never hit the LLM"


def test_hw_recommend():
    import firstrun
    hw = firstrun.probe_hardware()
    print(f"  probe: {hw}")
    # this dev machine has an RTX 4060 (8GB) — all three probes must see hardware
    assert hw["vram_mb"] > 0 and hw["ram_gb"] > 0 and hw["cores"] > 0
    # pure tier mapping
    assert firstrun.recommend(8000, 32) == "gpu"
    assert firstrun.recommend(0, 32) == "cpu"
    assert firstrun.recommend(4000, 32) == "cpu"   # small GPU -> CPU whisper
    assert firstrun.recommend(0, 8) == "lite"
    for name, tier in firstrun.TIERS.items():
        assert tier["model"] and tier["compute_type"] and tier["blurb"], name
        assert isinstance(tier["formatter_enabled"], bool), name


def test_config_set():
    import shutil
    import tempfile
    from pathlib import Path
    import firstrun
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "config.toml"
        shutil.copy("config.toml", tmp)
        before = tmp.read_text(encoding="utf-8")
        # [asr] model must change; [formatter] model must NOT (key appears in both)
        firstrun.config_set("asr", "model", "small.en", path=tmp)
        firstrun.config_set("formatter", "enabled", False, path=tmp)
        after = tmp.read_text(encoding="utf-8")
        import tomllib
        cfg = tomllib.loads(after)
        assert cfg["asr"]["model"] == "small.en"
        assert cfg["formatter"]["model"] == "gemma2:2b", "clobbered the wrong section"
        assert cfg["formatter"]["enabled"] is False
        assert cfg["history"]["enabled"] is True, "clobbered [history] enabled"
        # only the two targeted lines changed
        diff = [(a, b) for a, b in zip(before.splitlines(), after.splitlines()) if a != b]
        print(f"  changed lines: {[b for _, b in diff]}")
        assert len(diff) == 2


def test_apply_tier():
    import shutil
    import tempfile
    import tomllib
    from pathlib import Path
    import asr
    import firstrun
    # tier changes must take effect on the NEXT load_model call, so the
    # fallback ladder may no longer be frozen at import time
    assert not hasattr(asr, "_FALLBACKS"), "_FALLBACKS must move inside load_model"
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "config.toml"
        shutil.copy("config.toml", tmp)
        import vanni
        orig = dict(asr.CONFIG["asr"])
        orig_enabled = vanni.CONFIG["formatter"]["enabled"]
        try:
            firstrun.apply_tier("lite", path=tmp)
            cfg = tomllib.loads(tmp.read_text(encoding="utf-8"))
            assert cfg["asr"]["model"] == "small.en"
            assert cfg["asr"]["compute_type"] == "int8"
            assert cfg["formatter"]["enabled"] is False
            # in-memory configs updated too (choice applies this launch)
            assert asr.CONFIG["asr"]["model"] == "small.en"
            assert vanni.CONFIG["formatter"]["enabled"] is False
            print("  lite tier applied to file + in-memory config")
        finally:
            asr.CONFIG["asr"].update(orig)
            vanni.CONFIG["formatter"]["enabled"] = orig_enabled


def test_wizard_gate():
    import tempfile
    from pathlib import Path
    import firstrun
    assert callable(firstrun.wizard), "wizard dialog missing"
    orig = firstrun._MARKER
    try:
        with tempfile.TemporaryDirectory() as td:
            firstrun._MARKER = Path(td) / ".setup_done"
            assert firstrun.should_run_wizard(simulate=False) is True
            assert firstrun.should_run_wizard(simulate=True) is False, \
                "wizard must never block a headless simulate run"
            firstrun._MARKER.write_text("done")
            assert firstrun.should_run_wizard(simulate=False) is False
        print("  wizard gate: first-launch only, never under simulate")
    finally:
        firstrun._MARKER = orig


def test_ensure_cuda():
    import hashlib
    import io
    import json as _json
    import tempfile
    import zipfile
    from pathlib import Path
    import firstrun
    # the spec must no longer bundle CUDA (that's the whole point)
    spec = Path("Vanni.spec").read_text(encoding="utf-8")
    assert "nvidia/cublas" not in spec and "nvidia/cudnn" not in spec

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("nvidia/cublas/bin/fake.dll", b"dll-bytes")
        z.writestr("nvidia/cublas/lib/other.txt", b"must not be extracted")
    wheel = buf.getvalue()
    good_hash = hashlib.sha256(wheel).hexdigest()

    def fake_urlopen(url, timeout=None):
        if str(url).endswith("/json"):
            return io.BytesIO(_json.dumps({"urls": [
                {"filename": "fake-1.0-py3-none-win_amd64.whl",
                 "url": "https://fake/wheel", "size": len(wheel)}]}).encode())
        return io.BytesIO(wheel)

    def boom(*a, **k):
        raise AssertionError("network hit when it must not be")

    orig = (firstrun.urllib.request.urlopen, firstrun.shutil.which, firstrun._CUDA_WHEELS)
    try:
        firstrun.urllib.request.urlopen = fake_urlopen
        firstrun.shutil.which = lambda n: "C:/fake/nvidia-smi.exe"
        firstrun._CUDA_WHEELS = [("fake-pkg", "1.0", good_hash)]
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            firstrun.ensure_cuda(base=base)
            assert (base / "nvidia/cublas/bin/fake.dll").read_bytes() == b"dll-bytes"
            assert not (base / "nvidia/cublas/lib/other.txt").exists(), "extracted non-DLL"
            firstrun.urllib.request.urlopen = boom  # already present -> no download
            firstrun.ensure_cuda(base=base)
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            firstrun.urllib.request.urlopen = fake_urlopen
            firstrun._CUDA_WHEELS = [("fake-pkg", "1.0", "0" * 64)]  # bad hash
            firstrun.ensure_cuda(base=base)  # must not raise
            assert not (base / "nvidia").exists(), "extracted despite bad sha256"
        with tempfile.TemporaryDirectory() as td:
            firstrun.shutil.which = lambda n: None  # no GPU -> no network
            firstrun.urllib.request.urlopen = boom
            firstrun.ensure_cuda(base=Path(td))
        print("  ensure_cuda: extract / skip-present / bad-hash / no-gpu OK")
    finally:
        (firstrun.urllib.request.urlopen, firstrun.shutil.which,
         firstrun._CUDA_WHEELS) = orig


def test_grab_selection():
    import injector

    class FakeClipboard:
        """Clipboard whose content changes only when 'the app' answers ctrl+c."""
        def __init__(self, app_selection):
            self.content, self.app_selection = "stale junk", app_selection
        def copy(self, text):
            self.content = text
        def paste(self):
            return self.content

    class FakeKeyboard:
        def __init__(self, clip):
            self.clip, self.sent = clip, []
        def send(self, keys):
            self.sent.append(keys)
            if keys == "ctrl+c" and self.clip.app_selection:
                self.clip.content = self.clip.app_selection

    orig = (injector.pyperclip, injector.keyboard, injector._GRAB_SETTLE_S)
    try:
        injector._GRAB_SETTLE_S = 0.0  # no real sleeps in tests
        clip = FakeClipboard("the selected words")
        injector.pyperclip, injector.keyboard = clip, FakeKeyboard(clip)
        got = injector.grab_selection()
        print(f"  grabbed: {got!r} keys={injector.keyboard.sent}")
        assert got == "the selected words"
        assert injector.keyboard.sent == ["ctrl+c"]
        # nothing selected: the sentinel clear means stale clipboard is NOT returned
        clip = FakeClipboard(app_selection="")
        injector.pyperclip, injector.keyboard = clip, FakeKeyboard(clip)
        assert injector.grab_selection() == ""
    finally:
        injector.pyperclip, injector.keyboard, injector._GRAB_SETTLE_S = orig


def test_assist_pipeline():
    import asr
    import formatter
    import history
    import injector
    import vanni
    p = vanni.Pipeline.__new__(vanni.Pipeline)
    p.model = None
    audio = np.zeros(vanni.SAMPLE_RATE, dtype="float32")
    calls = {}

    def boom(*a, **k):
        raise AssertionError("must not be called on this path")

    orig = (asr.transcribe, injector.grab_selection, formatter.transform,
            injector.inject, injector.is_foreground_elevated, history.record)
    try:
        injector.is_foreground_elevated = lambda: False
        history.record = lambda raw, final, **k: calls.update(
            hist_raw=raw, hist_final=final, hist_mode=k.get("mode"))
        # happy path: instruction + selection -> transform -> verbatim paste
        asr.transcribe = lambda m, a: ("make it formal", 0.0)
        injector.grab_selection = lambda: "hey what's up"
        formatter.transform = lambda i, t: ("Good afternoon.", "ok")
        injector.inject = lambda t: calls.update(injected=t) or True
        r = p.process_assist(audio)
        print(f"  assist ok: injected={calls['injected']!r}")
        assert r["status"] == "ok" and calls["injected"] == "Good afternoon."
        # the grabbed original must survive in history (clipboard now holds the result)
        assert "hey what's up" in calls["hist_raw"] and calls["hist_mode"] == "assist"
        # nothing selected -> no LLM call
        injector.grab_selection = lambda: ""
        formatter.transform = boom
        assert p.process_assist(audio)["status"] == "no_selection"
        # transform degraded -> must NOT paste
        injector.grab_selection = lambda: "some text"
        formatter.transform = lambda i, t: (None, "degraded")
        injector.inject = boom
        assert p.process_assist(audio)["status"] == "assist_failed"
        # no spoken instruction
        asr.transcribe = lambda m, a: ("", 0.0)
        assert p.process_assist(audio)["status"] == "no_speech"
        # oversized selection would silently truncate in the LLM -> refuse
        asr.transcribe = lambda m, a: ("summarize", 0.0)
        injector.grab_selection = lambda: "word " * 2000
        formatter.transform = boom
        assert p.process_assist(audio)["status"] == "selection_too_long"
        print("  statuses ok: no_selection / assist_failed / no_speech / selection_too_long")
    finally:
        (asr.transcribe, injector.grab_selection, formatter.transform,
         injector.inject, injector.is_foreground_elevated, history.record) = orig
    assert "assist" in vanni.CONFIG["hotkeys"], "config.toml needs [hotkeys] assist"


def test_failure_status():
    import asr
    import formatter
    import history
    import injector
    import vanni
    p = vanni.Pipeline.__new__(vanni.Pipeline)  # skip model load
    p.model = None
    audio = np.zeros(vanni.SAMPLE_RATE, dtype="float32")  # 1s, passes min-duration gate
    orig = (asr.transcribe, formatter.clean, injector.inject,
            injector.is_foreground_elevated, history.record)
    try:
        history.record = lambda *a, **k: None
        injector.is_foreground_elevated = lambda: False  # hermetic: ignore real fg window
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
        (asr.transcribe, formatter.clean, injector.inject,
         injector.is_foreground_elevated, history.record) = orig
    print("  statuses ok: no_speech / paste_failed / ollama_offline_raw")


def test_elevated_detect():
    import asr
    import formatter
    import history
    import injector
    import vanni
    # is_foreground_elevated() composes _foreground_pid + _process_elevated, guarded
    o_pid, o_elev = injector._foreground_pid, injector._process_elevated
    try:
        injector._foreground_pid = lambda: 4321
        injector._process_elevated = lambda pid=None: True
        assert injector.is_foreground_elevated() is True
        injector._process_elevated = lambda pid=None: False
        assert injector.is_foreground_elevated() is False
    finally:
        injector._foreground_pid, injector._process_elevated = o_pid, o_elev
    # process surfaces paste_blocked when the target is elevated but Vanni is not
    p = vanni.Pipeline.__new__(vanni.Pipeline)
    p.model = None
    audio = np.zeros(vanni.SAMPLE_RATE, dtype="float32")
    orig = (asr.transcribe, formatter.clean, injector.inject,
            injector.is_foreground_elevated, injector.self_elevated, history.record)
    try:
        history.record = lambda *a, **k: None
        asr.transcribe = lambda m, a: ("please clean this whole sentence up for me now thanks", 0.0)
        formatter.clean = lambda t: (t, "ok")
        injector.inject = lambda t: True
        injector.is_foreground_elevated = lambda: True
        injector.self_elevated = lambda: False
        assert p.process(audio)["status"] == "paste_blocked"
    finally:
        (asr.transcribe, formatter.clean, injector.inject,
         injector.is_foreground_elevated, injector.self_elevated, history.record) = orig
    print("  elevated detect + paste_blocked status OK")


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
       test_formatter_fidelity_gate, test_formatter_ollama_down, test_transform, test_corrections, test_history, test_hotwords,
       test_smartfmt, test_snippets, test_hw_recommend, test_config_set,
       test_apply_tier, test_wizard_gate, test_ensure_cuda,
       test_grab_selection, test_assist_pipeline,
       test_failure_status, test_elevated_detect,
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
