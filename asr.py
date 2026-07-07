"""Speech-to-text via faster-whisper (CTranslate2). Fully offline after first model download."""
import os
import time
import tomllib
from pathlib import Path

from paths import BASE, BUNDLE, FROZEN

# CTranslate2 needs cuBLAS/cuDNN DLLs: from the pip wheels in dev, from the
# bundle in a frozen (PyInstaller) build
_dll_root = BUNDLE if FROZEN else BASE / ".venv" / "Lib" / "site-packages"
for _pkg in ("nvidia/cublas/bin", "nvidia/cudnn/bin"):
    _p = _dll_root / _pkg
    if _p.is_dir():
        os.add_dll_directory(str(_p))
        os.environ["PATH"] = str(_p) + os.pathsep + os.environ["PATH"]

import numpy as np
from faster_whisper import WhisperModel

CONFIG = tomllib.loads((BASE / "config.toml").read_text(encoding="utf-8"))

# transcripts whisper fabricates on silence/noise — never inject these
JUNK = {
    "thank you.", "thanks for watching.", "thank you for watching.",
    "subtitles by the amara.org community", "you", "bye.", ".",
}

# Silero VAD ships inside faster-whisper and auto-downloads on first use —
# fully offline afterwards, no API/token cost (config: [asr] vad)
VAD = CONFIG["asr"].get("vad", True)

def load_model() -> tuple[WhisperModel, str]:
    # Degradation ladder so Vanni runs on low-performance laptops too: GPU model
    # -> lighter GPU model -> small.en on CPU. Built at call time (not import)
    # so a wizard/tray tier change applies to the next load without a restart.
    fallbacks = [
        (CONFIG["asr"]["model"], "cuda", CONFIG["asr"]["compute_type"]),
        ("distil-large-v3", "cuda", "int8_float16"),
        ("small.en", "cpu", "int8"),
    ]
    last_err = None
    for name, device, compute in fallbacks:
        try:
            model = WhisperModel(name, device=device, compute_type=compute)
            return model, f"{name} ({device}/{compute})"
        except Exception as e:  # CUDA/VRAM/download failure -> next tier
            last_err = e
    raise RuntimeError(f"no ASR model could be loaded: {last_err}")


def _hotwords() -> str:
    """Bias recognition toward known terms so names/jargon are heard right the
    first time instead of fixed afterwards: corrections.json targets (the
    intended spellings) + the user's [asr] vocabulary list. Re-read per call
    (tiny file) so edits apply to the next dictation without a restart."""
    import json
    terms = [str(t) for t in CONFIG["asr"].get("vocabulary", [])]
    p = BASE / "corrections.json"
    if p.exists():
        terms += list(json.loads(p.read_text(encoding="utf-8")).values())
    seen: set[str] = set()
    uniq = [t for t in terms if not (t.lower() in seen or seen.add(t.lower()))]
    return " ".join(uniq)


def transcribe(model: WhisperModel, audio: np.ndarray, language: str | None = None) -> tuple[str, float]:
    """audio: float32 mono 16kHz in [-1, 1]. Returns (text, latency_seconds)."""
    lang = language or CONFIG["asr"]["language"]
    t0 = time.perf_counter()
    segments, _ = model.transcribe(
        audio,
        language=None if lang == "auto" else lang,
        vad_filter=VAD,
        beam_size=1,
        hotwords=_hotwords() or None,
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    latency = time.perf_counter() - t0
    if text.lower().strip() in JUNK:
        text = ""
    return text, latency


if __name__ == "__main__":
    import soundfile as sf
    model, desc = load_model()
    print(f"loaded: {desc}")
    audio, sr = sf.read("fixtures/fox.wav", dtype="float32")
    assert sr == 16000
    # warm run then timed run (first call includes CUDA kernel warmup)
    transcribe(model, audio)
    text, latency = transcribe(model, audio)
    print(f"transcript: {text!r}  latency: {latency:.2f}s")
