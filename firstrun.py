"""First-launch setup: make sure the local model stack exists, download if not.

Vanni's promise: 100% offline dictation, zero tokens, zero subscription, runs
on low-performance laptops. The only thing that ever needs internet is this
one-time setup — pulling the Ollama cleanup model(s); the Whisper speech
model and Silero VAD auto-download on first ASR load. After that, nothing
leaves the machine.

Everything here is best-effort and non-blocking: if Ollama is missing or a
pull fails, Vanni still runs (raw transcripts, no LLM cleanup) — same
degradation philosophy as the rest of the pipeline.
"""
import json
import shutil
import subprocess
import tomllib
import urllib.request

from paths import BASE

CONFIG = tomllib.loads((BASE / "config.toml").read_text(encoding="utf-8"))
_MARKER = BASE / ".setup_done"


# Hardware-matched ASR tiers. LLM model names never change per tier —
# gemma2:2b runs fine on CPU via Ollama; only whether cleanup is ON varies.
TIERS = {
    "gpu": {"model": "large-v3-turbo", "compute_type": "int8_float16",
            "formatter_enabled": True,
            "blurb": "Best accuracy, near-instant (NVIDIA GPU, ~1.5GB Whisper "
                     "+ ~700MB CUDA + ~2GB cleanup model download)"},
    "cpu": {"model": "small.en", "compute_type": "int8",
            "formatter_enabled": True,
            "blurb": "Good accuracy, English only, runs on any CPU "
                     "(~500MB Whisper + ~2GB cleanup model download)"},
    "lite": {"model": "small.en", "compute_type": "int8",
             "formatter_enabled": False,
             "blurb": "Lightest: raw dictation without LLM cleanup, "
                      "for low-memory laptops (~500MB download)"},
}


def probe_hardware() -> dict:
    """Best-effort {vram_mb, ram_gb, cores}; zeros when undetectable, never raises."""
    vram_mb = 0
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        vram_mb = int(out.stdout.split()[0]) if out.returncode == 0 else 0
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        pass
    ram_gb = 0
    try:
        import ctypes
        kb = ctypes.c_ulonglong(0)
        if ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(kb)):
            ram_gb = round(kb.value / 2**20)
    except Exception:
        pass
    import os
    return {"vram_mb": vram_mb, "ram_gb": ram_gb, "cores": os.cpu_count() or 0}


def recommend(vram_mb: int, ram_gb: int) -> str:
    """Pure tier mapper: GPU with >=6GB VRAM -> gpu; >=16GB RAM -> cpu; else lite."""
    if vram_mb >= 6000:
        return "gpu"
    return "cpu" if ram_gb >= 16 else "lite"


def _installed_models(url: str) -> list[str] | None:
    """Model names known to the local Ollama, or None if it's unreachable."""
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=3) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except OSError:
        return None


def _pull(model: str) -> bool:
    """`ollama pull <model>` with progress streamed to the console."""
    print(f"first launch: downloading cleanup model '{model}' (one time, a few GB)...")
    try:
        return subprocess.run(["ollama", "pull", model]).returncode == 0
    except OSError:
        return False


def ensure_models() -> None:
    """One-time check that the formatter models exist locally; pull if missing.
    Cheap no-op on every launch after the marker file exists."""
    if _MARKER.exists() or not CONFIG["formatter"]["enabled"]:
        return
    if not shutil.which("ollama"):
        print("SETUP: Ollama not installed — dictation works, but LLM cleanup is off.\n"
              "       Install it free (local, no account needed): winget install Ollama.Ollama\n"
              "       Then restart Vanni to auto-download the cleanup model.")
        return  # no marker: retry the check next launch, after Ollama is installed
    have = _installed_models(CONFIG["formatter"]["ollama_url"])
    if have is None:
        return  # server not up yet; vanni.ensure_ollama handles starting it next time
    ok = True
    for model in (CONFIG["formatter"]["model"], CONFIG["formatter"].get("fallback_model", "")):
        if model and not any(m == model or m.startswith(model + ":") for m in have):
            ok = _pull(model) and ok
    if ok:
        _MARKER.write_text("setup complete\n", encoding="utf-8")
        print("first-launch setup complete — Vanni is now fully offline.")


if __name__ == "__main__":
    _MARKER.unlink(missing_ok=True)
    ensure_models()
    assert shutil.which("ollama") is None or _installed_models(
        CONFIG["formatter"]["ollama_url"]) is None or _MARKER.exists()
    print("firstrun OK")
