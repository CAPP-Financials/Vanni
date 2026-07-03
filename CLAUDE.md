# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Vanni — a fully offline, local push-to-talk dictation app for Windows (a Wispr Flow clone).
Hold a hotkey, speak, release: audio is transcribed (faster-whisper), optionally cleaned up
by a local LLM (Ollama), and pasted at the cursor via clipboard + Ctrl+V. No cloud calls.

## Commands

```powershell
cd Wisperflow
.\.venv\Scripts\python vanni.py                                    # run the tray app (interactive)
.\.venv\Scripts\python test_pipeline.py                           # run full self-check suite
.\.venv\Scripts\python test_pipeline.py test_corrections           # run a single test by name
.\.venv\Scripts\python vanni.py --simulate fixtures/fox.wav --target notepad   # headless pipeline run, verified via Notepad injection
.\.venv\Scripts\python vanni.py --simulate fixtures/fox.wav --raw   # headless run, skip LLM cleanup
```

Individual modules also have `if __name__ == "__main__"` self-checks runnable directly
(`python asr.py`, `python formatter.py`, `python corrections.py`, `python injector.py`, `python overlay.py`).

There is no separate lint/typecheck config — keep changes consistent with existing style.

Ollama must be running for LLM cleanup (`ollama serve`); `vanni.py` auto-starts it if the port
is down (`ensure_ollama()`). Without it, Vanni still works but always pastes raw transcripts.

### Rebuilding the portable exe

```powershell
.\.venv\Scripts\pyinstaller --noconfirm --onedir --name Vanni --collect-all faster_whisper --collect-all ctranslate2 --collect-all pystray --add-binary ".venv/Lib/site-packages/nvidia/cublas/bin;nvidia/cublas/bin" --add-binary ".venv/Lib/site-packages/nvidia/cudnn/bin;nvidia/cudnn/bin" vanni.py
Copy-Item config.toml, corrections.json dist\Vanni\
```

Lifecycle scripts for the built exe: `Vanni-launcher.ps1` / `Vanni-launcher.ps1 -Stop`.

### Building the installer

```powershell
# after rebuilding the exe above
winget install JRSoftware.InnoSetup    # one time
& "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe" installer.iss   # -> Output\VanniSetup.exe
```

`installer.iss` installs to `%ProgramFiles%\Vanni` (per-user, no admin), offers a
"Start Vanni when Windows starts" task (HKCU Run key, removed on uninstall), and shows
`INSTALL_NOTES.md` (offline/no-cost promise, first-launch downloads, low-spec behaviour)
before install. First-launch model pulls are handled by `firstrun.ensure_models()`
(marker file `.setup_done` next to the exe).

## Architecture

Single-process pipeline, no web server / no DB. Each module is a thin, independently
runnable stage; `vanni.py` wires them together and owns the two runtime entry points
(`run_tray` for interactive use, `run_simulate` for headless verification).

**Pipeline order** (`Pipeline.process` in `vanni.py`): record → `asr.transcribe` →
`corrections.apply_verbose` (deterministic, pre-LLM) → `formatter.format_text` (LLM cleanup,
skipped for short utterances) → `injector.inject` → `history.record` (local daily markdown log).

- `asr.py` — faster-whisper (CTranslate2) wrapper. Tries GPU model, then CPU fallbacks
  (`_FALLBACKS` list) if CUDA/VRAM/download fails. Filters known Whisper silence-hallucination
  strings (`JUNK` set). CUDA DLL paths are patched at import time depending on frozen vs. source run.
- `corrections.py` — deterministic misheard-phrase → intended-text dictionary
  (`corrections.json`), applied before the LLM so common errors cost microseconds, not an
  inference call. Auto-reloads when the file's mtime changes. `learn()` persists new rules.
- `formatter.py` — calls local Ollama (`qwen3.5:4b` by default) to fix punctuation/casing and
  strip fillers. Utterances under `word_skip_threshold` words (config) skip the LLM entirely.
  Any Ollama failure degrades silently to the raw transcript — cleanup must never block dictation.
- `history.py` — appends every dictation (raw transcript, corrections that fired, final text)
  to `history/YYYY-MM-DD.md` next to the exe. Fully local; intended as a future ML training
  corpus. Best-effort: any failure is swallowed, never blocks dictation. `[history] enabled` in config.
- `firstrun.py` — one-time first-launch setup: checks Ollama is installed, `ollama pull`s the
  formatter model + fallback if missing, then writes `.setup_done`. Non-blocking; without
  Ollama, Vanni degrades to raw transcripts.
- `injector.py` — writes text to clipboard, pastes via `keyboard.send("ctrl+v")`. Clipboard is
  intentionally left populated afterward (re-paste elsewhere). `notepad_roundtrip` is a UIA-based
  test helper that verifies actual injection by reading back Notepad's text buffer.
- `overlay.py` — borderless, transparent, always-on-top Tk overlay showing a live waveform
  during recording/processing. Tk must own the main thread on Windows, so `run_tray` calls
  `indicator.run_forever()` last and runs pystray detached instead.
- `paths.py` — resolves `BASE` (config/editable files) and `BUNDLE` (read-only bundled
  resources) correctly whether running from source or as a frozen PyInstaller exe.
- `config.toml` — all tunables: hotkeys, ASR model/compute type, formatter model and
  word-count threshold, Ollama URL, min recording duration.

**Threading model**: hotkey press/release callbacks run on the `keyboard` library's thread;
actual processing (`work()` in `run_tray`) runs in a daemon thread guarded by a `busy` lock so
overlapping dictations are ignored. The Tk overlay's state setters (`recording()`,
`processing()`, `done()`, `set_level()`) are called cross-thread and are simple attribute
writes consumed by the Tk main-thread render loop (`_tick`/`_draw`), so no additional locking
is needed there.

**Two hotkeys, same pipeline**: `ctrl+windows` uses the formatter (if enabled and long enough);
`ctrl+alt+windows` always skips it (`use_formatter=False`) — useful for text where LLM cleanup
would be unwanted (code, exact phrasing).

## Known limits (from README)

- Pasting into elevated (admin) windows silently fails unless Vanni itself runs elevated; text
  still lands in the clipboard as a fallback.
- Some games/anti-cheat block synthetic Ctrl+V.
- `dist/Vanni/` portable build bundles CUDA and is large (~2.1GB); first launch is slow once
  due to Defender scanning.
