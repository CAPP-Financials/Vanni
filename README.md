# Vanni — local push-to-talk dictation (वाणी, "speech")

Fully offline voice dictation for Windows. Hold a hotkey, speak, release — polished
text is pasted at your cursor in whatever app has focus.

**No cloud. No account. No API tokens. No subscription.** Speech-to-text and text
cleanup both run on your machine (faster-whisper + a local Ollama LLM). Nothing you
say is ever sent anywhere. It's a from-scratch clone of the Wispr Flow experience,
built to run free forever, including on modest laptops.

## Why this exists

Commercial dictation tools like Wispr Flow are excellent but recurring-cost, cloud-based
products: your audio leaves your machine, and the product disappears if the company does
or if you stop paying. Vanni exists to prove the same UX — hold a key, speak, get clean
text at your cursor, low latency — is achievable with zero recurring cost and zero data
leaving your device, using only open local models. Every design choice below follows
from that constraint.

## How it's built (and why)

- **Speech-to-text: faster-whisper (CTranslate2), not the cloud.** `large-v3-turbo`
  on GPU (falling back to CPU `small.en` — see below) gives near-real-time transcription
  with no network round trip. Silero VAD (bundled, auto-downloads once) trims silence,
  which both speeds things up and prevents Whisper's well-known tendency to hallucinate
  text from silence/noise (`asr.py`'s `JUNK` filter catches the rest).
- **Cleanup: a local LLM via Ollama, not a hosted API.** Raw ASR output has no
  punctuation and includes filler words ("um", "so", "like"). A small local model
  (`phi3.5` by default) fixes that in one call. Five candidate models were benchmarked
  head-to-head (see `BUILD_LOG.md` Phase 8) — `phi3.5` won on latency (~0.3s warm vs.
  ~1.7s for the next best) with no quality loss. The system prompt is deliberately
  strict: fix punctuation/casing, strip fillers, **change nothing else** — early
  iterations let the model paraphrase, which silently rewrote what people actually said.
  Utterances under 10 words skip the LLM call entirely (not worth the latency).
- **Deterministic corrections before the LLM.** `corrections.py` runs a small
  regex dictionary (`corrections.json`) on the raw transcript before it reaches the LLM —
  a dictionary hit costs microseconds instead of an inference call. Useful for names,
  jargon, and product terms Whisper reliably mishears. Hot-reloads on edit.
  Corrections that fire are logged (see History below).
- **Injection via clipboard + Ctrl+V**, the only reliable way to insert text into an
  arbitrary focused Windows app without per-app integration. The clipboard is left
  populated afterward on purpose, so you can paste the same text elsewhere.
- **Everything degrades gracefully, never blocks dictation.** Ollama down? Paste the
  raw transcript. No GPU or out of VRAM? Fall back down the chain to a CPU model.
  History write fails? Swallow it. The one thing that must never happen is losing what
  you just said because a downstream stage broke.
- **Local dictation history.** Every dictation (raw transcript, corrections applied,
  final text) is appended to `history/YYYY-MM-DD.md` — plain markdown, on your disk,
  under your control. It's on by default but toggled off with one line in
  `config.toml`, or deleted any time. The intent is a corpus you can later use for
  your own analysis or model fine-tuning, not telemetry sent anywhere.
- **Low-spec laptops are a first-class target, not an afterthought.** No GPU → CPU
  Whisper fallback. Under 8GB VRAM → a smaller cleanup model (`qwen2.5:3b`) is
  retried automatically. VAD trims silence to cut CPU work. The goal is that this
  runs acceptably on a several-year-old laptop, not just a gaming rig.

## Quick start

Requires Windows, Python 3.12+, and (recommended, optional) [Ollama](https://ollama.com)
for text cleanup — without it Vanni still dictates, just without cleanup.

```powershell
git clone https://github.com/<your-org>/Vanni.git
cd Vanni
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python vanni.py
```

First run downloads the Whisper speech model (~1.5GB) and, if Ollama is installed,
pulls the configured cleanup model(s) automatically (`firstrun.py`). Everything after
that is offline.

| Hotkey | Action |
|---|---|
| hold `Ctrl+Win` | dictate → clean (if ≥10 words) → paste |
| hold `Ctrl+Alt+Win` | dictate → paste raw (never calls the LLM) |
| tray icon menu | toggle LLM cleanup · quit |

While recording, a small black pill with white waveform bars appears bottom-center
above the taskbar; the bars dance with your voice, settle to dots while Vanni
transcribes, and the pill hides once the text is pasted.

## Prebuilt installer

`Output\VanniSetup.exe` (see Building below) installs Vanni with an optional
**"Start Vanni when Windows starts"** checkbox, so it runs quietly in the background
ready to dictate. See `INSTALL_NOTES.md` for the full first-launch/low-spec explanation
shown during setup.

## Building the portable exe / installer

```powershell
.\.venv\Scripts\pip install pyinstaller
.\.venv\Scripts\pyinstaller --noconfirm Vanni.spec
Copy-Item config.toml, corrections.json dist\Vanni\
```

`dist/Vanni/` is then a self-contained portable folder (~2.1GB, CUDA bundled) — copy
it anywhere, no installer, no Python needed. First launch is slow once (Defender scans
the fresh DLLs).

```powershell
winget install JRSoftware.InnoSetup    # one time
& "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe" installer.iss   # -> Output\VanniSetup.exe
```

Lifecycle scripts for the built exe (hidden background run):

```powershell
powershell -ExecutionPolicy Bypass -File Vanni-launcher.ps1        # start (spawns Ollama if needed)
powershell -ExecutionPolicy Bypass -File Vanni-launcher.ps1 -Stop  # stop Vanni (+ Ollama only if we started it)
```

## Configuration

All tunables live in `config.toml`: hotkeys, ASR model/compute type + VAD, formatter
model/fallback/word-count threshold, Ollama URL, minimum recording length, history
toggle. Comments in the file explain each one.

## Measured latency (RTX 4060 Laptop, 8GB VRAM)

| Path | End-to-end |
|---|---|
| short utterance (raw, no LLM) | **~0.9s** |
| long utterance + LLM cleanup (phi3.5) | **~1.9s** |

## Self-check

```powershell
.\.venv\Scripts\python test_pipeline.py                 # full suite
.\.venv\Scripts\python vanni.py --simulate fixtures/fox.wav --target notepad
```

Individual modules also run standalone self-checks: `python asr.py`,
`python formatter.py`, `python corrections.py`, `python injector.py`,
`python overlay.py`, `python history.py`, `python firstrun.py`.

## Autostart (source checkout, no installer)

```powershell
schtasks /create /tn "Vanni dictation" /sc onlogon /tr "<repo-path>\.venv\Scripts\pythonw.exe <repo-path>\vanni.py"
```

## Multilingual

Set `language = "auto"` in `config.toml`. `large-v3-turbo` handles ~99 languages with
variable quality (high-resource languages are dictation-grade; long-tail are not).
For best accuracy in a specific language, swap `asr.model` per faster-whisper's model list.

## Known limits

- Pasting into **elevated (admin) windows** silently fails unless Vanni itself runs
  elevated. The text is still in your clipboard — paste manually.
- Some games/anti-cheat software block synthetic Ctrl+V.
- The `keyboard` library needs no admin rights for normal apps.

## Architecture

See `CLAUDE.md` for a full module-by-module breakdown of the pipeline (`asr.py` →
`corrections.py` → `formatter.py` → `injector.py` → `history.py`) and the threading
model. `BUILD_LOG.md` has the build history and the reasoning behind model/tooling
choices (benchmark numbers, rejected approaches, bugs found and fixed).

## Contributing

Issues and PRs welcome. There's no CI/lint config yet — keep changes consistent with
existing style, and run `test_pipeline.py` before submitting.

## License

MIT — see `LICENSE`.
