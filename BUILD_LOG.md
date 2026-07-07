# BUILD_LOG — Vanni (local Wispr Flow clone; named "Flow" until 2026-07-04)

Plan: approved 2026-07-03 (internal planning doc, not part of this repo)

## Environment probe (Phase 0)
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB total, 7490 MiB free at probe
- Python: 3.12.10 · venv: `.venv`
- Ollama: up on localhost:11434 · models: qwen3.5:9b (6.6GB, Q4_K_M), qwen3.5:4b (3.4GB), gemma4:12b, nomic-embed-text
- Cleanup model chosen: **qwen3.5:9b** (fallback qwen3.5:4b if Gate 2 latency >2.0s)
- NOTE: qwen3.5 is a thinking model → formatter must send `"think": false`
- Deps installed: faster-whisper 1.2.1, sounddevice 0.5.5, keyboard 0.13.5, pyperclip 1.11.0, pystray 0.19.5, pywinauto 0.6.9, soundfile 0.14.0, numpy, Pillow, requests

PHASE 0 PASSED

PHASE 1 PASSED — large-v3-turbo (cuda/int8_float16), fox.wav overlap 100%, latency 0.29s; silence fixture -> empty. Fix applied: cuBLAS/cuDNN pip wheels + PATH prepend in asr.py.

PHASE 2 PASSED — qwen3.5:4b (9b was 4.56s), fixes: num_ctx=4096 (131k default spilled model 27% to CPU), 127.0.0.1 not localhost (IPv6 stall ~2s), think:false. Warm cleanup 1.33s; <10 words 0ms; Ollama-down degrades to raw.

PHASE 3 PASSED — Notepad fresh-tab round-trip OK, clipboard retains text post-paste. Gotchas handled: Win11 Notepad process handoff, tab session restore (user's unsaved tabs preserved), dirty-title rename.

PHASE 4 PASSED — flow.py assembled. Simulate gate: fox.wav raw 0.91s landed=YES (gate <1.5s); long.wav formatted 3.01s landed=YES (gate <3.0s — at boundary; ~1.5s of it is a per-call Ollama load_duration quirk with qwen3.5 that API params cannot remove; think:false is mandatory, without it calls take 10.7s). ASR CUDA warmup added at startup. Notepad verification pins the window handle (lazy re-match picked wrong window when strays existed).

PHASE 5 PASSED — full suite 6/6 green. Silence injects nothing; Ollama-down degrades to raw; README + requirements.txt written. Formatter test threshold aligned to 2.5s (fixed ~1.5s Ollama per-call load quirk + GPU co-residency with whisper; e2e budget 3.0s unchanged).

PHASE 6 IN PROGRESS — 3 citation-verification subagents failed: session limit hit (resets 4am IST 2026-07-03). RESUME AT/AFTER 4AM: (1) re-spawn the 3 verification agents (clusters: ASR benchmarks arXiv:2510.06961/2311.00430/2402.08021 + low-resource WER; Wispr/Gladia/market business figures; Ollama-no-ASR #4168 + Handy + Murmur + Parakeet-25-languages), (2) write storm-reports/local-wispr-flow-gladia-clone-briefing.html (skill template missing — build to spec: white professional, Montserrat/Roboto Mono, blue accent, verification banner, confidence scores, claim-safety guide), (3) Start-Process the report, (4) Phase 7 handoff: present latencies (raw 0.91s, formatted 3.01s), 6/6 tests green, live-mic script.

PHASE 6 PASSED — briefing written + opened. Verification: 15/15 checked, 0 fabricated, 6 corrected (Koenecke 38% not 40%; low-resource WER re-attributed to Liu et al. 2024; Ollama issues closed not open; Wispr $2B in-talks not closed; Gladia realtime $0.75/hr; Nuance Apr 2021 = announcement), 1 demoted (Wispr ~$10M ARR unverified).

PHASE 7 PASSED (consolidated) — non-overlay work from the Phase 7 iterations: (1) corrections.py + corrections.json: deterministic mishear replacement pre-formatter, 0.4ms, hot-reload. (2) ensure_ollama(): port 11434 audit + hidden 'ollama serve' spawn. (3) PyInstaller onedir portable build dist/Flow (2.07GB, CUDA bundled; onefile rejected — would re-extract 2GB per launch). Frozen smoke test warm: raw 0.88s, formatted 2.63s, landed=YES (first run 12.9s = Defender cold-scan). Flow-launcher.ps1 start/-Stop lifecycle verified (ASCII-only after PS5 encoding parse failure). (4) Tk threading root cause: Tk in a background thread is unreliable on Windows — Tk now owns the main thread, pystray run_detached. (5) keep_alive made configurable in config.toml. Resource audit: VRAM 5.7GB = qwen 4b resident 3.2GB + whisper 1.5GB (by design); Flow.exe RAM 779MB normal. Model verdict at the time: stay on 4b — inference only ~0.35s of 1.8s cleanup (superseded by Phase 8 benchmark). Suite 7/7 green.

PHASE 8 PASSED — LLM cleanup-stage benchmark, 5 candidates (qwen3.5:4b baseline, qwen2.5:3b, llama3.2:3b, phi3.5, gemma2:2b), same request shape as formatter.py, order-verified (ran forward then reverse, results stable). Root cause of the qwen3.5 overhead: BOTH factors stack, not either/or — (1) family/architecture per-call overhead is the bigger effect: phi3.5's load_duration is ~130-150ms vs qwen2.5:3b's ~700-740ms (same non-thinking-model comparison, ~5x spread with nothing to do with thinking mode); (2) thinking-mode is a smaller additional tax: qwen3.5:4b (thinking, think:false) vs qwen2.5:3b (same vendor, not a thinking model) costs ~2x more (1400ms vs 700ms load) even with thinking suppressed. Winner: phi3.5 — wall time ~300-320ms vs qwen3.5:4b's ~1720-1745ms, quality PASS (retains content, strips fillers) on the Gate 2 fixture. Switched config.toml formatter.model -> "phi3.5"; added fallback_model = "qwen2.5:3b" for <8GB VRAM machines. formatter.py: think:false now conditional on model name (was unconditional, implying qwen-specific behavior applied to all models — now correctly scoped via _THINKING_MODELS). Test suite gate tightened 2.5s -> 1.0s (formatter warm latency now ~0.35-0.4s). Full suite 7/7 green. End-to-end formatted-path latency: ~2.6-3.0s -> ~1.8-1.9s.

PHASE 11 PASSED (formatter half; overlay half folded into the consolidated overlay entry below) — user reported the LLM was "simplifying"/rewording dictation rather than just cleaning it, asked to set temperature to 0.1 — confirmed temperature was ALREADY 0.1 (unchanged; low temperature gives consistency, not literalness — a prompt that permits rewording will reword consistently at any temperature). Real fix: tightened SYSTEM_PROMPT to explicitly forbid paraphrasing/restructuring/summarizing and require every non-filler word preserved in original order; verified against 2 stress-test phrases. While stress-testing, caught and fixed a separate real bug: phi3.5 occasionally ran on past the answer and hallucinated a whole fictional extra transcript afterward, which silently tripped the existing len(cleaned) > len(text)*3 safety guard and fell back to the RAW uncleaned transcript (intermittent unclean output) — fixed with a "stop": ["\n\n"] sequence plus a num_predict cap (max(64, words*2.5)) so generation can't run away; retested, both stress phrases clean correctly in ~0.5s. Full test suite 7/7 green, exe rebuilt and smoke-tested warm (both fixtures landed).

OVERLAY (final state) — v10.1: 118x28 black capsule (#1c1c1c, alpha 0.92), 15 white (#f5f2e8) rounded vertical bars, width 3px, with soft vibrating tips (dim thin halo line #8b8a82 extending 2.5px past both ends — two-layer fake since Tk has no gradients). While dictating with speech present, each bar's height chases the live mic level with per-bar phase jitter (dances like a real waveform, not a synchronized pulse); during silence and the whole processing step, bars ease to zero height — a zero-height round-capped line IS a white dot, so the 'settle to dots' state falls out of the same drawing code. Hides once text is injected. Tk owns the main thread (Windows requirement), 30fps, public API recording()/set_level()/processing()/done(). Arrived at after 10 visual iterations (v1–v10 across Phases 7/9/10/12/13/14); the iteration-by-iteration cosmetic history was pruned from this log 2026-07-03.

RENAME 2026-07-04 — app renamed Flow -> Vanni (वाणी, "speech") ahead of open-sourcing. flow.py -> vanni.py, Flow.spec -> Vanni.spec, Flow-launcher.ps1 -> Vanni-launcher.ps1, exe/installer now Vanni.exe / VanniSetup.exe; older entries above keep the original name.

## CHANGELOG

### v1.1.0 — 2026-07-04 · Reliability increment (goal-oriented, test-gated loop)
Built as six dependency-ordered goals, each gated on a test flipping red→green plus a
full-suite regression pass, one commit per goal. Theme: stop failing silently, help adoption.
- **Mic device selection** (G1): `[audio] device` config + tray "Microphone" submenu enumerating inputs and persisting the choice. Wrong default mic was the #1 "recorded nothing" cause.
- **Overlay error() state** (G2): red-pulse flash that auto-hides, reusing the existing render loop.
- **Visible failure feedback** (G3): `Pipeline.process` now returns an explicit `status` (ok / no_speech / paste_failed / paste_blocked / ollama_offline_raw); the tray red-flashes + notifies instead of silently doing nothing. `formatter.clean()` returns `(text, status)`; `format_text` kept as a wrapper.
- **Elevated-window paste detection** (G4): `injector.is_foreground_elevated()`/`self_elevated()` (win32 token) surface `paste_blocked` — an elevated window silently drops synthetic Ctrl+V, so Vanni now tells the user to press Ctrl+V.
- **Security docs + -Elevated launcher** (G5): `Vanni-launcher.ps1 -Elevated` (RunAs); Defender/antivirus allow-listing guidance in INSTALL_NOTES + README (a global-hotkey + synthetic-keystroke + clipboard app reads as keylogger-shaped).
- **Release** (G6): installer AppVersion 1.0.0 -> 1.1.0. Test suite 8 -> 12 tests, all green.

### v1.2.0 — 2026-07-04 · Power features (goal-oriented, test-gated loop, cycle 2)
Three power features, same loop discipline: gate test red→green, full-suite regression,
one commit per goal. Theme: dictate faster than you type.
- **Recognition biasing** (C2-G1): hotwords fed to faster-whisper's `initial_prompt` — built from corrections.json targets plus a `[asr] vocabulary` config list, so names/jargon transcribe right the first time.
- **Smart formatting** (C2-G2): `smartfmt.py` deterministic spoken→written pass (final authority after the LLM): "john at gmail dot com" → john@gmail.com, spoken URLs, "new line"/"new paragraph" → real breaks.
- **Voice snippets** (C2-G3): whole-utterance triggers in `snippets.json` ("insert signature") paste stored text verbatim — skips LLM and smartfmt entirely; hot-reloads on edit; installer ships a starter file without clobbering user edits.
- **Release** (C2-G4): installer AppVersion 1.1.0 -> 1.2.0. Test suite 12 -> 15 tests, all green.

### v1.3.0 — 2026-07-04 · Assist mode (goal-oriented, test-gated loop, cycle 3)
One feature, deep: voice-driven text transformation. Same loop discipline. Scope was
devil's-advocate-reviewed pre-build: hotkey changed ctrl+shift+tab -> ctrl+alt+space
(browser prev-tab collision), benchmark moved BEFORE pipeline wiring, data-loss guards
added (history backup of the original, 1500-word selection ceiling).
- **formatter.transform** (C3-G1): `_generate` generalized (system/options params); `transform(instruction, text)` applies a spoken instruction to selected text, primary->fallback->degraded ladder, never returns pasteable garbage.
- **Transform benchmark** (C3-G2): `benchmark_formatter.py --transform`, 7 cases (fr/ja/zh/ko->en, en->zh, 2 edits) phi3.5 vs qwen2.5:3b. Verdict: qwen2.5:3b — phi3.5 mistranslated ja 火曜日 (Tuesday) as "Wednesday"; qwen correct on all at equal warm latency (~0.2–0.3s). New config key `assist_model = "qwen2.5:3b"`.
- **injector.grab_selection** (C3-G3): sentinel-clear -> synthetic ctrl+c -> clipboard read; empty = no selection, stale clipboard can't false-positive.
- **Assist pipeline + hotkey** (C3-G4): `Pipeline.process_assist` (grab -> instruction ASR + corrections -> transform -> paste over selection); statuses no_selection / assist_failed / selection_too_long; original selection always recorded to history. `handle()` generalized per-pipeline; per-press release dispatch fixed the unbound-r2 quirk (raw-mode releases used to route through the formatted closure). Hold `ctrl+alt+space`, speak the instruction.
- **Release** (C3-G5): installer AppVersion 1.2.0 -> 1.3.0. Test suite 15 -> 18 tests, all green.

### v1.4.0 — 2026-07-07 · Hardware-adaptive setup + slim installer (cycle 4)
Adoption cycle, same test-gated loop. The 947MB installer was 85% CUDA DLLs
(nvidia cublas+cudnn = 1,806MB of the 2.1GB bundle) while all models already
downloaded at first launch — so CUDA now downloads on demand too.
- **Hardware probe + tiers** (C4-G1): `firstrun.probe_hardware()` (nvidia-smi VRAM / GetPhysicallyInstalledSystemMemory / cpu_count) + pure `recommend()` over TIERS: gpu (≥6GB VRAM → large-v3-turbo), cpu (≥16GB RAM → small.en), lite (small.en + cleanup off).
- **Section-aware config writes** (C4-G2): `config_set(section, key, value)` — `model`/`enabled` appear in two sections each; the old single-regex pattern would clobber the wrong one.
- **apply_tier + tray menu** (C4-G3): writes the tier to config.toml AND the in-memory config dicts; `asr._FALLBACKS` moved inside `load_model()` (call-time, not import-time); tray "Model quality" radio submenu.
- **First-launch wizard** (C4-G4): Tk dialog (stdin is never wired — app starts hidden/Run-key): detected hardware, tier radios with trade-off blurbs, recommended preselected, 30s auto-accept, optional winget Ollama install. Never fires under --simulate.
- **CUDA on demand** (C4-G5): Vanni.spec no longer bundles nvidia DLLs; `ensure_cuda()` fetches pinned wheels (cublas 12.9.2.10, cudnn 9.24.0.43 — exact versions the build was tested against) from PyPI, sha256 pinned in code, extracts only `nvidia/*/bin/*.dll` next to the exe; asr DLL registration re-runs from `load_model` (first-launch ordering). Honest framing: total first-run download is similar (~1.2GB CUDA on GPU machines) — the win is the small installer.
- **Release** (C4-G6): installer AppVersion 1.3.0 -> 1.4.0. Test suite 18 -> 22 tests, all green.
