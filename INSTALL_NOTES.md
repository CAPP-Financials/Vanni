VANNI — LOCAL DICTATION FOR WINDOWS
===================================

What you are installing
-----------------------
Vanni is a push-to-talk dictation app that runs 100% ON THIS MACHINE.

  * No cloud, no account, no subscription, no API tokens — ever.
  * Your voice and transcripts never leave your computer.
  * Dictation history is saved locally (history\ folder next to Vanni.exe)
    as plain markdown you own and can delete at any time.

How it works
------------
Hold Ctrl+Win, speak, release: your words are typed at the cursor.
Hold Ctrl+Alt+Win instead to skip the LLM cleanup (exact/raw text).

First launch (one-time downloads, internet needed ONCE)
-------------------------------------------------------
  1. The Whisper speech model (~1.5 GB) downloads automatically.
  2. If Ollama is installed, the text-cleanup model(s) (~2-4 GB) are
     pulled automatically. Ollama is free and local:
         winget install Ollama.Ollama
     Without Ollama, Vanni still works — it just pastes raw transcripts.
  3. After these downloads Vanni is fully offline. Total disk use after
     setup: roughly 5 GB including models.

Low-performance laptops
-----------------------
Vanni degrades automatically — no configuration needed:
  * No/weak GPU: speech recognition falls back to a small CPU model.
  * Under 8 GB VRAM: cleanup falls back to a smaller LLM (qwen2.5:3b).
  * Voice-activity detection trims silence to keep CPU use low.
Everything stays local and free at every tier.

Antivirus / Windows Defender
----------------------------
Vanni uses a global hotkey, types text via synthetic keystrokes, and writes
to the clipboard. That combination looks structurally like a keylogger to
security software, so Defender or corporate endpoint tools may flag or block
it. This is expected for any dictation tool that types for you — Vanni is
fully local and open-source (read the code). If it is blocked:
  * Windows Security -> Virus & threat protection -> Manage settings ->
    Add or remove exclusions -> add the Vanni install folder.
  * On managed/work laptops, ask IT to allow-list it.

Admin (elevated) windows
------------------------
Pasting into an elevated window (e.g. an admin terminal) silently fails
unless Vanni itself runs elevated — the text still lands in your clipboard,
and Vanni now warns you to press Ctrl+V. To dictate directly into admin
windows, start Vanni elevated:
    powershell -ExecutionPolicy Bypass -File Vanni-launcher.ps1 -Elevated
(A UAC prompt appears once.) Some games/anti-cheat software also block
synthetic Ctrl+V.

Startup option
--------------
On the next page you can tick "Start Vanni when Windows starts" so it is
always running in the background, ready to dictate.
