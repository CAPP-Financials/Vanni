"""Phase 8: benchmark cleanup-LLM alternatives against qwen3.5:4b.

Same request shape as formatter.py's _generate(): same system prompt, same
fixture, think:false where supported, num_ctx=4096, keep_alive=-1. Measures
Ollama's own load/prompt_eval/eval duration breakdown (not just wall time) so
we can tell a thinking-mode artifact apart from general Ollama call overhead.
"""
import json
import time

import requests

URL = "http://127.0.0.1:11434/api/generate"
SYSTEM_PROMPT = (
    "You clean up dictated speech transcripts. Fix punctuation and capitalization, "
    "remove filler words (um, uh, like, you know, basically), and fix obvious "
    "dictation artifacts. Do NOT change the meaning, add content, or answer "
    "questions in the text. Output ONLY the cleaned text, nothing else."
)
FIXTURE = "um so basically the meeting is uh moved to tuesday because of the client call"

# think:False is only meaningful for qwen3.5 (a thinking model); harmless no-op
# for the rest, but we tag it so the table records which models even have the toggle.
CANDIDATES = [
    ("qwen3.5:4b", True),   # current baseline
    ("qwen2.5:3b", False), # same vendor, NOT a thinking model -> isolates the cause
    ("llama3.2:3b", False),
    ("phi3.5", False),
    ("gemma2:2b", False),
]


def call(model: str, has_think: bool, prompt: str = FIXTURE) -> dict:
    body = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "keep_alive": -1,
        "options": {"temperature": 0.1, "num_ctx": 4096},
    }
    if has_think:
        body["think"] = False
    t0 = time.perf_counter()
    r = requests.post(URL, json=body, timeout=60)
    wall = time.perf_counter() - t0
    r.raise_for_status()
    j = r.json()
    j["_wall"] = wall
    return j


def quality_ok(text: str) -> bool:
    low = f" {text.lower()} "
    return "tuesday" in low and " um " not in low and " uh " not in low


def bench(model: str, has_think: bool) -> dict:
    call(model, has_think)  # warm-up / load into VRAM (not timed for the report)
    r = call(model, has_think)  # timed, warm
    text = r.get("response", "").strip()
    return {
        "model": model,
        "load_ms": r.get("load_duration", 0) / 1e6,
        "prompt_eval_ms": r.get("prompt_eval_duration", 0) / 1e6,
        "eval_ms": r.get("eval_duration", 0) / 1e6,
        "total_ms": r.get("total_duration", 0) / 1e6,
        "wall_ms": r["_wall"] * 1000,
        "quality": "PASS" if quality_ok(text) else "FAIL",
        "output": text,
    }


if __name__ == "__main__":
    rows = []
    for model, has_think in CANDIDATES:
        print(f"benchmarking {model}...")
        try:
            row = bench(model, has_think)
        except requests.RequestException as e:
            row = {"model": model, "load_ms": -1, "prompt_eval_ms": -1, "eval_ms": -1,
                   "total_ms": -1, "wall_ms": -1, "quality": f"ERROR: {e}", "output": ""}
        rows.append(row)
        print(f"  {row}")

    print("\n| Model | Load (ms) | Prompt-eval (ms) | Eval (ms) | Total (ms) | Wall (ms) | Quality |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r['load_ms']:.0f} | {r['prompt_eval_ms']:.0f} | "
              f"{r['eval_ms']:.0f} | {r['total_ms']:.0f} | {r['wall_ms']:.0f} | {r['quality']} |")

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
