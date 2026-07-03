"""Deterministic correction engine: maps known misheard phrases to intended text.

Runs on every transcript BEFORE the LLM formatter — a dictionary hit costs
microseconds instead of an inference round-trip. Users grow corrections.json
over time as they notice recurring mishears (names, jargon, product terms).

corrections.json format: {"misheard phrase": "intended text", ...}
Matching is case-insensitive on word boundaries; replacement preserves nothing
from the mishear (the value is inserted verbatim).
"""
import json
import re

from paths import BASE

_PATH = BASE / "corrections.json"
_cache: tuple[float, list[tuple[re.Pattern, str, str]]] | None = None


def _rules() -> list[tuple[re.Pattern, str, str]]:
    """Compiled (pattern, replacement, original_key) rules, reloaded
    automatically when corrections.json changes."""
    global _cache
    mtime = _PATH.stat().st_mtime if _PATH.exists() else 0.0
    if _cache is None or _cache[0] != mtime:
        mapping = json.loads(_PATH.read_text(encoding="utf-8")) if _PATH.exists() else {}
        rules = [
            (re.compile(rf"\b{re.escape(k)}\b", re.IGNORECASE), v, k)
            for k, v in sorted(mapping.items(), key=lambda kv: -len(kv[0]))  # longest first
        ]
        _cache = (mtime, rules)
    return _cache[1]


def apply(text: str) -> str:
    return apply_verbose(text)[0]


def apply_verbose(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Like apply(), but also returns [(misheard_key, replacement), ...] for
    the rules that actually fired — recorded in the dictation history."""
    applied: list[tuple[str, str]] = []
    for pattern, replacement, key in _rules():
        text, n = pattern.subn(replacement, text)
        if n:
            applied.append((key, replacement))
    return text, applied


def learn(misheard: str, intended: str) -> None:
    """Add/update one correction and persist it."""
    mapping = json.loads(_PATH.read_text(encoding="utf-8")) if _PATH.exists() else {}
    mapping[misheard.strip().lower()] = intended.strip()
    _PATH.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    learn("whisper flow", "Wispr Flow")
    learn("o lama", "Ollama")
    out = apply("I built a Whisper Flow clone with O Lama backing it")
    print(out)
    assert out == "I built a Wispr Flow clone with Ollama backing it"
    print("corrections OK")
