"""Voice snippets: say a trigger phrase, paste stored text verbatim.

snippets.json format: {"trigger phrase": "expansion text", ...}
The WHOLE utterance (case/punctuation-insensitive) must equal a trigger —
snippets never fire mid-sentence. Matches skip the LLM entirely: the stored
expansion is pasted exactly as written (sign-offs, addresses, boilerplate).
Hot-reloads on edit, same pattern as corrections.py.
"""
import json

from paths import BASE

_PATH = BASE / "snippets.json"
_cache: tuple[float, dict[str, str]] | None = None


def _normalize(s: str) -> str:
    return s.strip().strip(".,!?").lower()


def _mapping() -> dict[str, str]:
    global _cache
    mtime = _PATH.stat().st_mtime if _PATH.exists() else 0.0
    if _cache is None or _cache[0] != mtime:
        raw = json.loads(_PATH.read_text(encoding="utf-8")) if _PATH.exists() else {}
        _cache = (mtime, {_normalize(k): v for k, v in raw.items()})
    return _cache[1]


def match(text: str) -> str | None:
    """Expansion for a whole-utterance trigger, or None."""
    return _mapping().get(_normalize(text))


def learn(trigger: str, expansion: str) -> None:
    """Add/update one snippet and persist it."""
    raw = json.loads(_PATH.read_text(encoding="utf-8")) if _PATH.exists() else {}
    raw[trigger.strip().lower()] = expansion
    _PATH.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    assert match("Insert signature.") is not None
    assert match("insert signature") == match("  INSERT SIGNATURE! ")
    assert match("just a normal sentence") is None
    print("snippets OK")
