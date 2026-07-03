"""Local dictation history: one markdown file per day in history/.

Every dictation is appended with its raw transcript, any corrections that
fired, and the final injected text. Fully local (the whole point of Vanni) —
this doubles as a growing corpus for future ML work (mishear patterns,
personal language model fine-tuning, correction mining).

Writes must never block or break dictation: any failure is swallowed.
"""
import time
import tomllib

from paths import BASE

CONFIG = tomllib.loads((BASE / "config.toml").read_text(encoding="utf-8"))
ENABLED = CONFIG.get("history", {}).get("enabled", True)

DIR = BASE / "history"


def record(raw: str, final: str, *, mode: str = "formatted",
           corrections_applied: list[tuple[str, str]] | None = None,
           duration_s: float = 0.0) -> None:
    """Append one dictation to today's history file. Never raises."""
    if not ENABLED or not final:
        return
    try:
        DIR.mkdir(exist_ok=True)
        lines = [f"## {time.strftime('%H:%M:%S')}  ({mode} · {duration_s:.1f}s)"]
        if raw != final:  # only log the raw stage when something changed it
            lines.append(f"- raw: {raw}")
        if corrections_applied:
            fixes = ", ".join(f"{a} → {b}" for a, b in corrections_applied)
            lines.append(f"- corrections: {fixes}")
        lines += [f"- final: {final}", ""]
        path = DIR / f"{time.strftime('%Y-%m-%d')}.md"
        with path.open("a", encoding="utf-8") as f:
            if f.tell() == 0:
                f.write(f"# Vanni dictation history — {time.strftime('%Y-%m-%d')}\n\n")
            f.write("\n".join(lines) + "\n")
    except OSError:
        pass  # ponytail: history is best-effort; dictation must never fail over logging


if __name__ == "__main__":
    record("the quick brown fox with o lama", "The quick brown fox with Ollama.",
           mode="formatted", corrections_applied=[("o lama", "Ollama")], duration_s=1.2)
    today = DIR / f"{time.strftime('%Y-%m-%d')}.md"
    text = today.read_text(encoding="utf-8")
    assert "The quick brown fox with Ollama." in text
    assert "o lama → Ollama" in text
    print(f"history OK -> {today}")
