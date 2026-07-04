"""Smart formatting: deterministic spoken→written conversions, zero LLM latency.

Runs LAST in the pipeline (after the LLM formatter) so its output is the final
authority. Deliberately conservative — each rule requires an unambiguous spoken
pattern so ordinary sentences ("we met at noon") are never touched:

  "john dot smith at gmail dot com"  -> john.smith@gmail.com   (needs a real TLD)
  "example dot com"                  -> example.com            (needs a real TLD)
  "new line" / "new paragraph"       -> \\n / \\n\\n (spoken layout commands)
"""
import re

# common TLDs gate the email/URL rules so "at nine dot thirty" can't match
_TLD = r"(?:com|org|net|io|ai|dev|app|co|uk|in|de|edu|gov)"
_EMAIL = re.compile(
    rf"\b(\w+(?:\s+dot\s+\w+)*)\s+at\s+(\w+(?:\s+dot\s+\w+)*\s+dot\s+{_TLD})\b",
    re.IGNORECASE)
_URL = re.compile(rf"\b(\w+(?:\s+dot\s+\w+)*\s+dot\s+{_TLD})\b", re.IGNORECASE)
_NEWPARA = re.compile(r"\s*\bnew paragraph\b[,.]?\s*", re.IGNORECASE)
_NEWLINE = re.compile(r"\s*\bnew line\b[,.]?\s*", re.IGNORECASE)

_dots = lambda s: re.sub(r"\s+dot\s+", ".", s, flags=re.IGNORECASE).lower()


def apply(text: str) -> str:
    text = _EMAIL.sub(lambda m: f"{_dots(m[1])}@{_dots(m[2])}", text)
    text = _URL.sub(lambda m: _dots(m[1]), text)  # after emails (they contain "dot com" too)
    text = _NEWPARA.sub("\n\n", text)
    text = _NEWLINE.sub("\n", text)
    return text


if __name__ == "__main__":
    assert apply("mail John dot Smith at gmail dot com now") == "mail john.smith@gmail.com now"
    assert apply("see example dot com") == "see example.com"
    assert apply("a new line b new paragraph c") == "a\nb\n\nc"
    assert apply("we met at noon dot thirty") == "we met at noon dot thirty"  # no TLD, untouched
    print("smartfmt OK")
