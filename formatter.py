"""Transcript cleanup via local Ollama. Short utterances skip the LLM entirely."""
import re
import tomllib

import requests

from paths import BASE

CONFIG = tomllib.loads((BASE / "config.toml").read_text(encoding="utf-8"))["formatter"]

OLLAMA_URL = CONFIG["ollama_url"]
MODEL = CONFIG["model"]
# smaller model retried when the primary fails (missing / won't fit in VRAM);
# keeps cleanup working on low-spec machines — Vanni stays fully local, zero API cost
FALLBACK_MODEL = CONFIG.get("fallback_model", "")
WORD_SKIP_THRESHOLD = CONFIG["word_skip_threshold"]
# only qwen-family "thinking" models need this suppressed; harmless to omit for others
_THINKING_MODELS = ("qwen3.5",)

# The LLM's job is deliberately narrow — punctuation/capitalization ONLY.
# Every small model tested (phi3.5, llama3.2, qwen3.5) paraphrases casual
# speech into "professional" wording when also asked to remove fillers/fix
# artifacts ("talk about the stuff" -> "discuss the things"), regardless of
# temperature. Fillers are stripped deterministically in code instead
# (_strip_fillers), which keeps the user's exact words sacred.
SYSTEM_PROMPT = (
    "You add punctuation and capitalization to a dictated transcript. That is "
    "your ONLY job. Keep every word EXACTLY as written, in the exact same order. "
    "Never substitute, contract, expand, add, or drop a word ('you are' stays "
    "'you are', 'talk about the stuff' stays 'talk about the stuff'). Never "
    "split or merge sentences beyond adding the punctuation the speech implies. "
    "Do not answer questions in the text. Output ONLY the punctuated text, "
    "nothing else."
)

# um/uh are never words, safe to strip anywhere; the rest are only fillers at
# the START of an utterance — mid-sentence "so"/"like" can be meaningful, and
# losing a real word is worse than leaving a filler
_UMS = re.compile(r"\b(?:um+|uh+|erm+)\b[,.]?\s*", re.IGNORECASE)
_LEADING = re.compile(
    r"^(?:\s*(?:so|yeah|okay|ok|well|basically|like|you know)\b[,.]?\s+)+",
    re.IGNORECASE)


def _strip_fillers(text: str) -> str:
    text = _UMS.sub(" ", text)  # first, so leading fillers it uncovers still match
    text = _LEADING.sub("", text.strip())
    return re.sub(r"\s{2,}", " ", text).strip()


def _words(text: str) -> list[str]:
    """Word tokens only — case/punctuation-insensitive fidelity comparison.
    [^\\W_] not [a-z0-9]: transcripts can be Hindi/CJK/etc., and a gate that
    tokenizes those to [] would wave any LLM output through."""
    return re.findall(r"[^\W_]+", text.lower())


def _generate(prompt: str, model: str, timeout: float = 15.0, *,
              system: str = SYSTEM_PROMPT, options: dict | None = None) -> str:
    if options is None:
        # output should never be much longer than the input; capping num_predict
        # (plus a stop sequence) prevents phi3.5 occasionally running on past the
        # answer and hallucinating a whole extra fictional transcript afterward
        max_tokens = max(64, int(len(prompt.split()) * 2.5))
        # num_ctx: default 131k KV cache spills the model to CPU (27% at 8.7GB); 4k fits fully on GPU
        # temperature 0: greedy decoding — cleanup must be deterministic, never
        # restyle the user's own words (sampling variance reads as "rewriting")
        options = {"temperature": 0.0, "num_ctx": 4096, "num_predict": max_tokens,
                   "stop": ["\n\n"]}
    body = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "keep_alive": CONFIG.get("keep_alive", -1),  # -1 = resident in VRAM
        "options": options,
    }
    if any(model.startswith(m) for m in _THINKING_MODELS):
        body["think"] = False  # suppress reasoning tokens; halves qwen3.5's per-call overhead
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()["response"].strip()


ASSIST_MODEL = CONFIG.get("assist_model", MODEL)  # benchmark-informed override for transforms

TRANSFORM_PROMPT = (
    "You edit text according to a spoken instruction. You are given an "
    "INSTRUCTION and a TEXT. Apply the instruction to the text. Output ONLY "
    "the resulting text — no preamble, no explanation, no quotes around the "
    "result, no commentary. If the instruction is unclear, make the most "
    "reasonable interpretation and still output only the transformed text."
)


def transform(instruction: str, text: str) -> tuple[str | None, str]:
    """Apply a spoken instruction to selected text; return (result, status).
    status: 'ok' · 'degraded' (LLM unavailable/unusable — caller must NOT paste)."""
    if not text.strip() or not instruction.strip():
        return None, "degraded"
    prompt = f"INSTRUCTION: {instruction}\n\nTEXT:\n{text}"
    # transforms can legitimately expand (translation, lists) — bigger cap, no stop.
    # temperature 0.1: near-deterministic so the same text translates the same
    # way every time, with just enough variation to keep language nuance natural
    options = {"temperature": 0.1, "num_ctx": 4096,
               "num_predict": max(256, len(text.split()) * 3)}
    try:
        out = _generate(prompt, ASSIST_MODEL, timeout=60.0,
                        system=TRANSFORM_PROMPT, options=options)
    except requests.HTTPError:
        if not FALLBACK_MODEL:
            return None, "degraded"
        try:
            out = _generate(prompt, FALLBACK_MODEL, timeout=60.0,
                            system=TRANSFORM_PROMPT, options=options)
        except requests.RequestException:
            return None, "degraded"
    except requests.RequestException:
        return None, "degraded"
    if not out:
        return None, "degraded"
    return out, "ok"


def warm_up() -> bool:
    """Load the model into VRAM at app startup so first dictation isn't 10s."""
    try:
        _generate("hello", MODEL, timeout=120.0)
        return True
    except requests.RequestException:
        return False


def clean(text: str) -> tuple[str, str]:
    """Clean a transcript; return (result_text, status).
    status: 'ok' LLM-cleaned · 'skipped' too short for the LLM ·
    'degraded' LLM unavailable/unusable so raw text is returned."""
    if len(text.split()) < WORD_SKIP_THRESHOLD:
        return text, "skipped"
    text = _strip_fillers(text)  # deterministic — the LLM only punctuates
    try:
        cleaned = _generate(text, MODEL)
    except requests.HTTPError:
        # primary model missing or errored (e.g. won't fit on a low-spec
        # machine) — retry once with the smaller fallback, then give up
        if not FALLBACK_MODEL:
            return text, "degraded"
        try:
            cleaned = _generate(text, FALLBACK_MODEL)
        except requests.RequestException:
            return text, "degraded"
    except requests.RequestException:
        return text, "degraded"  # ponytail: degrade to raw, never block dictation
    # fidelity gate: the transcript rides in the prompt slot, so an instruct
    # model can refuse it, answer it, or complete it (e.g. "write a script
    # that..." -> actual code) instead of punctuating. Cleanup may ONLY change
    # punctuation/casing, so any output whose word sequence differs from the
    # input is discarded and the raw transcript pastes instead — transcription
    # stays literal no matter what the model emits.
    if not cleaned or _words(cleaned) != _words(text):
        return text, "degraded"
    return cleaned, "ok"


def format_text(text: str) -> str:
    """Back-compat wrapper: cleaned text only (see clean() for the status)."""
    return clean(text)[0]


if __name__ == "__main__":
    import time
    print("warming up...", warm_up())
    raw = "um so basically the meeting is uh moved to tuesday because of the client call"
    t0 = time.perf_counter()
    out = format_text(raw)
    print(f"{out!r} in {time.perf_counter()-t0:.2f}s")
