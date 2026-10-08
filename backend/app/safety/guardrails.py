"""
guardrails.py — input / output safety checks (plan step 4.5).

Input  (before retrieval, before any LLM spend):
    JailbreakDetector  — local prompt-injection classifier + fast regex pre-filter.
Output (after the citation guard):
    PII detection      — regex PII (email, phone, SSN, card w/ Luhn) -> redact.
    Secret detection   — credential-shaped strings and the literal values of this
                         process's own secrets -> block.

Deviation from the written plan: the `guardrails-ai` Hub validators
(DetectJailbreak / DetectPII / SecretsPresent) are not installable as plain PyPI
packages (they need a Hub token + `guardrails hub install`), so the same three
checks are implemented here locally with the same fail-safe semantics. Models run
locally; no extra hosted dependency or API key.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

from backend.app import config

log = logging.getLogger(__name__)

# ── Input: jailbreak / prompt-injection ──────────────────────────────────────

_INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
    r"disregard (all |any |the |your )?(previous|prior|above|earlier|system)? ?(instructions|prompts?|rules)",
    r"forget (everything|all|your) (you|instructions|rules)",
    r"(reveal|show|print|repeat|output) (me )?(your|the) (system|hidden|initial) (prompt|instructions)",
    r"you are (now )?(dan|in developer mode|jailbroken)",
    r"do anything now",
    r"pretend (you have|there are) no (rules|restrictions|guidelines)",
    r"\bjailbreak\b",
    r"override (your|the) (safety|system|instructions)",
    r"</?\s*(system|assistant)\s*>",
]
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in _INJECTION_PATTERNS), re.I)


@dataclass
class InputVerdict:
    ok: bool
    reason: str = ""
    score: float = 0.0


class JailbreakDetector:

    def __init__(self, model_name: str | None = None, threshold: float = 0.85,
                 use_model: bool = True, context_threshold: float = 0.999):
        self.model_name = model_name or config.JAILBREAK_MODEL
        self.threshold = threshold
        self.context_threshold = context_threshold
        self.use_model = use_model
        self._clf = None
        self._load_failed = False

    def _classifier(self):
        if self._clf is None and not self._load_failed:
            try:
                from transformers import pipeline
                self._clf = pipeline("text-classification", model=self.model_name,
                                     truncation=True, max_length=512)
            except Exception as e:
                log.warning("jailbreak model unavailable (%s); regex pre-filter only", e)
                self._load_failed = True
        return self._clf

    def score(self, text: str) -> float:
        """P(injection) from the classifier; 0.0 if the model is unavailable."""
        clf = self._classifier() if self.use_model else None
        if clf is None:
            return 0.0
        out = clf(text)[0]
        label = str(out["label"]).upper()
        bad = label in ("INJECTION", "JAILBREAK", "LABEL_1", "UNSAFE")
        return float(out["score"]) if bad else 1.0 - float(out["score"])

    @staticmethod
    def _is_prose(text: str) -> bool:
        """Numeric tables / reference lists look 'injection-like' to the classifier; skip them."""
        letters = sum(ch.isalpha() for ch in text)
        return len(text.split()) >= 15 and letters / max(len(text), 1) >= 0.6

    def check_context(self, text: str) -> InputVerdict:
        """
        Screen a *retrieved chunk* (untrusted paper text). Regex rules always apply; the
        classifier only on prose and at a much higher bar than for user queries,
        because dense paper text (tables, citations) produces false positives at 0.85.
        """
        if _INJECTION_RE.search(text):
            return InputVerdict(False, "matches a known prompt-injection pattern", 1.0)
        if self.use_model and self._is_prose(text):
            s = self.score(text)
            if s >= self.context_threshold:
                return InputVerdict(False, "classified as an embedded injection", s)
        return InputVerdict(True)

    def check_context_many(self, texts: list[str]) -> list[InputVerdict]:
        """Batched `check_context`: regex per text, then ONE classifier call over the prose candidates."""
        verdicts: list[InputVerdict | None] = [None] * len(texts)
        todo: list[int] = []
        for i, t in enumerate(texts):
            if _INJECTION_RE.search(t):
                verdicts[i] = InputVerdict(False, "matches a known prompt-injection pattern", 1.0)
            elif self.use_model and self._is_prose(t):
                todo.append(i)
            else:
                verdicts[i] = InputVerdict(True)
        clf = self._classifier() if todo else None
        if todo and clf is not None:
            outs = clf([texts[i] for i in todo], batch_size=8)
            for i, out in zip(todo, outs):
                out = out[0] if isinstance(out, list) else out
                bad = str(out["label"]).upper() in ("INJECTION", "JAILBREAK", "LABEL_1", "UNSAFE")
                s = float(out["score"]) if bad else 1.0 - float(out["score"])
                verdicts[i] = (InputVerdict(False, "classified as an embedded injection", s)
                               if s >= self.context_threshold else InputVerdict(True, "", s))
        return [v or InputVerdict(True) for v in verdicts]

    def check(self, text: str) -> InputVerdict:
        if _INJECTION_RE.search(text):
            return InputVerdict(False, "matches a known prompt-injection pattern", 1.0)
        s = self.score(text)
        if s >= self.threshold:
            return InputVerdict(False, "classified as a prompt-injection / jailbreak attempt", s)
        return InputVerdict(True, "", s)


# ── Output: PII + secrets ────────────────────────────────────────────────────

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4}(?!\d)")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")

_SECRET_PATTERNS = {
    "groq_key":        re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b"),
    "openai_key":      re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    "aws_access_key":  re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_token":    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "slack_token":     re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    "hf_token":        re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "private_key":     re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "bearer_token":    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{24,}"),
    "assigned_secret": re.compile(
        r"(?i)\b(?:api[_-]?key|secret|token|password|passwd)\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-./+=]{16,}"),
    "jwt":             re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
}


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def _own_secret_values() -> list[str]:
    vals = [config.GROQ_API_KEY, config.QDRANT_API_KEY]
    for k, v in os.environ.items():
        if v and re.search(r"(KEY|TOKEN|SECRET|PASSWORD)", k, re.I):
            vals.append(v)
    return [v for v in set(vals) if v and len(v) >= 12]


@dataclass
class OutputVerdict:
    ok: bool                       # True = answer may be returned as `text`
    text: str                      # possibly redacted answer ("" when blocked)
    blocked: bool = False
    redacted: bool = False
    findings: list[str] = field(default_factory=list)
    message: str = ""              # plain-language note for the user


class OutputGuard:

    def check(self, answer: str) -> OutputVerdict:
        findings: list[str] = []

        # secrets -> block outright (never partially return a credential-shaped answer)
        for name, rx in _SECRET_PATTERNS.items():
            if rx.search(answer):
                findings.append(name)
        for v in _own_secret_values():
            if v in answer:
                findings.append("configured_secret")
        if findings:
            return OutputVerdict(
                ok=False, text="", blocked=True, findings=sorted(set(findings)),
                message="The answer was withheld because it contained a credential-like string.")

        # PII -> redact
        text = answer
        pii: list[str] = []

        def sub(rx, label, s, validator=None):
            def repl(m):
                if validator and not validator(m.group(0)):
                    return m.group(0)
                pii.append(label)
                return f"[REDACTED {label}]"
            return rx.sub(repl, s)

        text = sub(_EMAIL, "EMAIL", text)
        text = sub(_SSN, "SSN", text)
        text = sub(_CARD, "CARD", text,
                   lambda s: 13 <= len(re.sub(r"\D", "", s)) <= 19 and _luhn(re.sub(r"\D", "", s)))
        text = sub(_PHONE, "PHONE", text)
        if pii:
            return OutputVerdict(
                ok=True, text=text, redacted=True, findings=sorted(set(pii)),
                message="Personal information detected in the answer was redacted.")
        return OutputVerdict(ok=True, text=text)
