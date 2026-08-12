"""
Cortex safety layer.

Three components:
  1. NeMo Guardrails — policy engine for input/output rails
  2. PII Scanner — Presidio-based detection and redaction
  3. Injection Detector — prompt injection pattern matching + LLM-as-judge

All components are optional (controlled by config flags) so the platform
degrades gracefully when safety libraries aren't installed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, ClassVar

from cortex.config import settings
from cortex.exceptions import PromptInjectionError
from cortex.logging_config import get_logger
from cortex.obs.metrics import safety_violations_total
from cortex.safety.moderation import ModerationError, get_moderator

logger = get_logger(__name__)


# ── PII Scanner ───────────────────────────────────────────────────────────────


class PIIScanner:
    """Detect and redact PII.

    Presidio and the regex patterns run *together*, not as alternatives.
    The original code treated them as an either/or fallback, which quietly
    made detection worse when Presidio was installed: Presidio's English
    configuration does not register `CreditCardRecognizer` and has no UK
    National Insurance recogniser at all, so installing the "better" engine
    silently stopped redacting card numbers and NINOs that the regex
    fallback had been catching.

    Presidio is also restricted to a configured allowlist of high-confidence
    identifiers. Its defaults flag DATE_TIME, PERSON, LOCATION and URL, so
    "the annual report" was being rewritten to "the <DATE_TIME> report" -
    a false positive that destroys the user's goal before an agent sees it,
    which is a worse failure than missing PII.
    """

    def __init__(self) -> None:
        self._analyzer: Any = None
        self._presidio_available = False
        self._init()

    def _init(self) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine

            self._analyzer = AnalyzerEngine()
            self._presidio_available = True
            logger.info("pii.presidio_loaded")
        except ImportError:
            logger.warning("pii.presidio_not_available - using regex patterns only")

    # Compiled regex patterns for PII Presidio does not cover for English.
    _PATTERNS: ClassVar[dict[str, re.Pattern[str]]] = {
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "phone_uk": re.compile(r"\b(?:\+44|0)[\d\s]{9,12}\b"),
        "phone_us": re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        "nino": re.compile(r"\b[A-Z]{2}\d{6}[A-D]\b"),  # UK National Insurance
    }

    def _presidio_spans(self, text: str) -> list[dict[str, Any]]:
        if not self._presidio_available:
            return []
        entities = list(settings.pii_entities)
        threshold = float(settings.pii_score_threshold)
        try:
            results = self._analyzer.analyze(text=text, language="en", entities=entities)
        except Exception:  # pragma: no cover - engine-internal failure
            logger.warning("pii.presidio_analyze_failed", exc_info=True)
            return []
        return [
            {"type": r.entity_type, "start": r.start, "end": r.end, "score": r.score}
            for r in results
            if r.score >= threshold
        ]

    def _regex_spans(self, text: str) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for pii_type, pattern in self._PATTERNS.items():
            for m in pattern.finditer(text):
                found.append({"type": pii_type, "start": m.start(), "end": m.end(), "score": 0.8})
        return found

    @staticmethod
    def _merge(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop spans overlapping an already-accepted, longer/stronger span.

        Two engines finding the same phone number must produce one
        redaction, not a redaction nested inside a redaction.
        """
        ordered = sorted(spans, key=lambda s: (-(s["end"] - s["start"]), -s["score"], s["start"]))
        kept: list[dict[str, Any]] = []
        for span in ordered:
            if any(span["start"] < k["end"] and k["start"] < span["end"] for k in kept):
                continue
            kept.append(span)
        return sorted(kept, key=lambda s: s["start"])

    def scan(self, text: str) -> list[dict[str, Any]]:
        """Return detected PII entities, from both engines, de-overlapped."""
        return self._merge(self._presidio_spans(text) + self._regex_spans(text))

    def redact(self, text: str) -> str:
        """Replace PII with type placeholders."""
        spans = self.scan(text)
        redacted = text
        for span in reversed(spans):
            placeholder = f"[{str(span['type']).upper()}_REDACTED]"
            redacted = redacted[: span["start"]] + placeholder + redacted[span["end"] :]
        return redacted


# ── Injection Detector ────────────────────────────────────────────────────────


class InjectionDetector:
    """
    Detects prompt injection attempts via pattern matching.
    For production, augment with an LLM-as-judge check.
    """

    _PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        # Role-breaking patterns
        re.compile(r"ignore\s+(?:all\s+)?(?:previous|above|prior)\s+instructions?", re.I),
        re.compile(r"forget\s+(?:everything|all)\s+(?:you|above)", re.I),
        re.compile(r"new\s+(?:system\s+)?prompt\s*:", re.I),
        re.compile(r"you\s+are\s+now\s+(?:a\s+)?(?:dan|jailbreak|uncensored)", re.I),
        # Exfiltration attempts
        re.compile(r"print\s+(?:your\s+)?(?:system\s+)?prompt", re.I),
        re.compile(r"reveal\s+(?:your\s+)?(?:instructions?|system\s+prompt)", re.I),
        re.compile(r"output\s+everything\s+(?:above|before)", re.I),
        # Override attempts
        re.compile(r"override\s+(?:safety|restriction|filter)", re.I),
        re.compile(r"\[INST\]|\[\/INST\]|<\|system\|>|<\|user\|>"),  # Llama-style injection
    ]

    def scan(self, text: str) -> list[str]:
        """Return list of matched injection pattern names."""
        matches = []
        for pattern in self._PATTERNS:
            if pattern.search(text):
                matches.append(pattern.pattern)
        return matches


# ── Guardrail middleware ──────────────────────────────────────────────────────


class SafetyMiddleware:
    """
    Unified safety check. Call check_input() before any LLM call
    and check_output() before returning results to users.
    """

    def __init__(self) -> None:
        self._pii = PIIScanner() if settings.pii_detection_enabled else None
        self._injection = InjectionDetector() if settings.injection_detection_enabled else None
        self._rails_app: Any = None
        if settings.guardrails_enabled:
            self._init_rails()

    def _init_rails(self) -> None:
        try:
            from nemoguardrails import LLMRails, RailsConfig

            rails_path = Path(__file__).resolve().parents[3] / "config" / "rails"
            if rails_path.exists():
                config = RailsConfig.from_path(rails_path)
                self._rails_app = LLMRails(config)
                logger.info("guardrails.nemo_loaded")
        except ImportError:
            logger.warning("guardrails.nemo_not_available")
        except Exception as exc:
            logger.warning("guardrails.nemo_init_failed", error=str(exc))

    async def check_input(self, text: str, user_id: str) -> str:
        """
        Validate and optionally sanitise user input.
        Returns sanitised text or raises a safety exception.
        """
        if not settings.guardrails_enabled:
            return text

        # Scored jailbreak detection runs first: it covers paraphrase that
        # the regex layer below cannot, and it is the cheaper of the two to
        # fail on.
        verdict = await get_moderator().check_input(text)
        if not verdict.allowed:
            logger.warning(
                "safety.jailbreak_blocked",
                user_id=user_id,
                score=verdict.score,
                reason=verdict.reason,
            )
            raise PromptInjectionError(f"Input refused by the moderation layer: {verdict.reason}")

        # 1. Injection check
        if self._injection:
            matches = self._injection.scan(text)
            if matches:
                safety_violations_total.labels(violation_type="prompt_injection").inc()
                logger.warning("safety.injection_detected", user_id=user_id, matches=matches[:3])
                raise PromptInjectionError(
                    "Input contains suspected prompt injection",
                    details={"pattern_count": len(matches)},
                )

        # 2. PII check — log and redact (don't block, to preserve usability)
        if self._pii:
            entities = self._pii.scan(text)
            if entities:
                safety_violations_total.labels(violation_type="pii_input").inc()
                logger.warning("safety.pii_in_input", user_id=user_id, entity_count=len(entities))
                text = self._pii.redact(text)

        return text

    async def check_output(self, text: str, context: list[str] | None = None) -> str:
        """
        Validate LLM output before returning to user.
        Returns sanitised text or raises a safety exception.
        """
        if not settings.guardrails_enabled:
            return text

        # Nothing used to look at what the model actually SAID - input was
        # filtered and PII redacted, but no control examined the output.
        # That is the wrong way round if you can only afford one: prompt
        # injection is a means, harmful output is the end.
        verdict = await get_moderator().check_output(text)
        if not verdict.allowed:
            logger.warning(
                "safety.output_blocked",
                categories=[c.value for c in verdict.categories],
                reason=verdict.reason,
            )
            raise ModerationError(f"Response withheld: {verdict.reason}")

        # PII redaction on output
        if self._pii:
            entities = self._pii.scan(text)
            if entities:
                safety_violations_total.labels(violation_type="pii_output").inc()
                logger.warning("safety.pii_in_output", entity_count=len(entities))
                text = self._pii.redact(text)

        return text


# Module-level singleton
_middleware: SafetyMiddleware | None = None


def get_safety_middleware() -> SafetyMiddleware:
    global _middleware
    if _middleware is None:
        _middleware = SafetyMiddleware()
    return _middleware
