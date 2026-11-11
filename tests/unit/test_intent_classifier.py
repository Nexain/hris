import asyncio

import pytest

from app.core.config import Settings
from app.models.chat import IntentType
from app.schemas.chat import ChatMessage
from app.services.intent import IntentClassifier
from app.services.router import ScopeRouter


class FakeGemini:
    """Returns a fixed label, or raises / hangs, and records calls."""

    def __init__(self, label=None, error=None, delay=0.0):
        self.label, self.error, self.delay = label, error, delay
        self.calls = []

    async def generate_json(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return {"intent": self.label}


def _classifier(gemini, **settings):
    return IntentClassifier(vertex_service=gemini, settings=Settings(**settings))


@pytest.mark.asyncio
@pytest.mark.parametrize("message", ["am I done yet?", "masih ada yang perlu kukerjakan?", "how far along am I"])
async def test_gemini_label_is_used_when_no_rule_matches(message):
    gemini = FakeGemini("MY_ONBOARDING")
    assert await _classifier(gemini).classify(message) == (IntentType.MY_ONBOARDING, True, "")
    assert gemini.calls[0]["prompt"].endswith(message)


@pytest.mark.asyncio
@pytest.mark.parametrize("message, expected", [
    ("what's left for me", IntentType.MY_ONBOARDING),
    ("apa tugas saya hari ini?", IntentType.MY_ONBOARDING),
    ("my checklist", IntentType.MY_ONBOARDING),
    ("Write me a poem about cats", IntentType.OUT_OF_SCOPE),
])
async def test_regex_match_skips_gemini(message, expected):
    gemini = FakeGemini("COMPANY_KNOWLEDGE")
    intent, _, _ = await _classifier(gemini).classify(message)
    assert intent == expected
    assert gemini.calls == []


@pytest.mark.asyncio
async def test_out_of_scope_and_sensitive_labels_are_refused():
    intent, allowed, reason = await _classifier(FakeGemini("OUT_OF_SCOPE")).classify("Tell me something fun about cats")
    assert (intent, allowed, reason) == (IntentType.OUT_OF_SCOPE, False, ScopeRouter.OUT_OF_SCOPE_REASON)

    intent, allowed, reason = await _classifier(FakeGemini("SENSITIVE")).classify("What does Budi take home?")
    assert (intent, allowed, reason) == (IntentType.OUT_OF_SCOPE, False, ScopeRouter.SENSITIVE_REASON)


@pytest.mark.asyncio
async def test_sensitive_regex_runs_before_gemini():
    gemini = FakeGemini("COMPANY_KNOWLEDGE")
    intent, allowed, _ = await _classifier(gemini).classify("What is the salary of my manager?")
    assert (intent, allowed) == (IntentType.OUT_OF_SCOPE, False)
    assert gemini.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("gemini", [
    FakeGemini(error=RuntimeError("quota")),
    FakeGemini("NOT_A_LABEL"),
    FakeGemini("MY_ONBOARDING", delay=1.0),
])
async def test_falls_back_to_regex_router(gemini):
    classifier = _classifier(gemini, intent_timeout_seconds=0.05)
    # No rule matches and Gemini is unusable: treat it as a document question.
    assert await classifier.classify("am I done yet?") == (IntentType.COMPANY_KNOWLEDGE, True, "")
    assert gemini.calls


@pytest.mark.asyncio
async def test_disabled_classifier_skips_gemini():
    gemini = FakeGemini("OUT_OF_SCOPE")
    intent, _, _ = await _classifier(gemini, intent_classifier_enabled=False).classify("am I done yet?")
    assert intent == IntentType.COMPANY_KNOWLEDGE
    assert gemini.calls == []


@pytest.mark.asyncio
async def test_recent_history_is_sent_for_follow_ups():
    gemini = FakeGemini("COMPANY_KNOWLEDGE")
    history = [ChatMessage(role="user", content=f"q{i}") for i in range(5)]
    await _classifier(gemini).classify("and the rest?", history)
    assert [m.content for m in gemini.calls[0]["history"]] == ["q3", "q4"]


@pytest.mark.parametrize("message, expected", [
    ("What tasks do I still have left?", IntentType.MY_ONBOARDING),
    ("what's left for me", IntentType.MY_ONBOARDING),
    ("apa tugas saya hari ini?", IntentType.MY_ONBOARDING),
    ("tugasku apa saja?", IntentType.MY_ONBOARDING),
    ("my checklist", IntentType.MY_ONBOARDING),
    ("Write me a poem about cats", IntentType.OUT_OF_SCOPE),
    ("buatkan puisi tentang kantor", IntentType.OUT_OF_SCOPE),
    ("How many days of annual leave do I get?", IntentType.COMPANY_KNOWLEDGE),
    ("berapa hari cuti tahunan?", IntentType.COMPANY_KNOWLEDGE),
    ("berapa gaji atasan saya?", IntentType.OUT_OF_SCOPE),
    ("berapa gaji pokok saya menurut kebijakan?", IntentType.COMPANY_KNOWLEDGE),
])
def test_regex_fallback_patterns(message, expected):
    assert ScopeRouter.route(message)[0] == expected


@pytest.mark.parametrize("message, lang", [
    ("apa tugas saya hari ini?", "id"),
    ("tugasku apa saja?", "id"),
    ("berapa hari cuti tahunan?", "id"),
    ("How many days of annual leave do I get?", "en"),
    ("my checklist", "en"),
    ("What is the dress code in the Jakarta office?", "en"),
])
def test_detect_language(message, lang):
    from app.services.language import detect_language
    assert detect_language(message) == lang
