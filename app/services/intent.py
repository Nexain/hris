"""Intent classification for chat messages.

Order of checks:
1. Regex rules (fast, no network): sensitive requests are blocked, and explicit
   matches (out of scope, my tasks, onboarding actions, general onboarding) are used
   as-is, skipping Gemini.
2. Otherwise one short Gemini call with structured JSON output picks the intent. It
   handles paraphrases and Indonesian that fixed patterns miss. The chat endpoint
   runs it in parallel with the document search.
3. If Gemini is disabled, fails, times out or returns something unexpected, the
   message is treated as a company-knowledge question.
"""

import asyncio
import logging
from typing import List, Optional, Sequence, Tuple

from fastapi import Depends
from google.genai import types

from app.core.config import Settings, get_settings
from app.models.chat import IntentType
from app.schemas.chat import ChatMessage
from app.services.router import ScopeRouter
from app.services.vertex_service import VertexGeminiService, get_vertex_service

logger = logging.getLogger(__name__)

SENSITIVE_LABEL = "SENSITIVE"
LABELS = [
    IntentType.COMPANY_KNOWLEDGE.value,
    IntentType.MY_ONBOARDING.value,
    IntentType.ONBOARDING_ACTION.value,
    IntentType.GENERAL_ONBOARDING.value,
    IntentType.OUT_OF_SCOPE.value,
    SENSITIVE_LABEL,
]

CLASSIFIER_PROMPT = """You classify messages sent to DayOne AI, an employee onboarding copilot.
Messages may be in English or Indonesian. Pick exactly one intent:

- MY_ONBOARDING: about the sender's own onboarding tasks, checklist, progress, what is left,
  what to do next or today, or their own profile.
  e.g. "what's left for me", "apa tugas saya hari ini?", "my checklist", "am I done yet?"
- ONBOARDING_ACTION: asks to mark, complete or finish one of the sender's tasks.
  e.g. "mark the laptop task as done", "saya sudah selesai tugas handbook"
- GENERAL_ONBOARDING: how the onboarding program works in general (process, timeline, stages),
  not the sender's own tasks. e.g. "how long is onboarding?"
- COMPANY_KNOWLEDGE: company policies, benefits, leave, office, IT, conduct, procedures, or any
  workplace question answered by company documents. e.g. "how many days of annual leave do I get?",
  "berapa hari cuti tahunan?"
- SENSITIVE: another person's salary or pay, or private HR records such as performance reviews,
  disciplinary or termination matters about someone else.
- OUT_OF_SCOPE: not about work at this company: jokes, poems, stories, songs, trivia, weather,
  coding help, general chit-chat.

Use the earlier turns only to understand short follow-ups. When unsure between
COMPANY_KNOWLEDGE and OUT_OF_SCOPE for a workplace-related question, choose COMPANY_KNOWLEDGE."""

RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={"intent": types.Schema(type=types.Type.STRING, enum=LABELS)},
    required=["intent"],
)

MAX_CONTEXT_TURNS = 2
MAX_CONTEXT_CHARS = 500


class IntentClassifier:
    """Classifies a chat message into an intent and decides whether it may be answered."""

    def __init__(
        self,
        vertex_service: Optional[VertexGeminiService] = None,
        settings: Optional[Settings] = None,
    ):
        self.vertex_service = vertex_service or get_vertex_service()
        self.settings = settings or get_settings()

    async def classify(
        self,
        message: str,
        history: Optional[Sequence[ChatMessage]] = None,
    ) -> Tuple[IntentType, bool, str]:
        """Return (intent, is_allowed, refusal_reason)."""
        fast = self.fast_route(message)
        if fast is not None:
            return fast
        return await self.classify_slow(message, history)

    @staticmethod
    def fast_route(message: str) -> Optional[Tuple[IntentType, bool, str]]:
        """Regex decision with no network call: sensitive, out of scope, my tasks, etc.

        Returns None when no rule fired; the caller then needs `classify_slow`.
        """
        return ScopeRouter.match(message)

    async def classify_slow(
        self,
        message: str,
        history: Optional[Sequence[ChatMessage]] = None,
    ) -> Tuple[IntentType, bool, str]:
        """Gemini classification for messages the regex rules did not match."""
        if self.settings.intent_classifier_enabled:
            label = await self._classify_with_gemini(message, history)
            if label == SENSITIVE_LABEL:
                return IntentType.OUT_OF_SCOPE, False, ScopeRouter.SENSITIVE_REASON
            if label == IntentType.OUT_OF_SCOPE.value:
                return IntentType.OUT_OF_SCOPE, False, ScopeRouter.OUT_OF_SCOPE_REASON
            if label is not None:
                return IntentType(label), True, ""

        return ScopeRouter.route(message)

    async def _classify_with_gemini(
        self,
        message: str,
        history: Optional[Sequence[ChatMessage]],
    ) -> Optional[str]:
        """Return a label from LABELS, or None if Gemini could not be used."""
        context: List[ChatMessage] = [
            ChatMessage(role=m.role, content=m.content[:MAX_CONTEXT_CHARS])
            for m in (history or [])
            if m.role != "system"
        ][-MAX_CONTEXT_TURNS:]
        try:
            result = await asyncio.wait_for(
                self.vertex_service.generate_json(
                    prompt=f"Message to classify:\n{message}",
                    response_schema=RESPONSE_SCHEMA,
                    system_instruction=CLASSIFIER_PROMPT,
                    history=context or None,
                    model=self.settings.intent_model or None,
                    max_output_tokens=256,
                ),
                timeout=self.settings.intent_timeout_seconds,
            )
        except Exception as e:  # includes timeouts and missing GCP config
            logger.info(f"Intent classifier unavailable ({type(e).__name__}: {e}); using regex router.")
            return None

        label = str(result.get("intent", "")).strip().upper()
        if label not in LABELS:
            logger.warning(f"Intent classifier returned unknown label {label!r}; using regex router.")
            return None
        return label


def get_intent_classifier(
    vertex_service: VertexGeminiService = Depends(get_vertex_service),
    settings: Settings = Depends(get_settings),
) -> IntentClassifier:
    """Dependency provider. Built per request so overrides of the Gemini service apply."""
    return IntentClassifier(vertex_service=vertex_service, settings=settings)
