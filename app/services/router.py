import re
from typing import Tuple

from app.models.chat import IntentType


class ScopeRouter:
    """Classifies user messages and enforces scope and security gates before RAG/Gemini."""

    SENSITIVE_PATTERNS = [
        # A specific person's pay / earnings (personal compensation is private).
        r"\bhow\s+much\s+does\b.*\b(make|earn|earns|get\s+paid|take\s+home)\b",
        r"\b(salary|compensation|wage|wages|earnings?|payslip|pay\s?slip)\s+(of|for)\b",
        r"\b(colleague|coworker|co-worker|teammate|employee|manager|someone|he|she|they)\b"
        r".{0,40}\b(salary|earn|earns|make|makes|paid|compensation|wage|wages)\b",
        # Private HR records / actions.
        r"\b(private|confidential)\s+hr\s+record",
        r"\bperformance\s+review\s+of\b",
        r"\bterminate\s+\w+\b",
        r"\bfire\s+\w+\b",
    ]

    OUT_OF_SCOPE_PATTERNS = [
        r"\b(tell\s+me\s+a\s+joke|write\s+a\s+poem|sing\s+a\s+song)\b",
        r"\b(weather\s+in|capital\s+of|recipe\s+for)\b",
        r"\b(who\s+won\s+the|play\s+game|crypto|bitcoin)\b",
    ]

    MY_ONBOARDING_PATTERNS = [
        r"\b(my\s+tasks?|my\s+checklist|my\s+onboarding|my\s+progress|what\s+should\s+i\s+do\s+next|next\s+step)\b",
        r"\b(my\s+profile|my\s+details|who\s+am\s+i)\b",
    ]

    ONBOARDING_ACTION_PATTERNS = [
        r"\b(mark|complete|finish)\s+task\b",
        r"\b(done\s+with|finished)\s+task\b",
    ]

    GENERAL_ONBOARDING_PATTERNS = [
        r"\b(what\s+is|what'?s|explain|tell\s+me\s+about|how\s+does|overview\s+of|how\s+long)\b.*\bonboarding\b",
        r"\bonboarding\b.*\b(process|programme?|experience|timeline|schedule|overview|journey|plan|stages?)\b",
    ]

    @classmethod
    def route(cls, message: str) -> Tuple[IntentType, bool, str]:
        """
        Route message to intent.
        Returns: (intent, is_allowed, fallback_or_reason)
        """
        msg_lower = message.strip().lower()

        # 1. Security Check: Sensitive or forbidden employee data
        for pat in cls.SENSITIVE_PATTERNS:
            if re.search(pat, msg_lower):
                return (
                    IntentType.OUT_OF_SCOPE,
                    False,
                    "I cannot access confidential HR records or personal employee salary information.",
                )

        # 2. Scope Gate: Unrelated non-work prompts
        for pat in cls.OUT_OF_SCOPE_PATTERNS:
            if re.search(pat, msg_lower):
                return (
                    IntentType.OUT_OF_SCOPE,
                    False,
                    "I am DayOne AI, your company onboarding copilot. I can only assist with company policies, onboarding guides, and your onboarding tasks.",
                )

        # 3. Onboarding Actions
        for pat in cls.ONBOARDING_ACTION_PATTERNS:
            if re.search(pat, msg_lower):
                return (IntentType.ONBOARDING_ACTION, True, "")

        # 4. Personal Onboarding Data
        for pat in cls.MY_ONBOARDING_PATTERNS:
            if re.search(pat, msg_lower):
                return (IntentType.MY_ONBOARDING, True, "")

        # 5. General onboarding questions (not personal, not company policy)
        for pat in cls.GENERAL_ONBOARDING_PATTERNS:
            if re.search(pat, msg_lower):
                return (IntentType.GENERAL_ONBOARDING, True, "")

        # 6. Default to Company Knowledge (RAG)
        return (IntentType.COMPANY_KNOWLEDGE, True, "")
