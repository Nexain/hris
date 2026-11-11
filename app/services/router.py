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
        r"\b(write|compose|make)\s+(me\s+)?(a\s+|an\s+)?(poem|song|story|haiku|limerick|rap)\b",
        r"\b(poem|haiku|limerick)\s+about\b",
        r"\b(buatkan|tuliskan|bikin)\b.*\b(puisi|lagu|cerita|pantun)\b",
        r"\b(weather\s+in|capital\s+of|recipe\s+for)\b",
        r"\b(who\s+won\s+the|play\s+game|crypto|bitcoin)\b",
    ]

    MY_ONBOARDING_PATTERNS = [
        r"\b(my\s+tasks?|my\s+checklist|my\s+onboarding|my\s+progress|what\s+should\s+i\s+do\s+next|next\s+step)\b",
        r"\b(my\s+profile|my\s+details|who\s+am\s+i)\b",
        r"\b(what'?s|what\s+is|anything)\s+(still\s+)?left\s+(for\s+me|to\s+do)\b",
        r"\b(tasks?|items?|things?)\b.{0,30}\b(left|remaining|pending|outstanding|to\s+do)\b",
        r"\b(remaining|pending|outstanding)\s+(tasks?|items?)\b",
        r"\bwhat\s+(do\s+i|should\s+i)\s+(still\s+)?(have\s+to|need\s+to)\s+do\b",
        # Indonesian: "tugas saya", "tugasku", "checklist saya", "progres saya", "apa yang harus saya lakukan"
        r"\b(tugas|daftar\s+tugas|checklist|progres|progress|onboarding)\s*(saya|aku|ku)\b",
        r"\btugasku\b",
        r"\b(apa\s+yang\s+harus\s+(saya|aku)\s+(lakukan|kerjakan)|tugas\s+(yang\s+)?(tersisa|belum))\b",
    ]

    ONBOARDING_ACTION_PATTERNS = [
        r"\b(mark|complete|finish)\s+task\b",
        r"\b(done\s+with|finished)\s+task\b",
    ]

    GENERAL_ONBOARDING_PATTERNS = [
        r"\b(what\s+is|what'?s|explain|tell\s+me\s+about|how\s+does|overview\s+of|how\s+long)\b.*\bonboarding\b",
        r"\bonboarding\b.*\b(process|programme?|experience|timeline|schedule|overview|journey|plan|stages?)\b",
    ]

    SENSITIVE_REASON = "I cannot access confidential HR records or personal employee salary information."
    OUT_OF_SCOPE_REASON = (
        "I am DayOne AI, your company onboarding copilot. I can only assist with company policies, "
        "onboarding guides, and your onboarding tasks."
    )

    @classmethod
    def is_sensitive(cls, message: str) -> bool:
        """Fast first gate: requests for confidential HR data or other people's pay."""
        msg_lower = message.strip().lower()
        return any(re.search(pat, msg_lower) for pat in cls.SENSITIVE_PATTERNS)

    @classmethod
    def route(cls, message: str) -> Tuple[IntentType, bool, str]:
        """
        Route message to intent.
        Returns: (intent, is_allowed, fallback_or_reason)
        """
        msg_lower = message.strip().lower()

        # 1. Security Check: Sensitive or forbidden employee data
        if cls.is_sensitive(message):
            return (IntentType.OUT_OF_SCOPE, False, cls.SENSITIVE_REASON)

        # 2. Scope Gate: Unrelated non-work prompts
        for pat in cls.OUT_OF_SCOPE_PATTERNS:
            if re.search(pat, msg_lower):
                return (IntentType.OUT_OF_SCOPE, False, cls.OUT_OF_SCOPE_REASON)

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
