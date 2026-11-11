"""Lightweight language detection and the fixed chat texts in English and Indonesian.

Detection is a word-list heuristic: good enough to pick the language of short chat
questions without a network call. Anything not detected as Indonesian is English.
"""

import re
from typing import Dict, List

from app.models.onboarding import OnboardingProgress, OnboardingTask

INDONESIAN_WORDS = {
    "apa", "apakah", "saya", "aku", "ku", "kami", "kita", "anda", "tugas", "tugasku", "hari", "ini",
    "itu", "yang", "dan", "atau", "untuk", "dengan", "tidak", "nggak", "gak", "belum", "sudah",
    "bagaimana", "gimana", "berapa", "kapan", "dimana", "mana", "siapa", "kenapa", "mengapa", "di",
    "ke", "dari", "ada", "bisa", "harus", "saja", "aja", "tolong", "mohon", "cuti", "kantor", "gaji",
    "karyawan", "perusahaan", "lakukan", "kerjakan", "selesai", "sisa", "tersisa", "lagi", "dong",
    "buat", "buatkan", "minggu", "pertama", "jam", "kerja", "aturan", "kebijakan", "boleh", "punya",
}


def detect_language(text: str) -> str:
    """Return "id" for Indonesian, else "en"."""
    words: List[str] = re.findall(r"[a-zA-Z]+", text.lower())
    if not words:
        return "en"
    hits = sum(1 for w in words if w in INDONESIAN_WORDS)
    return "id" if hits >= 2 or (hits == 1 and len(words) <= 2) else "en"


TEXTS: Dict[str, Dict[str, str]] = {
    "en": {
        "no_profile": "You don't have an onboarding profile yet, so there are no onboarding tasks to show.",
        "no_tasks": "You currently have no onboarding tasks assigned.",
        "all_done": "Congratulations! You have completed all {total} onboarding tasks (100% complete)!",
        "progress": "You have completed {completed} of {total} tasks ({percent}% complete). ",
        "left_one": "You still have 1 task left:",
        "left_many": "You still have {count} tasks left:",
        "task_line": "- Day {day}: {title}",
        "next": "Next up: **{title}** (Day {day}). {description}",
        "general_intro": "Your onboarding program walks you through company policies and your first-week tasks. ",
        "sensitive": "I cannot access confidential HR records or personal employee salary information.",
        "out_of_scope": (
            "I am DayOne AI, your company onboarding copilot. I can only assist with company policies, "
            "onboarding guides, and your onboarding tasks."
        ),
        "fallback": (
            "I couldn't find reliable information about that in the available company documentation. "
            "Please contact People Operations for assistance."
        ),
    },
    "id": {
        "no_profile": "Anda belum memiliki profil onboarding, jadi belum ada tugas onboarding yang bisa ditampilkan.",
        "no_tasks": "Saat ini Anda belum memiliki tugas onboarding.",
        "all_done": "Selamat! Anda telah menyelesaikan semua {total} tugas onboarding (100% selesai)!",
        "progress": "Anda telah menyelesaikan {completed} dari {total} tugas ({percent}% selesai). ",
        "left_one": "Masih ada 1 tugas tersisa:",
        "left_many": "Masih ada {count} tugas tersisa:",
        "task_line": "- Hari {day}: {title}",
        "next": "Berikutnya: **{title}** (Hari {day}). {description}",
        "general_intro": (
            "Program onboarding memandu Anda memahami kebijakan perusahaan dan tugas-tugas minggu pertama. "
        ),
        "sensitive": (
            "Saya tidak dapat mengakses catatan HR yang bersifat rahasia atau informasi gaji karyawan lain."
        ),
        "out_of_scope": (
            "Saya DayOne AI, copilot onboarding perusahaan Anda. Saya hanya dapat membantu terkait kebijakan "
            "perusahaan, panduan onboarding, dan tugas onboarding Anda."
        ),
        "fallback": (
            "Saya tidak menemukan informasi yang dapat diandalkan tentang hal itu di dokumentasi perusahaan. "
            "Silakan hubungi People Operations untuk bantuan."
        ),
    },
}


def text(lang: str, key: str, **kwargs) -> str:
    return TEXTS.get(lang, TEXTS["en"])[key].format(**kwargs)


def onboarding_status_answer(
    lang: str,
    tasks: List[OnboardingTask],
    pending: List[OnboardingTask],
    progress: OnboardingProgress,
) -> str:
    """Progress summary listing every pending task, in the user's language."""
    if not tasks:
        return text(lang, "no_tasks")
    if not pending:
        return text(lang, "all_done", total=progress.total)

    pending = sorted(pending, key=lambda t: (t.day, t.id))
    first = pending[0]
    left = text(lang, "left_one") if len(pending) == 1 else text(lang, "left_many", count=len(pending))
    lines = "\n".join(text(lang, "task_line", day=t.day, title=t.title) for t in pending)
    return (
        text(lang, "progress", completed=progress.completed, total=progress.total, percent=progress.progress_percent)
        + f"{left}\n\n{lines}\n\n"
        + text(lang, "next", title=first.title, day=first.day, description=first.description)
    )
