"""Compatibility shim.

The canonical configuration module is :mod:`app.core.config`. This module
re-exports ``Settings`` and ``get_settings`` so that legacy imports
(``from app.config import ...``) continue to work after the config was moved
under ``app/core``.
"""

from app.core.config import Settings, get_settings

__all__ = ["Settings", "get_settings"]


