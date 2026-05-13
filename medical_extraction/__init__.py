"""Root package shim so the src layout works without editable install."""

from __future__ import annotations

from pathlib import Path

_CURRENT_DIR = Path(__file__).resolve().parent
_SRC_PACKAGE = _CURRENT_DIR.parent / "src" / "medical_extraction"

__path__ = [str(_CURRENT_DIR)]
if _SRC_PACKAGE.exists():
    __path__.append(str(_SRC_PACKAGE))
