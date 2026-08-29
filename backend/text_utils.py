"""Shared text normalization and model-response cleanup helpers."""

from __future__ import annotations

import re
from typing import Any


def clean_json_response(content: str) -> str:
    """Remove accidental Markdown fences before JSON parsing."""
    content = (content or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    return content.strip()


def normalize(value: Any) -> str:
    """Normalize text for matching and duplicate detection."""
    if value is None:
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^\w\s<>=.-]", "", text)
