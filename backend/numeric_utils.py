"""Numeric parsing and source-anchored numeric derivation helpers."""

from __future__ import annotations

import re
from typing import Optional


NUMBER_WORD_VALUES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def extract_numeric_ranges(text: str) -> list[tuple[int, int]]:
    """Extract simple numeric ranges from requirement text."""
    if not text:
        return []

    normalized = str(text).lower()
    normalized = normalized.replace("$", " ")
    normalized = normalized.replace("€", " ")
    normalized = normalized.replace("£", " ")

    patterns = [
        r"between\s+(\d[\d,]*)\s+and\s+(\d[\d,]*)",
        r"from\s+(\d[\d,]*)\s+to\s+(\d[\d,]*)",
        r"(\d[\d,]*)\s+to\s+(\d[\d,]*)",
        r"(\d[\d,]*)\s*(?:-|–|—)\s*(\d[\d,]*)",
        r"minimum\s+(?:of\s+)?(\d[\d,]*)\s+.*?maximum\s+(?:of\s+)?(\d[\d,]*)",
        r"at\s+least\s+(\d[\d,]*)\s+.*?at\s+most\s+(\d[\d,]*)",
    ]

    ranges = []
    for pattern in patterns:
        for match in re.finditer(pattern, normalized, flags=re.IGNORECASE | re.DOTALL):
            low = int(match.group(1).replace(",", ""))
            high = int(match.group(2).replace(",", ""))
            if low < high:
                ranges.append((low, high))

    unique = []
    for item in ranges:
        if item not in unique:
            unique.append(item)
    return unique


def extract_numbers(text: str) -> list[int]:
    """Extract integers from formatted values such as ``$99,999``."""
    if not text:
        return []

    cleaned = str(text)
    cleaned = cleaned.replace("$", " ")
    cleaned = cleaned.replace("€", " ")
    cleaned = cleaned.replace("£", " ")
    cleaned = cleaned.replace(",", "")
    return [int(value) for value in re.findall(r"[-+]?\d+", cleaned)]


def _number_word_values(text: str) -> set[int]:
    """Extract simple numeric words used as requirement counts."""
    normalized_text = str(text or "").lower()
    return {
        value
        for word, value in NUMBER_WORD_VALUES.items()
        if re.search(rf"\b{re.escape(word)}\b", normalized_text)
    }


def _numeric_requirement_bounds(requirement_text: str) -> dict:
    """Extract one-sided numeric bounds and exact numeric requirements."""
    text = str(requirement_text or "").lower()
    result = {"lower": None, "upper": None, "exact": set()}

    lower_patterns = [
        r"\bat\s+least\s+(\d[\d,]*)",
        r"\bminimum(?:\s+of)?\s+(\d[\d,]*)",
        r"\bno\s+less\s+than\s+(\d[\d,]*)",
    ]
    upper_patterns = [
        r"\bat\s+most\s+(\d[\d,]*)",
        r"\bmaximum(?:\s+of)?\s+(\d[\d,]*)",
        r"\bno\s+more\s+than\s+(\d[\d,]*)",
    ]

    for pattern in lower_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            result["lower"] = int(match.group(1).replace(",", ""))
            break

    for pattern in upper_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            result["upper"] = int(match.group(1).replace(",", ""))
            break

    for match in re.finditer(r"\bexactly\s+(\d[\d,]*)", text, flags=re.IGNORECASE):
        result["exact"].add(int(match.group(1).replace(",", "")))

    return result


def _is_supported_generated_number(value: int, requirement_text: str, category: str) -> bool:
    """Check whether a generated numeric value is requirement-supported."""
    if value in set(extract_numbers(requirement_text)):
        return True

    if value in _number_word_values(requirement_text):
        return True

    for low, high in extract_numeric_ranges(requirement_text):
        if low <= value <= high:
            return True
        if category in {"negative", "bva", "ep"} and (value < low or value > high):
            return True

    bounds = _numeric_requirement_bounds(requirement_text)
    lower = bounds["lower"]
    upper = bounds["upper"]

    if lower is not None:
        if value >= lower:
            return True
        if category in {"negative", "bva", "ep"} and value < lower:
            return True

    if upper is not None:
        if value <= upper:
            return True
        if category in {"negative", "bva", "ep"} and value > upper:
            return True

    return value in bounds["exact"]


def _derived_numeric_values(requirement_text: str, category: str) -> set[int]:
    """Return obvious requirement-derived numeric representatives."""
    values = set(extract_numbers(requirement_text))
    values.update(_number_word_values(requirement_text))

    for low, high in extract_numeric_ranges(requirement_text):
        if high - low <= 1000:
            values.update(range(low, high + 1))
        if category in {"negative", "bva", "ep"}:
            values.update({low - 1, high + 1})

    bounds = _numeric_requirement_bounds(requirement_text)
    if bounds["lower"] is not None:
        values.add(bounds["lower"])
        if category in {"negative", "bva", "ep"}:
            values.add(bounds["lower"] - 1)
    if bounds["upper"] is not None:
        values.add(bounds["upper"])
        if category in {"negative", "bva", "ep"}:
            values.add(bounds["upper"] + 1)

    values.update(bounds["exact"])
    return values


def extract_bva_boundary(requirement_text: str) -> Optional[dict]:
    """Return a conservative BVA definition for numeric boundaries."""
    ranges = extract_numeric_ranges(requirement_text)
    if ranges:
        low, high = ranges[0]
        return {"kind": "range", "low": low, "high": high}

    bounds = _numeric_requirement_bounds(requirement_text)
    if bounds["lower"] is not None:
        return {"kind": "lower", "low": bounds["lower"]}
    if bounds["upper"] is not None:
        return {"kind": "upper", "high": bounds["upper"]}
    return None


_extract_bva_boundary = extract_bva_boundary
