"""Duplicate detection and canonical test-case ID utilities."""

from __future__ import annotations

from backend.text_utils import normalize


CATEGORIES = [
    "functional",
    "negative",
    "bva",
    "ep",
    "edge_cases",
]


def _case_fingerprint(case: dict) -> tuple:
    """Return a conservative fingerprint of test intent and outcome."""
    if not isinstance(case, dict):
        return ()

    intent = (
        normalize(case.get("description"))
        or normalize(case.get("scenario"))
        or normalize(case.get("class"))
        or normalize(case.get("input"))
        or normalize(case.get("values"))
    )
    expected = normalize(case.get("expected"))

    if not intent and not expected:
        return ()
    if intent and expected:
        return (intent, expected)
    return (intent or expected,)


def duplicate_indexes(test_cases: dict) -> list:
    """Detect duplicate test intents within each category."""
    duplicates = []
    for category in CATEGORIES:
        items = test_cases.get(category, [])
        if not isinstance(items, list):
            continue

        seen = {}
        for index, case in enumerate(items):
            fingerprint = _case_fingerprint(case)
            if not any(fingerprint):
                continue
            if fingerprint in seen:
                duplicates.append({
                    "category": category,
                    "index": index,
                    "duplicate_of": seen[fingerprint],
                })
            else:
                seen[fingerprint] = {"category": category, "index": index}
    return duplicates


def renumber_test_cases(test_cases: dict) -> dict:
    """Rebuild sequential IDs within each test-design category."""
    prefixes = {
        "functional": "TC",
        "negative": "NTC",
        "bva": "BVA",
        "ep": "EP",
        "edge_cases": "EC",
    }
    for category, prefix in prefixes.items():
        items = test_cases.get(category, [])
        if not isinstance(items, list):
            continue
        for index, case in enumerate(items, start=1):
            if isinstance(case, dict):
                case["id"] = f"{prefix}{index:02d}"
    return test_cases


def remove_duplicate_test_cases(test_cases: dict) -> dict:
    """Remove duplicate intents, including edge cases duplicated elsewhere."""
    ordered_categories = ["functional", "negative", "bva", "ep", "edge_cases"]
    seen_by_category = {}

    for category in ordered_categories:
        items = test_cases.get(category, [])
        if not isinstance(items, list):
            test_cases[category] = []
            continue

        unique = []
        seen = set()
        for case in items:
            if not isinstance(case, dict):
                continue
            fingerprint = _case_fingerprint(case)
            if not any(fingerprint) or fingerprint in seen:
                continue
            seen.add(fingerprint)
            unique.append(case)
        test_cases[category] = unique
        seen_by_category[category] = seen

    prior_fingerprints = set()
    for category in ["functional", "negative", "bva", "ep"]:
        prior_fingerprints.update(seen_by_category.get(category, set()))

    edge_unique = []
    edge_seen = set()
    for case in test_cases["edge_cases"]:
        fingerprint = _case_fingerprint(case)
        if not any(fingerprint) or fingerprint in prior_fingerprints or fingerprint in edge_seen:
            continue
        edge_seen.add(fingerprint)
        edge_unique.append(case)
    test_cases["edge_cases"] = edge_unique
    return test_cases
