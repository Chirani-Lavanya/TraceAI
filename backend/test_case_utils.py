"""Canonical test-case collection and traceability helpers."""

from __future__ import annotations

from backend.duplicate_detection import CATEGORIES


def all_test_cases(test_cases: dict) -> list:
    """Flatten category lists while preserving canonical category order."""
    cases = []
    for category in CATEGORIES:
        items = test_cases.get(category, [])
        if isinstance(items, list):
            cases.extend(items)
    return cases


def ensure_category_lists(test_cases: dict) -> dict:
    """Ensure every canonical test-case category contains a list."""
    for category in CATEGORIES:
        if not isinstance(test_cases.get(category), list):
            test_cases[category] = []
    return test_cases


def apply_requirement_id(test_cases: dict, requirement_id: str) -> dict:
    """Apply the source requirement ID to every test-case object."""
    for category in CATEGORIES:
        for case in test_cases.get(category, []):
            if isinstance(case, dict):
                case["requirement_id"] = requirement_id
    return test_cases


def build_traceability_records(test_cases: dict, requirement_id: str) -> list:
    """Build traceability rows from canonical test-case fields."""
    traceability = []
    for category in CATEGORIES:
        for case in test_cases.get(category, []):
            if not isinstance(case, dict):
                continue
            description = (
                case.get("description")
                or case.get("scenario")
                or case.get("class")
                or ""
            )
            traceability.append({
                "requirement_id": case.get("requirement_id", requirement_id),
                "test_case": case.get("id"),
                "type": case.get("type"),
                "description": description,
                "status": "Covered",
            })
    return traceability


_ensure_category_lists = ensure_category_lists
_apply_requirement_id = apply_requirement_id
_build_traceability_records = build_traceability_records
