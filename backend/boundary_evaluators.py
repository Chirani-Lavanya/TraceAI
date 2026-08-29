"""Boundary-value and equivalence-partition evaluators."""

from __future__ import annotations

import re

from backend.numeric_utils import _extract_bva_boundary, extract_numbers, extract_numeric_ranges
from backend.text_utils import normalize


def evaluate_bva(test_cases: dict, requirement_text: str) -> dict:
    """Validate BVA for explicit ranges and one-sided numeric bounds."""
    boundary = _extract_bva_boundary(requirement_text)
    bva_cases = test_cases.get("bva", [])

    if boundary is None:
        return {
            "applicable": False,
            "score": 100,
            "message": "No simple numeric boundary detected; BVA is not mandatory.",
        }

    if not isinstance(bva_cases, list) or not bva_cases:
        return {
            "applicable": True,
            "score": 0,
            "message": "A numeric boundary was detected but no BVA cases were generated.",
        }

    if boundary["kind"] == "range":
        low = boundary["low"]
        high = boundary["high"]
        expected_values = {low - 1, low, low + 1, high - 1, high, high + 1}
    elif boundary["kind"] == "lower":
        low = boundary["low"]
        expected_values = {low - 1, low, low + 1}
    else:
        high = boundary["high"]
        expected_values = {high - 1, high, high + 1}

    generated_values = set()
    for case in bva_cases:
        if isinstance(case, dict):
            generated_values.update(extract_numbers(case.get("values", "")))

    matched = expected_values.intersection(generated_values)
    score = round((len(matched) / len(expected_values)) * 100, 2)
    if score == 100:
        message = "BVA contains cases covering all required boundary positions."
    else:
        missing = sorted(expected_values - generated_values)
        message = "Some expected BVA boundary positions are missing: " + ", ".join(map(str, missing))

    return {
        "applicable": True,
        "score": score,
        "message": message,
    }


def evaluate_ep(test_cases: dict, requirement_text: str) -> dict:
    """Rule-based Equivalence Partitioning evaluation."""
    ranges = extract_numeric_ranges(requirement_text)
    ep_cases = test_cases.get("ep", [])

    if not isinstance(ep_cases, list):
        ep_cases = []

    if ranges:
        low, high = ranges[0]
        lower_present = False
        valid_present = False
        upper_present = False

        for case in ep_cases:
            if not isinstance(case, dict):
                continue

            input_text = normalize(case.get("input"))
            lower_match = False
            valid_match = False
            upper_match = False

            numeric_values = extract_numbers(case.get("input", ""))
            for value in numeric_values:
                if value < low:
                    lower_match = True
                elif low <= value <= high:
                    valid_match = True
                elif value > high:
                    upper_match = True

            lower_present = lower_present or lower_match
            valid_present = valid_present or valid_match
            upper_present = upper_present or upper_match

        partitions = sum([lower_present, valid_present, upper_present])
        score = round((partitions / 3) * 100, 2)

        missing = []
        if not lower_present:
            missing.append("below minimum")
        if not valid_present:
            missing.append("valid range")
        if not upper_present:
            missing.append("above maximum")

        if score == 100:
            message = (
                "EP correctly identifies the three expected partitions: "
                "below the minimum, within the valid range, and above the maximum."
            )
        else:
            message = f"EP covers {partitions}/3 expected partitions. Missing: {', '.join(missing)}."

        return {"applicable": True, "score": score, "message": message}

    text_lower = requirement_text.lower()
    validation_keywords = [
        "valid", "invalid", "only", "must", "required",
        "accept", "reject", "credentials", "email", "password",
        "input", "allowed", "not allowed", "cannot",
        "validation", "constraint",
    ]

    if not any(keyword in text_lower for keyword in validation_keywords):
        return {
            "applicable": False,
            "score": 100,
            "message": "No meaningful equivalence classes detected; EP is not mandatory.",
        }

    if not ep_cases:
        return {
            "applicable": True,
            "score": 0,
            "message": "A meaningful input domain was detected but no EP cases were generated.",
        }

    valid_present = False
    invalid_present = False

    for case in ep_cases:
        if not isinstance(case, dict):
            continue
        partition_text = " ".join([
            normalize(case.get("class")),
            normalize(case.get("input")),
            normalize(case.get("description")),
            normalize(case.get("expected")),
        ])
        if re.search(r"\bvalid\b|\baccepted\b|\bsuccess", partition_text):
            valid_present = True
        if re.search(r"\binvalid\b|\brejected\b|\berror\b|\bfail", partition_text):
            invalid_present = True

    partitions = sum([valid_present, invalid_present])
    score = round((partitions / 2) * 100, 2)
    missing = []
    if not valid_present:
        missing.append("valid input partition")
    if not invalid_present:
        missing.append("invalid input partition")

    if score == 100:
        message = "EP correctly identifies the valid and invalid input partitions for the requirement."
    else:
        message = f"EP covers {partitions}/2 expected validation partitions. Missing: {', '.join(missing)}."

    return {"applicable": True, "score": score, "message": message}
