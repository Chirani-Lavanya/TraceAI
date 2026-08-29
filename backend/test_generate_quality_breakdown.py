import json
import os
import sqlite3
from types import SimpleNamespace

from fastapi.testclient import TestClient

import main


class FakeChatCompletions:
    def __init__(self, payload):
        self.payload = payload

    def create(self, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(self.payload))
                )
            ]
        )


def test_generate_includes_quality_breakdown(monkeypatch):
    payload = {
        "requirement": {
            "id": "R01",
            "text": "Password must be between 8 and 20 characters.",
            "type": "Validation",
            "priority": "High",
            "conditions": ["Users must supply a password."],
            "constraints": ["Password length must be between 8 and 20 characters."],
            "inputs": ["password"],
            "expected_behavior": "The system accepts valid-length passwords and rejects invalid ones."
        },
        "test_cases": {
            "functional": [
                {
                    "id": "TC01",
                    "requirement_id": "R01",
                    "type": "Functional",
                    "description": "User enters a valid password length.",
                    "expected": "Login succeeds."
                }
            ],
            "negative": [
                {
                    "id": "NTC01",
                    "requirement_id": "R01",
                    "type": "Negative",
                    "description": "User enters a password shorter than 8 characters.",
                    "expected": "Login fails."
                }
            ],
            "bva": [
                {
                    "id": "BVA01",
                    "requirement_id": "R01",
                    "type": "BVA",
                    "scenario": "Short password boundary",
                    "values": "7",
                    "expected": "Password shorter than minimum is rejected."
                },
                {
                    "id": "BVA02",
                    "requirement_id": "R01",
                    "type": "BVA",
                    "scenario": "Minimum valid password",
                    "values": "8",
                    "expected": "Password at minimum length is accepted."
                },
                {
                    "id": "BVA03",
                    "requirement_id": "R01",
                    "type": "BVA",
                    "scenario": "Maximum valid password",
                    "values": "20",
                    "expected": "Password at maximum length is accepted."
                }
            ],
            "ep": [
                {
                    "id": "EP01",
                    "requirement_id": "R01",
                    "type": "EP",
                    "class": "Below minimum",
                    "input": "7 characters",
                    "expected": "Rejected"
                },
                {
                    "id": "EP02",
                    "requirement_id": "R01",
                    "type": "EP",
                    "class": "Valid range",
                    "input": "12 characters",
                    "expected": "Accepted"
                },
                {
                    "id": "EP03",
                    "requirement_id": "R01",
                    "type": "EP",
                    "class": "Above maximum",
                    "input": "21 characters",
                    "expected": "Rejected"
                }
            ],
            "edge_cases": []
        }
    }

    monkeypatch.setattr(main, "client", SimpleNamespace(chat=SimpleNamespace(completions=FakeChatCompletions(payload))))

    client = TestClient(main.app)
    response = client.post("/generate/", json={"text": "Password must be between 8 and 20 characters."})

    assert response.status_code == 200
    body = response.json()
    assert "error" not in body
    assert "quality_breakdown" in body["evaluation"]
    assert set(body["evaluation"]["quality_breakdown"]).issuperset({
        "structure",
        "traceability",
        "duplicates",
        "bva_quality",
        "ep_quality",
        "category_completeness",
    })


def test_duplicate_cleanup_keeps_distinct_boundary_scenarios():
    test_cases = {
        "functional": [],
        "negative": [],
        "bva": [
            {
                "id": "BVA01",
                "requirement_id": "R01",
                "type": "BVA",
                "scenario": "Minimum length accepted",
                "values": "8",
                "expected": "Password is accepted."
            },
            {
                "id": "BVA02",
                "requirement_id": "R01",
                "type": "BVA",
                "scenario": "Minimum length warning shown",
                "values": "8",
                "expected": "System displays a boundary warning."
            },
        ],
        "ep": [],
        "edge_cases": [],
    }

    cleaned = main.remove_duplicate_test_cases(test_cases)

    assert len(cleaned["bva"]) == 2
    assert {case["scenario"] for case in cleaned["bva"]} == {
        "Minimum length accepted",
        "Minimum length warning shown",
    }


def test_bva_quality_handles_currency_and_thousands_separator_for_atm_requirement():
    requirement_text = (
        "The ATM withdrawal amount must be between $500 and $100,000, "
        "and the account must have sufficient balance."
    )
    test_cases = {
        "functional": [],
        "negative": [],
        "bva": [
            {"id": "BVA01", "requirement_id": "R01", "type": "BVA", "scenario": "Below minimum", "values": "$499", "expected": "Rejected."},
            {"id": "BVA02", "requirement_id": "R01", "type": "BVA", "scenario": "At minimum", "values": "$500", "expected": "Accepted."},
            {"id": "BVA03", "requirement_id": "R01", "type": "BVA", "scenario": "Above minimum", "values": "$501", "expected": "Accepted."},
            {"id": "BVA04", "requirement_id": "R01", "type": "BVA", "scenario": "Below maximum", "values": "$99,999", "expected": "Accepted."},
            {"id": "BVA05", "requirement_id": "R01", "type": "BVA", "scenario": "At maximum", "values": "$100,000", "expected": "Accepted."},
            {"id": "BVA06", "requirement_id": "R01", "type": "BVA", "scenario": "Above maximum", "values": "$100,001", "expected": "Rejected."},
        ],
        "ep": [],
        "edge_cases": [],
    }

    result = main.evaluate_bva(test_cases, requirement_text)

    assert main.extract_numeric_ranges(requirement_text) == [(500, 100000)]
    assert main.extract_numbers("$99,999") == [99999]
    assert result["applicable"] is True
    assert result["score"] == 100.0


def test_original_requirement_text_is_preserved_for_atm_bva(monkeypatch):
    original_requirement = (
        "The ATM withdrawal amount must be between $500 and $100,000, "
        "and the account must have sufficient balance."
    )
    payload = {
        "requirement": {
            "id": "R01",
            "text": "The ATM withdrawal amount must be between 0 and 10,000, and the account must have sufficient balance.",
            "type": "Validation",
            "priority": "High",
            "conditions": [],
            "constraints": [],
            "inputs": [],
            "expected_behavior": "The ATM should allow withdrawals only if the amount is between 0 and 10,000 and the account has sufficient balance."
        },
        "test_cases": {
            "functional": [],
            "negative": [],
            "bva": [
                {"id": "BVA01", "requirement_id": "R01", "type": "BVA", "scenario": "Below minimum", "values": "499", "expected": "Rejected."},
                {"id": "BVA02", "requirement_id": "R01", "type": "BVA", "scenario": "At minimum", "values": "500", "expected": "Accepted."},
                {"id": "BVA03", "requirement_id": "R01", "type": "BVA", "scenario": "Above minimum", "values": "501", "expected": "Accepted."},
                {"id": "BVA04", "requirement_id": "R01", "type": "BVA", "scenario": "Below maximum", "values": "99999", "expected": "Accepted."},
                {"id": "BVA05", "requirement_id": "R01", "type": "BVA", "scenario": "At maximum", "values": "100000", "expected": "Accepted."},
                {"id": "BVA06", "requirement_id": "R01", "type": "BVA", "scenario": "Above maximum", "values": "100001", "expected": "Rejected."},
            ],
            "ep": [],
            "edge_cases": [],
        }
    }

    monkeypatch.setattr(main, "client", SimpleNamespace(chat=SimpleNamespace(completions=FakeChatCompletions(payload))))

    client = TestClient(main.app)
    response = client.post("/generate/", json={"text": original_requirement})

    assert response.status_code == 200
    body = response.json()
    assert body["requirement"]["text"] == original_requirement

    bva_values = sorted([
        int(str(case["values"]).replace("$", "").replace(",", ""))
        for case in body["test_cases"]["bva"]
    ])
    assert bva_values == [499, 500, 501, 99999, 100000, 100001]
    assert body["evaluation"]["details"]["bva_applicable"] is True
    assert body["evaluation"]["details"]["bva_quality"] == 100.0


def test_requirement_coverage_tracks_password_conditions():
    requirement = {
        "id": "R01",
        "text": "The password must contain between 8 and 20 characters.",
        "type": "Validation",
        "priority": "High",
        "conditions": [
            "Users must provide a password.",
            "Password length must be between 8 and 20 characters.",
        ],
        "constraints": [],
        "inputs": ["password"],
        "expected_behavior": "Valid passwords are accepted and invalid passwords are rejected.",
    }

    test_cases = {
        "functional": [
            {
                "id": "TC01",
                "requirement_id": "R01",
                "type": "Functional",
                "description": "User enters a valid password.",
                "expected": "Login succeeds."
            }
        ],
        "negative": [
            {
                "id": "NTC01",
                "requirement_id": "R01",
                "type": "Negative",
                "description": "User enters a password shorter than 8 characters.",
                "expected": "Login fails."
            }
        ],
        "bva": [
            {
                "id": "BVA01",
                "requirement_id": "R01",
                "type": "BVA",
                "scenario": "Below minimum boundary",
                "values": "7",
                "expected": "Rejected."
            },
            {
                "id": "BVA02",
                "requirement_id": "R01",
                "type": "BVA",
                "scenario": "Valid minimum boundary",
                "values": "8",
                "expected": "Accepted."
            },
            {
                "id": "BVA03",
                "requirement_id": "R01",
                "type": "BVA",
                "scenario": "Valid maximum boundary",
                "values": "20",
                "expected": "Accepted."
            },
        ],
        "ep": [
            {
                "id": "EP01",
                "requirement_id": "R01",
                "type": "EP",
                "class": "Below minimum",
                "input": "7 characters",
                "expected": "Rejected"
            },
            {
                "id": "EP02",
                "requirement_id": "R01",
                "type": "EP",
                "class": "Valid range",
                "input": "12 characters",
                "expected": "Accepted"
            },
            {
                "id": "EP03",
                "requirement_id": "R01",
                "type": "EP",
                "class": "Above maximum",
                "input": "21 characters",
                "expected": "Rejected"
            },
        ],
        "edge_cases": [],
    }

    result = main.evaluate_requirement_coverage(
        requirement,
        requirement["text"],
        test_cases,
    )

    assert result["total_conditions"] >= 2
    assert result["covered_conditions"] >= 2
    assert result["percentage"] > 0
    assert all(item["id"].startswith("RC") for item in result["conditions"])
    assert {item["id"] for item in result["conditions"]}.issubset({
        f"RC{i:02d}" for i in range(1, 10)
    })
    assert any(item["status"] == "covered" for item in result["conditions"])


def test_history_lists_latest_runs_ordered_newest_first():
    db_path = os.path.join(os.path.dirname(__file__), "..", "traceai.db")
    conn = sqlite3.connect(db_path)
    try:
        latest_id = conn.execute(
            "SELECT id FROM generation_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert latest_id is not None
        latest_id = latest_id[0]
    finally:
        conn.close()

    client = TestClient(main.app)
    response = client.get("/history/")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] >= 1
    assert body["runs"][0]["id"] >= latest_id
    assert body["runs"][0]["id"] == max(run["id"] for run in body["runs"])


def test_history_returns_full_run_details_for_selected_generation():
    db_path = os.path.join(os.path.dirname(__file__), "..", "traceai.db")
    conn = sqlite3.connect(db_path)
    try:
        latest = conn.execute(
            "SELECT id, requirement_id FROM generation_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert latest is not None
        run_id, requirement_id = latest
        requirement_text = conn.execute(
            "SELECT text FROM requirements WHERE id = ?",
            (requirement_id,),
        ).fetchone()
        assert requirement_text is not None
        requirement_text = requirement_text[0]
    finally:
        conn.close()

    client = TestClient(main.app)
    response = client.get(f"/history/{run_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == run_id
    assert body["requirement"]["text"] == requirement_text
    assert "test_cases" in body
    assert "evaluation" in body
    assert "traceability" in body
    assert body["status"] in {"completed", "failed"}
