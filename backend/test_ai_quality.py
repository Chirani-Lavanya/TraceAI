import json
import os
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main


BASE_REQUIREMENT = {
    "id": "R01",
    "type": "Functional",
    "priority": "Medium",
    "conditions": [],
    "constraints": [],
    "inputs": [],
    "expected_behavior": "The system performs the requested behavior.",
}


def case(case_id, case_type, description, expected, **fields):
    return {
        "id": case_id,
        "requirement_id": "R01",
        "type": case_type,
        "description": description,
        "expected": expected,
        **fields,
    }


def make_payload(requirement_text, test_cases, **requirement_fields):
    requirement = {
        **BASE_REQUIREMENT,
        "text": "Model rewritten text that must not escape.",
        **requirement_fields,
    }
    return {"requirement": requirement, "test_cases": test_cases}


def fixture_cases():
    return [
        pytest.param(
            "The profile page displays the user's name after login.",
            make_payload(
                "The profile page displays the user's name after login.",
                {
                    "functional": [case("TC01", "Functional", "Open profile after login.", "The profile page displays the user's name.")],
                    "negative": [], "bva": [], "ep": [], "edge_cases": [],
                },
                expected_behavior="The profile page displays the user's name.",
            ),
            id="simple-functional",
        ),
        pytest.param(
            "The email address must be valid before submission.",
            make_payload(
                "The email address must be valid before submission.",
                {
                    "functional": [case("TC01", "Functional", "Submit a valid email address.", "The form is submitted.")],
                    "negative": [case("NTC01", "Negative", "Submit an invalid email address.", "The form is rejected.")],
                    "bva": [],
                    "ep": [
                        case("EP01", "EP", "Invalid email partition", "The form is rejected.", **{"class": "Invalid email", "input": "invalid email"}),
                        case("EP02", "EP", "Valid email partition", "The form is submitted.", **{"class": "Valid email", "input": "user@example.com"}),
                    ],
                    "edge_cases": [],
                },
                type="Validation",
                expected_behavior="The form accepts valid email addresses and rejects invalid email addresses.",
            ),
            id="input-validation",
        ),
        pytest.param(
            "The password length must be between 8 and 20 characters.",
            make_payload(
                "The password length must be between 8 and 20 characters.",
                {
                    "functional": [case("TC01", "Functional", "Enter a password within the allowed length.", "The password is accepted.")],
                    "negative": [case("NTC01", "Negative", "Enter a password below the minimum length.", "The password is rejected.")],
                    "bva": [
                        {"id": f"BVA{i:02d}", "requirement_id": "R01", "type": "BVA", "scenario": "Password boundary", "values": str(value), "expected": "The password is accepted." if 8 <= value <= 20 else "The password is rejected."}
                        for i, value in enumerate([7, 8, 9, 19, 20, 21], 1)
                    ],
                    "ep": [
                        case("EP01", "EP", "Lower partition", "The password is rejected.", **{"class": "Below minimum", "input": "7 characters"}),
                        case("EP02", "EP", "Valid partition", "The password is accepted.", **{"class": "Valid range", "input": "12 characters"}),
                        case("EP03", "EP", "Upper partition", "The password is rejected.", **{"class": "Above maximum", "input": "21 characters"}),
                    ],
                    "edge_cases": [],
                },
                type="Validation",
                expected_behavior="Valid password lengths are accepted and invalid lengths are rejected.",
            ),
            id="boundary-value",
        ),
        pytest.param(
            "Users can log in only with valid credentials.",
            make_payload(
                "Users can log in only with valid credentials.",
                {
                    "functional": [case("TC01", "Functional", "Log in with valid credentials.", "The user is authenticated.")],
                    "negative": [case("NTC01", "Negative", "Log in with invalid credentials.", "Authentication is rejected.")],
                    "bva": [],
                    "ep": [
                        case("EP01", "EP", "Valid credentials", "The user is authenticated.", **{"class": "Valid credentials", "input": "valid username and password"}),
                        case("EP02", "EP", "Invalid credentials", "Authentication is rejected.", **{"class": "Invalid credentials", "input": "invalid username and password"}),
                    ],
                    "edge_cases": [],
                },
                type="Security",
                expected_behavior="Valid credentials authenticate the user; invalid credentials are rejected.",
            ),
            id="authentication",
        ),
        pytest.param(
            "The service rejects requests with an invalid token.",
            make_payload(
                "The service rejects requests with an invalid token.",
                {
                    "functional": [case("TC01", "Functional", "Send a request with a valid token.", "The service processes the request.")],
                    "negative": [case("NTC01", "Negative", "Send a request with an invalid token.", "The service rejects the request.")],
                    "bva": [], "ep": [], "edge_cases": [],
                },
                type="Security",
                expected_behavior="The service processes valid-token requests and rejects invalid-token requests.",
            ),
            id="negative-error-handling",
        ),
        pytest.param(
            "An order is accepted when payment is authorized and inventory is available.",
            make_payload(
                "An order is accepted when payment is authorized and inventory is available.",
                {
                    "functional": [case("TC01", "Functional", "Submit an order with authorized payment and available inventory.", "The order is accepted.")],
                    "negative": [case("NTC01", "Negative", "Submit an order when payment is declined.", "The order is rejected.")],
                    "bva": [], "ep": [], "edge_cases": [],
                },
                type="Business Rule",
                conditions=["Payment is authorized.", "Inventory is available."],
                expected_behavior="The order is accepted only when payment is authorized and inventory is available.",
            ),
            id="multiple-conditions",
        ),
        pytest.param(
            "Improve the account experience.",
            make_payload(
                "Improve the account experience.",
                {
                    "functional": [case("TC01", "Functional", "Record the supplied requirement for review.", "The requirement is recorded for review.")],
                    "negative": [], "bva": [], "ep": [], "edge_cases": [],
                },
                type="Other",
                conditions=[],
                constraints=[],
                inputs=[],
                expected_behavior="The wording is recorded for clarification.",
            ),
            id="ambiguous-conservative",
        ),
    ]


class FakeChatCompletions:
    def __init__(self, payload):
        self.payload = payload

    def create(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.payload)))]
        )


@pytest.mark.parametrize("requirement_text,payload", fixture_cases())
def test_representative_ai_quality_fixture(monkeypatch, requirement_text, payload):
    monkeypatch.setattr(
        main,
        "client",
        SimpleNamespace(chat=SimpleNamespace(completions=FakeChatCompletions(payload))),
    )

    response = TestClient(main.app).post("/generate/", json={"text": requirement_text})
    assert response.status_code == 200
    body = response.json()
    assert "error" not in body
    assert body["requirement"]["id"] == "R01"
    assert body["requirement"]["text"] == requirement_text

    all_cases = main.all_test_cases(body["test_cases"])
    assert all_cases
    assert all(case["requirement_id"] == "R01" for case in all_cases)
    assert all(main._expected_result_is_specific(case) for case in all_cases)
    assert len({main._case_fingerprint(case) for case in all_cases}) == len(all_cases)
    assert len(body["traceability"]) == len(all_cases)
    assert body["evaluation"]["requirement_traceability"] == 100.0
    assert body["evaluation"]["quality_breakdown"]["duplicates"] == 100


def test_bva_ignores_boundary_numbers_in_description():
    result = main.evaluate_bva(
        {"bva": [{"values": "7", "scenario": "Covers 7, 8, 9, 19, 20, and 21."}]},
        "The password length must be between 8 and 20 characters.",
    )
    assert result["score"] == round((1 / 6) * 100, 2)


def test_user_story_normalization_preserves_original_story_and_extracts_supported_fields():
    story = "As a customer, I want to transfer money between 1000 and 500000 when I have sufficient balance, so that I can securely transfer funds."

    normalized = main.normalize_user_story(story)

    assert normalized["source_type"] == "user_story"
    assert normalized["text"] == story
    assert normalized["role"] == "customer"
    assert "transfer money" in normalized["action"].lower()
    assert "securely transfer funds" in normalized["goal"].lower()
    assert normalized["conditions"] == [
        "User has sufficient balance.",
        "Transfer amount is between 1000 and 500000.",
    ]
    assert normalized["constraints"] == [
        "Transfer amount must be between 1000 and 500000."
    ]
    assert normalized["expected_behavior"]


def test_user_story_structure_analysis_extracts_supported_fields_and_preserves_story():
    story = "As a customer, I want to transfer money between 1000 and 500000 when I have sufficient balance, so that I can securely transfer funds."

    analysis = main.analyze_user_story(story)

    assert analysis["source_type"] == "user_story"
    assert analysis["story_id"] == "US01"
    assert analysis["actor"] == "customer"
    assert "transfer money" in analysis["action"].lower()
    assert "securely transfer funds" in analysis["goal"].lower()
    assert analysis["conditions"] == [
        "User has sufficient balance.",
        "Transfer amount is between 1000 and 500000.",
    ]
    assert analysis["constraints"] == [
        "Transfer amount must be between 1000 and 500000."
    ]
    assert analysis["original_source"] == story
    assert analysis["acceptance_criteria"] == []
    assert "otp" not in " ".join(analysis["constraints"]).lower()
    assert "2fa" not in " ".join(analysis["constraints"]).lower()


def test_user_story_prompt_isolates_negative_conditions():
    story = "As a customer, I want to transfer money between 1000 and 500000 when I have sufficient balance, so that I can securely transfer funds."

    prompt = " ".join(main._build_generation_prompt(story).split())

    assert "Test each invalid condition in isolation" in prompt
    assert "hold all other explicit conditions valid" in prompt
    assert "invalid transfer amount with sufficient balance" in prompt


def test_user_story_analysis_handles_free_form_story_without_acceptance_criteria():
    story = "A traveler needs to book a flight for a family trip before the departure date without creating duplicate reservations."

    analysis = main.analyze_user_story(story)

    assert analysis["source_type"] == "user_story"
    assert analysis["actor"]
    assert analysis["action"]
    assert analysis["goal"]
    assert analysis["conditions"]
    assert analysis["acceptance_criteria"] == []
    assert analysis["original_source"] == story


def test_normalize_user_story_maps_story_to_existing_requirement_shape():
    story = "As a customer, I want to transfer money between 1000 and 500000 when I have sufficient balance, so that I can securely transfer funds."

    normalized = main.normalize_user_story(story)

    assert normalized["source_type"] == "user_story"
    assert normalized["text"] == story
    assert normalized["id"] == "US01"
    assert normalized["type"] == "Business Rule"
    assert normalized["conditions"]
    assert normalized["constraints"]
    assert normalized["inputs"]
    assert normalized["expected_behavior"]
    assert normalized["story_analysis"]["original_source"] == story


def test_user_story_generate_endpoint_accepts_user_story_input():
    story = "As a customer, I want to transfer money between 1000 and 500000 when I have sufficient balance, so that I can securely transfer funds."

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        main,
        "client",
        SimpleNamespace(chat=SimpleNamespace(completions=FakeChatCompletions({
            "requirement": {
                "id": "US01",
                "text": story,
                "type": "Business Rule",
                "priority": "Medium",
                "conditions": ["Transfer amount is between 1000 and 500000.", "The account has sufficient balance."],
                "constraints": ["Transfer amount must be between 1000 and 500000."],
                "inputs": ["Transfer amount", "Account balance"],
                "expected_behavior": "The transfer is accepted when the amount is within the allowed range and the account has sufficient balance.",
            },
            "test_cases": {
                "functional": [case("TC01", "Functional", "Transfer 250000 with sufficient balance.", "The transfer is successful.")],
                "negative": [case("NTC01", "Negative", "Transfer amount 999 with sufficient balance.", "The transfer is rejected.")],
                "bva": [
                    {"id": "BVA01", "requirement_id": "US01", "type": "BVA", "scenario": "Low boundary", "values": "999", "expected": "Rejected"},
                    {"id": "BVA02", "requirement_id": "US01", "type": "BVA", "scenario": "Minimum allowed", "values": "1000", "expected": "Accepted"},
                    {"id": "BVA03", "requirement_id": "US01", "type": "BVA", "scenario": "Maximum allowed", "values": "500000", "expected": "Accepted"},
                    {"id": "BVA04", "requirement_id": "US01", "type": "BVA", "scenario": "High boundary", "values": "500001", "expected": "Rejected"},
                ],
                "ep": [
                    {"id": "EP01", "requirement_id": "US01", "type": "EP", "class": "Below minimum", "input": "999", "expected": "Rejected"},
                    {"id": "EP02", "requirement_id": "US01", "type": "EP", "class": "Valid range", "input": "250000", "expected": "Accepted"},
                    {"id": "EP03", "requirement_id": "US01", "type": "EP", "class": "Above maximum", "input": "500001", "expected": "Rejected"},
                ],
                "edge_cases": [],
            },
        }))),
    )

    response = TestClient(main.app).post("/generate/", json={"text": story, "source_type": "user_story"})
    assert response.status_code == 200
    body = response.json()
    assert "error" not in body
    assert body["requirement"]["text"] == story
    assert body["requirement"]["source_type"] == "user_story"
    assert body["requirement"]["id"] == "US01"
    assert body["requirement"]["conditions"] == [
        "User has sufficient balance.",
        "Transfer amount is between 1000 and 500000.",
    ]
    assert body["requirement"]["constraints"] == [
        "Transfer amount must be between 1000 and 500000."
    ]
    ntc01 = next(test_case for test_case in body["test_cases"]["negative"] if test_case["id"] == "NTC01")
    assert "999" in ntc01["description"]
    assert "sufficient balance" in ntc01["description"].lower()
    assert "insufficient balance" not in ntc01["description"].lower()
    assert body["traceability"]
    assert [case["values"] for case in body["test_cases"]["bva"]] == [
        "999", "1000", "1001", "499999", "500000", "500001"
    ]
    assert len(body["test_cases"]["ep"]) == 3
    monkeypatch.undo()


def test_login_user_story_analysis_preserves_supported_behavior_only():
    story = "As a registered user, I want to log in with my email and password so that I can access my account."

    analysis = main.analyze_user_story(story)

    assert analysis["story_id"] == "US01"
    assert analysis["actor"] == "registered user"
    assert "log in with my email and password" in analysis["action"].lower()
    assert analysis["goal"] == "I can access my account"
    assert len(analysis["conditions"]) == 1
    assert "log in" in analysis["conditions"][0].lower()
    assert "email" in analysis["conditions"][0].lower()
    assert "password" in analysis["conditions"][0].lower()
    assert analysis["constraints"] == []
    assert "requested capability is supported" not in str(analysis).lower()
    assert "otp" not in str(analysis).lower()
    assert "2fa" not in str(analysis).lower()


def test_login_user_story_generation_has_positive_coverage_and_no_bva(monkeypatch):
    story = "As a registered user, I want to log in with my email and password so that I can access my account."
    payload = {
        "requirement": {
            "id": "er",
            "conditions": ["The requested capability is supported by the user story."],
            "constraints": [],
            "inputs": ["Email address", "Password"],
        },
        "test_cases": {
            "functional": [],
            "negative": [case("NTC01", "Negative", "Log in with an invalid password.", "Login is rejected.")],
            "bva": [],
            "ep": [
                {"id": "EP01", "type": "EP", "class": "Invalid credentials", "input": "Invalid email or password", "expected": "Login is rejected."},
                {"id": "EP02", "type": "EP", "class": "Valid credentials", "input": "Valid email and password", "expected": "Login succeeds."},
            ],
            "edge_cases": [],
        },
    }
    monkeypatch.setattr(
        main,
        "client",
        SimpleNamespace(chat=SimpleNamespace(completions=FakeChatCompletions(payload))),
    )

    response = TestClient(main.app).post(
        "/generate/",
        json={"text": story, "source_type": "user_story"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["requirement"]["id"] == "US01"
    assert body["requirement"]["text"] == story
    assert body["requirement"]["conditions"]
    assert all("requested capability is supported" not in condition.lower() for condition in body["requirement"]["conditions"])
    assert len(body["test_cases"]["functional"]) >= 1
    functional_case = body["test_cases"]["functional"][0]
    assert "email" in functional_case["description"].lower()
    assert "password" in functional_case["description"].lower()
    assert "access" in functional_case["expected"].lower()
    assert body["test_cases"]["negative"]
    assert body["test_cases"]["ep"]
    assert body["test_cases"]["bva"] == []
    assert body["evaluation"]["test_design_coverage"] == 100.0
    assert body["evaluation"]["requirement_coverage"]["percentage"] == 100.0


def test_password_reset_user_story_extracts_inputs_and_covers_invalid_ep(monkeypatch):
    story = "As a user, I want to reset my password using my registered email address, a verification code, and a new password with at least 8 characters so that I can securely regain access to my account."
    payload = make_payload(
        story,
        {
            "functional": [case("TC01", "Functional", "Reset the password with valid inputs.", "The password is reset successfully.")],
            "negative": [],
            "bva": [],
            "ep": [{
                "id": "EP01", "type": "EP", "class": "Valid password length",
                "input": "8 characters", "expected": "Password reset succeeds.",
            }],
            "edge_cases": [],
        },
    )
    monkeypatch.setattr(
        main,
        "client",
        SimpleNamespace(chat=SimpleNamespace(completions=FakeChatCompletions(payload))),
    )

    response = TestClient(main.app).post(
        "/generate/",
        json={"text": story, "source_type": "user_story"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["requirement"]["text"] == story
    assert body["requirement"]["inputs"] == [
        "Email address",
        "Verification code",
        "New password",
    ]
    assert len(body["test_cases"]["functional"]) >= 1
    assert len(body["test_cases"]["ep"]) == 2
    assert body["evaluation"]["details"]["ep_quality"] == 100.0


def test_password_reset_input_extraction_preserves_explicit_password_field():
    story = "As a user, I want to reset my password using my registered email address, a verification code, and a new password with at least 8 characters so that I can securely regain access to my account."

    analysis = main.analyze_user_story(story)

    assert analysis["inputs"] == [
        "Email address",
        "Verification code",
        "New password",
    ]
    assert "Password" not in analysis["inputs"]


def test_password_reset_user_story_bva_restores_minimum_boundary(monkeypatch):
    story = "As a user, I want to reset my password using my registered email address, a verification code, and a new password with at least 8 characters so that I can securely regain access to my account."
    payload = make_payload(
        story,
        {
            "functional": [case("TC01", "Functional", "Reset password with valid inputs.", "Password reset succeeds.")],
            "negative": [],
            "bva": [
                {"id": "BVA01", "type": "BVA", "values": "7", "scenario": "Below minimum", "expected": "Rejected."},
                {"id": "BVA02", "type": "BVA", "values": "9", "scenario": "Above minimum", "expected": "Accepted."},
            ],
            "ep": [{"id": "EP01", "type": "EP", "class": "Valid", "input": "8 characters", "expected": "Accepted."}],
            "edge_cases": [],
        },
    )
    monkeypatch.setattr(
        main,
        "client",
        SimpleNamespace(chat=SimpleNamespace(completions=FakeChatCompletions(payload))),
    )

    response = TestClient(main.app).post(
        "/generate/",
        json={"text": story, "source_type": "user_story"},
    )

    assert response.status_code == 200
    body = response.json()
    assert [case["values"] for case in body["test_cases"]["bva"]] == ["7", "8", "9"]
    assert body["evaluation"]["details"]["bva_quality"] == 100.0


def test_ep_normalization_completes_all_numeric_partitions():
    requirement_text = "The withdrawal amount must be between 500 and 100000."
    test_cases = {
        "functional": [],
        "negative": [],
        "bva": [],
        "ep": [{
            "id": "EP01",
            "type": "EP",
            "class": "Valid range",
            "input": "500 to 100000",
            "expected": "Accepted.",
        }],
        "edge_cases": [],
    }

    normalized = main._normalize_lower_bound_ep_cases(
        test_cases,
        requirement_text,
        "R01",
    )

    inputs = [case["input"] for case in normalized["ep"]]
    assert any("499" in value for value in inputs)
    assert any("500" in value and "100000" in value for value in inputs)
    assert any("100001" in value for value in inputs)
    assert main.evaluate_ep(normalized, requirement_text)["score"] == 100.0


def test_ep_normalization_fixes_one_sided_upper_bound_partitions():
    requirement_text = "The username must contain no more than 20 characters."
    test_cases = {
        "functional": [],
        "negative": [],
        "bva": [],
        "ep": [{
            "id": "EP01",
            "type": "EP",
            # Mislabeled: 21 is above the maximum, not below any minimum.
            "class": "Below valid range",
            "input": "21 characters",
            "expected": "Accepted.",
        }],
        "edge_cases": [],
    }

    normalized = main._normalize_lower_bound_ep_cases(
        test_cases,
        requirement_text,
        "R01",
    )

    ep_cases = normalized["ep"]

    assert not any(
        "below" in main.normalize(str(c.get("class", "")))
        for c in ep_cases
    )
    assert any(
        main.extract_numbers(c.get("input", "")) and
        all(value <= 20 for value in main.extract_numbers(c.get("input", "")))
        for c in ep_cases
    )
    assert any(
        main.extract_numbers(c.get("input", "")) and
        all(value > 20 for value in main.extract_numbers(c.get("input", "")))
        and main.normalize(str(c.get("expected", ""))) in {"rejected", "invalid", "error"}
        for c in ep_cases
    )

    above_max_values = [
        value
        for c in ep_cases
        for value in main.extract_numbers(c.get("input", ""))
        if value > 20
    ]
    assert len(above_max_values) == 1
    assert main.evaluate_ep(normalized, requirement_text)["score"] == 100.0


def test_upper_bound_bva_normalization_restores_canonical_values_and_labels():
    requirement_text = "The username must contain no more than 20 characters."
    test_cases = {
        "functional": [], "negative": [], "bva": [
            {"id": "BVA01", "values": "19", "scenario": "Below maximum", "expected": "Accepted"},
            {"id": "BVA02", "values": "20", "scenario": "Maximum", "expected": "Accepted"},
            {"id": "BVA03", "values": "21", "scenario": "Below minimum", "expected": "Accepted"},
        ],
        "ep": [], "edge_cases": [],
    }

    normalized = main._normalize_numeric_bva_cases(test_cases, requirement_text, "R01")

    assert [case["values"] for case in normalized["bva"]] == ["19", "20", "21"]
    assert [case["expected"] for case in normalized["bva"]] == [
        "Accepted", "Accepted", "Rejected"
    ]
    assert main.evaluate_bva(normalized, requirement_text)["score"] == 100.0


def test_full_range_bva_max_plus_one_description_says_above_maximum():
    requirement_text = "The ATM withdrawal amount must be between 500 and 100000."
    test_cases = {
        "functional": [], "negative": [], "ep": [], "edge_cases": [],
        "bva": [
            {"id": "BVA01", "values": "499", "scenario": "Minimum boundary check.", "expected": "Rejected"},
            {"id": "BVA02", "values": "500", "scenario": "Minimum valid check.", "expected": "Accepted"},
            {"id": "BVA03", "values": "501", "scenario": "Just past the minimum threshold.", "expected": "Accepted"},
            {"id": "BVA04", "values": "99999", "scenario": "Just before the maximum threshold.", "expected": "Accepted"},
            {"id": "BVA05", "values": "100000", "scenario": "Maximum valid check.", "expected": "Accepted"},
            {"id": "BVA06", "values": "100001", "scenario": "Withdraw amount just above the minimum limit.", "expected": "Rejected"},
        ],
    }

    normalized = main._normalize_numeric_bva_cases(test_cases, requirement_text, "R01")

    assert [case["values"] for case in normalized["bva"]] == [
        "499", "500", "501", "99999", "100000", "100001",
    ]
    bva06 = normalized["bva"][5]
    assert bva06["values"] == "100001"
    assert bva06["expected"] == "Rejected"
    assert bva06["scenario"] == "Withdraw amount just above the maximum limit."
    # Other boundary descriptions must remain untouched.
    assert normalized["bva"][2]["scenario"] == "Just past the minimum threshold."



    cases = [
        (
            "The password must contain at least 8 characters.",
            [
                {"class": "Invalid", "input": "7 characters", "expected": "Rejected"},
                {"class": "Below minimum", "input": "Password shorter than 8", "expected": "Rejected"},
                {"class": "Valid", "input": "8 characters", "expected": "Accepted"},
            ],
            ["below", "valid"],
        ),
        (
            "The username must contain no more than 20 characters.",
            [
                {"class": "Invalid", "input": "21 characters", "expected": "Rejected"},
                {"class": "Above maximum", "input": "Username length above 20", "expected": "Rejected"},
                {"class": "Valid", "input": "20 characters", "expected": "Accepted"},
            ],
            ["valid", "above"],
        ),
        (
            "The amount must be between 500 and 100000.",
            [
                {"class": "Invalid", "input": "499", "expected": "Rejected"},
                {"class": "Below minimum", "input": "Amount below 500", "expected": "Rejected"},
                {"class": "Valid", "input": "500 to 100000", "expected": "Accepted"},
                {"class": "Invalid", "input": "100001", "expected": "Rejected"},
            ],
            ["below", "valid", "above"],
        ),
    ]

    for requirement_text, ep_cases, expected_classes in cases:
        normalized = main._normalize_lower_bound_ep_cases(
            {"functional": [], "negative": [], "bva": [], "ep": ep_cases, "edge_cases": []},
            requirement_text,
            "R01",
        )
        assert [main.normalize(case["class"]) for case in normalized["ep"]] == [
            main.normalize({"below": "Below minimum", "valid": "Valid input", "above": "Above maximum"}[name])
            if "between" not in requirement_text
            else main.normalize({"below": "Below minimum", "valid": "Valid range", "above": "Above maximum"}[name])
            for name in expected_classes
        ]
        assert len(normalized["ep"]) == len(expected_classes)
        assert main.evaluate_ep(normalized, requirement_text)["score"] == 100.0


def test_lower_bound_ep_normalization_repairs_contradictory_ai_case():
    requirement_text = "The password must contain at least 8 characters."
    test_cases = {
        "functional": [],
        "negative": [],
        "bva": [],
        "ep": [
            {
                "id": "EP01",
                "requirement_id": "R01",
                "type": "EP",
                "class": "Valid input",
                "input": "Password with less than 8 characters",
                "expected": "Accepted",
            },
            {
                "id": "EP02",
                "requirement_id": "R01",
                "type": "EP",
                "class": "Below minimum",
                "input": "7",
                "expected": "Rejected",
            },
            {
                "id": "EP03",
                "requirement_id": "R01",
                "type": "EP",
                "class": "Valid input",
                "input": "8",
                "expected": "Accepted",
            },
        ],
        "edge_cases": [],
    }

    normalized = main._normalize_lower_bound_ep_cases(
        test_cases,
        requirement_text,
        "R01",
    )

    assert normalized["ep"] == [
        {
            "id": "EP01",
            "requirement_id": "R01",
            "type": "EP",
            "class": "Below minimum",
            "input": "7",
            "expected": "Rejected",
        },
        {
            "id": "EP03",
            "requirement_id": "R01",
            "type": "EP",
            "class": "Valid input",
            "input": "8",
            "expected": "Accepted",
        },
    ]
    assert main.evaluate_ep(normalized, requirement_text)["score"] == 100.0


def test_conflicting_expected_behavior_is_surfaced_and_reduces_quality():
    requirement_text = "The amount must be between 500 and 100000."
    test_cases = {
        "functional": [{
            "id": "TC01",
            "requirement_id": "R01",
            "type": "Functional",
            "description": "Use an amount within the allowed range.",
            "expected": "The amount is accepted.",
        }],
        "negative": [],
        "bva": [],
        "ep": [],
        "edge_cases": [],
    }
    requirement = {
        "id": "R01",
        "text": requirement_text,
        "expected_behavior": "The amount must be between 0 and 10000.",
    }

    result = main._evaluate_generated_result(
        requirement,
        requirement_text,
        test_cases,
        "R01",
    )

    assert result["requirement_conflicts"]
    assert result["requirement_conflicts"][0]["type"] == (
        "requirement_expected_behavior_conflict"
    )
    assert result["quality_result"]["score"] < 100.0


def test_numeric_natural_language_bounds_and_bva_prose_are_normalized():
    lower = "The age must be 18 or older."
    upper = "The username must not exceed 20 characters."
    lower_cases = {"functional": [], "negative": [], "bva": [], "ep": [], "edge_cases": []}
    upper_cases = {"functional": [], "negative": [], "bva": [], "ep": [], "edge_cases": []}
    lower_cases["bva"] = [{"values": "8", "scenario": "less than 8", "expected": "Accepted"}]
    upper_cases["bva"] = [{"values": "20", "scenario": "greater than 20", "expected": "Accepted"}]

    lower_cases = main._normalize_numeric_bva_cases(lower_cases, lower, "R01")
    upper_cases = main._normalize_numeric_bva_cases(upper_cases, upper, "R01")

    assert [case["values"] for case in lower_cases["bva"]] == ["17", "18", "19"]
    assert [case["expected"] for case in lower_cases["bva"]] == ["Rejected", "Accepted", "Accepted"]
    assert [case["values"] for case in upper_cases["bva"]] == ["19", "20", "21"]
    assert [case["expected"] for case in upper_cases["bva"]] == ["Accepted", "Accepted", "Rejected"]
    assert main.analyze_user_story("As a user, I want to update my username.")["inputs"] == ["Username"]


def test_gap_analysis_surfaces_requirement_expected_behavior_conflict():
    requirement_text = "The amount must be between 500 and 100000."
    result = main.analyze_test_gaps(
        requirement_text,
        {"functional": [], "negative": [], "bva": [], "ep": [], "edge_cases": []},
        {"id": "R01", "expected_behavior": "The amount must be between 0 and 10000."},
    )

    assert result["evidence"]["requirement_expected_behavior_conflicts"]
    assert any(
        gap["category"] == "requirement_expected_behavior_conflict"
        for gap in result["gaps"]
    )


def test_confirm_password_input_is_preserved_as_an_explicit_field():
    story = "As a user, I want to update my password with a new password and confirm password."

    analysis = main.analyze_user_story(story)

    assert analysis["inputs"] == ["New password", "Confirm password"]


def test_successful_user_story_restores_functional_case_after_cleanup(monkeypatch):
    story = "As a customer, I want to download my invoices so that I can review my purchases."
    payload = make_payload(
        story,
        {
            "functional": [case(
                "TC01", "Functional", "Assume invoice downloads are emailed automatically.",
                "The purchase history is updated.",
            )],
            "negative": [], "bva": [], "ep": [], "edge_cases": [],
        },
    )
    monkeypatch.setattr(
        main,
        "client",
        SimpleNamespace(chat=SimpleNamespace(completions=FakeChatCompletions(payload))),
    )

    response = TestClient(main.app).post(
        "/generate/",
        json={"text": story, "source_type": "user_story"},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["test_cases"]["functional"]) >= 1
    functional_case = body["test_cases"]["functional"][0]
    assert "download" in functional_case["description"].lower()
    assert "invoice" in functional_case["description"].lower()
    assert "review" in functional_case["expected"].lower()
    assert "assume" not in functional_case["description"].lower()
    assert body["evaluation"]["requirement_traceability"] == 100.0


def test_profile_update_user_story_extracts_only_explicit_inputs_and_rules(monkeypatch):
    story = "As a registered user, I want to update my profile name and phone number so that my account information remains up to date."

    analysis = main.analyze_user_story(story)

    assert analysis["original_source"] == story
    assert analysis["inputs"] == ["Profile name", "Phone number"]
    assert "Booking date" not in analysis["inputs"]
    assert analysis["constraints"] == []
    assert analysis["conditions"] == [
        "The registered user can update my profile name and phone number.",
        "Account information remains up to date after the update.",
    ]

    payload = {
        "requirement": {
            "id": "US01",
            "conditions": ["The profile name cannot contain special characters."],
            "constraints": ["Phone number cannot contain letters."],
            "inputs": ["Booking date", "Account details"],
        },
        "test_cases": {
            "functional": [{
                "id": "TC01", "type": "Functional",
                "description": "Update profile name and phone number.",
                "expected": "Profile information is updated.",
            }],
            "negative": [{
                "id": "NTC01", "type": "Negative",
                "description": "Use special characters in the profile name.",
                "expected": "Update is rejected.",
            }],
            "bva": [],
            "ep": [{
                "id": "EP01", "type": "EP", "class": "Valid Name",
                "input": "Valid name format", "expected": "Update succeeds.",
            }],
            "edge_cases": [],
        },
    }
    monkeypatch.setattr(
        main,
        "client",
        SimpleNamespace(chat=SimpleNamespace(completions=FakeChatCompletions(payload))),
    )

    response = TestClient(main.app).post(
        "/generate/",
        json={"text": story, "source_type": "user_story"},
    )

    assert response.status_code == 200
    body = response.json()
    requirement = body["requirement"]
    assert requirement["text"] == story
    assert requirement["inputs"] == ["Profile name", "Phone number"]
    assert requirement["constraints"] == []
    assert requirement["conditions"] == [
        "The registered user can update my profile name and phone number.",
        "Account information remains up to date after the update.",
    ]
    assert len(body["test_cases"]["functional"]) >= 1
    functional_case = body["test_cases"]["functional"][0]
    assert "profile name" in functional_case["description"].lower()
    assert "phone number" in functional_case["description"].lower()
    assert "updated successfully" in functional_case["expected"].lower()
    assert body["test_cases"]["bva"] == []
    assert body["test_cases"]["negative"] == []
    assert len(body["test_cases"]["ep"]) == 1
    assert "profile name" in body["test_cases"]["ep"][0]["class"].lower()
    assert "phone number" in body["test_cases"]["ep"][0]["class"].lower()


def test_profile_update_grounding_ignores_generic_valid_wording():
    story = "As a registered user, I want to update my profile name and phone number so that my account information remains up to date."
    test_cases = {
        "functional": [{
            "id": "TC01",
            "type": "Functional",
            "description": "Update profile name and phone number with valid values.",
            "expected": "Profile name and phone number are updated successfully.",
        }],
        "negative": [],
        "bva": [],
        "ep": [{
            "id": "EP01",
            "type": "EP",
            "class": "Valid Name",
            "input": "Valid name format",
            "expected": "Profile update succeeds.",
        }],
        "edge_cases": [],
    }

    result = main.validate_requirement_grounding(story, test_cases)

    assert result["grounded"] is True
    assert result["findings"] == []
    assert test_cases["functional"]


def test_password_minimum_requirement_does_not_invent_maximum(monkeypatch):
    requirement_text = "The password must contain at least 8 characters and at least one special character."
    payload = make_payload(
        requirement_text,
        {
            "functional": [case("TC01", "Functional", "Use a password with at least 8 characters and a special character.", "Password is accepted.")],
            "negative": [
                case("NTC01", "Negative", "Use a password with fewer than 8 characters.", "Password is rejected."),
                case("NTC02", "Negative", "Use a password with no special character.", "Password is rejected."),
            ],
            "bva": [
                {"id": "BVA01", "type": "BVA", "values": "7", "scenario": "Below minimum", "expected": "Rejected."},
                {"id": "BVA02", "type": "BVA", "values": "8", "scenario": "Minimum", "expected": "Accepted."},
                {"id": "BVA03", "type": "BVA", "values": "9", "scenario": "Above minimum", "expected": "Accepted."},
            ],
            "ep": [
                {"id": "EP01", "type": "EP", "class": "Below minimum", "input": "7 characters", "expected": "Rejected."},
                {"id": "EP02", "type": "EP", "class": "Valid length", "input": "8 characters", "expected": "Accepted."},
                {"id": "EP03", "type": "EP", "class": "Above maximum", "input": "21 characters", "expected": "Rejected."},
            ],
            "edge_cases": [],
        },
        conditions=["The user can The password must contain at least 8 characters and at least one special character."],
    )
    monkeypatch.setattr(
        main,
        "client",
        SimpleNamespace(chat=SimpleNamespace(completions=FakeChatCompletions(payload))),
    )

    response = TestClient(main.app).post("/generate/", json={"text": requirement_text})

    assert response.status_code == 200
    body = response.json()
    assert body["requirement"]["text"] == requirement_text
    assert body["requirement"]["conditions"] == [
        "The password must contain at least 8 characters.",
        "The password must contain at least one special character.",
    ]
    assert [case["values"] for case in body["test_cases"]["bva"]] == ["7", "8", "9"]
    assert all("maximum" not in case.get("class", "").lower() for case in body["test_cases"]["ep"])
    assert body["evaluation"]["details"]["bva_applicable"] is True
    assert body["evaluation"]["requirement_coverage"]["percentage"] == 100.0


def test_one_sided_password_bva_normalization_restores_missing_minimum():
    requirement_text = "The password must contain at least 8 characters and at least one special character."
    test_cases = {
        "functional": [],
        "negative": [],
        "bva": [
            {"id": "BVA01", "type": "BVA", "values": "7", "scenario": "Below minimum"},
            {"id": "BVA02", "type": "BVA", "values": "9", "scenario": "Above minimum"},
        ],
        "ep": [],
        "edge_cases": [],
    }

    normalized = main._normalize_numeric_bva_cases(test_cases, requirement_text, "R01")

    assert [case["values"] for case in normalized["bva"]] == ["7", "8", "9"]
    assert main.evaluate_bva(normalized, requirement_text)["score"] == 100.0


def test_requirement_grounding_allows_derived_bva_and_ep_values():
    requirement = "Password length must be between 8 and 20 characters."
    test_cases = {
        "functional": [],
        "negative": [],
        "bva": [
            {"id": "BVA01", "scenario": "Below minimum", "values": "7", "expected": "Rejected."},
            {"id": "BVA02", "scenario": "Minimum", "values": "8", "expected": "Accepted."},
            {"id": "BVA03", "scenario": "Maximum", "values": "20", "expected": "Accepted."},
            {"id": "BVA04", "scenario": "Above maximum", "values": "21", "expected": "Rejected."},
        ],
        "ep": [
            {"id": "EP01", "class": "Below minimum", "input": "7 characters", "expected": "Rejected."},
            {"id": "EP02", "class": "Valid range", "input": "12 characters", "expected": "Accepted."},
            {"id": "EP03", "class": "Above maximum", "input": "21 characters", "expected": "Rejected."},
        ],
        "edge_cases": [],
    }

    result = main.validate_requirement_grounding(requirement, test_cases)

    assert result["grounded"] is True
    assert result["score"] == 100.0
    assert result["findings"] == []


def test_atm_requirement_preserves_withdrawal_terminology(monkeypatch):
    requirement_text = "The ATM withdrawal amount must be between 500 and 100,000, and the account must have sufficient balance."
    payload = make_payload(
        requirement_text,
        {"functional": [], "negative": [], "bva": [], "ep": [], "edge_cases": []},
        conditions=[
            "Transfer amount is between 500 and 100000.",
            "The account has sufficient balance.",
        ],
        constraints=["Transfer amount must be between 500 and 100000."],
        expected_behavior="The transfer is accepted within the allowed range.",
    )
    monkeypatch.setattr(
        main,
        "client",
        SimpleNamespace(chat=SimpleNamespace(completions=FakeChatCompletions(payload))),
    )

    response = TestClient(main.app).post("/generate/", json={"text": requirement_text})

    assert response.status_code == 200
    requirement = response.json()["requirement"]
    assert requirement["conditions"] == [
        "Withdrawal amount is between 500 and 100000.",
        "The account has sufficient balance.",
    ]
    assert requirement["constraints"] == [
        "Withdrawal amount must be between 500 and 100000."
    ]
    assert "withdrawal" in requirement["expected_behavior"].lower()
    assert "transfer" not in requirement["expected_behavior"].lower()


def test_profile_ep_preserves_both_explicit_inputs_without_format_rules(monkeypatch):
    story = "As a registered user, I want to update my profile name and phone number so that my account information remains up to date."
    payload = make_payload(
        story,
        {
            "functional": [case("TC01", "Functional", "Update profile name and phone number with valid values.", "Profile name and phone number are updated successfully.")],
            "negative": [],
            "bva": [],
            "ep": [{
                "id": "EP01", "type": "EP", "class": "Valid Name",
                "input": "Valid name format", "expected": "Update succeeds.",
            }, {
                "id": "EP02", "type": "EP", "class": "Invalid Name",
                "input": "Error message indicating invalid name format", "expected": "Update fails.",
            }, {
                "id": "EP03", "type": "EP", "class": "Invalid Phone",
                "input": "Error message indicating invalid phone number format", "expected": "Update fails.",
            }],
            "edge_cases": [],
        },
    )
    monkeypatch.setattr(
        main,
        "client",
        SimpleNamespace(chat=SimpleNamespace(completions=FakeChatCompletions(payload))),
    )

    response = TestClient(main.app).post(
        "/generate/",
        json={"text": story, "source_type": "user_story"},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["test_cases"]["ep"]) == 1
    assert body["evaluation"]["requirement_grounding_score"] == 100.0
    assert body["evaluation"]["ai_quality_score"] == 100.0
    ep_case = body["test_cases"]["ep"][0]
    assert "profile name" in ep_case["class"].lower()
    assert "phone number" in ep_case["class"].lower()
    assert "format" not in ep_case["class"].lower()
    assert "profile name" in ep_case["input"].lower()
    assert "phone number" in ep_case["input"].lower()


def test_profile_update_drops_unsupported_negative_validation_cases(monkeypatch):
    """Regression for the exact reported symptom: NTC01/NTC02 invent name and
    phone-number validation rules that the source User Story never states,
    so the entire negative category must be removed while the functional and
    combined EP cases are preserved and BVA stays inapplicable."""
    story = "As a registered user, I want to update my profile name and phone number so that my account information remains up to date."
    payload = make_payload(
        story,
        {
            "functional": [case(
                "TC01", "Functional",
                "Update profile name and phone number with valid values.",
                "Profile name and phone number are updated successfully, and the account information reflects the changes.",
            )],
            "negative": [
                case(
                    "NTC01", "Negative",
                    "User attempts to update profile name with invalid characters.",
                    "System rejects the update due to invalid characters in the name.",
                ),
                case(
                    "NTC02", "Negative",
                    "User attempts to update phone number with invalid format.",
                    "System rejects the update due to invalid phone number format.",
                ),
            ],
            "bva": [],
            "ep": [{
                "id": "EP01", "type": "EP", "class": "Profile name and phone number",
                "input": "Profile name and phone number",
                "expected": "Profile name and phone number are accepted for update.",
            }],
            "edge_cases": [],
        },
    )
    monkeypatch.setattr(
        main,
        "client",
        SimpleNamespace(chat=SimpleNamespace(completions=FakeChatCompletions(payload))),
    )

    response = TestClient(main.app).post(
        "/generate/",
        json={"text": story, "source_type": "user_story"},
    )

    assert response.status_code == 200
    body = response.json()

    assert len(body["test_cases"]["functional"]) == 1
    assert "profile name" in body["test_cases"]["functional"][0]["description"].lower()
    assert "phone number" in body["test_cases"]["functional"][0]["description"].lower()

    assert body["test_cases"]["negative"] == []

    assert len(body["test_cases"]["ep"]) == 1
    ep_case = body["test_cases"]["ep"][0]
    assert "profile name" in ep_case["class"].lower()
    assert "phone number" in ep_case["class"].lower()

    assert body["test_cases"]["bva"] == []
    assert body["evaluation"]["details"]["bva_applicable"] is False

    assert body["evaluation"]["requirement_grounding_score"] == 100.0
    assert body["evaluation"]["requirement_grounding"]["findings"] == []
    assert body["evaluation"]["requirement_coverage"]["percentage"] == 100.0
    assert body["evaluation"]["ai_quality_score"] == 100.0


def test_requirement_grounding_reports_invented_content_by_category():
    requirement = "The profile page displays the user's name after login."
    test_cases = {
        "functional": [{
            "id": "TC01",
            "description": "Assume the page redirects to a dashboard after login.",
            "expected": "The system persists the name in the database.",
        }],
        "negative": [],
        "bva": [],
        "ep": [{
            "id": "EP01",
            "class": "Valid email format",
            "input": "user@example.com",
            "expected": "The profile is displayed.",
        }],
        "edge_cases": [],
    }

    result = main.validate_requirement_grounding(requirement, test_cases)
    categories = {finding["category"] for finding in result["findings"]}

    assert result["grounded"] is False
    assert categories == {
        "unsupported_assumptions",
        "invented_formats",
        "invented_behaviors",
    }
    assert result["score"] < 100.0


def test_generate_exposes_grounding_score_separately_from_ai_quality(monkeypatch):
    requirement_text = "The profile page displays the user's name after login."
    payload = make_payload(
        requirement_text,
        {
            "functional": [case(
                "TC01",
                "Functional",
                "The profile page displays the user's name.",
                "The profile is displayed.",
            )],
            "negative": [], "bva": [], "ep": [], "edge_cases": [],
        },
    )
    monkeypatch.setattr(main, "client", SimpleNamespace(chat=SimpleNamespace(completions=FakeChatCompletions(payload))))

    response = TestClient(main.app).post("/generate/", json={"text": requirement_text})
    evaluation = response.json()["evaluation"]

    assert "ai_quality_score" in evaluation
    assert "requirement_grounding_score" in evaluation
    assert evaluation["requirement_grounding_score"] == evaluation["requirement_grounding"]["score"]


def test_agent_rejects_unsupported_intent():
    response = TestClient(main.app).post("/agent/", json={"intent": "run_tests"})

    assert response.status_code == 200
    assert response.json() == {
        "status": "unsupported",
        "error": "Unsupported agent intent. Use complete, analyze, generate, refine, explain, or history.",
    }


def test_agent_explains_selected_history_without_model_call(monkeypatch):
    history = {
        "id": 12,
        "evaluation": {
            "ai_quality_score": 91.0,
            "requirement_grounding_score": 100.0,
            "requirement_grounding": {"grounded": True, "findings": []},
            "quality_breakdown": {"structure": 100},
            "requirement_coverage": {"percentage": 100.0},
        },
    }
    monkeypatch.setattr(main, "get_generation_history", lambda run_id: history)

    response = TestClient(main.app).post(
        "/agent/",
        json={"intent": "explain", "generation_run_id": 12},
    )

    assert response.status_code == 200
    assert response.json()["action"] == "explain"
    assert response.json()["result"]["requirement_grounding_score"] == 100.0


def test_requirement_intelligence_is_source_anchored():
    requirement_text = "Password length must be between 8 and 20 characters."

    result = main.build_requirement_intelligence(requirement_text)

    assert result["source_text"] == requirement_text
    assert result["source_of_truth"] == "user_requirement"
    assert result["numeric_ranges"] == [{"low": 8, "high": 20}]
    assert "bva" in result["applicable_techniques"]
    assert "ep" in result["applicable_techniques"]
    assert result["conditions"]


def test_gap_analysis_reports_missing_condition_categories_and_bva_positions():
    requirement_text = "Password length must be between 8 and 20 characters."
    test_cases = {
        "functional": [],
        "negative": [],
        "bva": [{
            "id": "BVA01",
            "requirement_id": "R01",
            "type": "BVA",
            "scenario": "Minimum valid length",
            "values": "8",
            "expected": "Accepted.",
        }],
        "ep": [],
        "edge_cases": [],
    }

    result = main.analyze_test_gaps(requirement_text, test_cases)
    gap_categories = {gap["category"] for gap in result["gaps"]}

    assert result["has_gaps"] is True
    assert "functional" in gap_categories
    assert "negative" in gap_categories
    assert "bva" in gap_categories
    assert "ep" in gap_categories
    assert result["scores"]["requirement_grounding"] == 100.0


def test_agent_analyze_returns_intelligence_and_gap_evidence():
    response = TestClient(main.app).post(
        "/agent/",
        json={
            "intent": "analyze_requirement",
            "text": "The email address must be valid before submission.",
            "test_cases": {"functional": [], "negative": [], "bva": [], "ep": [], "edge_cases": []},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["action"] == "analyze"
    assert "requirement_intelligence" in body["result"]
    assert "gaps" in body["result"]
    assert body["result"]["evidence"]["grounding"]["grounded"] is True


class SequenceChatCompletions:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response))]
        )


def targeted_case_payload(description="Display the profile page."):
    return json.dumps({
        "test_cases": {
            "functional": [{
                "id": "TC99",
                "requirement_id": "R01",
                "type": "Functional",
                "description": description,
                "expected": "The profile page is displayed.",
            }],
            "negative": [],
            "bva": [],
            "ep": [],
            "edge_cases": [],
        }
    })


def test_agent_targets_only_categories_with_identified_gaps(monkeypatch):
    completions = SequenceChatCompletions([targeted_case_payload()])
    monkeypatch.setattr(
        main,
        "client",
        SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    result = main.run_agent_workflow(
        "The profile page displays the user's name after login.",
        {"functional": [], "negative": [], "bva": [], "ep": [], "edge_cases": []},
    )

    assert result["status"] == "completed"
    assert result["result"]["model_calls"] == 1
    assert result["result"]["generated_additions"]["functional"]
    assert not result["result"]["generated_additions"]["negative"]
    assert "IDENTIFIED GAPS TO ADDRESS" in completions.calls[0]["messages"][1]["content"]


def test_agent_rejects_grounding_failure_and_returns_evidence(monkeypatch):
    bad_case = targeted_case_payload("Assume the page redirects to a dashboard.")
    completions = SequenceChatCompletions([bad_case, bad_case])
    monkeypatch.setattr(
        main,
        "client",
        SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    result = main.run_agent_workflow(
        "The profile page displays the user's name after login.",
        {"functional": [], "negative": [], "bva": [], "ep": [], "edge_cases": []},
    )

    assert result["status"] == "needs_review"
    assert result["result"]["refinement_count"] == 1
    assert result["result"]["model_calls"] == 2
    assert any(
        finding["category"] == "unsupported_assumptions"
        for finding in result["result"]["grounding_findings"]
    )


def test_agent_refines_once_and_completes(monkeypatch):
    bad_case = targeted_case_payload("Assume the page redirects to a dashboard.")
    completions = SequenceChatCompletions([bad_case, targeted_case_payload()])
    monkeypatch.setattr(
        main,
        "client",
        SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    result = main.run_agent_workflow(
        "The profile page displays the user's name after login.",
        {"functional": [], "negative": [], "bva": [], "ep": [], "edge_cases": []},
    )

    assert result["status"] == "completed"
    assert result["result"]["refinement_count"] == 1
    assert result["result"]["model_calls"] == 2
    assert result["result"]["final_status"] == "completed"
    refinement_prompt = completions.calls[1]["messages"][1]["content"]
    assert "ORIGINAL REQUIREMENT:" in refinement_prompt
    assert "FAILING CASE DETAILS:" in refinement_prompt
    assert "GROUNDING AND COVERAGE FINDINGS:" in refinement_prompt
    assert "ALLOWED SOURCE FACTS:" in refinement_prompt
    assert "IDENTIFIED GAPS TO ADDRESS" not in refinement_prompt


def test_agent_returns_error_for_malformed_model_output(monkeypatch):
    completions = SequenceChatCompletions(["not valid json"])
    monkeypatch.setattr(
        main,
        "client",
        SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    result = main.run_agent_workflow("The profile page displays the user's name.")

    assert result["status"] == "error"
    assert result["result"]["model_calls"] == 1
    assert "invalid JSON" in result["error"]


def test_agent_returns_error_for_empty_model_output(monkeypatch):
    completions = SequenceChatCompletions([""])
    monkeypatch.setattr(
        main,
        "client",
        SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    result = main.run_agent_workflow("The profile page displays the user's name.")

    assert result["status"] == "error"
    assert result["result"]["model_calls"] == 1
    assert "empty response" in result["error"]


def test_agent_returns_error_for_model_api_failure(monkeypatch):
    completions = SequenceChatCompletions([RuntimeError("provider unavailable")])
    monkeypatch.setattr(
        main,
        "client",
        SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    result = main.run_agent_workflow("The profile page displays the user's name.")

    assert result["status"] == "error"
    assert result["result"]["model_calls"] == 1
    assert result["error"] == "provider unavailable"


def test_agent_enforces_two_call_limit(monkeypatch):
    bad_case = targeted_case_payload("Assume the page redirects to a dashboard.")
    completions = SequenceChatCompletions([bad_case, bad_case, targeted_case_payload()])
    monkeypatch.setattr(
        main,
        "client",
        SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    result = main.run_agent_workflow("The profile page displays the user's name.")

    assert result["status"] == "needs_review"
    assert result["result"]["model_calls"] == main.MAX_AGENT_MODEL_CALLS == 2
    assert len(completions.calls) == 2


@pytest.mark.parametrize(
    "category, model_case, required_fields",
    [
        (
            "functional",
            {"id": "TC01", "requirement_id": "R01", "type": "functional", "input": "Enter a valid password", "expected": "Password is accepted."},
            {"description", "expected"},
        ),
        (
            "negative",
            {"id": "NTC01", "requirement_id": "R01", "type": "negative", "input": "Enter an invalid password", "expected": "Password is rejected."},
            {"description", "expected"},
        ),
        (
            "bva",
            {"id": "BVA01", "requirement_id": "R01", "type": "bva", "input": "7", "expected": "Password is rejected."},
            {"scenario", "values", "expected"},
        ),
        (
            "ep",
            {"id": "EP01", "requirement_id": "R01", "type": "ep", "input": "Below minimum", "expected": "Password is rejected."},
            {"class", "input", "expected"},
        ),
        (
            "edge_cases",
            {"id": "EC01", "requirement_id": "R01", "type": "edge case", "input": "An edge situation", "expected": "The requirement remains satisfied."},
            {"description", "expected"},
        ),
    ],
)
def test_targeted_model_fields_normalize_to_canonical_schema(category, model_case, required_fields):
    normalized = main._normalize_agent_test_cases({category: [model_case]})[category][0]

    assert required_fields.issubset(normalized)
    assert normalized["expected"] == model_case["expected"]
    assert main.evaluate_structure({
        "functional": [normalized] if category == "functional" else [],
        "negative": [normalized] if category == "negative" else [],
        "bva": [normalized] if category == "bva" else [],
        "ep": [normalized] if category == "ep" else [],
        "edge_cases": [normalized] if category == "edge_cases" else [],
    })["score"] == 100.0


def test_targeted_expected_result_alias_normalizes_before_validation():
    normalized = main._normalize_agent_test_cases({
        "functional": [{
            "id": "TC01",
            "requirement_id": "R01",
            "input": "Display the profile.",
            "expected_result": "The profile is displayed.",
        }]
    })["functional"][0]

    assert normalized["description"] == "Display the profile."
    assert normalized["expected"] == "The profile is displayed."
    assert "expected_result" in normalized
    assert main.evaluate_structure({
        "functional": [normalized], "negative": [], "bva": [], "ep": [], "edge_cases": [],
    })["score"] == 100.0


def test_agent_merges_normalized_additions(monkeypatch):
    completions = SequenceChatCompletions([json.dumps({
        "test_cases": {
            "functional": [{"id": "TC99", "type": "functional", "input": "Display the profile.", "expected": "The profile is displayed."}],
            "negative": [], "bva": [], "ep": [], "edge_cases": [],
        }
    })])
    monkeypatch.setattr(main, "client", SimpleNamespace(chat=SimpleNamespace(completions=completions)))

    result = main.run_agent_workflow(
        "The profile page displays the user's name.",
        {"functional": [], "negative": [], "bva": [], "ep": [], "edge_cases": []},
    )

    assert result["status"] == "completed"
    assert result["result"]["test_cases"]["functional"][0]["id"] == "TC01"
    assert result["result"]["test_cases"]["functional"][0]["description"] == "Display the profile."


def test_password_range_workflow_completes_from_empty_cases(monkeypatch):
    model_cases = {
        "functional": [{"id": "TC01", "type": "functional", "input": "Enter a password within the allowed length", "expected": "Password is accepted."}],
        "negative": [{"id": "NTC01", "type": "negative", "input": "Enter an invalid-length password", "expected": "Password is rejected."}],
        "bva": [
            {"id": f"BVA{i:02d}", "type": "bva", "input": str(value), "expected": "Password is accepted." if 8 <= value <= 20 else "Password is rejected."}
            for i, value in enumerate([7, 8, 9, 19, 20, 21], 1)
        ],
        "ep": [
            {"id": "EP01", "type": "ep", "input": "7", "expected": "Password is rejected."},
            {"id": "EP02", "type": "ep", "input": "12", "expected": "Password is accepted."},
            {"id": "EP03", "type": "ep", "input": "21", "expected": "Password is rejected."},
        ],
        "edge_cases": [],
    }
    completions = SequenceChatCompletions([json.dumps({"test_cases": model_cases})])
    monkeypatch.setattr(main, "client", SimpleNamespace(chat=SimpleNamespace(completions=completions)))

    result = main.run_agent_workflow(
        "The password must be between 8 and 20 characters",
        {"functional": [], "negative": [], "bva": [], "ep": [], "edge_cases": []},
    )

    assert result["status"] == "completed"
    assert result["result"]["model_calls"] == 1
    assert result["result"]["refinement_count"] == 0
    assert len(result["result"]["test_cases"]["bva"]) == 6
    assert len(result["result"]["test_cases"]["ep"]) == 3


def test_password_string_inputs_normalize_to_numeric_bva_and_ep_representatives():
    cases = {
        "functional": [{"id": "TC01", "requirement_id": "R01", "type": "functional", "input": "Enter a valid password", "expected": "Password is accepted."}],
        "negative": [{"id": "NTC01", "requirement_id": "R01", "type": "negative", "input": "Enter an invalid-length password", "expected": "Password is rejected."}],
        "bva": [
            {"id": f"BVA{i:02d}", "requirement_id": "R01", "type": "bva", "input": value, "expected": "Password is accepted." if len(value) in {8, 9, 19, 20} else "Password is rejected."}
            for i, value in enumerate(["abcdefg", "abcdefgh", "abcdefghi", "abcdefghijklmnopqrs", "abcdefghijklmnopqrst", "abcdefghijklmnopqrstu"], 1)
        ],
        "ep": [
            {"id": "EP01", "requirement_id": "R01", "type": "ep", "input": "abcdefg", "expected": "Password is rejected."},
            {"id": "EP02", "requirement_id": "R01", "type": "ep", "input": "abcdefghijkl", "expected": "Password is accepted."},
            {"id": "EP03", "requirement_id": "R01", "type": "ep", "input": "abcdefghijklmnopqrstuv", "expected": "Password is rejected."},
        ],
        "edge_cases": [],
    }

    normalized = main._normalize_agent_test_cases(
        cases,
        "The password must be between 8 and 20 characters",
    )

    assert [case["values"] for case in normalized["bva"]] == ["7", "8", "9", "19", "20", "21"]
    assert [case["input"] for case in normalized["ep"]] == ["7", "12", "21"]
    assert main.evaluate_structure(normalized)["score"] == 100.0
    assert main.evaluate_bva(normalized, "The password must be between 8 and 20 characters")["score"] == 100.0
    assert main.evaluate_ep(normalized, "The password must be between 8 and 20 characters")["score"] == 100.0


def test_ep_uses_actual_input_not_descriptive_numbers():
    result = main.evaluate_ep(
        {
            "ep": [
                {"class": "Below minimum 8", "input": "unknown", "description": "Below 8"},
                {"class": "Valid range 8 to 20", "input": "unknown", "description": "Within 8 to 20"},
                {"class": "Above maximum 20", "input": "unknown", "description": "Above 20"},
            ]
        },
        "The password length must be between 8 and 20 characters.",
    )
    assert result["score"] == 0.0


def test_negative_quality_rejects_positive_scenario():
    quality = main.validate_generated_quality(
        "Users can log in only with valid credentials.",
        {"functional": [], "negative": [case("NTC01", "Negative", "Log in with credentials.", "The user is authenticated.")], "bva": [], "ep": [], "edge_cases": []},
        "R01",
        True,
    )
    assert quality["score"] < 100
    assert any("negative scenario" in finding for finding in quality["findings"])


def test_vague_expected_result_reduces_quality():
    quality = main.validate_generated_quality(
        "The system displays a profile.",
        {"functional": [case("TC01", "Functional", "Display the profile.", "Works successfully")], "negative": [], "bva": [], "ep": [], "edge_cases": []},
        "R01",
        False,
    )
    assert quality["score"] < 100
    assert any("vague expected" in finding for finding in quality["findings"])


def test_structural_validation_rejects_bva_case_missing_values():
    result = main.evaluate_structure({
        "functional": [],
        "negative": [],
        "bva": [{"id": "BVA01", "requirement_id": "R01", "type": "BVA", "scenario": "Below minimum", "expected": "Rejected."}],
        "ep": [],
        "edge_cases": [],
    })
    assert result["score"] == 0.0


def test_structural_validation_rejects_ep_case_missing_input():
    result = main.evaluate_structure({
        "functional": [],
        "negative": [],
        "bva": [],
        "ep": [{"id": "EP01", "requirement_id": "R01", "type": "EP", "class": "Below minimum", "expected": "Rejected."}],
        "edge_cases": [],
    })
    assert result["score"] == 0.0


def test_traceability_consistency_rejects_orphan_row():
    test_cases = {"functional": [case("TC01", "Functional", "Display profile.", "The profile is displayed.")], "negative": [], "bva": [], "ep": [], "edge_cases": []}
    result = main.validate_traceability_consistency(
        test_cases,
        [{"requirement_id": "R01", "test_case": "TC01"}, {"requirement_id": "R01", "test_case": "TC99"}],
        "R01",
    )
    assert result["consistent"] is False
    assert result["orphaned"] == ["TC99"]


def test_traceability_consistency_rejects_missing_row():
    test_cases = {"functional": [case("TC01", "Functional", "Display profile.", "The profile is displayed."), case("TC02", "Functional", "Refresh profile.", "The profile is refreshed.")], "negative": [], "bva": [], "ep": [], "edge_cases": []}
    result = main.validate_traceability_consistency(
        test_cases,
        [{"requirement_id": "R01", "test_case": "TC01"}],
        "R01",
    )
    assert result["consistent"] is False
    assert result["missing"] == ["TC02"]


def test_requirement_coverage_consistency_rejects_inconsistent_counts():
    result = main.validate_requirement_coverage_consistency({
        "total_conditions": 3,
        "covered_conditions": 1,
        "uncovered_conditions": 1,
        "percentage": 33.33,
    })
    assert result["consistent"] is False
    assert any("counts" in finding for finding in result["findings"])


def test_requirement_coverage_consistency_rejects_inconsistent_percentage():
    result = main.validate_requirement_coverage_consistency({
        "total_conditions": 2,
        "covered_conditions": 1,
        "uncovered_conditions": 1,
        "percentage": 100.0,
    })
    assert result["consistent"] is False
    assert result["expected_percentage"] == 50.0


def test_atm_bva_six_positions_remain_complete():
    requirement_text = "The ATM withdrawal amount must be between $500 and $100,000, and the account must have sufficient balance."
    values = ["$499", "$500", "$501", "$99,999", "$100,000", "$100,001"]
    result = main.evaluate_bva(
        {"bva": [{"values": value} for value in values]},
        requirement_text,
    )
    assert result["applicable"] is True
    assert result["score"] == 100.0


def test_negative_case_recognizes_insufficient_balance_wording():
    """Regression for the exact reported 93.8% Structural Completeness bug:
    an insufficient-balance negative case for the compound ATM requirement
    must be recognized as a negative scenario, not flagged as unclear."""
    requirement_text = (
        "The ATM withdrawal amount must be between 500 and 100000, "
        "and the account must have sufficient balance."
    )
    case = {
        "id": "NTC02", "requirement_id": "R01", "type": "Negative",
        "description": "Withdraw 1000 with insufficient account balance.",
        "expected": "Withdrawal is rejected.",
    }

    assert main._negative_case_is_negative(case, requirement_text) is True


def test_negative_case_recognizes_rejection_stated_only_in_expected_field():
    """Regression for the reported 93.75% (15/16) Structural Quality bug on
    the single-condition ATM range requirement: a negative case whose
    description has no failure keyword (e.g. "non-numeric withdrawal
    amount") but whose expected result explicitly states the rejection
    ("Withdrawal is rejected.") must still be recognized as a clear negative
    scenario, since _negative_case_is_negative previously only inspected
    description/scenario/class/input and ignored the expected field."""
    requirement_text = "The ATM withdrawal amount must be between 500 and 100000"
    case = {
        "id": "NTC03", "requirement_id": "R01", "type": "Negative",
        "description": "Withdraw a non-numeric withdrawal amount.",
        "expected": "Withdrawal is rejected.",
    }

    assert main._negative_case_is_negative(case, requirement_text) is True


def test_structural_quality_is_not_dragged_down_by_expected_only_rejection_wording():
    """Full reproduction of the reported 93.75% (15/16) Structural Quality
    case for "The ATM withdrawal amount must be between 500 and 100000":
    16 cases (3 functional, 4 negative, 6 BVA, 3 EP), all structurally
    complete per evaluate_structure, but validate_generated_quality
    previously flagged NTC03 ("non-numeric withdrawal amount") as unclear
    because its rejection was only stated in the expected field."""
    requirement_text = "The ATM withdrawal amount must be between 500 and 100000"
    requirement_id = "R01"
    test_cases = {
        "functional": [
            {"id": "TC01", "requirement_id": "R01", "type": "Functional", "description": "Withdraw 500 from the ATM.", "expected": "Withdrawal succeeds."},
            {"id": "TC02", "requirement_id": "R01", "type": "Functional", "description": "Withdraw 100000 from the ATM.", "expected": "Withdrawal succeeds."},
            {"id": "TC03", "requirement_id": "R01", "type": "Functional", "description": "Withdraw 50000 from the ATM.", "expected": "Withdrawal succeeds."},
        ],
        "negative": [
            {"id": "NTC01", "requirement_id": "R01", "type": "Negative", "description": "Withdraw 499 from the ATM.", "expected": "Withdrawal is rejected."},
            {"id": "NTC02", "requirement_id": "R01", "type": "Negative", "description": "Withdraw 100001 from the ATM.", "expected": "Withdrawal is rejected."},
            {"id": "NTC03", "requirement_id": "R01", "type": "Negative", "description": "Withdraw a non-numeric withdrawal amount.", "expected": "Withdrawal is rejected."},
            {"id": "NTC04", "requirement_id": "R01", "type": "Negative", "description": "Withdraw with an empty withdrawal amount field.", "expected": "Withdrawal is rejected."},
        ],
        "bva": [
            {"id": f"BVA{i:02d}", "requirement_id": "R01", "type": "BVA", "scenario": "Boundary", "values": str(v), "expected": "Accepted" if 500 <= v <= 100000 else "Rejected"}
            for i, v in enumerate([499, 500, 501, 99999, 100000, 100001], 1)
        ],
        "ep": [
            {"id": "EP01", "requirement_id": "R01", "type": "EP", "class": "Below minimum", "input": "499", "expected": "Rejected"},
            {"id": "EP02", "requirement_id": "R01", "type": "EP", "class": "Valid range", "input": "500 to 100000", "expected": "Accepted"},
            {"id": "EP03", "requirement_id": "R01", "type": "EP", "class": "Above maximum", "input": "100001", "expected": "Rejected"},
        ],
        "edge_cases": [],
    }

    assert len(main.all_test_cases(test_cases)) == 16

    structure_result = main.evaluate_structure(test_cases)
    generated_quality = main.validate_generated_quality(
        requirement_text, test_cases, requirement_id, True,
    )

    assert structure_result["score"] == 100.0
    assert generated_quality["score"] == 100.0
    assert generated_quality["findings"] == []
    assert min(structure_result["score"], generated_quality["score"]) == 100.0


def test_negative_case_still_flagged_when_expected_result_is_positive():
    """Regression guard: including the expected field in the negative-case
    check must not mask a genuinely mislabeled negative case whose expected
    result is itself positive (no rejection wording anywhere)."""
    quality = main.validate_generated_quality(
        "Users can log in only with valid credentials.",
        {
            "functional": [], "bva": [], "ep": [], "edge_cases": [],
            "negative": [{
                "id": "NTC01", "requirement_id": "R01", "type": "Negative",
                "description": "Log in with credentials.",
                "expected": "The user is authenticated.",
            }],
        },
        "R01",
        True,
    )

    assert quality["score"] < 100
    assert any("negative scenario" in finding for finding in quality["findings"])


def test_structural_completeness_is_not_dragged_down_by_insufficient_balance_wording():
    """Full reproduction of the reported 93.8% Structural Completeness case:
    a 16-case ATM+balance result where evaluate_structure itself is already
    100% (every required field is present), but validate_generated_quality
    previously misclassified the insufficient-balance negative case as
    'not a clear negative scenario', clamping the reported Structural
    Completeness down to 93.75% (rounded to 93.8%). It must now be 100%."""
    requirement_text = (
        "The ATM withdrawal amount must be between 500 and 100000, "
        "and the account must have sufficient balance."
    )
    requirement_id = "R01"
    test_cases = {
        "functional": [
            {"id": "TC01", "requirement_id": "R01", "type": "Functional", "description": "Withdraw 500 with sufficient balance.", "expected": "Withdrawal succeeds."},
            {"id": "TC02", "requirement_id": "R01", "type": "Functional", "description": "Withdraw 100000 with sufficient balance.", "expected": "Withdrawal succeeds."},
        ],
        "negative": [
            {"id": "NTC01", "requirement_id": "R01", "type": "Negative", "description": "Withdraw 499 with sufficient balance.", "expected": "Withdrawal is rejected."},
            {"id": "NTC02", "requirement_id": "R01", "type": "Negative", "description": "Withdraw 1000 with insufficient account balance.", "expected": "Withdrawal is rejected."},
        ],
        "bva": [
            {"id": f"BVA{i:02d}", "requirement_id": "R01", "type": "BVA", "scenario": "Boundary", "values": str(v), "expected": "Accepted" if 500 <= v <= 100000 else "Rejected"}
            for i, v in enumerate([499, 500, 501, 99999, 100000, 100001], 1)
        ],
        "ep": [
            {"id": "EP01", "requirement_id": "R01", "type": "EP", "class": "Below minimum", "input": "499", "expected": "Rejected"},
            {"id": "EP02", "requirement_id": "R01", "type": "EP", "class": "Valid range", "input": "500 to 100000", "expected": "Accepted"},
            {"id": "EP03", "requirement_id": "R01", "type": "EP", "class": "Above maximum", "input": "100001", "expected": "Rejected"},
        ],
        "edge_cases": [
            {"id": "EC01", "requirement_id": "R01", "type": "Edge Case", "description": "Withdraw exactly 500 with account balance exactly 500.", "expected": "Withdrawal succeeds."},
            {"id": "EC02", "requirement_id": "R01", "type": "Edge Case", "description": "Withdraw exactly 100000 with account balance exactly 100000.", "expected": "Withdrawal succeeds."},
            {"id": "EC03", "requirement_id": "R01", "type": "Edge Case", "description": "Withdraw 100000 when account balance is 99999.", "expected": "Withdrawal is rejected."},
        ],
    }

    assert len(main.all_test_cases(test_cases)) == 16

    structure_result = main.evaluate_structure(test_cases)
    generated_quality = main.validate_generated_quality(
        requirement_text, test_cases, requirement_id, True,
    )

    assert structure_result["score"] == 100.0
    assert generated_quality["score"] == 100.0
    assert generated_quality["findings"] == []
    assert min(structure_result["score"], generated_quality["score"]) == 100.0


def test_atm_derived_boundary_and_negative_values_are_grounded():
    requirement_text = (
        "The ATM withdrawal amount must be between 500 and 100000 "
        "and the account must have sufficient balance."
    )
    test_cases = {
        "functional": [],
        "negative": [
            {"id": "NTC01", "description": "Withdraw 400 with sufficient balance.", "expected": "Withdrawal is rejected."},
            {"id": "NTC02", "description": "Withdraw 100001 with sufficient balance.", "expected": "Withdrawal is rejected."},
        ],
        "bva": [
            {"id": "BVA01", "scenario": "Below minimum", "values": "400", "expected": "Rejected."},
            {"id": "BVA02", "scenario": "Just below minimum", "values": "499", "expected": "Rejected."},
            {"id": "BVA03", "scenario": "Just above minimum", "values": "501", "expected": "Accepted."},
            {"id": "BVA04", "scenario": "Just below maximum", "values": "99999", "expected": "Accepted."},
            {"id": "BVA05", "scenario": "Above maximum", "values": "100001", "expected": "Rejected."},
        ],
        "ep": [],
        "edge_cases": [],
    }

    result = main.validate_requirement_grounding(requirement_text, test_cases)

    assert result["grounded"] is True
    assert result["findings"] == []


def test_compound_condition_coverage_requires_condition_specific_evidence():
    requirement = {
        "id": "R01",
        "conditions": [
            "The withdrawal amount is between 500 and 100000.",
            "The account must have sufficient balance.",
        ],
    }
    test_cases = {
        "functional": [],
        "negative": [],
        "bva": [
            {"id": "BVA01", "scenario": "Withdrawal amount below minimum", "values": "499", "expected": "Withdrawal is rejected."},
            {"id": "BVA02", "scenario": "Withdrawal amount at minimum", "values": "500", "expected": "Withdrawal is accepted."},
            {"id": "BVA03", "scenario": "Withdrawal amount above maximum", "values": "100001", "expected": "Withdrawal is rejected."},
        ],
        "ep": [],
        "edge_cases": [],
    }

    result = main.evaluate_requirement_coverage(
        requirement,
        "The ATM withdrawal amount must be between 500 and 100000 and the account must have sufficient balance.",
        test_cases,
    )

    balance_condition = next(
        condition
        for condition in result["conditions"]
        if "sufficient balance" in condition["text"].lower()
    )
    assert balance_condition["status"] == "uncovered"
    assert balance_condition["covered_by"] == []


def test_targeted_functional_atm_fields_normalize_with_balance_evidence():
    normalized = main._normalize_agent_test_cases({
        "functional": [{
            "id": "TC01",
            "requirement_id": "R01",
            "type": "functional",
            "withdrawal_amount": "500",
            "account_balance": "sufficient",
            "expected": "Withdrawal succeeds.",
        }],
    })["functional"][0]

    assert normalized["description"] == "Withdrawal amount: 500; account balance: sufficient"
    assert main.evaluate_structure({
        "functional": [normalized], "negative": [], "bva": [], "ep": [], "edge_cases": [],
    })["score"] == 100.0


def test_targeted_negative_atm_fields_normalize_with_balance_evidence():
    normalized = main._normalize_agent_test_cases({
        "negative": [{
            "id": "NTC01",
            "requirement_id": "R01",
            "type": "negative",
            "withdrawal_amount": "500",
            "account_balance": "insufficient",
            "expected": "Withdrawal is rejected.",
        }],
    })["negative"][0]

    assert normalized["description"] == "Withdrawal amount: 500; account balance: insufficient"
    assert main.evaluate_structure({
        "functional": [], "negative": [normalized], "bva": [], "ep": [], "edge_cases": [],
    })["score"] == 100.0


def test_targeted_ep_withdrawal_amount_normalizes_to_canonical_input():
    normalized = main._normalize_agent_test_cases({
        "ep": [{
            "id": "EP01",
            "requirement_id": "R01",
            "type": "ep",
            "withdrawal_amount": "500",
            "account_balance": "sufficient",
            "expected": "Withdrawal succeeds.",
        }],
    })["ep"][0]

    assert normalized["input"] == "500"
    assert normalized["class"] == "sufficient"
    assert main.evaluate_structure({
        "functional": [], "negative": [], "bva": [], "ep": [normalized], "edge_cases": [],
    })["score"] == 100.0


def test_explicit_range_allows_derived_minimum_maximum_terminology():
    requirement_text = "The ATM withdrawal amount must be between 500 and 100000."
    test_cases = {
        "functional": [],
        "negative": [{
            "id": "NTC01",
            "description": "Withdraw below minimum limit.",
            "expected": "Withdrawal is rejected.",
        }, {
            "id": "NTC02",
            "description": "Withdraw above maximum limit.",
            "expected": "Withdrawal is rejected.",
        }],
        "bva": [], "ep": [], "edge_cases": [],
    }

    result = main.validate_requirement_grounding(requirement_text, test_cases)

    assert result["grounded"] is True
    assert result["findings"] == []


def test_atm_compound_coverage_requires_account_balance_evidence():
    requirement_text = (
        "The ATM withdrawal amount must be between 500 and 100000 "
        "and the account must have sufficient balance."
    )
    requirement = {"id": "R01", "conditions": []}
    test_cases = {
        "functional": [{
            "id": "TC01",
            "requirement_id": "R01",
            "description": "Withdrawal amount: 500; account balance: sufficient",
            "expected": "Withdrawal succeeds.",
        }],
        "negative": [], "bva": [], "ep": [], "edge_cases": [],
    }

    result = main.evaluate_requirement_coverage(
        requirement,
        requirement_text,
        test_cases,
    )

    assert result["covered_conditions"] == 2
    assert result["percentage"] == 100.0


def test_ep_two_of_three_identifies_below_minimum_and_refinement_repairs_it(monkeypatch):
    requirement_text = "The withdrawal amount must be between 500 and 100000."
    existing_cases = {
        "functional": [{
            "id": "TC01",
            "requirement_id": "R01",
            "type": "Functional",
            "description": "Withdraw an amount within the allowed range.",
            "expected": "Withdrawal succeeds.",
        }],
        "negative": [{
            "id": "NTC01",
            "requirement_id": "R01",
            "type": "Negative",
            "description": "Withdraw an amount above the maximum.",
            "expected": "Withdrawal is rejected.",
        }],
        "bva": [{
            "id": f"BVA{i:02d}",
            "requirement_id": "R01",
            "type": "BVA",
            "scenario": "Withdrawal amount boundary",
            "values": str(value),
            "expected": "Accepted." if 500 <= value <= 100000 else "Rejected.",
        } for i, value in enumerate([499, 500, 501, 99999, 100000, 100001], 1)],
        "ep": [
            {"id": "EP01", "requirement_id": "R01", "type": "EP", "class": "Valid range", "input": "500", "expected": "Accepted."},
            {"id": "EP02", "requirement_id": "R01", "type": "EP", "class": "Above maximum", "input": "100001", "expected": "Rejected."},
        ],
        "edge_cases": [],
    }
    initial_gap_analysis = main.analyze_test_gaps(requirement_text, existing_cases)

    assert initial_gap_analysis["evidence"]["ep"]["score"] == round((2 / 3) * 100, 2)
    assert initial_gap_analysis["evidence"]["ep"]["message"].endswith("Missing: below minimum.")

    lower_partition = json.dumps({
        "test_cases": {
            "ep": [{
                "id": "EP03",
                "requirement_id": "R01",
                "type": "EP",
                "class": "Below minimum",
                "input": "499",
                "expected": "Rejected.",
            }],
        }
    })
    completions = SequenceChatCompletions([lower_partition])
    monkeypatch.setattr(main, "client", SimpleNamespace(chat=SimpleNamespace(completions=completions)))

    result = main.run_agent_workflow(requirement_text, existing_cases)

    assert result["status"] == "completed"
    assert result["result"]["model_calls"] == 1
    assert result["result"]["analysis"]["scores"]["ep"] == 100.0
    assert result["result"]["analysis"]["gap_count"] == 0


def test_ep_two_of_three_refinement_prompt_contains_missing_partition(monkeypatch):
    requirement_text = "The withdrawal amount must be between 500 and 100000."
    existing_cases = {
        "functional": [{"id": "TC01", "requirement_id": "R01", "type": "Functional", "description": "Withdraw within range.", "expected": "Accepted."}],
        "negative": [{"id": "NTC01", "requirement_id": "R01", "type": "Negative", "description": "Withdraw below the minimum.", "expected": "Rejected."}],
        "bva": [{
            "id": f"BVA{i:02d}", "requirement_id": "R01", "type": "BVA",
            "scenario": "Boundary", "values": str(value), "expected": "Accepted." if 500 <= value <= 100000 else "Rejected.",
        } for i, value in enumerate([499, 500, 501, 99999, 100000, 100001], 1)],
        "ep": [],
        "edge_cases": [],
    }
    two_partitions = json.dumps({"test_cases": {"ep": [
        {"id": "EP01", "requirement_id": "R01", "type": "EP", "class": "Valid range", "input": "500", "expected": "Accepted."},
        {"id": "EP02", "requirement_id": "R01", "type": "EP", "class": "Above maximum", "input": "100001", "expected": "Rejected."},
    ]}})
    valid_lower = json.dumps({"test_cases": {"ep": [{
        "id": "EP03", "requirement_id": "R01", "type": "EP", "class": "Below minimum", "input": "499", "expected": "Rejected.",
    }]}})
    completions = SequenceChatCompletions([two_partitions, valid_lower])
    monkeypatch.setattr(main, "client", SimpleNamespace(chat=SimpleNamespace(completions=completions)))

    result = main.run_agent_workflow(requirement_text, existing_cases)

    assert result["status"] == "completed"
    assert result["result"]["model_calls"] == main.MAX_AGENT_MODEL_CALLS
    assert result["result"]["refinement_count"] == 1
    assert "Missing: below minimum." in completions.calls[1]["messages"][1]["content"]


def test_final_ep_validation_uses_post_refinement_cases(monkeypatch):
    requirement_text = (
        "The ATM withdrawal amount must be between 500 and 100000, "
        "and the account must have sufficient balance."
    )
    existing_cases = {
        "functional": [{
            "id": "TC01",
            "requirement_id": "R01",
            "type": "Functional",
            "description": "Withdraw $500 with sufficient balance.",
            "expected": "Withdrawal succeeds.",
        }],
        "negative": [{
            "id": "NTC01",
            "requirement_id": "R01",
            "type": "Negative",
            "description": "Withdraw $100001 with sufficient balance.",
            "expected": "Withdrawal is rejected.",
        }],
        "bva": [{
            "id": f"BVA{i:02d}",
            "requirement_id": "R01",
            "type": "BVA",
            "scenario": "Withdrawal amount boundary",
            "values": str(value),
            "expected": "Accepted." if 500 <= value <= 100000 else "Rejected.",
        } for i, value in enumerate([499, 500, 501, 99999, 100000, 100001], 1)],
        "ep": [
            {"id": "EP01", "requirement_id": "R01", "type": "EP", "class": "Valid range", "input": "500", "expected": "Accepted."},
            {"id": "EP02", "requirement_id": "R01", "type": "EP", "class": "Above maximum", "input": "100001", "expected": "Rejected."},
        ],
        "edge_cases": [],
    }
    missing_lower_partition = json.dumps({
        "test_cases": {
            "ep": [{
                "id": "EP03",
                "requirement_id": "R01",
                "type": "EP",
                "class": "Below minimum",
                "input": "499",
                "expected": "Rejected.",
            }],
        }
    })
    completions = SequenceChatCompletions([missing_lower_partition])
    monkeypatch.setattr(main, "client", SimpleNamespace(chat=SimpleNamespace(completions=completions)))

    result = main.run_agent_workflow(requirement_text, existing_cases)

    assert result["status"] == "completed"
    assert result["result"]["final_status"] == "completed"
    assert result["result"]["identified_gaps"] == []
    assert result["result"]["analysis"]["gaps"] == []
    assert result["result"]["analysis"]["scores"]["ep"] == 100.0
    assert result["result"]["evaluation"]["details"]["ep_quality"] == 100.0
    assert result["result"]["evaluation"]["quality_breakdown"]["ep_quality"] == 100.0
    assert {case["id"] for case in result["result"]["test_cases"]["ep"]} == {"EP01", "EP02", "EP03"}
    assert result["result"]["analysis"]["evidence"]["ep"]["message"].startswith("EP correctly identifies")


def test_dedup_does_not_collapse_ep_partitions_sharing_a_generic_class_label():
    """Regression for the true root cause behind the stale 'EP covers 1/3'
    UI symptom: below-minimum and above-maximum EP cases sharing a generic
    class label (e.g. both labeled "Invalid") must not be treated as
    duplicates of each other just because their descriptive label and
    expected outcome match. Their distinct input values must also count."""
    test_cases = {
        "functional": [], "negative": [], "bva": [], "edge_cases": [],
        "ep": [
            {"id": "EP01", "requirement_id": "R01", "type": "EP", "class": "Invalid", "input": "499", "expected": "Rejected."},
            {"id": "EP02", "requirement_id": "R01", "type": "EP", "class": "Valid", "input": "500", "expected": "Accepted."},
            {"id": "EP03", "requirement_id": "R01", "type": "EP", "class": "Invalid", "input": "100001", "expected": "Rejected."},
        ],
    }

    cleaned = main.remove_duplicate_test_cases({k: list(v) for k, v in test_cases.items()})

    assert {case["id"] for case in cleaned["ep"]} == {"EP01", "EP02", "EP03"}

    ep_result = main.evaluate_ep(cleaned, "The withdrawal amount must be between 500 and 100000.")
    assert ep_result["score"] == 100.0


def test_full_runtime_flow_generate_then_agent_reports_final_ep_and_completed_status(monkeypatch):
    """End-to-end regression matching the exact reported UI symptom: run the
    real /generate/ then /agent/ flow (as the Streamlit frontend does) with
    a model response that uses a generic shared EP class label for the two
    invalid partitions, and prove the UI-facing agent result shows EP
    quality 100%, no identified gaps, and a completed final status."""
    requirement_text = (
        "The ATM withdrawal amount must be between 500 and 100000, "
        "and the account must have sufficient balance."
    )
    generate_payload = {
        "requirement": {
            "id": "R01",
            "type": "Validation",
            "priority": "Medium",
            "conditions": [
                "The withdrawal amount is between 500 and 100000.",
                "The account must have sufficient balance.",
            ],
            "constraints": [],
            "inputs": [],
            "expected_behavior": "Withdrawal succeeds only within range and with sufficient balance.",
        },
        "test_cases": {
            "functional": [{
                "id": "TC01", "requirement_id": "R01", "type": "Functional",
                "description": "Withdraw 500 with sufficient balance.",
                "expected": "Withdrawal succeeds.",
            }],
            "negative": [{
                "id": "NTC01", "requirement_id": "R01", "type": "Negative",
                "description": "Withdraw 100001 with sufficient balance.",
                "expected": "Withdrawal is rejected.",
            }],
            "bva": [
                {"id": f"BVA{i:02d}", "requirement_id": "R01", "type": "BVA",
                 "scenario": "Withdrawal amount boundary", "values": str(value),
                 "expected": "Accepted." if 500 <= value <= 100000 else "Rejected."}
                for i, value in enumerate([499, 500, 501, 99999, 100000, 100001], 1)
            ],
            "ep": [
                {"id": "EP01", "requirement_id": "R01", "type": "EP", "class": "Invalid", "input": "499", "expected": "Rejected."},
                {"id": "EP02", "requirement_id": "R01", "type": "EP", "class": "Valid", "input": "500", "expected": "Accepted."},
                {"id": "EP03", "requirement_id": "R01", "type": "EP", "class": "Invalid", "input": "100001", "expected": "Rejected."},
            ],
            "edge_cases": [],
        },
    }
    monkeypatch.setattr(
        main, "client",
        SimpleNamespace(chat=SimpleNamespace(completions=FakeChatCompletions(generate_payload))),
    )

    client = TestClient(main.app)
    generate_response = client.post("/generate/", json={"text": requirement_text})
    assert generate_response.status_code == 200
    generated_body = generate_response.json()

    # Same request shape as frontend/api_client.run_agent_workflow.
    agent_response = client.post("/agent/", json={
        "intent": "complete",
        "text": generated_body["requirement"]["text"],
        "requirement": generated_body["requirement"],
        "test_cases": generated_body["test_cases"],
    })
    assert agent_response.status_code == 200
    agent_body = agent_response.json()

    assert agent_body["status"] == "completed"
    result = agent_body["result"]
    assert result["identified_gaps"] == []
    assert result["analysis"]["gaps"] == []
    assert "evaluation" in result
    assert result["evaluation"]["details"]["ep_quality"] == 100.0
    assert result["evaluation"]["quality_breakdown"]["ep_quality"] == 100.0
    assert {case["id"] for case in result["test_cases"]["ep"]} == {"EP01", "EP02", "EP03"}


def test_ep_survives_boundary_keyword_phrasing_and_reports_no_gap():
    """EP01 phrased as 'below 500' and EP03 as 'above 100000' must not be
    misread as the boundary number itself falling inside the valid range;
    the keyword must decide the partition, not the raw numeric comparison."""
    requirement_text = "The ATM withdrawal amount must be between 500 and 100000."
    test_cases = {
        "functional": [], "negative": [], "bva": [], "edge_cases": [],
        "ep": [
            {"id": "EP01", "requirement_id": "R01", "type": "EP", "class": "Below minimum", "input": "below 500", "expected": "Rejected."},
            {"id": "EP02", "requirement_id": "R01", "type": "EP", "class": "Valid range", "input": "500 to 100000", "expected": "Accepted."},
            {"id": "EP03", "requirement_id": "R01", "type": "EP", "class": "Above maximum", "input": "above 100000", "expected": "Rejected."},
        ],
    }

    result = main.evaluate_ep(test_cases, requirement_text)

    assert result["score"] == 100.0
    assert result["message"].startswith("EP correctly identifies")


def test_targeted_response_flat_list_of_cases_is_parsed_and_bucketed():
    """The targeted AI response contract must tolerate a flat list of case
    objects (each labeled with its own category), not only a dict keyed by
    category, so a differently-shaped-but-valid response is not rejected."""
    flat_response = json.dumps({
        "test_cases": [
            {"id": "EP03", "requirement_id": "R01", "type": "EP", "class": "Below minimum", "input": "499", "expected": "Rejected."},
            {"id": "TC02", "requirement_id": "R01", "type": "functional", "description": "Withdraw 500.", "expected": "Withdrawal succeeds."},
        ]
    })

    parsed = main._parse_agent_test_cases(flat_response, "The withdrawal amount must be between 500 and 100000.")

    assert [case["id"] for case in parsed["ep"]] == ["EP03"]
    assert [case["id"] for case in parsed["functional"]] == ["TC02"]


def test_grounding_allows_functional_cases_that_restate_requirement_minimum_maximum():
    """Functional cases describing 'the minimum allowed amount' or 'the
    maximum allowed amount' must not be flagged as inventing a constraint
    when the requirement itself states an explicit numeric range."""
    requirement_text = "The ATM withdrawal amount must be between 500 and 100000."
    test_cases = {
        "negative": [], "bva": [], "ep": [], "edge_cases": [],
        "functional": [
            {"id": "TC01", "requirement_id": "R01", "type": "Functional", "description": "Withdraw the minimum allowed amount of 500 with sufficient balance.", "expected": "Withdrawal succeeds."},
            {"id": "TC02", "requirement_id": "R01", "type": "Functional", "description": "Withdraw the maximum allowed amount of 100000 with sufficient balance.", "expected": "Withdrawal succeeds."},
        ],
    }

    result = main.validate_requirement_grounding(requirement_text, test_cases)

    assert result["grounded"] is True
    assert result["findings"] == []


def test_agent_workflow_completes_when_ep_and_functional_wording_are_already_valid():
    """Full run_agent_workflow regression matching the exact reported
    scenario: keyword-phrased EP boundaries plus functional cases restating
    minimum/maximum must not trigger any gap, and the final status must be
    completed with zero model calls needed."""
    requirement_text = (
        "The ATM withdrawal amount must be between 500 and 100000, "
        "and the account must have sufficient balance."
    )
    requirement = {
        "id": "R01",
        "conditions": [
            "The withdrawal amount is between 500 and 100000.",
            "The account must have sufficient balance.",
        ],
    }
    test_cases = {
        "functional": [
            {"id": "TC01", "requirement_id": "R01", "type": "Functional", "description": "Withdraw the minimum allowed amount of 500 with sufficient balance.", "expected": "Withdrawal succeeds."},
            {"id": "TC02", "requirement_id": "R01", "type": "Functional", "description": "Withdraw the maximum allowed amount of 100000 with sufficient balance.", "expected": "Withdrawal succeeds."},
        ],
        "negative": [{
            "id": "NTC01", "requirement_id": "R01", "type": "Negative",
            "description": "Withdraw 100001 with sufficient balance.",
            "expected": "Withdrawal is rejected.",
        }],
        "bva": [
            {"id": f"BVA{i:02d}", "requirement_id": "R01", "type": "BVA",
             "scenario": "Withdrawal amount boundary", "values": str(value),
             "expected": "Accepted." if 500 <= value <= 100000 else "Rejected."}
            for i, value in enumerate([499, 500, 501, 99999, 100000, 100001], 1)
        ],
        "ep": [
            {"id": "EP01", "requirement_id": "R01", "type": "EP", "class": "Below minimum", "input": "below 500", "expected": "Rejected."},
            {"id": "EP02", "requirement_id": "R01", "type": "EP", "class": "Valid range", "input": "500 to 100000", "expected": "Accepted."},
            {"id": "EP03", "requirement_id": "R01", "type": "EP", "class": "Above maximum", "input": "above 100000", "expected": "Rejected."},
        ],
        "edge_cases": [],
    }

    result = main.run_agent_workflow(requirement_text, test_cases, requirement)

    assert result["status"] == "completed"
    assert result["result"]["identified_gaps"] == []
    assert result["result"]["analysis"]["gaps"] == []
    assert result["result"]["model_calls"] == 0
    assert result["result"]["analysis"]["scores"]["ep"] == 100.0
    assert result["result"]["analysis"]["scores"]["requirement_grounding"] == 100.0
    assert "evaluation" in result["result"]
    assert result["result"]["evaluation"]["details"]["ep_quality"] == 100.0


def test_generate_strips_invented_empty_email_required_field_negative_case(monkeypatch):
    """Regression for the email requirement: a generated negative case
    asserting an unstated empty-input/required-field rule must not survive
    into the final result, while a genuine invalid-format negative case
    remains, and final grounding reflects only requirement-supported
    content."""
    requirement_text = "The user must enter a valid email address before registration can be completed."
    payload = make_payload(
        requirement_text,
        {
            "functional": [case("TC01", "Functional", "Register with a valid email address.", "Registration succeeds.")],
            "negative": [
                case(
                    "NTC01", "Negative",
                    "Register with an invalid email address such as example.com.",
                    "Registration fails because the email address is invalid.",
                ),
                case(
                    "NTC02", "Negative",
                    "Register with an empty email address.",
                    "Registration fails because the email is required.",
                ),
            ],
            "bva": [], "ep": [], "edge_cases": [],
        },
        type="Validation",
        expected_behavior="Registration completes only with a valid email address.",
    )
    monkeypatch.setattr(main, "client", SimpleNamespace(chat=SimpleNamespace(completions=FakeChatCompletions(payload))))

    response = TestClient(main.app).post("/generate/", json={"text": requirement_text})

    assert response.status_code == 200
    body = response.json()
    negative_cases = body["test_cases"]["negative"]

    assert [c["description"] for c in negative_cases] == [
        "Register with an invalid email address such as example.com."
    ]
    assert not any("empty" in c["description"].lower() for c in negative_cases)
    assert body["evaluation"]["requirement_grounding_score"] == 100.0
    assert body["evaluation"]["requirement_grounding"]["findings"] == []
    assert body["evaluation"]["details"]["bva_applicable"] is False


def _assert_quality_consistency(body):
    evaluation = body["evaluation"]
    details = evaluation["details"]
    breakdown = evaluation["quality_breakdown"]

    assert details["structural_quality"] == breakdown["structure"]
    assert details["duplicate_free"] == breakdown["duplicates"]
    assert details["bva_quality"] == breakdown["bva_quality"]
    assert details["ep_quality"] == breakdown["ep_quality"]
    assert details["requirement_traceability"] if "requirement_traceability" in details else True

    category_checks = [bool(body["test_cases"].get("functional"))]
    if details["negative_applicable"]:
        category_checks.append(bool(body["test_cases"].get("negative")))
    if details["bva_applicable"]:
        category_checks.append(bool(body["test_cases"].get("bva")))
    if details["ep_applicable"]:
        category_checks.append(bool(body["test_cases"].get("ep")))
    category_score = round((sum(category_checks) / len(category_checks)) * 100, 2)
    expected_score = round(
        breakdown["structure"] * 0.20
        + breakdown["traceability"] * 0.20
        + breakdown["duplicates"] * 0.15
        + breakdown["bva_quality"] * 0.20
        + breakdown["ep_quality"] * 0.20
        + category_score * 0.05,
        2,
    )
    assert evaluation["ai_quality_score"] == expected_score


def test_component_consistency_for_valid_atm_result(monkeypatch):
    requirement_text = "The ATM withdrawal amount must be between $500 and $100,000, and the account must have sufficient balance."
    payload = make_payload(
        requirement_text,
        {
            "functional": [case("TC01", "Functional", "Withdraw $500 with sufficient balance.", "Withdrawal succeeds.")],
            "negative": [case("NTC01", "Negative", "Withdraw $499 below the minimum.", "Withdrawal is rejected.")],
            "bva": [
                {"id": f"BVA{i:02d}", "requirement_id": "R01", "type": "BVA", "scenario": "Boundary", "values": value, "expected": "Withdrawal succeeds." if value in ["$500", "$501", "$99,999", "$100,000"] else "Withdrawal is rejected."}
                for i, value in enumerate(["$499", "$500", "$501", "$99,999", "$100,000", "$100,001"], 1)
            ],
            "ep": [
                case("EP01", "EP", "Lower amount", "Withdrawal is rejected.", **{"class": "Below minimum", "input": "$499"}),
                case("EP02", "EP", "Valid amount", "Withdrawal succeeds.", **{"class": "Valid range", "input": "$500"}),
                case("EP03", "EP", "Upper amount", "Withdrawal is rejected.", **{"class": "Above maximum", "input": "$100,001"}),
            ],
            "edge_cases": [],
        },
    )
    monkeypatch.setattr(main, "client", SimpleNamespace(chat=SimpleNamespace(completions=FakeChatCompletions(payload))))
    response = TestClient(main.app).post("/generate/", json={"text": requirement_text})
    assert response.status_code == 200
    _assert_quality_consistency(response.json())


def test_validator_impact_updates_all_quality_outputs(monkeypatch):
    requirement_text = "Users can log in only with valid credentials."
    payload = make_payload(
        requirement_text,
        {
            "functional": [case("TC01", "Functional", "Log in with valid credentials.", "The user is authenticated.")],
            "negative": [case("NTC01", "Negative", "Log in with credentials.", "The user is authenticated.")],
            "bva": [],
            "ep": [case("EP01", "EP", "Valid credentials", "The user is authenticated.", **{"class": "Valid", "input": "valid credentials"})],
            "edge_cases": [],
        },
    )
    monkeypatch.setattr(main, "client", SimpleNamespace(chat=SimpleNamespace(completions=FakeChatCompletions(payload))))
    response = TestClient(main.app).post("/generate/", json={"text": requirement_text})
    body = response.json()
    assert body["evaluation"]["details"]["structural_quality"] < 100
    _assert_quality_consistency(body)


def test_history_preserves_quality_consistency(monkeypatch):
    requirement_text = "The profile page displays the user's name after login."
    payload = make_payload(
        requirement_text,
        {
            "functional": [case("TC01", "Functional", "Open profile after login.", "The profile displays the user's name.")],
            "negative": [], "bva": [], "ep": [], "edge_cases": [],
        },
    )
    monkeypatch.setattr(main, "client", SimpleNamespace(chat=SimpleNamespace(completions=FakeChatCompletions(payload))))
    response = TestClient(main.app).post("/generate/", json={"text": requirement_text})
    assert response.status_code == 200

    db_path = os.path.join(os.path.dirname(__file__), "..", "traceai.db")
    with sqlite3.connect(db_path) as connection:
        run_id = connection.execute("SELECT MAX(id) FROM generation_runs").fetchone()[0]

    history_response = TestClient(main.app).get(f"/history/{run_id}")
    assert history_response.status_code == 200
    _assert_quality_consistency(history_response.json())


# ============================================================
# GROUNDING/COVERAGE HARDENING REGRESSIONS
# ============================================================

def test_grounding_accepts_any_out_of_range_value_for_negative_bva_ep():
    """Any clearly out-of-range magnitude is a legitimate boundary
    representative, not an invented business rule; only the exact
    literal is arbitrary, and the requirement supplies the range fact."""
    requirement_text = "The withdrawal amount must be between 500 and 100000."
    test_cases = {
        "functional": [],
        "negative": [case("NTC01", "Negative", "Withdraw 42 which is far below the minimum.", "Withdrawal is rejected.")],
        "bva": [case("BVA01", "BVA", "", "Withdrawal is rejected.", scenario="Below minimum", values="10")],
        "ep": [case("EP01", "EP", "", "Withdrawal is rejected.", **{"class": "Below minimum", "input": "999999"})],
        "edge_cases": [],
    }

    result = main.validate_requirement_grounding(requirement_text, test_cases)

    assert result["grounded"] is True
    assert result["findings"] == []


def test_grounding_still_rejects_unsupported_values_outside_negative_bva_ep():
    """Functional/edge-case categories are not boundary-testing categories,
    so an out-of-range number there is still flagged as unsupported."""
    requirement_text = "The withdrawal amount must be between 500 and 100000."
    test_cases = {
        "functional": [case("TC01", "Functional", "Withdraw 42 successfully.", "Withdrawal succeeds.")],
        "negative": [], "bva": [], "ep": [], "edge_cases": [],
    }

    result = main.validate_requirement_grounding(requirement_text, test_cases)

    assert result["grounded"] is False
    assert result["findings"][0]["category"] == "invented_values"


def test_grounding_scopes_derived_values_per_declared_range():
    """Documents current conservative behavior: a value that matches ANY
    declared numeric range is treated as supported, even when the case text
    concerns a different condition. This is a known limitation, not a
    silent invented-value bypass, because both ranges still come from the
    requirement itself."""
    requirement_text = (
        "The transfer amount must be between 10 and 500. "
        "The daily transfer limit must be between 1000 and 5000."
    )
    test_cases = {
        "functional": [],
        "negative": [case(
            "NTC01", "Negative", "Transfer amount of 1001, which belongs to the daily limit range.",
            "Transfer is rejected.",
        )],
        "bva": [], "ep": [], "edge_cases": [],
    }

    result = main.validate_requirement_grounding(requirement_text, test_cases)

    assert result["grounded"] is True


def test_grounding_derives_bare_minimum_maximum_from_range_without_literal_words():
    """Auto-generated BVA scenario labels such as "Test minimum boundary
    value." must not be flagged just because the requirement phrases its
    range as "between X and Y" instead of using the words minimum/maximum."""
    requirement_text = "The withdrawal amount must be between 500 and 100000."
    test_cases = main._normalize_numeric_bva_cases(
        {"bva": [], "functional": [], "negative": [], "ep": [], "edge_cases": []},
        requirement_text,
        "R01",
    )

    result = main.validate_requirement_grounding(requirement_text, test_cases)

    assert result["grounded"] is True
    assert result["findings"] == []


def test_grounding_number_extraction_tolerates_units_attached_to_numbers():
    """Units like "days" attached to a number must not prevent the number
    from being recognized as requirement-supported."""
    requirement_text = "The report must be generated within 30 days."
    test_cases = {
        "functional": [case("TC01", "Functional", "Generate the report within 30 days.", "The report is generated on time.")],
        "negative": [], "bva": [], "ep": [], "edge_cases": [],
    }

    result = main.validate_requirement_grounding(requirement_text, test_cases)

    assert result["grounded"] is True


def test_coverage_matches_negated_paraphrase_of_condition():
    """A negated word (invalid/valid) sharing the same root as a condition
    word should still count toward the required multi-token overlap."""
    requirement = {"id": "R01", "conditions": ["The password must be valid."]}
    test_cases = {
        "functional": [], "bva": [], "ep": [], "edge_cases": [],
        "negative": [case("NTC01", "Negative", "An invalid password results in rejection.", "The password is rejected.")],
    }

    result = main.evaluate_requirement_coverage(requirement, requirement["conditions"][0], test_cases)

    assert result["covered_conditions"] == 1
    assert result["conditions"][0]["status"] == "covered"


def test_coverage_still_requires_condition_specific_evidence_for_unrelated_case():
    """The negation-aware match must not turn into a generic pass: an
    amount-only case still does not cover an unrelated balance condition."""
    requirement = {
        "id": "R01",
        "conditions": [
            "The withdrawal amount is between 500 and 100000.",
            "The account must have sufficient balance.",
        ],
    }
    test_cases = {
        "functional": [], "negative": [], "ep": [], "edge_cases": [],
        "bva": [case("BVA01", "BVA", "", "Withdrawal is accepted.", scenario="Withdrawal amount at minimum", values="500")],
    }

    result = main.evaluate_requirement_coverage(
        requirement,
        "The ATM withdrawal amount must be between 500 and 100000 and the account must have sufficient balance.",
        test_cases,
    )

    balance_condition = next(c for c in result["conditions"] if "sufficient balance" in c["text"].lower())
    assert balance_condition["status"] == "uncovered"


def test_grounding_attributes_findings_to_correct_category_for_compound_requirement():
    """A requirement combining a numeric range and a required format must
    have each unsupported fact attributed to its own finding category."""
    requirement_text = "The transfer amount must be between 10 and 500."
    test_cases = {
        "functional": [case(
            "TC01", "Functional",
            "Transfer 10000 using an email confirmation.",
            "Transfer is processed.",
        )],
        "negative": [], "bva": [], "ep": [], "edge_cases": [],
    }

    result = main.validate_requirement_grounding(requirement_text, test_cases)

    categories = {finding["category"] for finding in result["findings"]}
    assert categories == {"invented_values", "invented_formats"}


def test_direct_to_range_is_parsed_and_bva_is_applicable():
    requirement_text = "The password must contain 8 to 20 characters."

    assert main.extract_numeric_ranges(requirement_text) == [(8, 20)]
    assert main._extract_bva_boundary(requirement_text) == {
        "kind": "range",
        "low": 8,
        "high": 20,
    }
    assert main.evaluate_bva(
        {"functional": [], "negative": [], "bva": [{"values": "7"}], "ep": [], "edge_cases": []},
        requirement_text,
    )["applicable"] is True


def test_derived_range_values_are_grounded_for_bva_and_ep():
    requirement_text = "The password must contain 8 to 20 characters."
    test_cases = {
        "functional": [],
        "negative": [],
        "bva": [
            {"id": f"BVA{index:02d}", "values": str(value), "scenario": "Password length boundary", "expected": "Accepted."}
            for index, value in enumerate([7, 8, 9, 15, 19, 20, 21], start=1)
        ],
        "ep": [],
        "edge_cases": [],
    }

    result = main.validate_requirement_grounding(requirement_text, test_cases)

    assert result["grounded"] is True
    assert result["findings"] == []


def test_merge_test_cases_deduplicates_password_minimum_boundary_across_rounds():
    """Regression for the reported duplicate-BVA bug: each agent refinement
    round normalizes only its own additions, so re-generated boundary
    values with different wording must still collapse to exactly one case
    per canonical position (7, 8, 9) once merged into the accumulated set,
    instead of surviving as text-based near-duplicates."""
    requirement_text = "The password must contain at least 8 characters and an uppercase letter."

    round_one = main._merge_test_cases(
        None,
        {
            "functional": [], "negative": [], "ep": [], "edge_cases": [],
            "bva": [
                {"id": "BVA01", "type": "BVA", "scenario": "Below minimum", "values": "7", "expected": "Rejected."},
                {"id": "BVA02", "type": "BVA", "scenario": "Above minimum", "values": "9", "expected": "Accepted."},
            ],
        },
        "R01",
        requirement_text,
    )

    assert [c["values"] for c in round_one["bva"]] == ["7", "8", "9"]

    # A second refinement round regenerates the same boundary positions
    # with different wording (the exact symptom reported: 7, 9, 7, 8, 9).
    round_two = main._merge_test_cases(
        round_one,
        {
            "functional": [], "negative": [], "ep": [], "edge_cases": [],
            "bva": [
                {"id": "BVA01", "type": "BVA", "scenario": "Test value below minimum boundary.", "values": "7", "expected": "Rejected."},
                {"id": "BVA02", "type": "BVA", "scenario": "Minimum valid value.", "values": "8", "expected": "Accepted."},
                {"id": "BVA03", "type": "BVA", "scenario": "Test value just above minimum.", "values": "9", "expected": "Accepted."},
            ],
        },
        "R01",
        requirement_text,
    )

    values = [c["values"] for c in round_two["bva"]]
    assert values == ["7", "8", "9"]
    assert len(values) == len(set(values))
    assert [c["expected"] for c in round_two["bva"]] == ["Rejected", "Accepted", "Accepted"]
    assert [c["id"] for c in round_two["bva"]] == ["BVA01", "BVA02", "BVA03"]


def test_merge_test_cases_still_produces_atm_six_position_range_without_regression():
    """Regression guard: the merge-time BVA renormalization must not change
    behavior for an explicit range requirement (500..100000); the six
    canonical positions must remain intact and deterministic across
    merges."""
    requirement_text = "The ATM withdrawal amount must be between 500 and 100000."

    round_one = main._merge_test_cases(
        None,
        {
            "functional": [], "negative": [], "ep": [], "edge_cases": [],
            "bva": [
                {"id": "BVA01", "type": "BVA", "scenario": "Below minimum", "values": "499", "expected": "Rejected."},
                {"id": "BVA02", "type": "BVA", "scenario": "Minimum", "values": "500", "expected": "Accepted."},
            ],
        },
        "R01",
        requirement_text,
    )
    round_two = main._merge_test_cases(
        round_one,
        {
            "functional": [], "negative": [], "ep": [], "edge_cases": [],
            "bva": [
                {"id": "BVA01", "type": "BVA", "scenario": "Just above minimum", "values": "501", "expected": "Accepted."},
                {"id": "BVA02", "type": "BVA", "scenario": "Just below maximum", "values": "99999", "expected": "Accepted."},
                {"id": "BVA03", "type": "BVA", "scenario": "Maximum", "values": "100000", "expected": "Accepted."},
                {"id": "BVA04", "type": "BVA", "scenario": "Above maximum", "values": "100001", "expected": "Rejected."},
            ],
        },
        "R01",
        requirement_text,
    )

    values = [c["values"] for c in round_two["bva"]]
    assert values == ["499", "500", "501", "99999", "100000", "100001"]
    assert len(values) == len(set(values))
    assert [c["expected"] for c in round_two["bva"]] == [
        "Rejected", "Accepted", "Accepted", "Accepted", "Accepted", "Rejected",
    ]


def test_password_functional_and_negative_cover_length_and_uppercase_conditions():
    """The uppercase-letter condition is not a numeric boundary and must be
    covered through functional/negative cases rather than BVA, while the
    length condition remains separately covered."""
    requirement_text = "The password must contain at least 8 characters and an uppercase letter."
    requirement = {
        "id": "R01",
        "conditions": [
            "The password must contain at least 8 characters.",
            "The password must contain an uppercase letter.",
        ],
    }
    test_cases = {
        "functional": [{
            "id": "TC01", "requirement_id": "R01", "type": "Functional",
            "description": "Enter a password with at least 8 characters and an uppercase letter.",
            "expected": "The password is accepted.",
        }],
        "negative": [
            {
                "id": "NTC01", "requirement_id": "R01", "type": "Negative",
                "description": "Enter a password with fewer than 8 characters.",
                "expected": "The password is rejected.",
            },
            {
                "id": "NTC02", "requirement_id": "R01", "type": "Negative",
                "description": "Enter a password with 8 or more characters but no uppercase letter.",
                "expected": "The password is rejected.",
            },
        ],
        "bva": [
            {"id": "BVA01", "requirement_id": "R01", "type": "BVA", "scenario": "Below minimum", "values": "7", "expected": "Rejected."},
            {"id": "BVA02", "requirement_id": "R01", "type": "BVA", "scenario": "Minimum valid value", "values": "8", "expected": "Accepted."},
            {"id": "BVA03", "requirement_id": "R01", "type": "BVA", "scenario": "Just above minimum", "values": "9", "expected": "Accepted."},
        ],
        "ep": [],
        "edge_cases": [],
    }

    coverage = main.evaluate_requirement_coverage(requirement, requirement_text, test_cases)

    assert coverage["total_conditions"] == 2
    assert coverage["covered_conditions"] == 2
    assert coverage["percentage"] == 100.0
    assert all(c["status"] == "covered" for c in coverage["conditions"])


def test_run_agent_workflow_deduplicates_prior_password_boundary_and_keeps_no_gaps(monkeypatch):
    """End-to-end regression: an existing test-case set already carrying the
    exact reported duplicate BVA symptom (7, 9, 7, 8, 9) must come out of
    run_agent_workflow with the canonical, deduplicated 7/8/9 positions,
    without needing any model call, and without dropping the
    functional/negative cases covering the uppercase-letter condition."""
    requirement_text = "The password must contain at least 8 characters and an uppercase letter."
    requirement = {
        "id": "R01",
        "conditions": [
            "The password must contain at least 8 characters.",
            "The password must contain an uppercase letter.",
        ],
    }
    existing_cases = {
        "functional": [{
            "id": "TC01", "requirement_id": "R01", "type": "Functional",
            "description": "Enter a password with at least 8 characters and an uppercase letter.",
            "expected": "The password is accepted.",
        }],
        "negative": [
            {
                "id": "NTC01", "requirement_id": "R01", "type": "Negative",
                "description": "Enter a password with fewer than 8 characters.",
                "expected": "The password is rejected.",
            },
            {
                "id": "NTC02", "requirement_id": "R01", "type": "Negative",
                "description": "Enter a password with 8 or more characters but no uppercase letter.",
                "expected": "The password is rejected.",
            },
        ],
        "bva": [
            {"id": "BVA01", "requirement_id": "R01", "type": "BVA", "scenario": "Below minimum", "values": "7", "expected": "Rejected."},
            {"id": "BVA02", "requirement_id": "R01", "type": "BVA", "scenario": "Above minimum", "values": "9", "expected": "Accepted."},
            {"id": "BVA03", "requirement_id": "R01", "type": "BVA", "scenario": "Test value below minimum boundary.", "values": "7", "expected": "Rejected."},
            {"id": "BVA04", "requirement_id": "R01", "type": "BVA", "scenario": "Minimum valid value.", "values": "8", "expected": "Accepted."},
            {"id": "BVA05", "requirement_id": "R01", "type": "BVA", "scenario": "Test value just above minimum.", "values": "9", "expected": "Accepted."},
        ],
        "ep": [
            {"id": "EP01", "requirement_id": "R01", "type": "EP", "class": "Valid password", "input": "password with 8 characters and an uppercase letter", "expected": "Accepted."},
            {"id": "EP02", "requirement_id": "R01", "type": "EP", "class": "Invalid password", "input": "password with 7 characters and no uppercase letter", "expected": "Rejected."},
        ],
        "edge_cases": [],
    }

    # Any model call would indicate the merge-level fix failed to resolve
    # the duplicate BVA cases without agent involvement.
    def _fail_if_called(**kwargs):
        raise AssertionError("No model call should be needed once BVA positions are deduplicated.")

    monkeypatch.setattr(
        main, "client", SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_fail_if_called))),
    )

    result = main.run_agent_workflow(requirement_text, existing_cases, requirement)

    assert result["status"] == "completed"
    final_cases = result["result"]["test_cases"]
    bva_values = [c["values"] for c in final_cases["bva"]]

    assert bva_values == ["7", "8", "9"]
    assert len(bva_values) == len(set(bva_values))
    assert [c["expected"] for c in final_cases["bva"]] == ["Rejected", "Accepted", "Accepted"]
    assert [c["id"] for c in final_cases["bva"]] == ["BVA01", "BVA02", "BVA03"]
    assert len(final_cases["functional"]) == 1
    assert len(final_cases["negative"]) == 2
    assert any("uppercase" in c["description"].lower() for c in final_cases["negative"])
