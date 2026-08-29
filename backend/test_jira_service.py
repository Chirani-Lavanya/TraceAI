import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

import main
from backend.jira_service import JiraService


TEST_CASE = {
    "id": "TC01",
    "requirement_id": "R01",
    "type": "Functional",
    "description": "Submit valid credentials.",
    "expected": "The user is authenticated.",
}


def configure_jira(monkeypatch):
    monkeypatch.setenv("JIRA_BASE_URL", "https://traceai.atlassian.net")
    monkeypatch.setenv("JIRA_EMAIL", "qa@example.com")
    monkeypatch.setenv("JIRA_API_TOKEN", "token")
    monkeypatch.setenv("JIRA_PROJECT_KEY", "TRACE")


def stub_issue_type_lookup(monkeypatch, issue_types=None, status_code=200, captured_urls=None):
    """Mock the issue-type-metadata GET call used to resolve issue-type IDs.
    Defaults to no matching issue types, so the caller falls back to the
    prior name-based behavior unless a test explicitly supplies mappings.
    """
    response = SimpleNamespace(
        status_code=status_code,
        text="",
        json=lambda: {"issueTypes": issue_types or []},
    )

    def fake_get(url, *args, **kwargs):
        if captured_urls is not None:
            captured_urls.append(url)
        return response

    monkeypatch.setattr("backend.jira_service.requests.get", fake_get)


def test_create_issue_returns_issue_details(monkeypatch):
    configure_jira(monkeypatch)
    stub_issue_type_lookup(monkeypatch)
    response = SimpleNamespace(
        status_code=201,
        text="",
        json=lambda: {"key": "TRACE-123", "id": "10001"},
    )
    monkeypatch.setattr("backend.jira_service.requests.post", lambda *args, **kwargs: response)

    result = JiraService().create_issue(TEST_CASE, "Users can log in.")

    assert result["success"] is True
    assert result["issue_key"] == "TRACE-123"
    assert result["issue_url"].endswith("/browse/TRACE-123")


def test_create_issue_preserves_requirement_and_case_ids_in_payload(monkeypatch):
    """The Jira summary/description must retain the requirement ID and
    test-case ID, plus the case description and expected result."""
    configure_jira(monkeypatch)
    stub_issue_type_lookup(monkeypatch)
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return SimpleNamespace(status_code=201, text="", json=lambda: {"key": "TRACE-123", "id": "10001"})

    monkeypatch.setattr("backend.jira_service.requests.post", fake_post)

    result = JiraService().create_issue(TEST_CASE, "Users can log in.", category="Bug")

    assert result["success"] is True
    fields = captured["json"]["fields"]
    assert fields["issuetype"]["name"] == "Bug"
    assert "R01" in fields["summary"]
    assert "TC01" in fields["summary"]

    description_text = json.dumps(fields["description"])
    assert "R01" in description_text
    assert "TC01" in description_text
    assert "Submit valid credentials." in description_text
    assert "The user is authenticated." in description_text

    auth_header = captured["headers"]["Authorization"]
    assert auth_header.startswith("Basic ")
    assert "qa@example.com" not in auth_header
    assert "token" not in auth_header


def test_create_issue_maps_category_to_project_issue_type_id(monkeypatch):
    """When the configured project defines a matching issue type, its real
    ID must be sent instead of an assumed display name, since Jira rejects
    names that don't exactly match the project's configured types."""
    configure_jira(monkeypatch)
    stub_issue_type_lookup(monkeypatch, issue_types=[
        {"id": "10001", "name": "Task"},
        {"id": "10002", "name": "Bug"},
        {"id": "10003", "name": "Story"},
    ])
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["json"] = json
        return SimpleNamespace(status_code=201, text="", json=lambda: {"key": "TRACE-124", "id": "10002"})

    monkeypatch.setattr("backend.jira_service.requests.post", fake_post)

    result = JiraService().create_issue(TEST_CASE, "Users can log in.", category="Bug")

    assert result["success"] is True
    assert captured["json"]["fields"]["issuetype"] == {"id": "10002"}


def test_create_issue_queries_createmeta_issuetypes_endpoint_for_project(monkeypatch):
    """The issue-type lookup must use the createmeta issuetypes endpoint
    (which Jira reliably populates for team-managed projects like KAN),
    not the plain project endpoint, which does not reliably expose
    issueTypes for team-managed projects and caused 'Specify a valid issue
    type' even though authentication and the project key were correct."""
    configure_jira(monkeypatch)
    urls = []
    stub_issue_type_lookup(monkeypatch, issue_types=[
        {"id": "10002", "name": "Bug"},
    ], captured_urls=urls)
    monkeypatch.setattr(
        "backend.jira_service.requests.post",
        lambda *args, **kwargs: SimpleNamespace(status_code=201, text="", json=lambda: {"key": "TRACE-126", "id": "10002"}),
    )

    result = JiraService().create_issue(TEST_CASE, "Users can log in.", category="Bug")

    assert result["success"] is True
    assert urls == ["https://traceai.atlassian.net/rest/api/3/issue/createmeta/TRACE/issuetypes"]


def test_create_issue_falls_back_to_name_when_issue_type_lookup_unavailable(monkeypatch):
    """If the project's issue types cannot be retrieved (e.g. network error
    or non-200 response), issue creation must still proceed using the
    category name, preserving prior behavior instead of failing outright."""
    configure_jira(monkeypatch)
    stub_issue_type_lookup(monkeypatch, status_code=500)
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["json"] = json
        return SimpleNamespace(status_code=201, text="", json=lambda: {"key": "TRACE-125", "id": "10003"})

    monkeypatch.setattr("backend.jira_service.requests.post", fake_post)

    result = JiraService().create_issue(TEST_CASE, "Users can log in.", category="Bug")

    assert result["success"] is True
    assert captured["json"]["fields"]["issuetype"] == {"name": "Bug"}


def test_create_issue_maps_bug_to_alias_when_project_has_no_bug_type(monkeypatch):
    """Regression for the real KAN project: a team-managed project can be
    missing a literal "Bug" issue type (only Epic/Subtask/Task/Story). The
    "Bug" category must resolve to a semantic alias (e.g. Task) instead of
    sending an issue-type name Jira will reject."""
    configure_jira(monkeypatch)
    stub_issue_type_lookup(monkeypatch, issue_types=[
        {"id": "10006", "name": "Epic"},
        {"id": "10007", "name": "Subtask"},
        {"id": "10008", "name": "Task"},
        {"id": "10009", "name": "Story"},
    ])
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["json"] = json
        return SimpleNamespace(status_code=201, text="", json=lambda: {"key": "KAN-11", "id": "10101"})

    monkeypatch.setattr("backend.jira_service.requests.post", fake_post)

    result = JiraService().create_issue(TEST_CASE, "Users can log in.", category="Bug")

    assert result["success"] is True
    assert captured["json"]["fields"]["issuetype"] == {"id": "10008"}


def test_create_issue_reports_invalid_issue_type_from_jira(monkeypatch):
    """Jira's 'Specify a valid issue type' error must still surface as a
    specific, actionable message even after the ID-mapping change."""
    configure_jira(monkeypatch)
    stub_issue_type_lookup(monkeypatch)
    body = {"errorMessages": [], "errors": {"issuetype": "Specify a valid issue type"}}
    response = SimpleNamespace(status_code=400, text="ignored", json=lambda: body)
    monkeypatch.setattr("backend.jira_service.requests.post", lambda *args, **kwargs: response)

    result = JiraService().create_issue(TEST_CASE, "Users can log in.", category="Bug")

    assert result["success"] is False
    assert "issuetype: Specify a valid issue type" in result["message"]


def test_create_issue_handles_jira_api_error(monkeypatch):
    configure_jira(monkeypatch)
    stub_issue_type_lookup(monkeypatch)
    response = SimpleNamespace(status_code=400, text="Invalid project", json=lambda: {})
    monkeypatch.setattr("backend.jira_service.requests.post", lambda *args, **kwargs: response)

    result = JiraService().create_issue(TEST_CASE, "Users can log in.")

    assert result == {
        "success": False,
        "message": "Jira rejected the request. Check the project key and issue type.",
        "error_details": "Invalid project",
    }


def test_create_issue_reports_required_field_errors(monkeypatch):
    configure_jira(monkeypatch)
    stub_issue_type_lookup(monkeypatch)
    body = {"errorMessages": [], "errors": {"reporter": "Reporter is required."}}
    response = SimpleNamespace(status_code=400, text="ignored", json=lambda: body)
    monkeypatch.setattr("backend.jira_service.requests.post", lambda *args, **kwargs: response)

    result = JiraService().create_issue(TEST_CASE, "Users can log in.")

    assert result["success"] is False
    assert "reporter: Reporter is required." in result["message"]


def test_create_issue_distinguishes_auth_failure(monkeypatch):
    configure_jira(monkeypatch)
    stub_issue_type_lookup(monkeypatch, status_code=401)
    response = SimpleNamespace(status_code=401, text="Unauthorized", json=lambda: {})
    monkeypatch.setattr("backend.jira_service.requests.post", lambda *args, **kwargs: response)

    result = JiraService().create_issue(TEST_CASE, "Users can log in.")

    assert result["success"] is False
    assert "authentication failed" in result["message"].lower()


def test_create_issue_handles_missing_configuration(monkeypatch):
    for variable in ("JIRA_BASE_URL", "JIRA_EMAIL", "JIRA_API_TOKEN", "JIRA_PROJECT_KEY"):
        monkeypatch.delenv(variable, raising=False)

    result = JiraService().create_issue(TEST_CASE, "Users can log in.")

    assert result["success"] is False
    assert result["message"].startswith("Jira is not configured")
    assert set(result["missing_config"]) == {
        "JIRA_BASE_URL",
        "JIRA_EMAIL",
        "JIRA_API_TOKEN",
        "JIRA_PROJECT_KEY",
    }


def test_jira_endpoint_delegates_to_service(monkeypatch):
    expected = {
        "success": True,
        "issue_key": "TRACE-123",
        "issue_url": "https://traceai.atlassian.net/browse/TRACE-123",
    }

    class StubJiraService:
        def create_issue(self, test_case, requirement_text, category):
            assert test_case == TEST_CASE
            assert requirement_text == "Users can log in."
            assert category == "Task"
            return expected

    monkeypatch.setattr(main, "JiraService", StubJiraService)
    with TestClient(main.app) as client:
        response = client.post(
            "/jira/create-issue",
            json={
                "test_case": TEST_CASE,
                "requirement_text": "Users can log in.",
                "category": "Task",
            },
        )

    assert response.status_code == 200
    assert response.json() == expected


def test_jira_endpoint_passes_selected_issue_type(monkeypatch):
    class StubJiraService:
        def create_issue(self, test_case, requirement_text, category):
            assert category == "Bug"
            return {"success": True, "issue_key": "TRACE-124"}

    monkeypatch.setattr(main, "JiraService", StubJiraService)
    with TestClient(main.app) as client:
        response = client.post(
            "/jira/create-issue",
            json={
                "test_case": TEST_CASE,
                "requirement_text": "Users can log in.",
                "category": "Bug",
            },
        )

    assert response.status_code == 200
    assert response.json()["issue_key"] == "TRACE-124"
