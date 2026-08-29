"""HTTP client functions for the TraceAI frontend."""

from __future__ import annotations

import requests


BACKEND_ROOT_URL = "http://127.0.0.1:9022"


def fetch_history():
    response = requests.get(f"{BACKEND_ROOT_URL}/history/", timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_history_run(generation_run_id):
    response = requests.get(
        f"{BACKEND_ROOT_URL}/history/{generation_run_id}",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def run_agent_workflow(requirement, test_cases):
    response = requests.post(
        f"{BACKEND_ROOT_URL}/agent/",
        json={
            "intent": "complete",
            "text": requirement.get("text", ""),
            "requirement": requirement,
            "test_cases": test_cases,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def create_jira_issue(test_case, requirement_text, issue_type):
    response = requests.post(
        f"{BACKEND_ROOT_URL}/jira/create-issue",
        json={
            "test_case": test_case,
            "requirement_text": requirement_text,
            "category": issue_type,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()
