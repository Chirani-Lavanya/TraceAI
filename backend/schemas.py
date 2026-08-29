"""FastAPI request models used by the TraceAI API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class InputText(BaseModel):
    text: str


class AgentRequest(BaseModel):
    """Bounded, stateless agent request contract."""

    intent: str
    text: Optional[str] = None
    generation_run_id: Optional[int] = None
    requirement: Optional[dict] = None
    test_cases: Optional[dict] = None


class JiraRequest(BaseModel):
    """Request to create a Jira issue from a test case."""

    test_case: dict
    requirement_text: str
    category: Optional[str] = "Task"
