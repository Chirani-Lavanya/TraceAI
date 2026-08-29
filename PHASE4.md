# TraceAI Phase 4

## What was added

- `backend/jira_service.py` provides a small Jira Cloud REST integration using the existing `requests` dependency.
- `POST /jira/create-issue` accepts a test case, requirement text, and optional Jira issue type.
- The Streamlit result view now calls the existing `/agent/` workflow with generated cases as context and displays:
  - requirement analysis
  - generation
  - coverage evaluation
  - detected gaps
  - refinement passes
  - final validation status
- The Streamlit result view includes Jira issue creation for an individually selected test case.
- `backend/test_jira_service.py` covers successful issue creation, Jira API errors, missing configuration, and endpoint delegation.

## Agent architecture

The existing `run_agent_workflow()` remains the orchestration layer. It performs deterministic gap analysis, requests only targeted additions, validates those additions against grounding and structure rules, and allows one bounded refinement pass. The UI displays the returned status and analysis without replacing the existing generation or evaluation pipeline.

## Jira configuration

Set these environment variables before starting the backend:

```text
JIRA_BASE_URL=https://your-domain.atlassian.net
JIRA_EMAIL=your-jira-email@example.com
JIRA_API_TOKEN=your-jira-api-token
JIRA_PROJECT_KEY=TRACE
```

The Jira integration degrades gracefully when any variable is missing. No Jira request is sent until all four values are configured.

The Streamlit UI lets you pick the issue type (Task/Bug/Story) per issue. Some
Jira Cloud projects require additional fields beyond project/summary/
description/issuetype (most commonly `reporter`); if creation fails with a
field-required error, the returned message names the missing field, and
`JiraService.create_issue()` accepts an `extra_fields` dict for callers that
need to supply it.

## Demo workflow

1. Start the FastAPI backend on port `9022`.
2. Start the Streamlit app from the `frontend` directory.
3. Enter a requirement and select `Analyze & generate test cases`.
4. Review the Test Analyst Agent flow and any remaining coverage gaps.
5. Configure the Jira environment variables and restart the backend if needed.
6. Select a generated test case under Jira integration.
7. Select `Create Jira issue` and use the returned link to open the issue.

## Validation

The focused Jira tests pass (`4 passed`). The complete non-live suite passes (`67 passed` before the Jira test module was added); rerun the command below after all local changes:

```text
python -m pytest -q --ignore=backend/test_live.py
```
