"""Streamlit renderers for agent workflow and Jira actions."""

from __future__ import annotations

import requests
import streamlit as st

try:
    from frontend.api_client import create_jira_issue
    from frontend.rendering_utils import render_section_title
except ModuleNotFoundError:
    from api_client import create_jira_issue
    from rendering_utils import render_section_title


def render_agent_flow(agent_result):
    render_section_title("TraceAI Test Analyst Agent")
    if not isinstance(agent_result, dict):
        st.warning("The agent workflow did not return a valid response.")
        return

    workflow_result = agent_result.get("result", {})
    analysis = workflow_result.get("analysis", {}) if isinstance(workflow_result, dict) else {}
    gaps = workflow_result.get("identified_gaps", []) if isinstance(workflow_result, dict) else []
    if not gaps:
        gaps = analysis.get("gaps", []) if isinstance(analysis, dict) else []
    refinement_count = workflow_result.get("refinement_count", 0) if isinstance(workflow_result, dict) else 0

    with st.container(border=True):
        step_columns = st.columns(3)
        with step_columns[0]:
            st.success("Requirement analyzed")
        with step_columns[1]:
            st.success("Test cases generated")
        with step_columns[2]:
            st.success("Coverage evaluated")

        if gaps:
            gap_text = "; ".join(
                str(gap.get("message", gap.get("description", gap)))
                if isinstance(gap, dict) else str(gap)
                for gap in gaps
            )
            st.warning(f"Coverage gaps detected: {gap_text}")
        else:
            st.info("No coverage gaps were identified.")
        if refinement_count:
            st.info(f"Agent refinement performed ({refinement_count} pass)")

        if agent_result.get("status") == "completed":
            st.success("Final validation completed")
        elif agent_result.get("status") == "needs_review":
            st.warning("Final validation needs review")
        else:
            st.error(agent_result.get("error", "Agent workflow failed."))


def render_jira_controls(requirement_text, test_cases):
    render_section_title("Jira Integration")
    case_options = {
        f"{case.get('id', 'Unknown')} - {case.get('description', case.get('scenario', 'Test case'))}": case
        for category in ("functional", "negative", "bva", "ep", "edge_cases")
        for case in test_cases.get(category, [])
        if isinstance(case, dict)
    }
    if not case_options:
        st.info("No test cases are available for Jira integration.")
        return

    with st.container(border=True):
        st.caption("Create a Jira issue directly from a generated test case.")
        select_col, type_col = st.columns([3, 1])
        with select_col:
            selected_label = st.selectbox("Select a test case to create a Jira issue", list(case_options))
        with type_col:
            issue_type = st.selectbox("Issue type", ["Task", "Bug", "Story"])

        if st.button(
            "Create Jira issue",
            icon=":material/bug_report:",
            key="create_jira_issue",
            type="primary",
        ):
            try:
                jira_result = create_jira_issue(
                    case_options[selected_label],
                    requirement_text,
                    issue_type,
                )
                if jira_result.get("success"):
                    st.success(f"Jira issue created successfully: {jira_result.get('issue_key', '')}")
                    if jira_result.get("issue_url"):
                        st.link_button("Open in Jira", jira_result["issue_url"], icon=":material/open_in_new:")
                else:
                    st.info(jira_result.get("message", "Jira is not configured."))
                    if jira_result.get("missing_config"):
                        st.caption(
                            "Set these environment variables and restart the backend: "
                            + ", ".join(jira_result["missing_config"])
                        )
                    if jira_result.get("error_details"):
                        with st.expander("View Jira error details"):
                            st.code(jira_result["error_details"])
            except requests.exceptions.RequestException as error:
                st.error(f"Jira request failed: {error}")
