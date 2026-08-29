import streamlit as st
import requests
import json
import hashlib
import pandas as pd

try:
    from frontend.api_client import (
        BACKEND_ROOT_URL,
        fetch_history,
        fetch_history_run,
        run_agent_workflow,
    )
    from frontend.rendering_utils import (
        format_score,
        get_evaluation_details,
        render_export_controls,
        render_section_title,
    )
    from frontend.workflow_ui import render_agent_flow, render_jira_controls
except ModuleNotFoundError:
    from api_client import (
        BACKEND_ROOT_URL,
        fetch_history,
        fetch_history_run,
        run_agent_workflow,
    )
    from rendering_utils import (
        format_score,
        get_evaluation_details,
        render_export_controls,
        render_section_title,
    )
    from workflow_ui import render_agent_flow, render_jira_controls


def _style_table_success(rows, column, is_success):
    """Return a Styler that subtly highlights cells where is_success(value) is True.

    Purely a presentation helper: it never changes the row data, only how
    matching cells in `column` are rendered.
    """
    frame = pd.DataFrame(rows)
    if frame.empty or column not in frame.columns:
        return frame

    def _highlight(value):
        return (
            "background-color: #e6f4ea; color: #1e7e34; font-weight: 600;"
            if is_success(value)
            else ""
        )

    return frame.style.map(_highlight, subset=[column])


def _is_perfect_score(value):
    try:
        return float(str(value).rstrip("%")) >= 100
    except (TypeError, ValueError):
        return False


def render_bva_status(evaluation):
    details = evaluation.get("details", {}) if isinstance(evaluation, dict) else {}
    if "bva_applicable" not in details:
        return

    if details["bva_applicable"]:
        bva_message = details.get("bva_message", "BVA is applicable to this requirement.")
        if details.get("bva_quality") == 100:
            st.success(f"BVA applicable: {bva_message}")
        else:
            st.warning(f"BVA applicable: {bva_message}")
    else:
        st.info("BVA is not applicable to this requirement.")


def render_history_result(result):
    requirement = result.get("requirement", {})
    evaluation = result.get("evaluation", {})
    test_cases = result.get("test_cases", {})

    render_export_controls(result, f"history_{result.get('id', 'run')}")

    render_section_title("Historical Result")
    st.caption("Read-only view of a persisted generation run (not the current live session).")
    render_section_title("Requirement Analysis")
    with st.container(border=True):
        metric1, metric2, metric3 = st.columns(3)
        with metric1:
            st.metric("Requirement ID", requirement.get("id", "R01"))
        with metric2:
            st.metric("Type", requirement.get("type", "Not specified"))
        with metric3:
            st.metric("Priority", requirement.get("priority", "Not specified"))

        st.info(requirement.get("text", ""))

    if requirement.get("conditions"):
        st.write("**Conditions:**")
        for condition in requirement["conditions"]:
            st.write(f"• {condition}")

    if requirement.get("constraints"):
        st.write("**Constraints:**")
        for constraint in requirement["constraints"]:
            st.write(f"• {constraint}")

    if requirement.get("inputs"):
        st.write("**Inputs:**")
        for item in requirement["inputs"]:
            st.write(f"• {item}")

    if requirement.get("expected_behavior"):
        st.write("**Expected Behaviour:**")
        st.success(requirement["expected_behavior"])

    st.divider()
    render_section_title("Score Dashboard")
    evaluation_data = get_evaluation_details(evaluation)
    requirement_coverage = evaluation_data["requirement_coverage"]
    metrics = [
        ("Test Design Coverage", evaluation_data["test_design_coverage"]),
        ("Requirement Traceability", evaluation_data["requirement_traceability"]),
        (
            "Requirement Coverage",
            requirement_coverage.get("percentage")
            if isinstance(requirement_coverage, dict)
            else None,
        ),
    ]
    if evaluation_data["ai_quality_score"] is not None:
        metrics.append(("AI Quality Score", evaluation_data["ai_quality_score"]))
    if evaluation_data["requirement_grounding_score"] is not None:
        metrics.append(("Requirement Grounding", evaluation_data["requirement_grounding_score"]))

    dashboard = st.container(border=True)
    metric_columns = dashboard.columns(len(metrics))
    for column, (label, value) in zip(metric_columns, metrics):
        with column:
            st.metric(label, format_score(value))

    render_bva_status(evaluation)

    if isinstance(requirement_coverage, dict) and requirement_coverage.get("conditions"):
        total_conditions = requirement_coverage.get("total_conditions", 0)
        covered_conditions = requirement_coverage.get("covered_conditions", 0)
        uncovered_conditions = requirement_coverage.get("uncovered_conditions", 0)
        render_section_title("Requirement Condition Coverage")
        coverage_summary = (
            f"{covered_conditions} of {total_conditions} requirement condition(s) covered "
            f"({uncovered_conditions} uncovered)."
        )
        if uncovered_conditions == 0:
            st.success(coverage_summary)
        else:
            st.warning(coverage_summary)
        condition_rows = [
            {
                "ID": condition.get("id", "RC00"),
                "Status": condition.get("status", "uncovered").title(),
                "Condition": condition.get("text", ""),
                "Covered By": ", ".join(condition.get("covered_by", [])) or "-",
            }
            for condition in requirement_coverage["conditions"]
        ]
        st.dataframe(
            _style_table_success(condition_rows, "Status", lambda value: value == "Covered"),
            width="stretch",
            hide_index=True,
        )

    quality_breakdown = evaluation_data["quality_breakdown"]
    if quality_breakdown:
        render_section_title("AI Quality Breakdown")
        breakdown_rows = [
            {
                "Evaluation": key.replace("_", " ").title(),
                "Score": format_score(value),
            }
            for key, value in quality_breakdown.items()
        ]
        st.dataframe(
            _style_table_success(breakdown_rows, "Score", _is_perfect_score),
            width="stretch",
            hide_index=True,
        )

    grounding = evaluation_data["requirement_grounding"]
    if isinstance(grounding, dict) and grounding.get("findings"):
        st.write("**Requirement Grounding Findings**")
        st.dataframe(
            [
                {
                    "Test Case": finding.get("test_case", "Unknown"),
                    "Category": finding.get("category", "").replace("_", " ").title(),
                    "Finding": finding.get("message", ""),
                }
                for finding in grounding["findings"]
            ],
            width="stretch",
            hide_index=True,
        )

    st.divider()
    render_section_title("Test Cases")
    _history_counts_summary = " | ".join(
        f"{len(test_cases.get(key, []))} {label}"
        for key, label in [
            ("functional", "Functional"),
            ("negative", "Negative"),
            ("bva", "BVA"),
            ("ep", "EP"),
            ("edge_cases", "Edge Cases"),
        ]
        if test_cases.get(key)
    )
    if _history_counts_summary:
        st.caption(_history_counts_summary)
    for category, title in [
        ("functional", "✅ Functional Test Cases"),
        ("negative", "❌ Negative Test Cases"),
        ("bva", "📏 Boundary Value Analysis (BVA)"),
        ("ep", "🔀 Equivalence Partitioning (EP)"),
        ("edge_cases", "⚠️ Edge Cases"),
    ]:
        cases = test_cases.get(category, [])
        if not cases:
            continue
        st.subheader(title)
        for case in cases:
            case_title = case.get("scenario") or case.get("class") or case.get("description") or "Test case"
            with st.expander(f"{case.get('id', 'TC')} — {case_title}"):
                st.write(f"**Type:** {case.get('type', category)}")
                st.write(f"**Requirement ID:** {case.get('requirement_id', 'R01')}")
                if case.get("description"):
                    st.write("**Description:**")
                    st.write(case["description"])
                if case.get("scenario"):
                    st.write("**Scenario:**")
                    st.write(case["scenario"])
                if case.get("class"):
                    st.write("**Equivalence Class:**")
                    st.write(case["class"])
                if case.get("input"):
                    st.write("**Input:**")
                    st.write(case["input"])
                if case.get("values"):
                    st.write("**Boundary Value:**")
                    st.write(case["values"])
                if case.get("expected"):
                    st.write("**Expected Result:**")
                    st.write(case["expected"])

    st.divider()
    render_section_title("Requirement Traceability")
    traceability = result.get("traceability", [])
    if traceability:
        st.dataframe(traceability, width="stretch", hide_index=True)
    else:
        st.info("No traceable test cases were persisted for this generation run.")

    render_reanalysis_control(
        result,
        f"history_{result.get('id', 'run')}",
    )


# ============================================================
def _result_content_key(result: dict) -> str:
    """Stable key for caching agent analysis across Streamlit reruns.

    Export buttons, Jira actions, selectboxes, and other Streamlit widgets
    rerun the script. The requirement/test-case content is the correct cache
    identity; evaluation metadata is intentionally excluded so re-analysis
    does not trigger the agent workflow again.
    """
    payload = {
        "requirement": result.get("requirement", {}),
        "test_cases": result.get("test_cases", {}),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _get_cached_agent_result(result: dict) -> dict:
    """Run the agent workflow once per generated test-case set."""
    cache_key = _result_content_key(result)
    if st.session_state.get("agent_result_key") != cache_key:
        requirement = result.get("requirement", {})
        try:
            agent_result = run_agent_workflow(
                requirement,
                result.get("test_cases", {}),
            )
        except requests.exceptions.RequestException as error:
            agent_result = {
                "status": "error",
                "error": f"Agent workflow unavailable: {error}",
            }
        st.session_state["agent_result_key"] = cache_key
        st.session_state["agent_result"] = agent_result
    return st.session_state.get(
        "agent_result",
        {"status": "error", "error": "Agent result is unavailable."},
    )


# ============================================================
def render_generated_result(result, user_input=""):
    """Render the current TraceAI result and its actions."""
    render_section_title("Generated Result")
    st.caption("Review the requirement interpretation, quality signals, test cases, and traceability below.")
    render_export_controls(result, "current_result")

    # ==================================================
    # REQUIREMENT ANALYSIS
    # ==================================================

    requirement = result.get(
        "requirement",
        {}
    )

    # IMPORTANT: Streamlit reruns on Excel/PDF/Jira clicks. Cache the agent
    # workflow so those clicks never trigger another AI workflow run.
    agent_result = _get_cached_agent_result(result)
    render_agent_flow(agent_result)

    final_agent_result = agent_result.get("result", {})
    if (
        agent_result.get("status") == "completed"
        and isinstance(final_agent_result, dict)
        and isinstance(final_agent_result.get("evaluation"), dict)
    ):
        result = {
            **result,
            "test_cases": final_agent_result.get("test_cases", result.get("test_cases", {})),
            "evaluation": final_agent_result["evaluation"],
        }


    render_section_title("Requirement Analysis")

    # ------------------------------------------------
    # Requirement information
    # ------------------------------------------------

    with st.container(border=True):

        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "Requirement ID",
                requirement.get(
                    "id",
                    "R01"
                )
            )


        with col2:

            st.metric(
                "Type",
                requirement.get(
                    "type",
                    "Not specified"
                )
            )


        with col3:

            st.metric(
                "Priority",
                requirement.get(
                    "priority",
                    "Not specified"
                )
            )


        with col4:

            st.metric(
                "Source Type",
                requirement.get(
                    "source_type",
                    "requirement"
                ).replace("_", " ").title()
            )


        st.write(
            "**Original requirement:**"
        )

        st.info(
            requirement.get(
                "text",
                user_input
            )
        )


    # ------------------------------------------------
    # Conditions
    # ------------------------------------------------

    conditions = requirement.get(
        "conditions",
        []
    )

    if conditions:

        st.write(
            "**Conditions:**"
        )

        for condition in conditions:

            st.write(
                f"• {condition}"
            )


    # ------------------------------------------------
    # Constraints
    # ------------------------------------------------

    constraints = requirement.get(
        "constraints",
        []
    )

    if constraints:

        st.write(
            "**Constraints:**"
        )

        for constraint in constraints:

            st.write(
                f"• {constraint}"
            )


    # ------------------------------------------------
    # Inputs
    # ------------------------------------------------

    inputs = requirement.get(
        "inputs",
        []
    )

    if inputs:

        st.write(
            "**Inputs:**"
        )

        for item in inputs:

            st.write(
                f"• {item}"
            )


    # ------------------------------------------------
    # Expected Behaviour
    # ------------------------------------------------

    expected_behavior = requirement.get(
        "expected_behavior",
        ""
    )

    if expected_behavior:

        st.write(
            "**Expected Behaviour:**"
        )

        st.success(
            expected_behavior
        )


    st.divider()


    # ==================================================
    # TRACEAI EVALUATION
    # ==================================================

    render_section_title("Score Dashboard")

    evaluation = result.get(
        "evaluation",
        {}
    )

    # Supports both the upgraded backend and the
    # previous response format.
    evaluation_data = get_evaluation_details(
        evaluation
    )

    test_design_coverage = evaluation_data[
        "test_design_coverage"
    ]

    requirement_traceability = evaluation_data[
        "requirement_traceability"
    ]

    ai_quality_score = evaluation_data[
        "ai_quality_score"
    ]

    requirement_grounding_score = evaluation_data[
        "requirement_grounding_score"
    ]

    requirement_grounding = evaluation_data[
        "requirement_grounding"
    ]

    requirement_coverage = evaluation_data[
        "requirement_coverage"
    ]

    quality_breakdown = evaluation_data[
        "quality_breakdown"
    ]

    requirement_coverage_value = (
        requirement_coverage.get("percentage")
        if isinstance(requirement_coverage, dict)
        else None
    )

    metric_values = 3
    if ai_quality_score is not None:
        metric_values += 1
    if requirement_grounding_score is not None:
        metric_values += 1

    dashboard = st.container(border=True)
    metric_columns = dashboard.columns(metric_values)

    with metric_columns[0]:
        st.metric(
            "Test Design Coverage",
            format_score(test_design_coverage)
        )

    with metric_columns[1]:
        st.metric(
            "Requirement Traceability",
            format_score(requirement_traceability)
        )

    with metric_columns[2]:
        st.metric(
            "Requirement Coverage",
            format_score(requirement_coverage_value)
            if requirement_coverage_value is not None
            else "N/A"
        )

    metric_index = 3
    if ai_quality_score is not None:
        with metric_columns[metric_index]:
            st.metric(
                "AI Quality Score",
                format_score(ai_quality_score)
            )
        metric_index += 1

    if requirement_grounding_score is not None:
        with metric_columns[metric_index]:
            st.metric(
                "Requirement Grounding",
                format_score(requirement_grounding_score)
            )

    render_bva_status(evaluation)

    st.caption(
        "Test Design Coverage measures whether the requested "
        "test-design categories were generated. Requirement "
        "Traceability measures whether generated test cases "
        "are linked to the requirement. Requirement Coverage "
        "checks whether the explicit requirement conditions, "
        "constraints, inputs and expected behaviour are actually "
        "covered by the generated tests. AI Quality Score is "
        "a heuristic evaluation of structural and test-design "
        "quality; it is not a proof of semantic AI accuracy."
    )

    if isinstance(requirement_grounding, dict) and requirement_grounding.get("findings"):
        st.write("**Requirement Grounding Findings**")
        st.dataframe(
            [
                {
                    "Test Case": finding.get("test_case", "Unknown"),
                    "Category": finding.get("category", "").replace("_", " ").title(),
                    "Finding": finding.get("message", ""),
                }
                for finding in requirement_grounding["findings"]
            ],
            use_container_width=True,
            hide_index=True,
        )

    if isinstance(requirement_coverage, dict):
        total_conditions = requirement_coverage.get("total_conditions", 0)
        covered_conditions = requirement_coverage.get("covered_conditions", 0)
        uncovered_conditions = requirement_coverage.get("uncovered_conditions", 0)
        condition_rows = requirement_coverage.get("conditions", [])
        missing_recommendations = requirement_coverage.get("missing_recommendations", [])

        if total_conditions:
            render_section_title("Requirement Condition Coverage")
            coverage_summary = (
                f"{covered_conditions} of {total_conditions} requirement condition(s) covered "
                f"({uncovered_conditions} uncovered)."
            )
            if uncovered_conditions == 0:
                st.success(coverage_summary)
            else:
                st.warning(coverage_summary)

            condition_table = []
            for condition in condition_rows:
                condition_table.append({
                    "ID": condition.get("id", "RC00"),
                    "Status": condition.get("status", "uncovered").title(),
                    "Condition": condition.get("text", ""),
                    "Covered By": ", ".join(condition.get("covered_by", [])) or "-",
                })

            if condition_table:
                st.dataframe(
                    _style_table_success(condition_table, "Status", lambda value: value == "Covered"),
                    use_container_width=True,
                    hide_index=True,
                )

            if missing_recommendations:
                st.write("**Recommended Missing Requirement Tests**")
                for item in missing_recommendations[:5]:
                    st.info(item.get("recommended_test_case", item.get("condition", "")))

    st.divider()

    # --------------------------------------------------
    # Quality breakdown from the upgraded backend
    # --------------------------------------------------

    if quality_breakdown:

        render_section_title("AI Quality Breakdown")

        breakdown_labels = {
            "structure": "Structural Completeness",
            "traceability": "Traceability",
            "duplicates": "Duplicate-Free",
            "bva_quality": "BVA Quality",
            "ep_quality": "EP Quality",
            "category_completeness": "Category Completeness"
        }

        breakdown_rows = []

        for key, value in quality_breakdown.items():

            label = breakdown_labels.get(
                key,
                key.replace("_", " ").title()
            )

            breakdown_rows.append({
                "Evaluation": label,
                "Score": format_score(value)
            })

        if breakdown_rows:
            st.dataframe(
                _style_table_success(breakdown_rows, "Score", _is_perfect_score),
                use_container_width=True,
                hide_index=True
            )

    st.divider()


    # ==================================================
    # GET TEST CASES
    # ==================================================

    test_cases = result.get(
        "test_cases",
        {}
    )

    render_section_title("Test Cases")
    _category_labels = [
        ("functional", "Functional"),
        ("negative", "Negative"),
        ("bva", "BVA"),
        ("ep", "EP"),
        ("edge_cases", "Edge Cases"),
    ]
    _counts_summary = " | ".join(
        f"{len(test_cases.get(key, []))} {label}"
        for key, label in _category_labels
        if test_cases.get(key)
    )
    if _counts_summary:
        st.caption(_counts_summary)


    # ==================================================
    # FUNCTIONAL TEST CASES
    # ==================================================

    functional = test_cases.get(
        "functional",
        []
    )


    if functional:

        st.subheader(
            "✅ Functional Test Cases"
        )


        for tc in functional:

            with st.expander(
                f"{tc.get('id', 'TC')} "
                f"— "
                f"{tc.get('requirement_id', 'R01')}"
            ):

                st.write(
                    "**Type:** "
                    + tc.get(
                        "type",
                        "Functional"
                    )
                )


                st.write(
                    "**Requirement ID:** "
                    + tc.get(
                        "requirement_id",
                        "R01"
                    )
                )


                st.write(
                    "**Description:**"
                )

                st.write(
                    tc.get(
                        "description",
                        ""
                    )
                )


                st.write(
                    "**Expected Result:**"
                )

                st.success(
                    tc.get(
                        "expected",
                        ""
                    )
                )


    # ==================================================
    # NEGATIVE TEST CASES
    # ==================================================

    negative = test_cases.get(
        "negative",
        []
    )


    if negative:

        st.subheader(
            "❌ Negative Test Cases"
        )


        for tc in negative:

            with st.expander(
                f"{tc.get('id', 'NTC')} "
                f"— "
                f"{tc.get('requirement_id', 'R01')}"
            ):

                st.write(
                    "**Type:** Negative"
                )


                st.write(
                    "**Requirement ID:** "
                    + tc.get(
                        "requirement_id",
                        "R01"
                    )
                )


                st.write(
                    "**Description:**"
                )

                st.write(
                    tc.get(
                        "description",
                        ""
                    )
                )


                st.write(
                    "**Expected Result:**"
                )

                st.error(
                    tc.get(
                        "expected",
                        ""
                    )
                )


    # ==================================================
    # BVA
    # ==================================================

    bva = test_cases.get(
        "bva",
        []
    )


    if bva:

        st.subheader(
            "📏 Boundary Value Analysis (BVA)"
        )


        for item in bva:

            with st.expander(
                f"{item.get('id', 'BVA')} "
                f"— "
                f"{item.get('scenario', 'Boundary Test')}"
            ):

                st.write(
                    "**Requirement ID:** "
                    + item.get(
                        "requirement_id",
                        "R01"
                    )
                )


                st.write(
                    "**Boundary Value:**"
                )

                st.write(
                    item.get(
                        "values",
                        ""
                    )
                )

                if item.get("expected"):
                    st.write(
                        "**Expected Result:**"
                    )

                    st.write(
                        item.get(
                            "expected",
                            ""
                        )
                    )



    # ==================================================
    # EP
    # ==================================================

    ep = test_cases.get(
        "ep",
        []
    )


    if ep:

        st.subheader(
            "🔀 Equivalence Partitioning (EP)"
        )


        for item in ep:

            with st.expander(
                f"{item.get('id', 'EP')} "
                f"— "
                f"{item.get('class', 'Equivalence Class')}"
            ):

                st.write(
                    "**Requirement ID:** "
                    + item.get(
                        "requirement_id",
                        "R01"
                    )
                )


                st.write(
                    "**Equivalence Class:**"
                )

                st.write(
                    item.get(
                        "class",
                        ""
                    )
                )


                st.write(
                    "**Input:**"
                )

                st.write(
                    item.get(
                        "input",
                        ""
                    )
                )


                st.write(
                    "**Expected Result:**"
                )

                st.write(
                    item.get(
                        "expected",
                        ""
                    )
                )


    # ==================================================
    # EDGE CASES
    # ==================================================

    edge_cases = test_cases.get(
        "edge_cases",
        []
    )


    if edge_cases:

        st.subheader(
            "⚠️ Edge Cases"
        )

        st.caption(
            "Only requirement-supported edge cases are shown. "
            "TraceAI avoids inventing unspecified business rules."
        )

        for item in edge_cases:

            with st.expander(
                f"{item.get('id', 'EC')} "
                f"— "
                f"{item.get('requirement_id', 'R01')}"
            ):

                st.write(
                    "**Requirement ID:** "
                    + item.get(
                        "requirement_id",
                        "R01"
                    )
                )


                st.write(
                    "**Description:**"
                )

                st.write(
                    item.get(
                        "description",
                        ""
                    )
                )


                st.write(
                    "**Expected Result:**"
                )

                st.write(
                    item.get(
                        "expected",
                        ""
                    )
                )

    elif test_cases:
        st.subheader(
            "⚠️ Edge Cases"
        )

        st.info(
            "No additional requirement-supported edge cases "
            "were identified. This is valid when the supplied "
            "requirement does not define additional edge-case rules."
        )


    # ==================================================
    # TRACEABILITY MATRIX
    # ==================================================

    st.divider()

    render_section_title("Requirement Traceability")

    # Prefer the traceability rows prepared by the backend.
    backend_traceability = result.get("traceability", [])

    if backend_traceability:
        trace_rows = backend_traceability
    else:
        # Backward-compatible fallback for older backend responses.
        requirement_id = requirement.get(
            "id",
            "R01"
        )

        trace_rows = []

        for tc in functional:
            trace_rows.append({
                "Requirement ID": requirement_id,
                "Test Case": tc.get("id", ""),
                "Type": "Functional",
                "Description": tc.get("description", ""),
                "Status": "Covered"
            })

        for tc in negative:
            trace_rows.append({
                "Requirement ID": requirement_id,
                "Test Case": tc.get("id", ""),
                "Type": "Negative",
                "Description": tc.get("description", ""),
                "Status": "Covered"
            })

        for item in bva:
            trace_rows.append({
                "Requirement ID": requirement_id,
                "Test Case": item.get("id", ""),
                "Type": "BVA",
                "Description": item.get("scenario", ""),
                "Status": "Covered"
            })

        for item in ep:
            trace_rows.append({
                "Requirement ID": requirement_id,
                "Test Case": item.get("id", ""),
                "Type": "EP",
                "Description": item.get("class", ""),
                "Status": "Covered"
            })

        for item in edge_cases:
            trace_rows.append({
                "Requirement ID": requirement_id,
                "Test Case": item.get("id", ""),
                "Type": "Edge Case",
                "Description": item.get("description", ""),
                "Status": "Covered"
            })

    if trace_rows:

        st.caption(
            f"{len(trace_rows)} traceable test case(s) "
            "linked to the analyzed requirement."
        )

        st.dataframe(
            trace_rows,
            use_container_width=True,
            hide_index=True
        )

        # CSV export for the traceability matrix.
        trace_df = pd.DataFrame(trace_rows)

        st.download_button(
            label="⬇️ Download Traceability as CSV",
            data=trace_df.to_csv(index=False).encode("utf-8"),
            file_name="traceai_traceability.csv",
            mime="text/csv"
        )
    else:
        st.info(
            "No traceable test cases were generated."
        )


    # ==================================================
    # RAW JSON
    # ==================================================

    with st.expander(
        "🔍 View Raw TraceAI Response"
    ):

        st.json(
            result
        )

    render_jira_controls(
        requirement.get("text", user_input),
        result.get("test_cases", {}),
    )



# ============================================================
# CURRENT RESULT / RE-ANALYSIS
# ============================================================

def analyze_current_result(result):
    """Re-evaluate the existing test cases without generating or modifying them."""
    if not isinstance(result, dict):
        raise ValueError("No TraceAI result is available.")

    requirement = result.get("requirement", {})
    test_cases = result.get("test_cases", {})

    if not isinstance(requirement, dict):
        raise ValueError("The current requirement data is invalid.")
    if not isinstance(test_cases, dict):
        raise ValueError("The current test-case data is invalid.")

    requirement_text = str(requirement.get("text", "")).strip()
    if not requirement_text:
        raise ValueError("The current requirement text is missing.")

    response = requests.post(
        f"{BACKEND_ROOT_URL}/agent/",
        json={
            "intent": "analyze",
            "text": requirement_text,
            "requirement": requirement,
            "test_cases": test_cases,
        },
        timeout=120,
    )
    response.raise_for_status()

    payload = response.json()
    if payload.get("status") != "completed":
        raise ValueError(
            payload.get("error", "TraceAI could not re-analyze the test cases.")
        )

    analysis = payload.get("result", {})
    if not isinstance(analysis, dict):
        raise ValueError("TraceAI returned an invalid re-analysis result.")

    # The backend now returns a complete deterministic evaluation. Replace
    # the displayed evaluation atomically so every metric reflects the same
    # test-case snapshot.
    evaluation = analysis.get("evaluation")
    if isinstance(evaluation, dict):
        result["evaluation"] = {
            **result.get("evaluation", {}),
            **evaluation,
            "reanalysis": analysis,
        }
    else:
        # Backward-compatible fallback for an older backend.
        evidence = analysis.get("evidence", {})
        scores = analysis.get("scores", {})
        current_evaluation = result.setdefault("evaluation", {})
        if "requirement_coverage" in evidence:
            current_evaluation["requirement_coverage"] = evidence["requirement_coverage"]
        if "grounding" in evidence:
            current_evaluation["requirement_grounding"] = evidence["grounding"]
        if "requirement_grounding" in scores:
            current_evaluation["requirement_grounding_score"] = scores["requirement_grounding"]
        current_evaluation["reanalysis"] = analysis

    if isinstance(analysis.get("traceability"), list):
        result["traceability"] = analysis["traceability"]

    return result, analysis


def render_reanalysis_control(result, key_prefix="current"):
    """Render an explicit action to re-analyze the current test cases."""
    if not isinstance(result, dict) or not result.get("test_cases"):
        return

    st.divider()
    st.subheader("🔄 Re-analyze current test cases")
    st.caption(
        "Run TraceAI analysis again on the existing test cases. "
        "This does not generate a new test-case set."
    )

    if st.button(
        "Analyze Test Cases Again",
        icon=":material/refresh:",
        key=f"reanalyze_{key_prefix}",
        type="secondary",
        width="stretch",
    ):
        with st.spinner("Re-analyzing the current test cases..."):
            try:
                updated_result, analysis = analyze_current_result(result)
                result_key = f"reanalysis_result_{key_prefix}"
                st.session_state[result_key] = updated_result
                st.session_state[f"reanalysis_analysis_{key_prefix}"] = analysis
                if key_prefix == "current_result":
                    st.session_state["current_result"] = updated_result
                st.rerun()
            except requests.exceptions.RequestException as error:
                st.error(f"Re-analysis request failed: {error}")
            except (ValueError, TypeError, KeyError) as error:
                st.error(f"Re-analysis failed: {error}")


# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TraceAI - AI Test Case & Traceability Generator",
    page_icon=":material/biotech:",
    layout="wide"
)

st.markdown(
    """
    <style>
        :root {
            --traceai-primary: #1f6da8;
            --traceai-primary-dark: #17324d;
            --traceai-success: #1e7e34;
            --traceai-success-bg: #e6f4ea;
            --traceai-warning: #8a6100;
            --traceai-warning-bg: #fff4e0;
            --traceai-border: rgba(31, 109, 168, 0.22);
        }

        .traceai-brand {
            padding: 0.35rem 0 1.15rem 0;
            border-bottom: 1px solid rgba(49, 51, 63, 0.16);
            margin-bottom: 1.5rem;
        }
        .traceai-kicker {
            display: inline-block;
            color: var(--traceai-primary);
            background: rgba(31, 109, 168, 0.09);
            border: 1px solid rgba(31, 109, 168, 0.25);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            margin-bottom: 0.6rem;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
        }
        .traceai-brand h1 {
            margin: 0;
            color: var(--traceai-primary-dark);
            font-size: 2.6rem;
            font-weight: 800;
            line-height: 1.1;
        }
        .traceai-brand p {
            color: #5b6770;
            font-size: 1rem;
            margin: 0.5rem 0 0;
            max-width: 60rem;
        }
        .traceai-feature-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin: 1.1rem 0 1.6rem;
        }
        .traceai-feature-chip {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            background: #f3f8fb;
            border: 1px solid rgba(31, 109, 168, 0.18);
            border-radius: 999px;
            padding: 0.45rem 1rem;
            color: var(--traceai-primary-dark);
            font-size: 0.88rem;
            font-weight: 600;
        }
        .traceai-feature-chip .dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--traceai-primary);
            flex-shrink: 0;
        }
        .traceai-main-section-title {
            color: var(--traceai-primary-dark);
            font-weight: 700;
            font-size: 1.65rem;
            line-height: 1.25;
            margin: 1.4rem 0 0.7rem;
            padding-left: 0.75rem;
            border-left: 4px solid var(--traceai-primary);
        }

        /* Rounded, subtly-shadowed cards for every bordered container
           (requirement input, requirement analysis, score dashboard,
           agent status, Jira, history) — one reusable rule instead of
           per-section styling. */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 10px !important;
            box-shadow: 0 1px 4px rgba(23, 50, 77, 0.06);
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background: #f7fafc;
            border: 1px solid rgba(31, 109, 168, 0.14);
            border-radius: 8px;
            padding: 0.6rem 0.8rem;
        }

        /* Consistent, cleaner data tables */
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--traceai-border);
            border-radius: 8px;
            overflow: hidden;
        }

        /* Rounded alert banners with a hint of left-border emphasis */
        div[data-testid="stAlert"] {
            border-radius: 8px;
        }

        /* Emphasized requirement-input card: larger, blue-bordered, roomier */
        .st-key-requirement_input_card > div {
            border: 1.5px solid rgba(31, 109, 168, 0.35) !important;
            border-radius: 14px !important;
            box-shadow: 0 2px 10px rgba(23, 50, 77, 0.08);
            padding: 0.5rem 0.25rem;
        }

        /* Primary action button: blue accent with a subtle hover lift */
        div.stButton > button[kind="primary"] {
            background-color: var(--traceai-primary);
            border-color: var(--traceai-primary);
            font-weight: 700;
            font-size: 1.02rem;
            padding: 0.7rem 1.2rem;
            transition: background-color 0.15s ease, box-shadow 0.15s ease, transform 0.05s ease;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: var(--traceai-primary-dark);
            border-color: var(--traceai-primary-dark);
            box-shadow: 0 2px 8px rgba(31, 109, 168, 0.35);
        }
        div.stButton > button[kind="primary"]:active {
            transform: translateY(1px);
        }
        div.stButton > button,
        div.stDownloadButton > button {
            border-radius: 8px;
            border-color: #9ab7c6;
            transition: box-shadow 0.15s ease, border-color 0.15s ease;
        }
        div.stButton > button:hover,
        div.stDownloadButton > button:hover {
            border-color: var(--traceai-primary);
            box-shadow: 0 1px 6px rgba(31, 109, 168, 0.18);
        }

        /* "Export this result" card: scoped via :has() to the bordered
           container that wraps the Excel/PDF export buttons, so no changes
           to rendering_utils.py are needed to identify it. */
        div[data-testid="stVerticalBlockBorderWrapper"]:has(
            button[class*="st-key-excel_export_"], button[class*="st-key-pdf_export_"]
        ) {
            background: linear-gradient(135deg, #f5faff 0%, #eef3f8 100%);
            border: 1px solid rgba(31, 109, 168, 0.2) !important;
            border-radius: 14px !important;
            box-shadow: 0 4px 14px rgba(23, 50, 77, 0.08);
            padding: 0.35rem 0.15rem;
        }

        /* Excel export button: subtle green accent card */
        div.stDownloadButton[class*="st-key-excel_export_"] button {
            background: linear-gradient(180deg, #ffffff 0%, #eaf7ee 100%);
            border: 1.5px solid rgba(30, 126, 52, 0.35) !important;
            color: var(--traceai-success) !important;
            font-weight: 600;
            border-radius: 10px !important;
            box-shadow: 0 1px 3px rgba(30, 126, 52, 0.14);
            transition: transform 0.12s ease, box-shadow 0.12s ease, border-color 0.12s ease;
        }
        div.stDownloadButton[class*="st-key-excel_export_"] button:hover {
            transform: translateY(-2px) scale(1.01);
            border-color: var(--traceai-success) !important;
            box-shadow: 0 6px 16px rgba(30, 126, 52, 0.22);
        }
        div.stDownloadButton[class*="st-key-excel_export_"] button:active {
            transform: translateY(0) scale(0.99);
        }

        /* PDF export button: subtle coral/red accent card */
        div.stDownloadButton[class*="st-key-pdf_export_"] button {
            background: linear-gradient(180deg, #ffffff 0%, #fdeeed 100%);
            border: 1.5px solid rgba(197, 60, 51, 0.35) !important;
            color: #c53c33 !important;
            font-weight: 600;
            border-radius: 10px !important;
            box-shadow: 0 1px 3px rgba(197, 60, 51, 0.14);
            transition: transform 0.12s ease, box-shadow 0.12s ease, border-color 0.12s ease;
        }
        div.stDownloadButton[class*="st-key-pdf_export_"] button:hover {
            transform: translateY(-2px) scale(1.01);
            border-color: #c53c33 !important;
            box-shadow: 0 6px 16px rgba(197, 60, 51, 0.22);
        }
        div.stDownloadButton[class*="st-key-pdf_export_"] button:active {
            transform: translateY(0) scale(0.99);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="traceai-brand">
        <h1>TraceAI</h1>
        <div class="traceai-kicker">AI-Powered Requirement Intelligence &amp; Test Design</div>
        <p>TraceAI converts software requirements and user stories into structured, traceable test
        cases &mdash; Functional, Negative, Boundary Value Analysis, and Equivalence Partitioning &mdash;
        with built-in quality scoring and requirement traceability.</p>
        <div class="traceai-feature-row">
            <div class="traceai-feature-chip"><span class="dot"></span>AI-powered analysis</div>
            <div class="traceai-feature-chip"><span class="dot"></span>Comprehensive test design</div>
            <div class="traceai-feature-chip"><span class="dot"></span>Quality &amp; traceability</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# REQUIREMENT INPUT
# ============================================================

render_section_title("Requirement / User Story")
st.caption("Describe the behavior to test. TraceAI preserves the original source text as the source of truth.")

current_requirement_text = (
    st.session_state.get("current_result", {})
    .get("requirement", {})
    .get("text", "")
)
current_source_type = (
    st.session_state.get("current_result", {})
    .get("requirement", {})
    .get("source_type", "requirement")
)

# Seed the radio once from the last result; afterwards its own widget key
# (not a recomputed index) governs the value, so it can't silently flip
# back to a stale source type on unrelated reruns.
if "source_type_selection" not in st.session_state:
    st.session_state["source_type_selection"] = (
        "User Story" if current_source_type == "user_story" else "Requirement"
    )

with st.container(border=True, key="requirement_input_card"):
    source_type = st.radio(
        "Choose the input type",
        ["Requirement", "User Story"],
        key="source_type_selection",
        horizontal=True,
        help="Select whether the text below is a formal requirement statement or a user story.",
    )
    source_type_value = "user_story" if source_type == "User Story" else "requirement"

    user_input = st.text_area(
        "Requirement or user story text",
        value=current_requirement_text,
        placeholder=(
            "Requirement example:\n"
            "The password must contain between 8 and 20 characters. "
            "Users can log in only when the password length is within this range.\n\n"
            "User story example:\n"
            "As a customer, I want to transfer money between 1000 and 500000 when I have sufficient balance, so that I can securely transfer funds."
        ),
        height=180,
        label_visibility="collapsed",
    )



# ============================================================
# GENERATE BUTTON
# ============================================================

if st.button(
    "Analyze & Generate Test Cases",
    type="primary",
    icon=":material/auto_awesome:",
    width="stretch",
):
    if not user_input.strip():
        st.warning("Please enter a software requirement or user story.")
    else:
        with st.spinner(
            "TraceAI is analyzing the selected source "
            f"({source_type_value.replace('_', ' ').title()}) "
            "and generating test cases..."
        ):
            try:
                response = requests.post(
                    f"{BACKEND_ROOT_URL}/generate/",
                    json={"text": user_input, "source_type": source_type_value},
                    timeout=120,
                )

                if response.status_code != 200:
                    st.error(f"Backend error: HTTP {response.status_code}")
                    st.code(response.text)
                else:
                    result = response.json()

                    if "error" in result:
                        st.error(f"{result['error']}")

                        if result.get("message"):
                            st.info(result["message"])

                        if result.get("raw_output"):
                            with st.expander("🔍 View Raw AI Output"):
                                st.code(result["raw_output"])
                    else:
                        # Persist the result because Excel/PDF/Jira clicks
                        # cause Streamlit to rerun the script.
                        st.session_state["current_result"] = result
                        st.session_state.pop("last_reanalysis", None)
                        st.session_state.pop("agent_result", None)
                        st.session_state.pop("agent_result_key", None)
                        st.session_state.pop("reanalysis_result_current_result", None)
                        st.session_state.pop("reanalysis_analysis_current_result", None)

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the FastAPI backend.")
                st.info(
                    "Make sure this is running:\n\n"
                    "python -m uvicorn main:app --reload"
                )
            except requests.exceptions.Timeout:
                st.error("The AI request took too long. Please try again.")
            except json.JSONDecodeError:
                st.error("The backend returned an invalid response.")
            except Exception as error:
                st.error(f"Unexpected error: {error}")


# ------------------------------------------------------------
# Persisted current result
# ------------------------------------------------------------
current_result = st.session_state.get("current_result")
current_reanalysis = st.session_state.get("reanalysis_result_current_result")
if isinstance(current_reanalysis, dict) and current_reanalysis:
    current_result = current_reanalysis
    st.session_state["current_result"] = current_reanalysis

if isinstance(current_result, dict) and current_result:
    render_generated_result(current_result, user_input)
    render_reanalysis_control(current_result, "current_result")


# HISTORY
# ============================================================

st.divider()
render_section_title("History")
st.caption(
    "Browse persisted generation runs and reopen any complete result without "
    "regenerating it. This is separate from the current result above."
)

try:
    with st.spinner("Loading generation history..."):
        history_payload = fetch_history()
    history_runs = history_payload.get("runs", [])

    if not history_runs:
        st.info("No previous generation runs were found.", icon=":material/history:")
    else:
        st.caption(f"{len(history_runs)} persisted generation run(s), newest first.")
        history_options = {
            (
                f"Run {run.get('id')} · "
                f"{run.get('requirement_id', 'R01')} · "
                f"{run.get('status', 'unknown').title()} · "
                f"{run.get('created_at', 'Unknown date')}"
            ): run.get("id")
            for run in history_runs
        }

        selected_label = st.selectbox(
            "Choose a generation run",
            options=list(history_options),
            key="history_run_selection",
            help="Runs are ordered newest first.",
        )
        selected_run_id = history_options[selected_label]
        selected_run = next(
            run for run in history_runs if run.get("id") == selected_run_id
        )

        with st.container(border=True):
            meta1, meta2, meta3 = st.columns(3)
            with meta1:
                st.metric("Generation run", selected_run.get("id", "-"))
            with meta2:
                st.metric("Requirement", selected_run.get("requirement_id", "R01"))
            with meta3:
                st.metric("Status", selected_run.get("status", "Unknown").title())
            st.caption(f"Created {selected_run.get('created_at', 'Unknown')}")
            st.write(selected_run.get("requirement_text", ""))

        with st.spinner("Loading the selected generation..."):
            history_result = fetch_history_run(selected_run_id)

        if "error" in history_result:
            st.error(f"{history_result['error']}")
        else:
            history_reanalysis = st.session_state.get(
                f"reanalysis_result_history_{selected_run_id}"
            )
            render_history_result(
                history_reanalysis if isinstance(history_reanalysis, dict) else history_result
            )

except requests.exceptions.ConnectionError:
    st.error("Cannot connect to the FastAPI backend for History.")
    st.info("Make sure the FastAPI backend is running on port 9022.")
except requests.exceptions.Timeout:
    st.error("The History request took too long. Please try again.")
except requests.exceptions.RequestException as error:
    st.error(f"History request failed: {error}")
except (KeyError, TypeError, ValueError) as error:
    st.error(f"The backend returned an invalid History response: {error}")

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "TraceAI runs as a Streamlit frontend connected to the local FastAPI "
    "backend. The AI generation is performed through the configured "
    "Generative AI API.")