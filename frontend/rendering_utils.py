"""Reusable Streamlit rendering helpers for result exports and metrics."""

from __future__ import annotations

import streamlit as st

try:
    from frontend.export_utils import build_excel_export, build_pdf_export
except ModuleNotFoundError:
    from export_utils import build_excel_export, build_pdf_export


def format_score(value):
    """Display a numeric score cleanly."""
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def render_section_title(text):
    """Consistent accent-colored heading shared by every major TraceAI section."""
    st.markdown(f'<div class="traceai-main-section-title">{text}</div>', unsafe_allow_html=True)


def get_evaluation_details(evaluation):
    """Read the evaluation response with compatibility defaults."""
    return {
        "test_design_coverage": evaluation.get("test_design_coverage", 0),
        "requirement_traceability": evaluation.get("requirement_traceability", 0),
        "requirement_coverage": evaluation.get("requirement_coverage", {}),
        "ai_quality_score": evaluation.get("ai_quality_score"),
        "requirement_grounding_score": evaluation.get("requirement_grounding_score"),
        "requirement_grounding": evaluation.get("requirement_grounding", {}),
        "quality_breakdown": evaluation.get("quality_breakdown", {}),
    }


def render_export_controls(result, key_prefix):
    """Render Excel and PDF downloads for a result payload."""
    if not isinstance(result, dict) or not result:
        st.info("No result is available to export.")
        return

    try:
        excel_data = build_excel_export(result)
        pdf_data = build_pdf_export(result)
    except ValueError as error:
        st.info(str(error))
        return

    with st.container(border=True):
        st.write("**Export this result**")
        st.caption("Download the currently displayed requirement, evaluation, test cases, and traceability.")
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            st.download_button(
                "Excel workbook",
                data=excel_data,
                file_name=f"traceai_{key_prefix}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"excel_export_{key_prefix}",
                icon=":material/table_view:",
                width="stretch",
            )
        with export_col2:
            st.download_button(
                "PDF report",
                data=pdf_data,
                file_name=f"traceai_{key_prefix}.pdf",
                mime="application/pdf",
                key=f"pdf_export_{key_prefix}",
                icon=":material/picture_as_pdf:",
                width="stretch",
            )
