from __future__ import annotations

import json
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


# Shared by both export formats so a category can never be dropped from only one.
TEST_CASE_CATEGORIES = (
    ("functional", "Functional"),
    ("negative", "Negative"),
    ("bva", "BVA"),
    ("ep", "EP"),
    ("edge_cases", "Edge Cases"),
)

# Only the columns meaningful to each category are exported, so
# Functional/Negative tables no longer show empty BVA/EP-only fields.
_CATEGORY_COLUMNS = {
    "functional": ("ID", "Requirement ID", "Type", "Description", "Expected"),
    "negative": ("ID", "Requirement ID", "Type", "Description", "Expected"),
    "bva": ("ID", "Requirement ID", "Type", "Description/Scenario", "Values", "Expected"),
    "ep": ("ID", "Requirement ID", "Type", "Class", "Input", "Expected"),
    "edge_cases": ("ID", "Requirement ID", "Type", "Description", "Expected"),
}


# Evaluation keys surfaced as headline metrics; everything else goes to the details table.
_EVALUATION_HEADLINE_LABELS = {
    "test_design_coverage": "Test design coverage",
    "requirement_traceability": "Requirement traceability",
    "ai_quality_score": "AI quality score",
    "requirement_grounding_score": "Requirement grounding score",
}

# Shared brand palette for both exports (presentation only — no effect on data).
_NAVY = "#17324d"
_BLUE = "#1f6da8"
_LIGHT_BLUE = "#eef4f8"
_BAND = "#f5f9fc"
_GREEN = "#1e7e34"
_GREEN_BG = "#e6f4ea"
_RED = "#b3261e"
_RED_BG = "#fbe9ea"
_GRID = "#b7c9d6"
_MUTED = "#5b6770"


def _status_tone(value: Any) -> str:
    """Classify a status/expected/score value as success, danger, or neutral.

    Purely a presentation helper: it only inspects the already-computed
    text/score, never derives or alters it.
    """
    text = str(value or "").strip().lower()
    if any(word in text for word in ("accept", "covered", "pass", "success")):
        return "success"
    if any(word in text for word in ("reject", "uncovered", "fail", "error", "denied")):
        return "danger"
    if text.endswith("%"):
        try:
            if float(text.rstrip("%")) >= 100:
                return "success"
        except ValueError:
            pass
    return "neutral"


def _require_result(result: Any) -> dict:
    if not isinstance(result, dict) or not result:
        raise ValueError("No result data is available for export.")
    if not any(result.get(key) for key in ("requirement", "test_cases", "evaluation", "traceability")):
        raise ValueError("No result data is available for export.")
    return result


def _json_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True)
    return "" if value is None else str(value)


def _test_case_rows(result: dict, category: str) -> list[dict]:
    cases = result.get("test_cases", {}).get(category, [])
    if not isinstance(cases, list):
        return []
    columns = _CATEGORY_COLUMNS.get(category, tuple(_CATEGORY_COLUMNS["functional"]))
    rows = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        full_row = {
            "ID": case.get("id", ""),
            "Requirement ID": case.get("requirement_id", ""),
            "Type": case.get("type", category),
            "Description": case.get("description", ""),
            "Description/Scenario": case.get("description", "") or case.get("scenario", ""),
            "Class": case.get("class", ""),
            "Input": case.get("input", ""),
            "Values": case.get("values", ""),
            "Expected": case.get("expected", ""),
        }
        rows.append({column: full_row.get(column, "") for column in columns})
    return rows


def _traceability_rows(result: dict) -> list[dict]:
    rows = result.get("traceability", [])
    if not isinstance(rows, list):
        return []
    return [
        {
            "Requirement ID": row.get("requirement_id", ""),
            "Test Case": row.get("test_case", ""),
            "Type": row.get("type", ""),
            "Description": row.get("description", ""),
            "Status": row.get("status", ""),
        }
        for row in rows
        if isinstance(row, dict)
    ]


def _format_metric_value(value: Any) -> str:
    if isinstance(value, bool) or isinstance(value, (dict, list)):
        return _json_value(value)
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return _json_value(value)


def _evaluation_summary_rows(evaluation: dict) -> list[dict]:
    rows = [
        {"Metric": label, "Value": _format_metric_value(evaluation[key])}
        for key, label in _EVALUATION_HEADLINE_LABELS.items()
        if key in evaluation
    ]
    if evaluation.get("explanation"):
        rows.append({"Metric": "Explanation", "Value": str(evaluation["explanation"])})
    return rows


def _evaluation_detail_rows(evaluation: dict) -> list[dict]:
    excluded = set(_EVALUATION_HEADLINE_LABELS) | {"explanation"}
    return [
        {"Field": key, "Value": _json_value(value)}
        for key, value in evaluation.items()
        if key not in excluded
    ]


def _summary_rows(result: dict) -> list[dict]:
    """Front-page dashboard rows assembled from data already in `result`."""
    requirement = result.get("requirement") or {}
    evaluation = result.get("evaluation") or {}
    rows = [
        {"Metric": "Requirement ID", "Value": _json_value(requirement.get("id") or "Unrecorded requirement")},
        {"Metric": "Type", "Value": _json_value(requirement.get("type"))},
        {"Metric": "Priority", "Value": _json_value(requirement.get("priority"))},
    ]
    rows.extend(_evaluation_summary_rows(evaluation))
    for category, label in TEST_CASE_CATEGORIES:
        case_rows = _test_case_rows(result, category)
        if case_rows:
            rows.append({"Metric": f"{label} test cases", "Value": str(len(case_rows))})
    return rows


def build_excel_export(result: dict) -> bytes:
    result = _require_result(result)
    requirement = result.get("requirement") or {}
    evaluation = result.get("evaluation") or {}
    output = BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book

        title_format = workbook.add_format({
            "bold": True, "font_size": 14, "font_color": "white",
            "bg_color": _NAVY, "valign": "vcenter", "indent": 1,
        })
        subtitle_format = workbook.add_format({
            "italic": True, "font_size": 9, "font_color": _MUTED, "indent": 1,
        })
        header_format = workbook.add_format({
            "bold": True, "font_color": "white", "bg_color": _BLUE,
            "border": 1, "border_color": _GRID, "valign": "vcenter", "text_wrap": True,
        })
        cell_format = workbook.add_format({"border": 1, "border_color": _GRID, "valign": "top", "text_wrap": True})
        band_format = workbook.add_format({"border": 1, "border_color": _GRID, "valign": "top", "text_wrap": True, "bg_color": _BAND})
        success_format = workbook.add_format({"border": 1, "border_color": _GRID, "valign": "top", "bg_color": _GREEN_BG, "font_color": _GREEN, "bold": True})
        danger_format = workbook.add_format({"border": 1, "border_color": _GRID, "valign": "top", "bg_color": _RED_BG, "font_color": _RED, "bold": True})

        requirement_id = str(requirement.get("id") or "Unrecorded requirement")
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        def _write_sheet(sheet_name, columns, rows, status_columns=()):
            worksheet = workbook.add_worksheet(sheet_name)
            writer.sheets[sheet_name] = worksheet
            last_col = max(len(columns) - 1, 1)
            worksheet.merge_range(0, 0, 0, last_col, f"TraceAI \u2014 {sheet_name}", title_format)
            worksheet.merge_range(1, 0, 1, last_col, f"Requirement {requirement_id}  \u00b7  Generated {generated_at}", subtitle_format)
            header_row = 3
            for col_idx, column in enumerate(columns):
                worksheet.write(header_row, col_idx, column, header_format)
            if rows:
                display_rows = rows
            else:
                placeholder = {column: "" for column in columns}
                if "Type" in columns:
                    placeholder["Type"] = sheet_name
                display_rows = [placeholder]
            for row_idx, row in enumerate(display_rows):
                banded = row_idx % 2 == 1
                for col_idx, column in enumerate(columns):
                    value = _json_value(row.get(column, ""))
                    if column in status_columns:
                        tone = _status_tone(value)
                        fmt = {"success": success_format, "danger": danger_format}.get(
                            tone, band_format if banded else cell_format
                        )
                    else:
                        fmt = band_format if banded else cell_format
                    worksheet.write(header_row + 1 + row_idx, col_idx, value, fmt)
            worksheet.freeze_panes(header_row + 1, 0)
            worksheet.set_column(0, 0, 24)
            worksheet.set_column(1, last_col, 30)
            return worksheet

        _write_sheet("Summary", ["Metric", "Value"], _summary_rows(result), status_columns=("Value",))

        requirement_rows = [{"Field": key, "Value": _json_value(value)} for key, value in requirement.items()]
        _write_sheet("Requirement", ["Field", "Value"], requirement_rows)

        evaluation_rows = [{"Metric": key, "Value": _json_value(value)} for key, value in evaluation.items()]
        _write_sheet("Evaluation", ["Metric", "Value"], evaluation_rows)

        for category, sheet_name in TEST_CASE_CATEGORIES:
            columns = list(_CATEGORY_COLUMNS.get(category, _CATEGORY_COLUMNS["functional"]))
            _write_sheet(sheet_name, columns, _test_case_rows(result, category), status_columns=("Expected",))

        _write_sheet(
            "Traceability",
            ["Requirement ID", "Test Case", "Type", "Description", "Status"],
            _traceability_rows(result),
            status_columns=("Status",),
        )

    return output.getvalue()


def _pdf_table(title: str, rows: list[dict], styles, status_columns: tuple = ()) -> list:
    elements = [Paragraph(title, styles["SectionHeading"])]
    if not rows:
        elements.append(Paragraph("No data available.", styles["BodyText"]))
        elements.append(Spacer(1, 0.12 * inch))
        return elements

    columns = list(rows[0].keys())
    table_data = [[Paragraph(str(column), styles["TableHeader"]) for column in columns]]
    for row in rows:
        table_row = []
        for column in columns:
            value = _json_value(row.get(column, ""))
            if column in status_columns:
                tone = _status_tone(value)
                style_name = {"success": "TableCellSuccess", "danger": "TableCellDanger"}.get(tone, "TableCell")
            else:
                style_name = "TableCell"
            table_row.append(Paragraph(value, styles[style_name]))
        table_data.append(table_row)

    table = Table(table_data, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(_BLUE)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor(_GRID)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(_LIGHT_BLUE)]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.extend([table, Spacer(1, 0.2 * inch)])
    return elements


def build_pdf_export(result: dict) -> bytes:
    result = _require_result(result)
    requirement = result.get("requirement") or {}
    evaluation = result.get("evaluation") or {}
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="BannerTitle", parent=styles["Title"], textColor=colors.white,
        alignment=TA_CENTER, fontSize=20, leading=24,
    ))
    styles.add(ParagraphStyle(
        name="BannerSubtitle", parent=styles["BodyText"], textColor=colors.white,
        alignment=TA_CENTER, fontSize=9,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeading", parent=styles["Heading2"], textColor=colors.HexColor(_NAVY),
        borderColor=colors.HexColor(_BLUE), borderWidth=0, spaceBefore=6, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(name="TableHeader", parent=styles["BodyText"], alignment=TA_CENTER, textColor=colors.white, fontSize=8, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="TableCell", parent=styles["BodyText"], fontSize=7, leading=9))
    styles.add(ParagraphStyle(name="TableCellSuccess", parent=styles["TableCell"], textColor=colors.HexColor(_GREEN), fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="TableCellDanger", parent=styles["TableCell"], textColor=colors.HexColor(_RED), fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="Subtitle", parent=styles["BodyText"], textColor=colors.HexColor(_MUTED)))

    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=letter,
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )
    requirement_id = requirement.get("id") or "Unrecorded requirement"
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    banner = Table(
        [
            [Paragraph("TraceAI Exported Result", styles["BannerTitle"])],
            [Paragraph(f"Requirement {requirement_id} \u2014 generated {generated_at}", styles["BannerSubtitle"])],
        ],
        colWidths=[7.5 * inch],
    )
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(_NAVY)),
        ("TOPPADDING", (0, 0), (0, 0), 10),
        ("BOTTOMPADDING", (0, 0), (0, 0), 2),
        ("TOPPADDING", (0, 1), (0, 1), 2),
        ("BOTTOMPADDING", (0, 1), (0, 1), 10),
    ]))
    story = [banner, Spacer(1, 0.2 * inch)]

    story.extend(_pdf_table("Summary", _summary_rows(result), styles, status_columns=("Value",)))
    story.extend(_pdf_table(
        "Requirement",
        [{"Field": key, "Value": _json_value(value)} for key, value in requirement.items()],
        styles,
    ))
    story.extend(_pdf_table("Evaluation summary", _evaluation_summary_rows(evaluation), styles))
    story.extend(_pdf_table("Evaluation details", _evaluation_detail_rows(evaluation), styles))
    for category, title in TEST_CASE_CATEGORIES:
        story.extend(_pdf_table(title, _test_case_rows(result, category), styles))
    story.extend(_pdf_table("Traceability", _traceability_rows(result), styles))
    document.build(story)
    return output.getvalue()