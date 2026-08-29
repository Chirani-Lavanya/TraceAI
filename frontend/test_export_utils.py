from io import BytesIO
from zipfile import ZipFile

import pytest

from frontend.export_utils import (
    build_excel_export,
    build_pdf_export,
    _evaluation_detail_rows,
    _evaluation_summary_rows,
    _test_case_rows,
)


RESULT = {
    "requirement": {
        "id": "R01",
        "text": "Password must be between 8 and 20 characters.",
        "type": "Validation",
        "priority": "High",
        "expected_behavior": "Valid passwords are accepted.",
    },
    "test_cases": {
        "functional": [{"id": "TC01", "type": "Functional", "expected": "Accepted."}],
        "negative": [{"id": "NTC01", "type": "Negative", "expected": "Rejected."}],
        "bva": [{"id": "BVA01", "type": "BVA", "values": "7", "expected": "Rejected."}],
        "ep": [{"id": "EP01", "type": "EP", "class": "Below minimum", "expected": "Rejected."}],
        "edge_cases": [{"id": "EC01", "type": "Edge Case", "description": "Empty password.", "expected": "Rejected."}],
    },
    "evaluation": {
        "test_design_coverage": 100.0,
        "requirement_traceability": 100.0,
        "ai_quality_score": 95.0,
        "requirement_grounding_score": 100.0,
        "explanation": "All categories are grounded and covered.",
        "requirement_coverage": {"percentage": 100.0},
    },
    "traceability": [{
        "requirement_id": "R01",
        "test_case": "TC01",
        "type": "Functional",
        "description": "Valid password",
        "status": "Covered",
    }],
}


def test_excel_export_contains_expected_worksheets():
    exported = build_excel_export(RESULT)

    assert exported.startswith(b"PK")
    with ZipFile(BytesIO(exported)) as workbook:
        workbook_xml = workbook.read("xl/workbook.xml").decode("utf-8")

    for sheet_name in ["Requirement", "Evaluation", "Functional", "Negative", "BVA", "EP", "Edge Cases", "Traceability"]:
        assert f'name="{sheet_name}"' in workbook_xml


def test_excel_export_includes_edge_case_rows():
    exported = build_excel_export(RESULT)

    with ZipFile(BytesIO(exported)) as workbook:
        shared_strings = workbook.read("xl/sharedStrings.xml").decode("utf-8")

    assert "EC01" in shared_strings


def test_pdf_export_is_valid_pdf_with_content():
    exported = build_pdf_export(RESULT)

    assert exported.startswith(b"%PDF-")
    assert len(exported) > 100


def test_evaluation_summary_rows_use_readable_labels_and_percentages():
    summary = _evaluation_summary_rows(RESULT["evaluation"])

    assert {"Metric": "Test design coverage", "Value": "100.0%"} in summary
    assert {"Metric": "AI quality score", "Value": "95.0%"} in summary
    assert {"Metric": "Explanation", "Value": "All categories are grounded and covered."} in summary


def test_evaluation_detail_rows_exclude_headline_metrics():
    details = _evaluation_detail_rows(RESULT["evaluation"])

    detail_fields = {row["Field"] for row in details}
    assert "requirement_coverage" in detail_fields
    assert "test_design_coverage" not in detail_fields
    assert "explanation" not in detail_fields


def test_history_result_can_be_exported_without_regeneration():
    history_result = dict(RESULT)
    history_result["id"] = 42
    history_result["status"] = "completed"

    assert len(build_excel_export(history_result)) > 100
    assert len(build_pdf_export(history_result)) > 100


@pytest.mark.parametrize("missing_result", [None, {}, {"error": "No result"}])
def test_empty_or_missing_result_is_rejected(missing_result):
    with pytest.raises(ValueError, match="No result data"):
        build_excel_export(missing_result)
    with pytest.raises(ValueError, match="No result data"):
        build_pdf_export(missing_result)


def test_functional_and_negative_rows_omit_bva_ep_only_columns():
    functional_rows = _test_case_rows(RESULT, "functional")
    negative_rows = _test_case_rows(RESULT, "negative")

    assert list(functional_rows[0].keys()) == ["ID", "Requirement ID", "Type", "Description", "Expected"]
    assert list(negative_rows[0].keys()) == ["ID", "Requirement ID", "Type", "Description", "Expected"]


def test_bva_rows_use_description_scenario_and_values_columns():
    bva_rows = _test_case_rows(RESULT, "bva")

    assert list(bva_rows[0].keys()) == [
        "ID", "Requirement ID", "Type", "Description/Scenario", "Values", "Expected",
    ]


def test_ep_rows_use_class_and_input_columns():
    ep_rows = _test_case_rows(RESULT, "ep")

    assert list(ep_rows[0].keys()) == ["ID", "Requirement ID", "Type", "Class", "Input", "Expected"]


def test_edge_cases_rows_unchanged_when_present_and_absent():
    present_rows = _test_case_rows(RESULT, "edge_cases")
    assert present_rows == [{
        "ID": "EC01",
        "Requirement ID": "",
        "Type": "Edge Case",
        "Description": "Empty password.",
        "Expected": "Rejected.",
    }]

    empty_result = dict(RESULT)
    empty_result["test_cases"] = dict(RESULT["test_cases"])
    empty_result["test_cases"]["edge_cases"] = []
    assert _test_case_rows(empty_result, "edge_cases") == []


def test_no_test_case_information_is_dropped_from_export():
    functional_rows = _test_case_rows(RESULT, "functional")

    assert functional_rows[0]["ID"] == "TC01"
    assert functional_rows[0]["Type"] == "Functional"
    assert functional_rows[0]["Expected"] == "Accepted."