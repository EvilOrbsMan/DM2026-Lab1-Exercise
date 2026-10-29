import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools as make_data_tools
from tools_dtm import make_tools as make_dtm_tools


def test_build_dtm_tool_known_answer_4_full_dataset():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    summary = build_dtm_tool.invoke({})

    # TEST_FIXTURE #4 assertions on full dataset
    assert summary["n_documents"] == 8
    assert summary["n_terms"] == 5
    assert summary["shape"] == [8, 5]
    assert summary["non_zero"] == 21
    assert summary["total_elements"] == 40
    assert summary["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)

    # State updates check
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert "count_vectorizer" in session.artifacts
    assert session.labels is not None
    assert len(session.labels) == 8


def test_build_dtm_tool_after_drop_duplicates():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    data_tools = make_data_tools(session)
    check_duplicates_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")
    check_duplicates_tool.invoke({"drop": True})

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    summary = build_dtm_tool.invoke({})

    # Check metrics on 6-row dataset
    assert summary["n_documents"] == 6
    assert summary["shape"] == [6, 5]
    assert summary["non_zero"] == 15
    assert summary["total_elements"] == 30
    assert summary["sparsity_pct"] == pytest.approx(50.0, abs=1e-3)

    assert session.labels is not None
    assert len(session.labels) == 6


def test_term_frequency_tool_known_answer_14():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    summary = term_frequency_tool.invoke({})

    # TEST_FIXTURE #14 assertions
    assert summary["n_terms"] == 5
    assert summary["total_count"] == 27

    expected_top_terms = [
        {"term": "always", "frequency": 7},
        {"term": "gamma", "frequency": 7},
        {"term": "alpha", "frequency": 6},
        {"term": "delta", "frequency": 4},
        {"term": "beta", "frequency": 3},
    ]

    assert summary["top_terms"] == expected_top_terms

    # Check stored result full_report
    result_id = summary["result_id"]
    stored_entry = session.results_store[result_id]
    full_report = stored_entry["full_report"]

    assert full_report is not None
    assert list(full_report.columns) == ["term", "frequency"]
    assert full_report["term"].tolist() == ["always", "gamma", "alpha", "delta", "beta"]
    assert full_report["frequency"].tolist() == [7, 7, 6, 4, 3]


def test_term_frequency_tool_top_n_and_tie_breaking():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    summary = term_frequency_tool.invoke({"top_n": 2})

    # Capped at top_n = 2
    assert len(summary["top_terms"]) == 2

    # Tie-breaker check: always=7 precedes gamma=7 alphabetically
    assert summary["top_terms"] == [
        {"term": "always", "frequency": 7},
        {"term": "gamma", "frequency": 7},
    ]


def test_term_frequency_tool_error_cases():
    session = SessionState()
    dtm_tools = make_dtm_tools(session)
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    # Uninitialized feature_matrix error
    summary_no_dtm = term_frequency_tool.invoke({})
    assert "error" in summary_no_dtm

    # Load dataset & build DTM to test top_n < 1 error
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    summary_invalid_top_n = term_frequency_tool.invoke({"top_n": 0})
    assert "error" in summary_invalid_top_n
