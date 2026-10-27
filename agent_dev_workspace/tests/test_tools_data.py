import os
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_tokenize_tool_known_answer_10():
    # Load premade load_dataset_tool bound to session
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load fixture CSV
    load_result = load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # Get student's tokenize_tool bound to the SAME session
    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")

    # Run tokenize_tool
    summary = tokenize_tool.invoke({})

    # Exact token sequences expected per document from TEST_FIXTURE.md #10
    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        [],
    ]

    expected_total_tokens = sum(len(t) for t in expected_tokens)

    # Verify return dict shape and metrics
    assert "result_id" in summary
    assert summary["n_documents"] == 8
    assert summary["total_tokens"] == expected_total_tokens

    # Verify session state updates
    assert "unigrams" in session.dataframe.columns
    assert "unigrams" in session.artifacts

    unigrams = session.dataframe["unigrams"].tolist()
    assert unigrams == expected_tokens


def test_list_files_tool_known_answer_11():
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    result = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert result["directories"] == []
    assert result["files"] == ["Reddit-stock-sentiment.csv"]


def test_check_missing_tool_known_answer_2():
    # Fresh session so dataset state is clean
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load fixture CSV
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # Get check_missing_tool bound to the SAME session
    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    summary = check_missing_tool.invoke({})

    # TEST_FIXTURE.md #2 known answer assertions
    assert "result_id" in summary
    assert summary["n_documents"] == 8
    assert summary["missing_count"] == 1
    assert summary["valid_count"] == 7
    assert summary["missing_indices"] == [7]


def test_check_duplicates_tool_known_answer_3_no_drop():
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

    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    summary = check_duplicates_tool.invoke({"drop": False})

    assert "result_id" in summary
    assert summary["n_documents_before"] == 8
    assert summary["duplicate_count"] == 1
    assert summary["rows_deleted"] == 0
    assert summary["n_documents_after"] == 8
    assert summary["duplicate_indices"] == [5, 6]
    assert len(session.dataframe) == 8


def test_check_duplicates_tool_known_answer_3_with_drop():
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

    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    summary = check_duplicates_tool.invoke({"drop": True})

    assert "result_id" in summary
    assert summary["n_documents_before"] == 8
    assert summary["duplicate_count"] == 1
    assert summary["rows_deleted"] == 2
    assert summary["n_documents_after"] == 6
    assert summary["duplicate_indices"] == [5, 6]
    assert len(session.dataframe) == 6


def test_inspect_data_tool_default_head():
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

    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")

    summary = inspect_data_tool.invoke({"n_rows": 5, "mode": "head"})

    assert "result_id" in summary
    assert summary["shape"][0] == 8
    assert "text" in summary["columns"]
    assert "category_name" in summary["columns"]
    assert summary["n_rows_returned"] == 5
    assert summary["category_counts"] == {"catA": 4, "catB": 4}

    preview = summary["preview_rows"]
    assert preview[0]["text"] == "always alpha alpha alpha beta"
    assert preview[0]["category_name"] == "catA"


def test_inspect_data_tool_tail():
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

    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")

    summary = inspect_data_tool.invoke({"n_rows": 2, "mode": "tail"})

    assert summary["n_rows_returned"] == 2
    preview = summary["preview_rows"]
    assert preview[0]["text"] == "always gamma delta"
    assert preview[0]["category_name"] == "catB"
    assert preview[1]["category_name"] == "catA"
    assert preview[1]["text"] is None or preview[1]["text"] == ""


def test_inspect_data_tool_clamping():
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

    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")

    summary = inspect_data_tool.invoke({"n_rows": 50})

    assert "error" not in summary
    assert summary["n_rows_returned"] == 8


def test_inspect_data_tool_error_cases():
    # Fresh session before loading dataset
    session = SessionState()
    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")

    summary_no_df = inspect_data_tool.invoke({})
    assert "error" in summary_no_df

    # Load dataset to test invalid n_rows
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    summary_invalid_rows = inspect_data_tool.invoke({"n_rows": 0})
    assert "error" in summary_invalid_rows
