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
