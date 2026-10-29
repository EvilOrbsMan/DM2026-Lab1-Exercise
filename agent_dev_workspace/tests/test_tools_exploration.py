import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_exploration import make_tools as make_exploration_tools
from tools_dtm import make_tools as make_dtm_tools


def test_cosine_similarity_tool_known_answers_16():
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

    exploration_tools = make_exploration_tools(session)
    cosine_similarity_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # 1. Near-identical documents in the same category
    res_0_1 = cosine_similarity_tool.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert res_0_1["cosine_similarity"] == pytest.approx(0.9847, abs=1e-4)

    res_3_4 = cosine_similarity_tool.invoke({"doc_index_1": 3, "doc_index_2": 4})
    assert res_3_4["cosine_similarity"] == pytest.approx(0.9847, abs=1e-4)

    # Identical documents
    res_5_6 = cosine_similarity_tool.invoke({"doc_index_1": 5, "doc_index_2": 6})
    assert res_5_6["cosine_similarity"] == pytest.approx(1.0, abs=1e-4)

    # 2. Cross-category documents sharing only "always"
    res_0_3 = cosine_similarity_tool.invoke({"doc_index_1": 0, "doc_index_2": 3})
    assert res_0_3["cosine_similarity"] == pytest.approx(0.0909, abs=1e-4)

    # 3. Empty document edge case (row 7)
    res_0_7 = cosine_similarity_tool.invoke({"doc_index_1": 0, "doc_index_2": 7})
    assert res_0_7["cosine_similarity"] == 0.0


def test_cosine_similarity_tool_error_handling():
    # Calling before loading dataset or building DTM
    session_empty = SessionState()
    exploration_tools_empty = make_exploration_tools(session_empty)
    cosine_similarity_tool_empty = next(
        t for t in exploration_tools_empty if t.name == "cosine_similarity_tool"
    )

    res_no_data = cosine_similarity_tool_empty.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert "error" in res_no_data

    # Load dataset & build DTM to test index out of bounds
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

    exploration_tools = make_exploration_tools(session)
    cosine_similarity_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # Out of bounds indices
    res_negative = cosine_similarity_tool.invoke({"doc_index_1": -1, "doc_index_2": 0})
    assert "error" in res_negative

    res_too_large = cosine_similarity_tool.invoke({"doc_index_1": 0, "doc_index_2": 10})
    assert "error" in res_too_large
