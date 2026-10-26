import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_patterns_tools


@pytest.mark.parametrize(
    "category_name, filtering_method, algorithm, min_sup, k, expected_n_kept, expected_patterns, expected_note",
    [
        # KNOWN ANSWER #17: variance & term_frequency filter keep 0 terms for both categories
        ("catA", "variance", "fpgrowth", 1, None, 0, [], "no terms survived filtering"),
        ("catB", "variance", "fpgrowth", 1, None, 0, [], "no terms survived filtering"),
        ("catA", "term_frequency", "fpgrowth", 1, None, 0, [], "no terms survived filtering"),
        ("catB", "term_frequency", "fpgrowth", 1, None, 0, [], "no terms survived filtering"),
        # KNOWN ANSWER #17: tfidf filter + fpgrowth (min_sup 1-3 for catA finds {alpha}; 1-4 for catB finds {gamma})
        ("catA", "tfidf", "fpgrowth", 1, None, 1, [{"pattern": ["alpha"], "support": 3}], None),
        ("catA", "tfidf", "fpgrowth", 3, None, 1, [{"pattern": ["alpha"], "support": 3}], None),
        ("catA", "tfidf", "fpgrowth", 4, None, 1, [], "no patterns found"),
        ("catB", "tfidf", "fpgrowth", 1, None, 1, [{"pattern": ["gamma"], "support": 4}], None),
        ("catB", "tfidf", "fpgrowth", 4, None, 1, [{"pattern": ["gamma"], "support": 4}], None),
        # KNOWN ANSWER #20: algorithm="topk" (k=1 and k=3)
        ("catA", "tfidf", "topk", None, 1, 1, [{"pattern": ["alpha"], "support": 3}], None),
        ("catB", "tfidf", "topk", None, 1, 1, [{"pattern": ["gamma"], "support": 4}], None),
        ("catA", "tfidf", "topk", None, 3, 1, [{"pattern": ["alpha"], "support": 3}], None),
        ("catB", "tfidf", "topk", None, 3, 1, [{"pattern": ["gamma"], "support": 4}], None),
        # KNOWN ANSWER #20: algorithm="maxfpgrowth" (min_sup=1 and min_sup=4, returns 0 patterns for both)
        ("catA", "tfidf", "maxfpgrowth", 1, None, 1, [], "no patterns found"),
        ("catB", "tfidf", "maxfpgrowth", 1, None, 1, [], "no patterns found"),
        ("catA", "tfidf", "maxfpgrowth", 4, None, 1, [], "no patterns found"),
        ("catB", "tfidf", "maxfpgrowth", 4, None, 1, [], "no patterns found"),
    ],
)
def test_mine_patterns_tool_fixture_cases(
    category_name,
    filtering_method,
    algorithm,
    min_sup,
    k,
    expected_n_kept,
    expected_patterns,
    expected_note,
):
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

    pattern_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    args = {
        "category_name": category_name,
        "filtering_method": filtering_method,
        "algorithm": algorithm,
    }
    if min_sup is not None:
        args["min_sup"] = min_sup
    if k is not None:
        args["k"] = k

    summary = mine_patterns_tool.invoke(args)

    assert summary["n_terms_kept"] == expected_n_kept
    assert summary["patterns"] == expected_patterns

    if expected_note is not None:
        assert summary.get("note") == expected_note

    # Check artifact storage key
    artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
    assert session.artifacts[artifact_key] == expected_patterns


def test_mine_patterns_tool_validation():
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

    pattern_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    # Missing category
    res = mine_patterns_tool.invoke({"category_name": "catUnknown", "filtering_method": "tfidf"})
    assert "error" in res

    # Missing min_sup for fpgrowth
    res = mine_patterns_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth"})
    assert "error" in res

    # Missing k for topk
    res = mine_patterns_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "topk"})
    assert "error" in res
