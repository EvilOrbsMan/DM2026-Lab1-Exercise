import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool_known_answer_5():
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

    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    summary = variance_filter_tool.invoke({"threshold": 0.15})

    # Assertions based on TEST_FIXTURE #5
    assert summary["n_terms_initial"] == 5
    assert summary["n_terms_kept"] == 4
    assert summary["n_terms_removed"] == 1

    # Check kept & removed terms (alphabetical in input vocabulary order)
    assert summary["kept_terms"] == ["alpha", "beta", "delta", "gamma"]
    assert summary["removed_terms"] == ["always"]
    assert session.artifacts["kept_terms_variance"] == ["alpha", "beta", "delta", "gamma"]

    # Verify global DTM feature matrix shape remains untouched (8, 5)
    assert session.feature_matrix.shape == (8, 5)

    # Check full_report stored in session.results_store
    result_id = summary["result_id"]
    stored_result = session.results_store[result_id]
    full_report = stored_result["full_report"]

    expected_features = ["alpha", "gamma", "delta", "beta", "always"]
    expected_variances = [1.1875, 1.109375, 0.25, 0.234375, 0.109375]

    assert full_report.iloc[:, 0].tolist() == expected_features
    actual_variances = full_report.iloc[:, 1].tolist()
    for actual, expected in zip(actual_variances, expected_variances):
        assert actual == pytest.approx(expected, abs=1e-3)


def test_pearson_filter_tool_known_answer_6():
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

    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    summary = pearson_filter_tool.invoke({"target_class": "catB"})

    result_id = summary["result_id"]
    full_report = session.results_store[result_id]["full_report"]

    expected_features = ["delta", "gamma", "beta", "alpha", "always"]
    expected_r = [1.0000, 0.8307, -0.7746, -0.6882, 0.3780]

    assert full_report.iloc[:, 0].tolist() == expected_features
    actual_r = full_report["pearson_r"].tolist()
    for actual, expected in zip(actual_r, expected_r):
        assert actual == pytest.approx(expected, abs=1e-4)


def test_pearson_filter_tool_threshold():
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

    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    summary = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.5})

    assert set(summary["kept_terms"]) == {"delta", "gamma", "beta", "alpha"}
    assert summary["removed_terms"] == ["always"]
    assert summary["n_terms_kept"] == 4
    assert summary["n_terms_removed"] == 1


def test_pearson_filter_tool_invalid_category():
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

    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    result = pearson_filter_tool.invoke({"target_class": "catC"})
    assert "error" in result


def test_spearman_filter_tool_known_answer_6():
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

    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    summary = spearman_filter_tool.invoke({"target_class": "catB"})

    assert summary["total_features"] == 5
    assert summary["n_evaluated"] == 5

    # Check top_features against TEST_FIXTURE #6 Spearman r values
    expected_mapping = {
        "delta": 1.0000,
        "gamma": 0.9363,
        "beta": -0.7746,
        "alpha": -0.7500,
        "always": 0.3780,
    }

    top_features = summary["top_features"]
    assert len(top_features) == 5

    for rec in top_features:
        feat = rec["feature"]
        assert feat in expected_mapping
        assert rec["spearman_r"] == pytest.approx(expected_mapping[feat], abs=1e-4)

    # Check top_positive and top_negative partitions
    pos_features = [rec["feature"] for rec in summary["top_positive"]]
    neg_features = [rec["feature"] for rec in summary["top_negative"]]

    assert set(pos_features) == {"delta", "gamma", "always"}
    assert set(neg_features) == {"beta", "alpha"}

    # Test with threshold = 0.5
    summary_thresh = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.5})
    assert summary_thresh["kept_count"] == 4


def test_spearman_filter_tool_max_terms():
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

    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    summary = spearman_filter_tool.invoke({"target_class": "catB", "max_terms": 2})

    assert summary["n_evaluated"] == 2
    top_features = summary["top_features"]
    assert len(top_features) == 2
    # Pre-filtered top 2 terms by variance are alpha (1.1875) and gamma (1.109375)
    # Sorted by |spearman_r| descending: gamma (~0.9363) then alpha (~-0.75)
    assert top_features[0]["feature"] == "gamma"
    assert top_features[0]["spearman_r"] == pytest.approx(0.9363, abs=1e-4)
    assert top_features[1]["feature"] == "alpha"
    assert top_features[1]["spearman_r"] == pytest.approx(-0.7500, abs=1e-4)


def test_spearman_filter_tool_error_handling():
    # Error when called with invalid target class
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

    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    result_invalid_class = spearman_filter_tool.invoke({"target_class": "catC"})
    assert "error" in result_invalid_class

    # Error when called on uninitialized SessionState before DTM creation
    session_no_dtm = SessionState()
    filtering_tools_no_dtm = make_filtering_tools(session_no_dtm)
    spearman_filter_tool_no_dtm = next(
        t for t in filtering_tools_no_dtm if t.name == "spearman_filter_tool"
    )

    result_no_dtm = spearman_filter_tool_no_dtm.invoke({"target_class": "catB"})
    assert "error" in result_no_dtm
