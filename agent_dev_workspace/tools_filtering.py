from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from langchain_core.tools import tool
from helpers.feature_filtering import pearson_filter


def make_tools(session):
    @tool
    def variance_filter_tool(threshold: float = 0.15) -> Dict[str, Any]:
        """Flags near-constant terms by population variance for removal based on session.feature_matrix. Typically used after build_dtm_tool and before feature correlation or pattern mining.

        Does not overwrite session.feature_matrix or session.feature_names. Stores full kept terms list in
        session.artifacts['kept_terms_variance'] and passes a full report DataFrame to session.store_result
        (column 0: feature, column 1: variance, column 2: status) sorted by variance descending.

        Args:
            threshold (float): Variance cutoff. Terms with variance > threshold are kept. Default is 0.15.

        Returns:
            Dict[str, Any]: Summary containing result_id, threshold, n_terms_initial,
            n_terms_kept, n_terms_removed, kept_terms (capped at 20), and removed_terms (capped at 20).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix available. Run build_dtm_tool first."}

        dtm = session.feature_matrix
        feature_names = session.feature_names

        # Compute population variance (ddof=0) efficiently without dense conversion
        if issparse(dtm):
            mean = dtm.mean(axis=0).A1
            mean_sq = dtm.power(2).mean(axis=0).A1
            variances = mean_sq - (mean ** 2)
        else:
            dense_dtm = np.asarray(dtm)
            variances = np.var(dense_dtm, axis=0, ddof=0)

        # Clip tiny negative numbers due to floating point precision
        variances = np.maximum(variances, 0.0)

        # Build full report DataFrame with terms, variance, and status
        report_df = pd.DataFrame(
            {
                "feature": feature_names,
                "variance": variances,
                "status": ["kept" if v > threshold else "removed" for v in variances],
            }
        )

        # Sort in descending order of variance
        report_df = report_df.sort_values(by="variance", ascending=False).reset_index(drop=True)

        # Partition kept and removed terms based on variance > threshold
        kept_mask = variances > threshold
        kept_terms = [name for name, keep in zip(feature_names, kept_mask) if keep]
        removed_terms = [name for name, keep in zip(feature_names, kept_mask) if not keep]

        # Store complete kept terms list in artifacts without modifying global DTM
        session.artifacts["kept_terms_variance"] = kept_terms

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "n_terms_initial": len(feature_names),
            "n_terms_kept": len(kept_terms),
            "n_terms_removed": len(removed_terms),
            "kept_terms": kept_terms[:20],
            "removed_terms": removed_terms[:20],
        }

        # Store result with full_report formatted for visualize_result_tool
        session.store_result(
            "variance_filter_tool",
            {"threshold": threshold},
            summary,
            full_report=report_df,
        )
        return summary

    @tool
    def pearson_filter_tool(
        target_class: str, threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Calculates Pearson correlation between term counts and a binary one-vs-rest target class indicator.

        Does not overwrite session.feature_matrix or session.feature_names. Stores the full list of kept
        terms in session.artifacts['kept_terms_pearson'] and passes full_report (DataFrame with columns
        feature, pearson_r, and pearson_p sorted by |pearson_r| descending) to session.store_result.

        Args:
            target_class (str): Target category for one-vs-rest correlation calculation.
            threshold (Optional[float]): Absolute correlation threshold cutoff. When specified, terms with
                |pearson_r| > threshold are kept. Defaults to None (all terms kept).

        Returns:
            Dict[str, Any]: Summary containing result_id, target_class, threshold, n_terms_initial,
            n_terms_kept, n_terms_removed, kept_terms (capped at 20), and removed_terms (capped at 20).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix available. Run build_dtm_tool first."}

        if session.labels is None:
            return {"error": "No document labels available. Run build_dtm_tool first."}

        if target_class not in session.categories:
            return {
                "error": f"Target class '{target_class}' not found in categories. Valid categories are: {session.categories}"
            }

        # Build one-vs-rest binary indicator vector
        y_binary = (np.asarray(session.labels) == target_class).astype(int)

        # Call helper to get full report DataFrame sorted by |pearson_r| descending
        report_df = pearson_filter(
            session.feature_matrix, y_binary, session.feature_names
        )

        # Determine kept vs removed terms based on threshold condition (|pearson_r| > threshold)
        if threshold is None:
            kept_df = report_df
            removed_df = report_df.iloc[0:0]
        else:
            kept_df = report_df[report_df["pearson_r"].abs() > threshold]
            removed_df = report_df[report_df["pearson_r"].abs() <= threshold]

        kept_terms = kept_df["feature"].tolist()
        removed_terms = removed_df["feature"].tolist()

        # Store full kept terms list in artifacts
        session.artifacts["kept_terms_pearson"] = kept_terms

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "n_terms_initial": len(session.feature_names),
            "n_terms_kept": len(kept_terms),
            "n_terms_removed": len(removed_terms),
            "kept_terms": kept_terms[:20],
            "removed_terms": removed_terms[:20],
        }

        # Store result with full_report
        session.store_result(
            "pearson_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=report_df,
        )
        return summary

    return [variance_filter_tool, pearson_filter_tool]
