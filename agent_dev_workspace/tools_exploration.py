from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import issparse
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def cosine_similarity_tool(doc_index_1: int, doc_index_2: int) -> Dict[str, Any]:
        """Computes cosine similarity between two documents in session.dataframe using raw counts.

        Vectorizes documents' text using session.artifacts['count_vectorizer'] (falling back to slicing
        session.feature_matrix if vectorizer is absent). Returns 0.0 cleanly if either document vector
        has zero norm (e.g. empty text).

        Args:
            doc_index_1 (int): 0-based index of the first document.
            doc_index_2 (int): 0-based index of the second document.

        Returns:
            Dict[str, Any]: Summary containing result_id, doc_index_1, doc_index_2, cosine_similarity,
            and category labels of both documents if available.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "No dataset loaded. Run load_dataset_tool first."}

        n_docs = len(session.dataframe)
        if doc_index_1 < 0 or doc_index_1 >= n_docs:
            return {"error": f"doc_index_1 ({doc_index_1}) out of bounds [0, {n_docs - 1}]."}

        if doc_index_2 < 0 or doc_index_2 >= n_docs:
            return {"error": f"doc_index_2 ({doc_index_2}) out of bounds [0, {n_docs - 1}]."}

        vectorizer = session.artifacts.get("count_vectorizer")

        if vectorizer is not None:
            # Vectorize using saved count_vectorizer
            text1 = str(session.dataframe["text"].iloc[doc_index_1]) if pd.notna(session.dataframe["text"].iloc[doc_index_1]) else ""
            text2 = str(session.dataframe["text"].iloc[doc_index_2]) if pd.notna(session.dataframe["text"].iloc[doc_index_2]) else ""

            v1 = vectorizer.transform([text1])
            v2 = vectorizer.transform([text2])
        elif session.feature_matrix is not None:
            # Fallback to slicing session.feature_matrix
            v1 = session.feature_matrix[doc_index_1 : doc_index_1 + 1]
            v2 = session.feature_matrix[doc_index_2 : doc_index_2 + 1]
        else:
            return {"error": "No document-term matrix or vectorizer available. Run build_dtm_tool first."}

        # Check for zero norm
        norm1 = np.linalg.norm(v1.toarray() if hasattr(v1, "toarray") else np.asarray(v1))
        norm2 = np.linalg.norm(v2.toarray() if hasattr(v2, "toarray") else np.asarray(v2))

        if norm1 == 0.0 or norm2 == 0.0:
            sim_score = 0.0
        else:
            sim_matrix = cosine_similarity(v1, v2)
            sim_score = float(sim_matrix[0, 0])

        # Get category labels if available
        label1: Optional[str] = None
        label2: Optional[str] = None

        if session.labels is not None and len(session.labels) == n_docs:
            label1 = str(session.labels[doc_index_1])
            label2 = str(session.labels[doc_index_2])
        elif "category_name" in session.dataframe.columns:
            label1 = str(session.dataframe["category_name"].iloc[doc_index_1])
            label2 = str(session.dataframe["category_name"].iloc[doc_index_2])

        result_id = session.next_result_id("cosine_similarity")
        summary = {
            "result_id": result_id,
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2,
            "cosine_similarity": round(sim_score, 4),
            "label_1": label1,
            "label_2": label2,
        }

        session.store_result(
            "cosine_similarity_tool",
            {"doc_index_1": doc_index_1, "doc_index_2": doc_index_2},
            summary,
        )
        return summary

    @tool
    def feature_correlation_matrix_tool(top_n: int = 20) -> Dict[str, Any]:
        """Computes feature-versus-feature Pearson correlation matrix over top terms by variance.

        Operates on global DTM (session.feature_matrix / session.feature_names). Selects top_n terms by variance,
        computes Pearson correlation matrix, renders an annotated heatmap into session.pending_figure, and returns
        correlation matrix summary.

        Args:
            top_n (int): Number of top terms by variance to include (max 50, default 20).

        Returns:
            Dict[str, Any]: Summary containing result_id, terms, correlation_matrix (2D list), and n_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix available. Run build_dtm_tool first."}

        if top_n < 1:
            return {"error": "top_n must be at least 1."}

        top_n = min(top_n, 50)

        dtm = session.feature_matrix
        feature_names = session.feature_names

        # Compute term variances directly from sparse matrix without converting full DTM to dense
        if issparse(dtm):
            # var = E[X^2] - (E[X])^2
            mean = np.array(dtm.mean(axis=0)).ravel()
            mean_sq = np.array(dtm.power(2).mean(axis=0)).ravel()
            variances = mean_sq - (mean ** 2)
            variances = np.maximum(variances, 0)
        else:
            variances = np.var(np.asarray(dtm), axis=0, ddof=0)

        # Build DataFrame for deterministic sorting: descending by variance, ascending by term
        var_df = pd.DataFrame(
            {
                "term": feature_names,
                "col_idx": list(range(len(feature_names))),
                "variance": variances,
            }
        ).sort_values(by=["variance", "term"], ascending=[False, True]).reset_index(drop=True)

        top_selected = var_df.head(top_n)
        top_terms = top_selected["term"].tolist()
        top_indices = top_selected["col_idx"].tolist()

        if len(top_indices) == 0:
            return {"error": "No terms available in feature matrix."}

        # Slice only the selected top_n columns into a dense sub-matrix
        sliced_cols = dtm[:, top_indices]
        sub_matrix = sliced_cols.toarray() if issparse(sliced_cols) else np.asarray(sliced_cols)

        # Compute Pearson correlation matrix
        corr_matrix = np.corrcoef(sub_matrix, rowvar=False)

        # Handle scalar output when top_n == 1
        if np.ndim(corr_matrix) == 0:
            corr_matrix = np.array([[1.0]])

        # Handle NaNs and ensure diagonal is exactly 1.0
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        np.fill_diagonal(corr_matrix, 1.0)

        corr_matrix_list = [
            [round(float(val), 4) for val in row] for row in corr_matrix
        ]

        # Generate heatmap plot
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            xticklabels=top_terms,
            yticklabels=top_terms,
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            ax=ax,
        )
        ax.set_title(f"Feature Correlation Matrix (Top {len(top_terms)} terms by variance)")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("feature_correlation")
        summary = {
            "result_id": result_id,
            "terms": top_terms,
            "correlation_matrix": corr_matrix_list,
            "n_terms": len(top_terms),
        }

        session.store_result(
            "feature_correlation_matrix_tool",
            {"top_n": top_n},
            summary,
        )
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
