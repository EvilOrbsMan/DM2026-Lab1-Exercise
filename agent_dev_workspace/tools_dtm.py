from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import issparse
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def build_dtm_tool(
        min_df: int = 1,
        max_df: float = 1.0,
        ngram_range: List[int] = [1, 1],
        stop_words: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Builds document-term matrix (DTM) from session.dataframe['text'] into session.feature_matrix and session.feature_names. Typically used after data quality checks (check_missing_tool, check_duplicates_tool) and before feature filtering/exploration tools.

        Stores fitted CountVectorizer in session.artifacts['count_vectorizer'], updates session.labels,
        and reports document-term matrix sparsity metrics.

        Args:
            min_df (int): Minimum document frequency cutoff for vectorizer. Default is 1.
            max_df (float): Maximum document frequency ratio/count cutoff for vectorizer. Default is 1.0.
            ngram_range (List[int]): Range of n-gram sizes [min_n, max_n]. Default is [1, 1] (unigrams).
            stop_words (Optional[str]): Stop words setting (e.g. 'english' or None). Default is None.

        Returns:
            Dict[str, Any]: Summary containing result_id, n_documents, n_terms, shape, non_zero,
            total_elements, and sparsity_pct.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "No dataframe loaded or 'text' column missing. Load a dataset first."}

        # Safe conversion of NaN/None to empty strings
        texts = session.dataframe["text"].fillna("").astype(str).tolist()

        ngram_tuple = (ngram_range[0], ngram_range[1]) if len(ngram_range) >= 2 else (1, 1)

        vectorizer = CountVectorizer(
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_tuple,
            stop_words=stop_words,
        )

        try:
            dtm = vectorizer.fit_transform(texts)
        except ValueError as e:
            return {"error": f"Failed to build document-term matrix: {str(e)}"}

        feature_names = vectorizer.get_feature_names_out().tolist()

        # Update session state attributes
        session.feature_matrix = dtm
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)

        n_rows, n_cols = dtm.shape
        non_zero = int(dtm.nnz)
        total_elements = int(n_rows * n_cols)
        sparsity_pct = float(100.0 * (1.0 - non_zero / total_elements)) if total_elements > 0 else 0.0

        result_id = session.next_result_id("build_dtm")
        summary = {
            "result_id": result_id,
            "n_documents": n_rows,
            "n_terms": n_cols,
            "shape": [n_rows, n_cols],
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
        }

        session.store_result(
            "build_dtm_tool",
            {
                "min_df": min_df,
                "max_df": max_df,
                "ngram_range": ngram_range,
                "stop_words": stop_words,
            },
            summary,
        )
        return summary

    @tool
    def term_frequency_tool(top_n: int = 20) -> Dict[str, Any]:
        """Calculates total term frequency across all documents in session.feature_matrix.

        Requires session.feature_matrix and session.feature_names to be populated by build_dtm_tool.
        Sums count per term across all documents, sorts terms by frequency descending (breaking ties
        alphabetically by term name), and passes full report DataFrame (columns 'term' and 'frequency')
        to session.store_result for visualization by visualize_result_tool.

        Args:
            top_n (int): Number of top terms to include in the summary output (default 20).

        Returns:
            Dict[str, Any]: Summary containing result_id, n_terms, total_count, and top_terms list/dict.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix available. Run build_dtm_tool first."}

        if top_n < 1:
            return {"error": "top_n must be at least 1."}

        dtm = session.feature_matrix
        feature_names = session.feature_names

        # Sum frequencies across axis=0
        if issparse(dtm):
            counts = dtm.sum(axis=0).A1
        else:
            counts = np.sum(np.asarray(dtm), axis=0)

        # Convert counts to plain Python numbers
        counts = [int(c) for c in counts]

        # Build full report DataFrame
        report_df = pd.DataFrame(
            {
                "term": feature_names,
                "frequency": counts,
            }
        )

        # Sort descending by frequency, then ascending by term for tie-breaking
        report_df = report_df.sort_values(
            by=["frequency", "term"], ascending=[False, True]
        ).reset_index(drop=True)

        top_terms = [
            {"term": str(r["term"]), "frequency": int(r["frequency"])}
            for _, r in report_df.head(top_n).iterrows()
        ]

        result_id = session.next_result_id("term_frequency")
        summary = {
            "result_id": result_id,
            "n_terms": len(feature_names),
            "total_count": int(sum(counts)),
            "top_terms": top_terms,
        }

        session.store_result(
            "term_frequency_tool",
            {"top_n": top_n},
            summary,
            full_report=report_df,
        )
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> Dict[str, Any]:
        """Plots and returns a heatmap slice of the raw document-term matrix (DTM).

        Takes a positional slice of session.feature_matrix and session.feature_names without sorting
        or filtering. Max cap is 50 for both n_terms and n_documents. If the available matrix has fewer
        terms or documents than requested, all available rows and columns are included.

        Args:
            n_terms (int): Number of terms (columns) from the start of feature_names to include (max 50, default 20).
            n_documents (int): Number of documents (rows) from the start of feature_matrix to include (max 50, default 20).

        Returns:
            Dict[str, Any]: Summary dictionary containing result_id, matrix slice, terms, document indices,
            category labels (if available), shape, and capped flag if clamped to 50.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix available. Run build_dtm_tool first."}

        if n_terms < 1 or n_documents < 1:
            return {"error": "n_terms and n_documents must both be at least 1."}

        capped = False
        if n_terms > 50:
            n_terms = 50
            capped = True
        if n_documents > 50:
            n_documents = 50
            capped = True

        dtm = session.feature_matrix
        feature_names = session.feature_names

        # Take raw positional slice without sorting or filtering
        sliced_dtm = dtm[:n_documents, :n_terms]
        sliced_terms = feature_names[:n_terms]

        if issparse(sliced_dtm):
            dense_matrix = sliced_dtm.toarray()
        else:
            dense_matrix = np.asarray(sliced_dtm)

        # Convert to dense 2D list of plain Python numbers
        matrix_list = [[int(val) if isinstance(val, (int, np.integer)) else float(val) for val in row] for row in dense_matrix]

        doc_indices = list(range(dense_matrix.shape[0]))

        labels = None
        if session.labels is not None:
            labels = [str(lbl) for lbl in session.labels[:n_documents]]
        elif session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = [str(lbl) for lbl in session.dataframe["category_name"].iloc[:n_documents]]

        # Generate plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            dense_matrix,
            annot=True,
            fmt="d" if np.issubdtype(dense_matrix.dtype, np.integer) else "g",
            xticklabels=sliced_terms,
            yticklabels=labels if labels is not None else doc_indices,
            cmap="Blues",
            ax=ax,
        )
        ax.set_title(f"Document-Term Matrix Heatmap (First {len(sliced_terms)} terms x {dense_matrix.shape[0]} docs)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "matrix": matrix_list,
            "terms": sliced_terms,
            "documents": doc_indices,
            "labels": labels,
            "shape": list(dense_matrix.shape),
            "capped": capped,
        }

        session.store_result(
            "dtm_heatmap_tool",
            {"n_terms": n_terms, "n_documents": n_documents},
            summary,
        )
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
