from typing import Dict, Any, List, Optional
import numpy as np
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

    return [build_dtm_tool]
