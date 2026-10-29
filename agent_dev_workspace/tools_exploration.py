from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
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

    return [cosine_similarity_tool]
