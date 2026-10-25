import os
from typing import Dict, Any, List, Optional
import nltk
import pandas as pd
from langchain_core.tools import tool

# Ensure punkt tokenizer models are available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def make_tools(session):
    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """Tokenizes the text column of session.dataframe into unigram tokens using NLTK word_tokenize.

        Requires session.dataframe to be populated with a 'text' column (typically run after loading data
        and before building a DTM). Converts text to lowercase before tokenizing each document.

        Updates session.dataframe['unigrams'] and session.artifacts['unigrams'].

        Returns:
            Dict[str, Any]: Summary containing result_id, n_documents, total_tokens,
            and sample tokens from the first document.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "No dataframe loaded or 'text' column missing. Load a dataset first."}

        unigrams_list = []
        total_tokens = 0

        for text in session.dataframe["text"]:
            text_str = str(text) if text is not None and not (isinstance(text, float) and text != text) else ""
            if not text_str.strip():
                tokens = []
            else:
                tokens = nltk.word_tokenize(text_str.lower())
            unigrams_list.append(tokens)
            total_tokens += len(tokens)

        # Update session state
        session.dataframe["unigrams"] = unigrams_list
        session.artifacts["unigrams"] = unigrams_list

        result_id = session.next_result_id("tokenize")
        sample_tokens = unigrams_list[0] if unigrams_list else []

        summary = {
            "result_id": result_id,
            "n_documents": len(unigrams_list),
            "total_tokens": total_tokens,
            "sample_tokens_doc_0": sample_tokens[:10],
        }

        session.store_result("tokenize_tool", {}, summary)
        return summary

    @tool
    def list_files_tool(subdirectory: str = "") -> Dict[str, Any]:
        """Lists files and directories in the repository or a given subdirectory prior to loading data.

        This tool is used before loading data to discover file paths and dataset locations in the workspace.
        Filters out hidden files/folders (starting with '.') and Python cache folders ('__pycache__').

        Args:
            subdirectory (str): Subdirectory path relative to the root workspace (default is "").

        Returns:
            Dict[str, Any]: Dictionary containing result_id, subdirectory, directories, and files.
        """
        target_dir = os.path.normpath(subdirectory) if subdirectory else "."

        if not os.path.exists(target_dir) or not os.path.isdir(target_dir):
            return {"error": f"Directory '{subdirectory}' does not exist or is not a directory."}

        try:
            entries = os.listdir(target_dir)
        except Exception as e:
            return {"error": f"Failed to list directory contents: {str(e)}"}

        files = []
        directories = []

        for entry in sorted(entries):
            # Ignore hidden files/folders and __pycache__
            if entry.startswith(".") or entry == "__pycache__":
                continue

            full_path = os.path.join(target_dir, entry)
            if os.path.isdir(full_path):
                directories.append(entry)
            else:
                files.append(entry)

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files,
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """Checks session.dataframe for missing or empty text values as part of data quality check.

        Typically used after loading dataset (load_dataset_tool) and before establishing document-term matrix
        (build_dtm_tool) to verify data quality. Standard pandas isnull()/isna() only catches NaNs/Nones, missing
        whitespace-only or empty strings. This tool catches both NaN/None and empty/whitespace text entries without
        modifying or deleting rows.

        Returns:
            Dict[str, Any]: Summary containing result_id, n_documents, missing_count, valid_count,
            and missing_indices (capped at 20 indices).
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "No dataframe loaded or 'text' column missing. Load a dataset first."}

        df = session.dataframe
        text_col = df["text"]

        # Standard NaN/None check combined with empty/whitespace string check
        is_missing_mask = text_col.isna() | (text_col.astype(str).str.strip() == "")

        missing_indices = df.index[is_missing_mask].tolist()
        missing_count = len(missing_indices)
        n_documents = len(df)
        valid_count = n_documents - missing_count

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "n_documents": n_documents,
            "missing_count": missing_count,
            "valid_count": valid_count,
            "missing_indices": missing_indices[:20],
        }

        session.store_result("check_missing_tool", {}, summary)
        return summary

    @tool
    def check_duplicates_tool(
        drop: bool = False, subset: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Checks for and optionally drops duplicate documents in session.dataframe. Typically used after loading data (load_dataset_tool) and before building a DTM (build_dtm_tool).

        Identifies redundant extra copies using keep='first' for duplicate_count, and uses keep=False when dropping rows (removing every copy of a duplicated document, matching Master's drop logic).

        Note on Master deviation: Master compares all columns by default. Here, subset defaults to ['text'] if present to avoid unhashable type errors if tokenization ('unigrams' list column) has already been run.

        Args:
            drop (bool): If True, drops all identified duplicate rows from session.dataframe using keep=False. Default is False.
            subset (Optional[List[str]]): List of column names to consider for identifying duplicates.
                If None, defaults to ['text'] if present in dataframe.

        Returns:
            Dict[str, Any]: Summary containing result_id, n_documents_before, duplicate_count,
            rows_deleted, n_documents_after, duplicate_indices (capped at max 20), and optional warning.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded. Load a dataset first."}

        df = session.dataframe

        # Determine subset of columns to check
        if subset is None:
            check_cols = ["text"] if "text" in df.columns else list(df.columns)
        else:
            check_cols = subset

        # Validate check_cols exist in dataframe
        missing_cols = [col for col in check_cols if col not in df.columns]
        if missing_cols:
            return {"error": f"Columns not found in dataframe: {missing_cols}"}

        # Calculate duplicate_count (extra copies, keep='first')
        dup_extra_mask = df.duplicated(subset=check_cols, keep="first")
        duplicate_count = int(dup_extra_mask.sum())

        # Identify all rows involved in duplication (keep=False)
        dup_all_mask = df.duplicated(subset=check_cols, keep=False)
        duplicate_indices = df.index[dup_all_mask].tolist()
        n_documents_before = len(df)

        warning_msg = None

        if drop and len(duplicate_indices) > 0:
            session.dataframe = df.drop(index=duplicate_indices)
            rows_deleted = len(duplicate_indices)

            # Check if per-row artifacts or DTM exist and add warning
            if (
                "unigrams" in session.dataframe.columns
                or session.feature_matrix is not None
                or "count_vectorizer" in session.artifacts
            ):
                warning_msg = (
                    "Rows were dropped from session.dataframe after tokenization or DTM creation. "
                    "Existing downstream attributes/artifacts (e.g. unigrams, feature_matrix, count_vectorizer) "
                    "are now out of sync with session.dataframe. Re-run tokenize_tool / build_dtm_tool."
                )
        else:
            rows_deleted = 0

        n_documents_after = len(session.dataframe)

        result_id = session.next_result_id("check_duplicates")
        summary: Dict[str, Any] = {
            "result_id": result_id,
            "n_documents_before": n_documents_before,
            "duplicate_count": duplicate_count,
            "rows_deleted": rows_deleted,
            "n_documents_after": n_documents_after,
            "duplicate_indices": duplicate_indices[:20],
        }
        if warning_msg:
            summary["warning"] = warning_msg

        session.store_result(
            "check_duplicates_tool",
            {"drop": drop, "subset": check_cols},
            summary,
        )
        return summary

    return [tokenize_tool, list_files_tool, check_missing_tool, check_duplicates_tool]
