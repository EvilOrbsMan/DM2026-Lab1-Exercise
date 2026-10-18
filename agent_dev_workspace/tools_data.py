import os
from typing import Dict, Any
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

    return [tokenize_tool, list_files_tool, check_missing_tool]
