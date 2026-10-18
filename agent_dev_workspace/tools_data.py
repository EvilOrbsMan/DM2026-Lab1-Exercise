from typing import Dict, Any
import nltk
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

    return [tokenize_tool]
