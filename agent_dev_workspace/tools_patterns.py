import os
import io
import tempfile
import contextlib
from typing import Dict, Any, List, Optional, Literal
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"],
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Mines frequent patterns for a single document category using PAMI algorithms (FPGrowth, Top-K/FAE, MaxFPGrowth).

        Builds a fresh category-local document-term matrix with CountVectorizer(stop_words="english") and applies
        one of three filtering methods (variance, tfidf, term_frequency). Converts kept terms into a temporary
        transactional DB and mines patterns using PAMI.

        Args:
            category_name (str): The target category name in session.dataframe.
            filtering_method (Literal["variance", "tfidf", "term_frequency"]): Vocabulary filtering method.
            algorithm (Literal["fpgrowth", "topk", "maxfpgrowth"]): PAMI mining algorithm to run. Default "fpgrowth".
            min_sup (Optional[int]): Minimum support threshold (raw count) for "fpgrowth" and "maxfpgrowth".
            k (Optional[int]): Number of top patterns to return for "topk".

        Returns:
            Dict[str, Any]: Summary containing result_id, category_name, filtering_method, algorithm,
            n_terms_kept, patterns (list of dicts with 'pattern' as list of str and 'support' as int, capped at 20),
            and optional note or error.
        """
        # Step 0: Input validation
        if session.dataframe is None:
            return {"error": "No working dataframe loaded. Run load_dataset_tool first."}

        df = session.dataframe
        if "category_name" not in df.columns or "text" not in df.columns:
            return {"error": "DataFrame missing 'category_name' or 'text' column."}

        if "category_name" in df.columns and category_name not in df["category_name"].values:
            return {"error": f"Category '{category_name}' not found in dataframe."}

        valid_filtering_methods = ["variance", "tfidf", "term_frequency"]
        if filtering_method not in valid_filtering_methods:
            return {"error": f"Invalid filtering_method '{filtering_method}'. Must be one of {valid_filtering_methods}."}

        valid_algorithms = ["fpgrowth", "topk", "maxfpgrowth"]
        if algorithm not in valid_algorithms:
            return {"error": f"Invalid algorithm '{algorithm}'. Must be one of {valid_algorithms}."}

        if algorithm in ["fpgrowth", "maxfpgrowth"] and min_sup is None:
            return {"error": f"min_sup is required for algorithm '{algorithm}'."}

        if algorithm == "topk" and k is None:
            return {"error": "k is required for algorithm 'topk'."}

        cat_df = df[df["category_name"] == category_name]

        # Step 1: Category-local CountVectorizer with English stopwords, handling NaNs
        clean_text = cat_df["text"].fillna("").astype(str)
        vectorizer = CountVectorizer(stop_words="english")
        try:
            dtm = vectorizer.fit_transform(clean_text)
            feature_names = vectorizer.get_feature_names_out()
        except ValueError:
            # Handles cases where no terms exist in vocabulary
            dtm = None
            feature_names = []

        if dtm is None or dtm.shape[1] == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_terms_kept": 0,
                "patterns": [],
                "note": "no terms survived filtering",
            }
            session.artifacts[f"patterns_{category_name}_{filtering_method}_{algorithm}"] = []
            session.store_result(
                "mine_patterns_tool",
                {
                    "category_name": category_name,
                    "filtering_method": filtering_method,
                    "algorithm": algorithm,
                    "min_sup": min_sup,
                    "k": k,
                },
                summary,
            )
            return summary

        term_df = pd.DataFrame(dtm.toarray(), columns=feature_names)

        # Step 2: Percentile-based filtering math with inclusive bounds (>= and <=)
        if filtering_method == "variance":
            variances = term_df.var(axis=0, ddof=1)
            p5 = np.percentile(variances, 5)
            p95 = np.percentile(variances, 95)
            kept_cols = [col for col, v in variances.items() if p5 <= v <= p95]
        elif filtering_method == "term_frequency":
            counts = term_df.sum(axis=0)
            p5 = np.percentile(counts, 5)
            p95 = np.percentile(counts, 95)
            kept_cols = [col for col, c in counts.items() if p5 <= c <= p95]
        elif filtering_method == "tfidf":
            tfidf = TfidfTransformer().fit_transform(dtm)
            mean_tfidf = np.array(tfidf.mean(axis=0)).flatten()
            p20 = np.percentile(mean_tfidf, 20)
            kept_cols = [col for col, score in zip(feature_names, mean_tfidf) if score >= p20]

        filtered_df = term_df[kept_cols]

        # Handle zero terms surviving filtering
        if filtered_df.shape[1] == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_terms_kept": 0,
                "patterns": [],
                "note": "no terms survived filtering",
            }
            session.artifacts[f"patterns_{category_name}_{filtering_method}_{algorithm}"] = []
            session.store_result(
                "mine_patterns_tool",
                {
                    "category_name": category_name,
                    "filtering_method": filtering_method,
                    "algorithm": algorithm,
                    "min_sup": min_sup,
                    "k": k,
                },
                summary,
            )
            return summary

        # Step 3: DF2DB and PAMI mining inside redirect_stdout context
        patterns_list = []

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
            tmp_path = tmp.name

        try:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                from PAMI.extras.convert import DF2DB
                converter = DF2DB.DF2DB(filtered_df)
                converter.convert2TransactionalDatabase(tmp_path, ">=", 1)

                if algorithm == "fpgrowth":
                    from PAMI.frequentPattern.basic import FPGrowth
                    miner = FPGrowth.FPGrowth(iFile=tmp_path, minSup=int(min_sup), sep="\t")
                elif algorithm == "topk":
                    from PAMI.frequentPattern.topk import FAE
                    miner = FAE.FAE(iFile=tmp_path, k=int(k), sep="\t")
                elif algorithm == "maxfpgrowth":
                    from PAMI.frequentPattern.maximal import MaxFPGrowth
                    miner = MaxFPGrowth.MaxFPGrowth(iFile=tmp_path, minSup=int(min_sup), sep="\t")

                miner.mine()
                patterns_df = miner.getPatternsAsDataFrame()

            if patterns_df is not None and not patterns_df.empty:
                p_col = "Patterns" if "Patterns" in patterns_df.columns else patterns_df.columns[0]
                s_col = "Support" if "Support" in patterns_df.columns else patterns_df.columns[1]

                for _, row in patterns_df.iterrows():
                    raw_pattern = str(row[p_col])
                    supp = int(row[s_col])
                    terms = raw_pattern.split("\t")
                    patterns_list.append({"pattern": terms, "support": supp})
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # Sort patterns by support descending
        patterns_list.sort(key=lambda x: x["support"], reverse=True)

        full_report_df = pd.DataFrame(
            {
                "pattern": [" ".join(p["pattern"]) for p in patterns_list],
                "support": [p["support"] for p in patterns_list],
            }
        )

        result_id = session.next_result_id("mine_patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "n_terms_kept": len(kept_cols),
            "patterns": patterns_list[:20],
        }

        if len(patterns_list) == 0:
            summary["note"] = "no patterns found"

        session.artifacts[f"patterns_{category_name}_{filtering_method}_{algorithm}"] = patterns_list
        session.store_result(
            "mine_patterns_tool",
            {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k,
            },
            summary,
            full_report=full_report_df,
        )

        return summary

    return [mine_patterns_tool]
