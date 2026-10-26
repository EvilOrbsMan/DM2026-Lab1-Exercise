from typing import Dict, Any, List, Optional, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"],
        n_components: int = 2,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Reduces document-term matrix dimensions (PCA, t-SNE, or UMAP) and generates a 2D scatter plot by category.

        Does not overwrite session.feature_matrix. Stores full coordinate array in session.artifacts[f'reduced_{method}'],
        sets a 2D scatter figure on session.pending_figure, and stores full_report DataFrame in session.results_store.
        Note: This tool generates its own scatter plot directly via session.pending_figure; full_report stores doc_id,
        category, and coordinates for evaluation rather than bar-charting via visualize_result_tool.

        Args:
            method (Literal["pca", "tsne", "umap"]): Dimensionality reduction algorithm.
            n_components (int): Target number of dimensions (default 2). Must be >= 2.
            perplexity (float): Perplexity parameter for t-SNE. Must be less than number of samples.
            n_neighbors (int): Number of neighbors parameter for UMAP. Must be >= 2 and <= n_samples.
            random_state (int): Seed for reproducible reduction. Default 42.

        Returns:
            Dict[str, Any]: Summary containing result_id, method, n_components, shape, coordinates (capped at 20 rows),
            and explained_variance_ratio (only for PCA).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix available. Run build_dtm_tool first."}

        if session.labels is None:
            return {"error": "No document labels available. Run build_dtm_tool first."}

        valid_methods = ["pca", "tsne", "umap"]
        if method not in valid_methods:
            return {"error": f"Invalid method '{method}'. Must be one of {valid_methods}."}

        dense_matrix = (
            session.feature_matrix.toarray()
            if hasattr(session.feature_matrix, "toarray")
            else np.asarray(session.feature_matrix)
        )
        n_samples, n_features = dense_matrix.shape

        if n_components < 2:
            return {"error": "n_components must be at least 2 for 2D visualization."}

        if method == "pca" and n_components > min(n_samples, n_features):
            return {
                "error": f"n_components ({n_components}) cannot exceed min(n_samples, n_features) ({min(n_samples, n_features)})."
            }

        # Parameter validation for t-SNE and UMAP
        if method == "tsne" and perplexity >= n_samples:
            return {
                "error": f"perplexity ({perplexity}) must be strictly less than the number of samples ({n_samples})."
            }

        if method == "umap" and not (2 <= n_neighbors <= n_samples):
            return {
                "error": f"n_neighbors ({n_neighbors}) must be between 2 and number of samples ({n_samples})."
            }

        # Run dimensionality reduction model
        explained_variance_ratio = None
        if method == "pca":
            model = PCA(n_components=n_components, random_state=random_state)
            coords = model.fit_transform(dense_matrix)
            explained_variance_ratio = model.explained_variance_ratio_.tolist()
        elif method == "tsne":
            model = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
            )
            coords = model.fit_transform(dense_matrix)
        elif method == "umap":
            import umap  # Lazy import to speed up module loading
            model = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=random_state,
            )
            coords = model.fit_transform(dense_matrix)

        # Store complete coordinate array in artifacts
        session.artifacts[f"reduced_{method}"] = coords

        # Build full_report DataFrame
        labels_arr = np.asarray(session.labels)
        col_names = [f"Dim{i+1}" for i in range(n_components)]
        report_df = pd.DataFrame(coords, columns=col_names)
        report_df.insert(0, "category", labels_arr)
        report_df.insert(0, "doc_id", np.arange(n_samples))

        # Generate 2D Scatter Plot directly on session.pending_figure for every valid run
        fig, ax = plt.subplots(figsize=(8, 6))
        categories = session.categories if session.categories else list(np.unique(labels_arr))
        for cat in categories:
            mask = labels_arr == cat
            if np.any(mask):
                ax.scatter(coords[mask, 0], coords[mask, 1], label=str(cat), alpha=0.7)

        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.legend(title="Category")
        ax.grid(True, linestyle="--", alpha=0.5)
        session.pending_figure = fig

        result_id = session.next_result_id(f"reduce_{method}")
        summary = {
            "result_id": result_id,
            "method": method,
            "n_components": n_components,
            "shape": list(coords.shape),
            "coordinates": coords[:20].tolist(),
        }

        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        session.store_result(
            "reduce_dimensions_tool",
            {
                "method": method,
                "n_components": n_components,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors,
                "random_state": random_state,
            },
            summary,
            full_report=report_df,
        )

        return summary

    return [reduce_dimensions_tool]
