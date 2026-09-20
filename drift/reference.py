import numpy as np
import pandas as pd
import torch


class ReferenceData:
    """
    Loads the Phase 1 reference artifact for one (model_id, version).
    Tabular -> Quantile bin edges per feature from the reference sample.
    Embedding models -> A unit-normalized reference centroid.
    """

    def __init__(self, model_id: str, reference_file: str, n_bins: int = 10):
        self.model_id = model_id
        self.kind = "tabular" if model_id == "fraud" else "embedding"

        if self.kind == "tabular":
            self.reference_df = pd.read_parquet(reference_file)
            self.bin_edges = {
                col: self._quantile_bins(self.reference_df[col].values, n_bins)
                for col in self.reference_df.columns
            }
        else:
            emb = torch.load(reference_file, map_location="cpu")
            self.reference_embeddings = (
                emb.numpy() if hasattr(emb, "numpy") else np.asarray(emb)
            )
            centroid = self.reference_embeddings.mean(axis=0)
            norm = np.linalg.norm(centroid)
            self.centroid_unit = centroid / norm if norm > 0 else centroid

    @staticmethod
    def _quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
        # Calculate percentile edges across requested bins
        percentiles = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(values, percentiles)

        # Deduplicate edges (handles repeated values/sparse features)
        edges = np.unique(edges)

        # If deduplication collapsed edges, fallback to min/max with small epsilon
        if len(edges) < 2:
            min_val = float(values.min()) if len(values) > 0 else 0.0
            max_val = float(values.max()) if len(values) > 0 else 1.0
            if min_val == max_val:
                min_val -= 1e-5
                max_val += 1e-5
            edges = np.array([min_val, max_val])

        # Force infinite outer boundaries so no sample falls out of bounds
        edges[0] = -np.inf
        edges[-1] = np.inf
        return edges