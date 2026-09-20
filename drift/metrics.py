import numpy as np


def compute_psi(
    reference_values: np.ndarray,
    current_values: np.ndarray,
    bin_edges: np.ndarray,
) -> float:
    """
    Population Stability Index (PSI) using count-based Laplace smoothing.

    Uses Add-One (Laplace) smoothing on raw counts before computing percentages.
    This prevents tiny windows (or empty buckets) from producing extreme log-ratio
    penalties that cause false-positive drift alerts.
    """
    ref_counts, _ = np.histogram(reference_values, bins=bin_edges)
    cur_counts, _ = np.histogram(current_values, bins=bin_edges)
    n_bins = len(bin_edges) - 1

    # Add 1 pseudo-count per bin
    ref_pct = (ref_counts + 1) / (ref_counts.sum() + n_bins)
    cur_pct = (cur_counts + 1) / (cur_counts.sum() + n_bins)

    # PSI sum over all bins
    psi_value = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi_value)


def compute_embedding_drift(
    reference_centroid_unit: np.ndarray, current_embeddings: np.ndarray
) -> dict:
    """
    Cosine distance between reference embedding centroid and current window centroid.
    """
    current_centroid = current_embeddings.mean(axis=0)
    norm = np.linalg.norm(current_centroid)
    current_unit = current_centroid / norm if norm > 0 else current_centroid
    cosine_distance = 1.0 - float(np.dot(reference_centroid_unit, current_unit))

    per_sample_sim = current_embeddings @ reference_centroid_unit / (
        np.linalg.norm(current_embeddings, axis=1) + 1e-8
    )
    return {
        "cosine_distance": cosine_distance,
        "mean_similarity": float(per_sample_sim.mean()),
        "std_similarity": float(per_sample_sim.std()),
    }