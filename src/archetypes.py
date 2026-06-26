"""
Archetypes classification system.

Derives the user's workflow archetype based on the 5 dimension scores.
Applies AGENCY weights to discount automated agent behaviors (Verification, Context)
so the archetype reflects the human's true intent.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# Dimensional weights to discount automated/default agent behaviors
AGENCY: dict[str, float] = {
    "Direction": 1.0,
    "Verification": 0.35,
    "Context": 0.15,
    "Iteration": 1.0,
    "Toolcraft": 0.8,
}

DIMENSIONS: tuple[str, ...] = ("Direction", "Verification", "Context", "Iteration", "Toolcraft")

# Standard ideal vectors for each archetype
IDEAL_VECTORS: dict[str, tuple[float, ...]] = {
    "Autonomous Agent": (0.2, 0.9, 0.9, 0.2, 0.9),
    "Architect": (0.9, 0.5, 0.9, 0.4, 0.6),
    "Debugger": (0.4, 0.9, 0.8, 0.9, 0.7),
    "Collaborator": (0.8, 0.6, 0.6, 0.8, 0.7),
    "Sprinter": (0.9, 0.2, 0.2, 0.4, 0.3),
}


def _z_score_normalize(vector: Sequence[float]) -> tuple[float, ...]:
    """Normalize a vector using z-score (subtract mean, divide by stdev)."""
    if not vector:
        return ()
    n = len(vector)
    if n == 1:
        return (0.0,)
    mean = sum(vector) / n
    variance = sum((x - mean) ** 2 for x in vector) / n
    stdev = math.sqrt(variance)
    if stdev == 0:
        return tuple(0.0 for _ in vector)
    return tuple((x - mean) / stdev for x in vector)


def _cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(v1, v2, strict=True))
    mag1 = math.sqrt(sum(x * x for x in v1))
    mag2 = math.sqrt(sum(y * y for y in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)


def classify_archetype(scores: dict[str, float]) -> str:
    """Classify the user's workflow into an archetype based on 5 dimensions.

    Args:
        scores: A dictionary mapping dimension names to their scores [0.0, 1.0].

    Returns:
        The name of the assigned archetype, or a blended label.
    """
    # 1. Ensure all dimensions are present, default to 0.5
    raw_vector = [scores.get(dim, 0.5) for dim in DIMENSIONS]

    # 2. Apply AGENCY weights
    agency_weights = [AGENCY.get(dim, 1.0) for dim in DIMENSIONS]
    weighted_vector = [val * weight for val, weight in zip(raw_vector, agency_weights, strict=True)]

    # 3. Z-score normalize the observed vector
    norm_obs = _z_score_normalize(weighted_vector)

    # 4. Compare against z-score normalized ideal vectors
    similarities: list[tuple[str, float]] = []
    for name, ideal in IDEAL_VECTORS.items():
        # The ideal vectors also get z-score normalized to represent relative profiles
        norm_ideal = _z_score_normalize(ideal)
        sim = _cosine_similarity(norm_obs, norm_ideal)
        similarities.append((name, sim))

    # 5. Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)

    # 6. Blended label if top 2 are very close (< 0.06 diff)
    if len(similarities) >= 2:
        top1_name, top1_sim = similarities[0]
        top2_name, top2_sim = similarities[1]
        if (top1_sim - top2_sim) < 0.06:
            return f"{top1_name} / {top2_name}"

    return similarities[0][0]
