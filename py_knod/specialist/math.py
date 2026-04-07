"""Shared math utilities for embedding operations."""

import numpy as np

_EPS = 1e-10


def cosine(a: np.ndarray, b: np.ndarray) -> float:
	"""Cosine similarity between two vectors."""
	a_norm = a / (np.linalg.norm(a) + _EPS)
	b_norm = b / (np.linalg.norm(b) + _EPS)
	return float(np.dot(a_norm, b_norm))


def normalize(v: np.ndarray) -> np.ndarray:
	"""L2-normalize a vector."""
	return v / (np.linalg.norm(v) + _EPS)
