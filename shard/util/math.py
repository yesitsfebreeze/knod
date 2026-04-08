import numpy as np

_EPS = 1e-10


def cosine(a: np.ndarray, b: np.ndarray) -> float:
	return float(np.dot(normalize(a), normalize(b)))


def normalize(v: np.ndarray) -> np.ndarray:
	return v / (np.linalg.norm(v) + _EPS)
