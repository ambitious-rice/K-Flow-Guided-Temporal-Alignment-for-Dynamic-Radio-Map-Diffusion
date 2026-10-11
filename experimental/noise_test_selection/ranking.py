"""Ranking on pooled-window unobserved MSE, with equal rate/sigma weight."""
import numpy as np


def score_video(matrices):
    a = np.asarray(matrices, dtype=np.float64)
    if a.shape != (3, 4, 4) or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError('Expected three finite nonnegative 4x4 error matrices')
    correct = np.diagonal(a, axis1=1, axis2=2)
    wrong = a.copy()
    wrong[:, np.arange(4), np.arange(4)] = np.inf
    best_wrong = wrong.min(axis=2)
    gains = 1 - correct / np.maximum(best_wrong, 1e-12)
    return dict(score=float(gains.mean()), score_percent=float(100*gains.mean()),
                correct_unobserved_mse=float(correct.mean()),
                correct_minimum_cells=int((correct <= best_wrong + 1e-12).sum()),
                gains_percent=(100*gains).tolist(), matrices=a.tolist())


def ranked(records):
    # Stable tie breaking does not add an all-cells-pass requirement.
    return sorted(records, key=lambda r: (-r['score'], r['correct_unobserved_mse'], r['video_id']))
