from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    micro_f1: float


def validate_arrays(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_score = np.asarray(y_score, dtype=np.float32)
    if y_true.ndim != 2 or y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same two-dimensional shape")
    if not np.isfinite(y_score).all():
        raise ValueError("y_score contains non-finite values")
    return y_true, y_score


def select_global_threshold(
    y_true,
    y_score,
    thresholds: Iterable[float] = (0.0,),
) -> ThresholdSelection:
    y_true, y_score = validate_arrays(y_true, y_score)
    candidates = sorted({float(value) for value in thresholds})
    if not candidates or candidates[0] <= 0 or candidates[-1] >= 1:
        raise ValueError("threshold candidates must lie strictly between zero and one")
    best = ThresholdSelection(candidates[0], -1.0)
    for threshold in candidates:
        prediction = y_score >= threshold
        score = float(f1_score(y_true, prediction, average="micro", zero_division=0))
        if score > best.micro_f1 or (
            np.isclose(score, best.micro_f1) and threshold > best.threshold
        ):
            best = ThresholdSelection(threshold, score)
    return best


def predict_threshold(y_score, threshold: float) -> np.ndarray:
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie strictly between zero and one")
    return (np.asarray(y_score) >= threshold).astype(np.float32)


def predict_cardinality(y_score, cardinality, max_cardinality: int = 0) -> np.ndarray:
    y_score = np.asarray(y_score, dtype=np.float32)
    cardinality = np.asarray(cardinality, dtype=np.int64)
    if y_score.ndim != 2 or cardinality.shape != (len(y_score),):
        raise ValueError("cardinality must contain one value per score row")
    output = np.zeros_like(y_score, dtype=np.float32)
    for row, count in enumerate(cardinality):
        if count < 0 or count > max_cardinality:
            raise ValueError("predicted cardinality is outside the configured range")
        if count:
            indices = np.argsort(-y_score[row], kind="stable")[: int(count)]
            output[row, indices] = 1.0
    return output


def precision_at_k(y_true, y_score, k: int, monitored_only: bool = False) -> float:
    y_true, y_score = validate_arrays(y_true, y_score)
    if monitored_only:
        keep = y_true.sum(axis=1) > 0
        y_true, y_score = y_true[keep], y_score[keep]
    if len(y_true) == 0:
        return 0.0
    order = np.argsort(-y_score, axis=1, kind="stable")[:, :k]
    hits = np.take_along_axis(y_true > 0, order, axis=1).sum(axis=1)
    return float(np.mean(hits / k))


def map_at_k(y_true, y_score, k: int, monitored_only: bool = False) -> float:
    y_true, y_score = validate_arrays(y_true, y_score)
    if monitored_only:
        keep = y_true.sum(axis=1) > 0
        y_true, y_score = y_true[keep], y_score[keep]
    if len(y_true) == 0:
        return 0.0
    relevant = y_true > 0
    order = np.argsort(-y_score, axis=1, kind="stable")[:, :k]
    ranked = np.take_along_axis(relevant, order, axis=1).astype(np.float64)
    precision = np.cumsum(ranked, axis=1) / np.arange(1, k + 1)
    denominator = np.minimum(relevant.sum(axis=1), k)
    average_precision = np.divide(
        np.sum(precision * ranked, axis=1),
        denominator,
        out=np.zeros(len(ranked), dtype=np.float64),
        where=denominator > 0,
    )
    return float(np.mean(average_precision))


def evaluate_predictions(
    y_true,
    y_score,
    y_pred,
    *,
    ranking_k: int,
    scenario: str,
) -> dict:
    y_true, y_score = validate_arrays(y_true, y_score)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    if y_pred.shape != y_true.shape or scenario not in {"closed", "open"}:
        raise ValueError("invalid prediction shape or scenario")
    monitored_only = scenario == "open"
    valid_auc = (y_true.sum(axis=0) > 0) & (y_true.sum(axis=0) < len(y_true))
    macro_auc = (
        float(roc_auc_score(y_true[:, valid_auc], y_score[:, valid_auc], average="macro"))
        if np.any(valid_auc)
        else 0.0
    )
    true_cardinality = y_true.sum(axis=1).astype(np.int64)
    predicted_cardinality = y_pred.sum(axis=1).astype(np.int64)
    result = {
        "micro_precision": float(
            precision_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "micro_recall": float(
            recall_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "p_at_k": precision_at_k(y_true, y_score, ranking_k, monitored_only),
        "map_at_k": map_at_k(y_true, y_score, ranking_k, monitored_only),
        "macro_auc": macro_auc,
        "cardinality_accuracy": float(np.mean(predicted_cardinality == true_cardinality)),
        "cardinality_mae": float(
            np.mean(np.abs(predicted_cardinality - true_cardinality))
        ),
        "avg_predicted_cardinality": float(np.mean(predicted_cardinality)),
        "n_samples": int(len(y_true)),
    }
    if scenario == "open":
        background = true_cardinality == 0
        result["background_zero_rate"] = (
            float(np.mean(predicted_cardinality[background] == 0))
            if np.any(background)
            else 0.0
        )
    return result
