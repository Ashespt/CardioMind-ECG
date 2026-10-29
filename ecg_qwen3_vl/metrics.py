from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def binary_metrics(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    targets = targets.astype(bool)
    predictions = predictions.astype(bool)
    tp = np.logical_and(targets, predictions).sum(axis=0).astype(np.float64)
    fp = np.logical_and(~targets, predictions).sum(axis=0).astype(np.float64)
    fn = np.logical_and(targets, ~predictions).sum(axis=0).astype(np.float64)
    tn = np.logical_and(~targets, ~predictions).sum(axis=0).astype(np.float64)
    sensitivity = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) > 0)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    total_positives = (tp + fn).sum()
    micro_recall = 0.0 if total_positives == 0 else tp.sum() / total_positives
    return {
        "macro_sensitivity": float(sensitivity.mean()),
        "macro_recall": float(sensitivity.mean()),
        "micro_recall": float(micro_recall),
        "macro_specificity": float(specificity.mean()),
        "macro_precision": float(precision.mean()),
        "macro_f1": float(f1.mean()),
        "ecg_exact_match": float(np.all(targets == predictions, axis=1).mean()),
    }


def recall_per_class(targets: np.ndarray, predictions: np.ndarray) -> List[float]:
    targets = targets.astype(bool)
    predictions = predictions.astype(bool)
    tp = np.logical_and(targets, predictions).sum(axis=0).astype(np.float64)
    fn = np.logical_and(targets, ~predictions).sum(axis=0).astype(np.float64)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    return recall.tolist()


def specificity_per_class(targets: np.ndarray, predictions: np.ndarray) -> List[float]:
    targets = targets.astype(bool)
    predictions = predictions.astype(bool)
    tn = np.logical_and(~targets, ~predictions).sum(axis=0).astype(np.float64)
    fp = np.logical_and(~targets, predictions).sum(axis=0).astype(np.float64)
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) > 0)
    return specificity.tolist()


def select_thresholds(
    targets: np.ndarray,
    probabilities: np.ndarray,
    candidates: Iterable[float] = np.arange(0.05, 0.951, 0.01),
) -> np.ndarray:
    thresholds = np.full(targets.shape[1], 0.5, dtype=np.float32)
    for label_index in range(targets.shape[1]):
        truth = targets[:, label_index].astype(bool)
        best_score = -1.0
        for threshold in candidates:
            prediction = probabilities[:, label_index] >= threshold
            tp = np.logical_and(truth, prediction).sum()
            fp = np.logical_and(~truth, prediction).sum()
            fn = np.logical_and(truth, ~prediction).sum()
            score = 0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)
            if score > best_score:
                best_score = score
                thresholds[label_index] = threshold
    return thresholds


def select_thresholds_for_specificity(
    targets: np.ndarray,
    probabilities: np.ndarray,
    target_specificity: float = 0.90,
    candidates: Iterable[float] = np.arange(0.01, 1.0, 0.01),
) -> np.ndarray:
    """Maximize per-class sensitivity subject to a validation specificity floor."""
    if not 0.0 < target_specificity < 1.0:
        raise ValueError("target_specificity must be in (0, 1)")
    thresholds = np.full(targets.shape[1], 0.5, dtype=np.float32)
    for label_index in range(targets.shape[1]):
        truth = targets[:, label_index].astype(bool)
        best = None
        for threshold in candidates:
            prediction = probabilities[:, label_index] >= threshold
            tp = np.logical_and(truth, prediction).sum()
            fn = np.logical_and(truth, ~prediction).sum()
            tn = np.logical_and(~truth, ~prediction).sum()
            fp = np.logical_and(~truth, prediction).sum()
            sensitivity = 0.0 if tp + fn == 0 else tp / (tp + fn)
            specificity = 0.0 if tn + fp == 0 else tn / (tn + fp)
            score = (sensitivity, specificity, float(threshold))
            if specificity >= target_specificity and (best is None or score > best):
                best = score
        thresholds[label_index] = 1.0 if best is None else best[2]
    return thresholds


def select_thresholds_for_youden(targets, probabilities, candidates=np.arange(0.01, 1.0, 0.01)):
    """Select per-class thresholds maximizing Youden's J statistic."""
    thresholds = np.full(targets.shape[1], 0.5, dtype=np.float32)
    for i in range(targets.shape[1]):
        truth = targets[:, i].astype(bool); best = None
        for threshold in candidates:
            pred = probabilities[:, i] >= threshold
            tp = np.logical_and(truth, pred).sum(); fn = np.logical_and(truth, ~pred).sum()
            tn = np.logical_and(~truth, ~pred).sum(); fp = np.logical_and(~truth, pred).sum()
            sens = 0.0 if tp + fn == 0 else tp / (tp + fn)
            spec = 0.0 if tn + fp == 0 else tn / (tn + fp)
            score = (sens + spec - 1.0, sens, spec, -float(threshold))
            if best is None or score > best[0]: best = (score, float(threshold))
        thresholds[i] = best[1] if best else 0.5
    return thresholds


def patient_exact_match(
    targets: np.ndarray,
    predictions: np.ndarray,
    patient_hashes: Sequence[str],
) -> float:
    grouped: Dict[str, Tuple[set[int], set[int]]] = {}
    for truth, prediction, patient_hash in zip(targets, predictions, patient_hashes):
        true_set, predicted_set = grouped.setdefault(str(patient_hash), (set(), set()))
        true_set.update(np.flatnonzero(truth).tolist())
        predicted_set.update(np.flatnonzero(prediction).tolist())
    return float(np.mean([truth == prediction for truth, prediction in grouped.values()]))


def average_precision_per_class(targets: np.ndarray, probabilities: np.ndarray) -> List[float]:
    try:
        from sklearn.metrics import average_precision_score
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for PR-AUC evaluation") from exc
    return [
        float(average_precision_score(targets[:, index], probabilities[:, index]))
        for index in range(targets.shape[1])
    ]


def roc_auc_per_class(targets: np.ndarray, probabilities: np.ndarray) -> List[float]:
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for ROC-AUC evaluation") from exc
    return [
        float(roc_auc_score(targets[:, index], probabilities[:, index]))
        for index in range(targets.shape[1])
    ]


def micro_roc_auc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for ROC-AUC evaluation") from exc
    return float(roc_auc_score(targets.ravel(), probabilities.ravel()))
