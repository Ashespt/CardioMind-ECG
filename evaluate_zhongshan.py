#!/usr/bin/env python3
"""Zhongshan 13-class evaluation following rule_evaluate.md (exclude SR)."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from evaluation_utils import load_jsonl, load_raw_targets, write_workbook
from ecg_qwen3_vl.output_space import LABEL_NAMES, exclude_sr

def safe_auc(target, score):
    return float(roc_auc_score(target, score)) if len(np.unique(target)) == 2 else None


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def counts(target: np.ndarray, prediction: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(np.logical_and(target, prediction).sum()),
        int(np.logical_and(~target, ~prediction).sum()),
        int(np.logical_and(~target, prediction).sum()),
        int(np.logical_and(target, ~prediction).sum()),
    )


def metric_values(tp: int, tn: int, fp: int, fn: int) -> dict[str, float | None]:
    sensitivity = safe_ratio(tp, tp + fn)
    specificity = safe_ratio(tn, tn + fp)
    if sensitivity is None or specificity is None:
        return {"BAcc": None, "Sensitivity": sensitivity, "Specificity": specificity, "F1": safe_ratio(2*tp, 2*tp+fp+fn), "LR+": None, "LR-": None}
    return {
        "BAcc": (sensitivity + specificity) / 2,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1": safe_ratio(2 * tp, 2 * tp + fp + fn),
        "LR+": safe_ratio(sensitivity, 1 - specificity),
        "LR-": safe_ratio(1 - sensitivity, specificity),
    }



def evaluate(targets, probabilities, thresholds):
    if targets.ndim != 2 or targets.shape[1] != 13 or probabilities.shape != targets.shape:
        raise ValueError("Expected matching N x 13 targets and probabilities")
    if not len(targets) or not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Invalid probabilities or empty dataset")
    if thresholds.shape != (13,) or not np.isfinite(thresholds).all():
        raise ValueError("Expected 13 finite thresholds")
    predictions = probabilities >= thresholds
    # Match the reference: micro/Exact Match use classes with both GT outcomes.
    evaluable = targets.any(axis=0) & (~targets).any(axis=0)
    if not evaluable.any():
        raise ValueError("No evaluable classes with positive and negative targets")
    y, pred, score = targets[:, evaluable], predictions[:, evaluable], probabilities[:, evaluable]
    overall = {
        "dataset": "Zhongshan", "unit": "ecg_sample", "num_samples": len(targets),
        "class_space": "C1 13-class ontology excluding SR", "num_classes": 13,
        "num_evaluable_classes": int(evaluable.sum()),
        "evaluable_classes": [name for name, valid in zip(LABEL_NAMES, evaluable) if valid],
        "ROC-AUC": safe_auc(y.ravel(), score.ravel()),
        **metric_values(*counts(y.ravel(), pred.ravel())),
        "Exact match": float(np.mean((y == pred).all(axis=1))),
        "normal_rule": "SR ignored; original NORM score and threshold retained",
    }
    per_class = []
    for i, label in enumerate(LABEL_NAMES):
        per_class.append({
            "class": label, "evaluable": bool(evaluable[i]), "threshold": float(thresholds[i]),
            "positives": int(targets[:, i].sum()), "negatives": int((~targets[:, i]).sum()),
            "ROC-AUC": safe_auc(targets[:, i], probabilities[:, i]),
            **metric_values(*counts(targets[:, i], predictions[:, i])),
        })
    return overall, per_class


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prediction-root", default="outputs")
    p.add_argument("--manifest", default="data/manifest.jsonl")
    p.add_argument("--ground-truth", default="data/labels.csv")
    p.add_argument("--thresholds", default=str(Path(__file__).parent / "config/youden_thresholds.json"))
    p.add_argument("--output", default="outputs/zhongshan_metrics.xlsx")
    args = p.parse_args()
    manifest = [row for row in load_jsonl(Path(args.manifest)) if row.get("subset") == "zs"]
    ids = [row["id"] for row in manifest]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Manifest must contain unique Zhongshan IDs")
    targets = exclude_sr(load_raw_targets(manifest, Path(args.ground_truth)))
    with np.load(Path(args.prediction_root) / "zs/predictions.npz") as artifact:
        if "sample_ids" not in artifact or artifact["sample_ids"].tolist() != ids:
            raise ValueError("Prediction sample_ids do not match manifest order")
        if "labels" not in artifact or artifact["labels"].tolist() != LABEL_NAMES:
            raise ValueError("Prediction labels must match the 13-class order excluding SR")
        probabilities = artifact["probabilities"].astype(np.float32)
        thresholds = artifact["thresholds"].astype(np.float32)
        fixed = json.loads(Path(args.thresholds).read_text())
        if fixed["labels"] != LABEL_NAMES or not np.array_equal(thresholds, np.asarray(fixed["thresholds"], dtype=np.float32)):
            raise ValueError("Artifact thresholds must match fixed validation Youden thresholds")
        if "predictions" in artifact and not np.array_equal(artifact["predictions"], probabilities >= thresholds):
            raise ValueError("Stored decisions disagree with probabilities and validation thresholds")
    overall, per_class = evaluate(targets, probabilities, thresholds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps({"overall_micro": [overall], "per_class": per_class}, indent=2, allow_nan=False) + "\n")
    # XLSX writer accepts scalar cells.
    workbook_overall = dict(overall, evaluable_classes="/".join(overall["evaluable_classes"]))
    write_workbook(output, {"Overall-Micro": [workbook_overall], "Per-Class": per_class})
    print(json.dumps(overall, indent=2))
    print(output.resolve())

if __name__ == "__main__":
    main()
