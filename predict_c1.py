#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoProcessor

from ecg_qwen3_vl.data import (
    DEFAULT_PROMPT,
    ECGEvalCollator,
    LABEL_NAMES as MODEL_LABEL_NAMES,
    canonical_clear_probabilities,
)
from ecg_qwen3_vl.output_space import LABEL_NAMES
from ecg_qwen3_vl.metrics import sigmoid
from ecg_qwen3_vl.model import model_classes


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def parse_args():
    parser = argparse.ArgumentParser(description="Run CardioMind C1 inference on Zhongshan")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--ema_state", default=None)
    parser.add_argument("--subset", default="zs")
    parser.add_argument("--manifest", default="data/manifest.jsonl")
    parser.add_argument("--pulse_dir", default="data/tools/PULSE")
    parser.add_argument(
        "--featuredb", default="data/tools/measurement.jsonl"
    )
    parser.add_argument(
        "--clear_cls", default="data/tools/clear_predictions.jsonl"
    )
    parser.add_argument("--include_clear_cls", action="store_true")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--min_pixels", type=int, default=200704)
    parser.add_argument("--max_pixels", type=int, default=501760)
    parser.add_argument("--attn_implementation", default="sdpa")
    return parser.parse_args()


class XiamenDataset(Dataset):
    def __init__(self, args, clear_late_fusion=False):
        self.rows = [
            row for row in load_jsonl(args.manifest) if row.get("subset") == args.subset
        ]
        if not self.rows:
            raise ValueError("No records for requested subset")
        ids = [r["id"] for r in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate manifest IDs")
        pulse = {}
        for path in sorted(Path(args.pulse_dir).glob("*.jsonl")):
            pulse.update(
                ((row.get("sample_id") or row["id"]), row)
                for row in load_jsonl(path)
                if (row.get("subset") or row.get("dataset")) == args.subset
            )
        featuredb = {
            (row.get("sample_id") or row["id"]): row for row in load_jsonl(args.featuredb)
        }
        clear_cls = (
            {
                row["sample_id"].removesuffix(".dat"): row
                for row in load_jsonl(args.clear_cls)
                if row.get("sample_id", "").startswith(f"{args.subset}/")
            }
            if args.include_clear_cls
            else {}
        )
        self.items = []
        rendered_fallback_root = Path(args.pulse_dir).parent / "rendered_missing_images"
        for index, row in enumerate(self.rows):
            sample_id = row["id"]
            feature = featuredb.get(sample_id, {})
            pulse_row = pulse.get(sample_id, {})
            clear_row = clear_cls.get(sample_id, {})
            tools = {}
            clear_probabilities = [0.5] * len(MODEL_LABEL_NAMES)
            clear_mask = [0.0] * len(MODEL_LABEL_NAMES)
            if feature.get("status") == "ok" and feature.get("Measurement"):
                tools["measurement"] = feature["Measurement"]
            if pulse_row.get("diagnosis"):
                tools["pulse_diagnosis"] = pulse_row["diagnosis"]
            if pulse_row.get("report"):
                tools["pulse_report"] = pulse_row["report"]
            if clear_row.get("status") == "ok":
                probabilities = clear_row.get("probabilities", {})
                canonical = canonical_clear_probabilities(probabilities)
                if len(canonical) == len(MODEL_LABEL_NAMES):
                    if clear_late_fusion:
                        clear_probabilities = [
                            float(canonical[label]) for label in MODEL_LABEL_NAMES
                        ]
                        clear_mask = [1.0] * len(MODEL_LABEL_NAMES)
                    else:
                        tools["clear_cls"] = {"probabilities": canonical}
            image_path = Path(row["image_path"])
            if not image_path.is_file():
                fallback = rendered_fallback_root / args.subset / image_path.name
                if fallback.is_file():
                    image_path = fallback
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if not tools.get("measurement") or not (tools.get("pulse_diagnosis") or tools.get("pulse_report")):
                raise ValueError(f"Missing FeatureDB/PULSE evidence: {sample_id}")
            if args.include_clear_cls and not any(clear_mask) and "clear_cls" not in tools:
                raise ValueError(f"Missing CLEAR_CLS evidence: {sample_id}")
            item = {
                "row": {"sample_id": sample_id, "level1_label_ids": []},
                "index": index,
                "tool_results": tools,
                "image_path": image_path,
            }
            if clear_late_fusion and args.include_clear_cls:
                item["clear_probabilities"] = clear_probabilities
                item["clear_mask"] = clear_mask
            self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def main():
    args = parse_args()
    accelerator = Accelerator()
    output_dir = Path(args.output_dir)
    threshold_payload = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    if threshold_payload.get("labels") != LABEL_NAMES:
        raise ValueError("Threshold labels do not match the model label order")
    thresholds = np.asarray(threshold_payload["thresholds"], dtype=np.float32)
    if thresholds.shape != (13,) or not np.isfinite(thresholds).all():
        raise ValueError("Expected 13 finite validation thresholds")

    processor = AutoProcessor.from_pretrained(
        args.checkpoint, min_pixels=args.min_pixels, max_pixels=args.max_pixels
    )
    processor.tokenizer.padding_side = "right"
    config = AutoConfig.from_pretrained(args.checkpoint)
    clear_late_fusion = bool(getattr(config, "ecg_clear_late_fusion", False))
    dataset = XiamenDataset(args, clear_late_fusion=clear_late_fusion)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        collate_fn=ECGEvalCollator(processor, DEFAULT_PROMPT),
    )
    _, wrapper_class = model_classes(config.model_type)
    model = wrapper_class.from_pretrained(
        args.checkpoint,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    if args.ema_state:
        ema_state = torch.load(args.ema_state, map_location="cpu", weights_only=True)
        loaded = 0
        for name, parameter in model.named_parameters():
            if name in ema_state:
                parameter.data.copy_(ema_state[name].to(device=parameter.device, dtype=parameter.dtype))
                loaded += 1
        if accelerator.is_main_process:
            print(f"Loaded EMA parameters: {loaded}", flush=True)
    model.eval().requires_grad_(False)
    model.to(accelerator.device)
    loader = accelerator.prepare(loader)

    gathered_indices = []
    gathered_logits = []
    with torch.inference_mode():
        for step, batch in enumerate(loader, start=1):
            indices = batch.pop("example_indices")
            batch.pop("class_targets")
            logits = model(**batch).classification_logits.float()
            indices, logits = accelerator.gather_for_metrics((indices, logits))
            gathered_indices.append(indices.cpu())
            gathered_logits.append(logits.cpu())
            if accelerator.is_main_process and step % 100 == 0:
                print(f"{args.subset}: {step}/{len(loader)} batches", flush=True)

    if not accelerator.is_main_process:
        accelerator.end_training()
        return
    indices = torch.cat(gathered_indices).numpy()
    logits = torch.cat(gathered_logits).numpy()
    order = np.argsort(indices)
    indices, logits = indices[order], logits[order]
    _, unique = np.unique(indices, return_index=True)
    indices, logits = indices[unique], logits[unique]
    probabilities = sigmoid(logits)
    if logits.shape != (len(dataset), 13):
        raise ValueError(f"Expected complete 13-class outputs, got {logits.shape}")
    predictions = probabilities >= thresholds

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, probs, predicted in zip(indices, probabilities, predictions):
            sample_id = dataset.rows[int(index)]["id"]
            labels = [LABEL_NAMES[i] for i in np.flatnonzero(predicted)]
            counts.update(labels)
            handle.write(json.dumps({
                "id": sample_id,
                "predicted_label_ids": np.flatnonzero(predicted).tolist(),
                "predicted_labels": labels,
                "probabilities": {
                    label: float(probs[i]) for i, label in enumerate(LABEL_NAMES)
                },
            }, ensure_ascii=False) + "\n")
    np.savez_compressed(
        output_dir / "predictions.npz",
        sample_ids=np.asarray([dataset.rows[int(i)]["id"] for i in indices]),
        labels=np.asarray(LABEL_NAMES),
        indices=indices,
        logits=logits,
        probabilities=probabilities,
        predictions=predictions,
        thresholds=thresholds,
    )
    summary = {
        "checkpoint": args.checkpoint,
        "thresholds": args.thresholds,
        "num_samples": int(len(indices)),
        "num_classes": 13,
        "labels": LABEL_NAMES,
        "projection": "exclude SR; preserve original NORM and other class scores",
        "subset": args.subset,
        "include_clear_cls": args.include_clear_cls,
        "clear_late_fusion": clear_late_fusion,
        "mean_labels_per_sample": float(predictions.sum(axis=1).mean()),
        "empty_prediction_count": int((predictions.sum(axis=1) == 0).sum()),
        "positive_counts": {label: counts[label] for label in LABEL_NAMES},
        "note": "No ground-truth labels are present; these are prediction counts, not metrics.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    accelerator.end_training()


if __name__ == "__main__":
    main()
