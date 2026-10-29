import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


LABEL_NAMES = [
    "sinus rhythm",
    "normal ECG",
    "ST/T change",
    "others",
    "atrial arrhythmia",
    "intraventricular block",
    "myocardial infarction",
    "ventricular arrhythmia",
    "ventricular hypertrophy",
    "conduction disturbance",
    "atrial enlargement",
    "junctional arrhythmia",
    "QT interval abnormality",
    "ventricular preexcitation",
]

CLEAR_LABEL_NAMES = [
    "\u7aa6\u6027\u5fc3\u5f8b",
    "\u6b63\u5e38\u5fc3\u7535\u56fe",
    "ST-T\u6539\u53d8",
    "\u5176\u5b83",
    "\u623f\u6027\u5fc3\u5f8b\u5931\u5e38",
    "\u5fc3\u5ba4\u5185\u963b\u6ede",
    "\u5fc3\u808c\u6897\u6b7b",
    "\u5ba4\u6027\u5fc3\u5f8b",
    "\u5fc3\u5ba4\u80a5\u5927",
    "\u4f20\u5bfc\u963b\u6ede",
    "\u5fc3\u623f\u80a5\u5927",
    "\u623f\u5ba4\u8fde\u63a5\u5904\u5fc3\u5f8b",
    "Q-T\u95f4\u671f\u5f02\u5e38",
    "\u5fc3\u5ba4\u9884\u6fc0",
]

DEFAULT_PROMPT = """Analyze this 12-lead ECG image. Return one strict JSON object with:
- level1_label_ids: all applicable IDs in ascending order
- level1_labels: the corresponding labels
- level3_labels_en: concise English ECG findings
- report_target_en: a concise English ECG report, or null when no reliable report is available

Allowed level-1 labels:
0 sinus rhythm; 1 normal ECG; 2 ST/T change; 3 others; 4 atrial arrhythmia;
5 intraventricular block; 6 myocardial infarction; 7 ventricular arrhythmia;
8 ventricular hypertrophy; 9 conduction disturbance; 10 atrial enlargement;
11 junctional arrhythmia; 12 QT interval abnormality; 13 ventricular preexcitation.
The tool outputs below are fallible candidate evidence. Adjudicate them against the ECG image,
resolve disagreements, and do not copy unsupported tool claims.
Return JSON only."""


def sample_key(row: Dict[str, Any]) -> str:
    """Return the split-qualified stem shared by metadata and tool outputs."""
    sample_id = row.get("sample_id") or row.get("ecg")
    if sample_id:
        path = Path(str(sample_id))
        split = row.get("split")
        if not split and len(path.parts) >= 2 and path.parts[-2] in {
            "train", "validation", "hard_mining"
        }:
            split = path.parts[-2]
        return f"{split or ''}/{path.stem}"
    split = str(row.get("split") or "")
    image = row.get("image")
    if image:
        identifier = Path(str(image)).stem
    else:
        identifier = str(row.get("id") or "")
        path = Path(identifier)
        if path.suffix.lower() in {".dat", ".png", ".jpg", ".jpeg"}:
            identifier = path.stem
    return f"{split}/{identifier}"


def load_tool_results(
    tools_root: str | Path,
    split: str,
    include_clear_cls: bool = False,
    round_clear_probabilities: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Load the available tool outputs for one dataset split."""
    root = Path(tools_root)
    results: Dict[str, Dict[str, Any]] = {}
    sources = [
        (root / "FeatureDB" / split / "measurement.jsonl", "measurement"),
        (root / "PULSE" / split / "diagnosis.jsonl", "pulse_diagnosis"),
        (root / "PULSE" / split / "report.jsonl", "pulse_report"),
    ]
    if include_clear_cls:
        sources.append((root / "CLEAR_CLS" / split / "predictions.jsonl", "clear_cls"))
    for path, field in sources:
        if not path.is_file():
            continue
        for row in load_jsonl(path):
            key = sample_key(row)
            if field == "measurement":
                if row.get("status") != "ok" or not row.get("Measurement"):
                    continue
                value = row["Measurement"]
            elif field == "clear_cls":
                probabilities = row.get("probabilities") or {}
                if row.get("status") != "ok" or not probabilities:
                    continue
                canonical = canonical_clear_probabilities(
                    probabilities,
                    round_digits=4 if round_clear_probabilities else None,
                )
                if len(canonical) != len(LABEL_NAMES):
                    continue
                value = {"probabilities": canonical}
            else:
                value = row.get("text")
                if not value:
                    continue
            results.setdefault(key, {})[field] = value
    return results


def canonical_clear_probabilities(
    probabilities: Dict[str, Any], round_digits: Optional[int] = None
) -> Dict[str, float]:
    canonical: Dict[str, float] = {}
    for index, label in enumerate(LABEL_NAMES):
        value = probabilities.get(label, probabilities.get(CLEAR_LABEL_NAMES[index]))
        if value is None:
            continue
        numeric = float(value)
        canonical[label] = round(numeric, round_digits) if round_digits is not None else numeric
    return canonical


def prompt_with_tools(prompt: str, tool_results: Optional[Dict[str, Any]]) -> str:
    evidence = tool_results or {}
    return (
        f"{prompt}\n\nTool outputs (candidate evidence):\n"
        + json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def canonical_level1(row: Dict[str, Any]) -> tuple[List[int], List[str]]:
    ids = sorted({int(value) for value in row.get("level1_label_ids", [])})
    if any(value < 0 or value >= len(LABEL_NAMES) for value in ids):
        raise ValueError(f"Invalid level1_label_ids for {row.get('sample_id')}: {ids}")
    return ids, [LABEL_NAMES[value] for value in ids]


def report_is_eligible(row: Dict[str, Any]) -> bool:
    return row.get("report_generation_eligible") is True and bool(row.get("report_target_en"))


def build_target(row: Dict[str, Any]) -> Dict[str, Any]:
    ids, names = canonical_level1(row)
    return {
        "level1_label_ids": ids,
        "level1_labels": names,
        "level3_labels_en": list(row.get("level3_labels_en") or []),
        "report_target_en": row["report_target_en"] if report_is_eligible(row) else None,
    }


def target_text(row: Dict[str, Any]) -> str:
    return json.dumps(build_target(row), ensure_ascii=False, separators=(",", ":"))


def resolve_image_path(
    row: Dict[str, Any],
    image_root: Path,
    image_field: Optional[str] = None,
    image_suffix: str = ".png",
) -> Path:
    if image_field and row.get(image_field):
        candidate = Path(str(row[image_field]))
        return candidate if candidate.is_absolute() else image_root / candidate

    source = Path(str(row["org_data_path"]))
    parts = source.parts[1:] if source.parts and source.parts[0] == "org_data" else source.parts
    return image_root.joinpath(*parts).with_suffix(image_suffix)


def make_messages(prompt: str, image: Image.Image, answer: Optional[str] = None) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


class ECGJsonlDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        image_root: str | Path,
        image_field: Optional[str] = None,
        image_suffix: str = ".png",
        tools_root: Optional[str | Path] = None,
        include_clear_cls: bool = False,
        clear_cls_as_text: bool = True,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.image_root = Path(image_root)
        self.image_field = image_field
        self.image_suffix = image_suffix
        self.include_clear_cls = include_clear_cls
        self.clear_cls_as_text = clear_cls_as_text
        self.rows = load_jsonl(self.jsonl_path)
        self.tool_results: Dict[str, Dict[str, Any]] = {}
        if tools_root:
            source_splits = {sample_key(row).split("/", 1)[0] for row in self.rows}
            for split in source_splits:
                self.tool_results.update(
                    load_tool_results(
                        tools_root,
                        split,
                        include_clear_cls=include_clear_cls,
                        round_clear_probabilities=clear_cls_as_text,
                    )
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        tool_results = dict(self.tool_results.get(sample_key(row), {}))
        item = {
            "row": row,
            "index": index,
            "tool_results": tool_results,
            "image_path": resolve_image_path(
                row, self.image_root, self.image_field, self.image_suffix
            ),
        }
        if self.include_clear_cls and not self.clear_cls_as_text:
            clear_cls = tool_results.pop("clear_cls", None)
            probabilities = (clear_cls or {}).get("probabilities", {})
            complete = all(label in probabilities for label in LABEL_NAMES)
            item["clear_probabilities"] = (
                [float(probabilities[label]) for label in LABEL_NAMES]
                if complete
                else [0.5] * len(LABEL_NAMES)
            )
            item["clear_mask"] = [1.0 if complete else 0.0] * len(LABEL_NAMES)
        return item


def _find_last_subsequence(sequence: List[int], pattern: List[int]) -> int:
    for start in range(len(sequence) - len(pattern), -1, -1):
        if sequence[start : start + len(pattern)] == pattern:
            return start
    return -1


def _pad_1d(values: Iterable[torch.Tensor], pad_value: int) -> torch.Tensor:
    values = list(values)
    max_length = max(value.numel() for value in values)
    output = torch.full((len(values), max_length), pad_value, dtype=values[0].dtype)
    for index, value in enumerate(values):
        output[index, : value.numel()] = value
    return output


class _BaseCollator:
    def __init__(self, processor: Any, prompt: str = DEFAULT_PROMPT) -> None:
        self.processor = processor
        self.prompt = prompt

    def _open_image(self, path: Path) -> Image.Image:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing ECG PNG: {path}. Generate PNGs first or set --image_root/--image_field."
            )
        with Image.open(path) as image:
            return image.convert("RGB")

    def _merge(self, encoded: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id
        batch: Dict[str, torch.Tensor] = {
            "input_ids": _pad_1d((item["input_ids"][0] for item in encoded), pad_token_id),
            "attention_mask": _pad_1d((item["attention_mask"][0] for item in encoded), 0),
        }
        for name in ("mm_token_type_ids", "token_type_ids"):
            if name in encoded[0]:
                batch[name] = _pad_1d((item[name][0] for item in encoded), 0)
        for name in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            if name in encoded[0]:
                batch[name] = torch.cat([item[name] for item in encoded], dim=0)
        return batch

    @staticmethod
    def _class_targets(items: List[Dict[str, Any]]) -> torch.Tensor:
        targets = torch.zeros((len(items), len(LABEL_NAMES)), dtype=torch.float32)
        for row_index, item in enumerate(items):
            ids, _ = canonical_level1(item["row"])
            targets[row_index, ids] = 1.0
        return targets

    @staticmethod
    def _clear_inputs(items: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not any("clear_probabilities" in item for item in items):
            return {}
        probabilities = torch.tensor(
            [item.get("clear_probabilities", [0.5] * len(LABEL_NAMES)) for item in items],
            dtype=torch.float32,
        )
        mask = torch.tensor(
            [item.get("clear_mask", [0.0] * len(LABEL_NAMES)) for item in items],
            dtype=torch.float32,
        )
        return {"clear_probabilities": probabilities, "clear_mask": mask}


class ECGTrainCollator(_BaseCollator):
    """Creates assistant-only language labels plus image-conditioned class targets."""

    def __init__(
        self,
        processor: Any,
        prompt: str = DEFAULT_PROMPT,
        clear_dropout_probability: float = 0.0,
        include_language_targets: bool = True,
    ) -> None:
        super().__init__(processor, prompt)
        if not 0.0 <= clear_dropout_probability < 1.0:
            raise ValueError("clear_dropout_probability must be in [0, 1)")
        self.clear_dropout_probability = clear_dropout_probability
        self.include_language_targets = include_language_targets
        self.assistant_header = processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )

    def __call__(self, items: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        encoded: List[Dict[str, torch.Tensor]] = []
        labels: List[torch.Tensor] = []
        class_positions: List[int] = []
        for item in items:
            image = self._open_image(item["image_path"])
            prompt = prompt_with_tools(self.prompt, item.get("tool_results"))
            if self.include_language_targets:
                result = self.processor.apply_chat_template(
                    make_messages(prompt, image, target_text(item["row"])),
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            else:
                result = self.processor.apply_chat_template(
                    make_messages(prompt, image),
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            ids = result["input_ids"][0]
            if self.include_language_targets:
                start = _find_last_subsequence(ids.tolist(), self.assistant_header)
                if start < 0:
                    raise RuntimeError(
                        "Could not locate the assistant response boundary in tokenized input"
                    )
                answer_start = start + len(self.assistant_header)
                label = ids.clone()
                label[:answer_start] = -100
                labels.append(label)
                class_positions.append(answer_start - 1)
            else:
                class_positions.append(int(result["attention_mask"][0].sum().item()) - 1)
            encoded.append(result)

        batch = self._merge(encoded)
        if labels:
            batch["labels"] = _pad_1d(labels, -100)
        batch["class_targets"] = self._class_targets(items)
        batch["classification_positions"] = torch.tensor(class_positions, dtype=torch.long)
        clear_inputs = self._clear_inputs(items)
        if clear_inputs and self.clear_dropout_probability > 0.0:
            keep = (
                torch.rand((len(items), 1)) >= self.clear_dropout_probability
            ).to(torch.float32)
            clear_inputs["clear_mask"] *= keep
        batch.update(clear_inputs)
        return batch


class ECGEvalCollator(_BaseCollator):
    """Builds prompt-only inputs so classification never sees the target answer."""

    def __call__(self, items: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        encoded: List[Dict[str, torch.Tensor]] = []
        class_positions: List[int] = []
        for item in items:
            image = self._open_image(item["image_path"])
            prompt = prompt_with_tools(self.prompt, item.get("tool_results"))
            result = self.processor.apply_chat_template(
                make_messages(prompt, image),
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            class_positions.append(int(result["attention_mask"][0].sum().item()) - 1)
            encoded.append(result)
        batch = self._merge(encoded)
        batch["class_targets"] = self._class_targets(items)
        batch["classification_positions"] = torch.tensor(class_positions, dtype=torch.long)
        batch["example_indices"] = torch.tensor([item["index"] for item in items], dtype=torch.long)
        batch.update(self._clear_inputs(items))
        return batch
