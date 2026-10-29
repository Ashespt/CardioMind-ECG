"""Public 13-class output: preserve NORM and all abnormal classes verbatim."""
import json
from pathlib import Path

LABEL_NAMES = json.loads((Path(__file__).resolve().parents[1] / "config/labels.json").read_text())

def exclude_sr(values):
    """Slice a NumPy array or torch tensor without changing retained values."""
    if values.shape[-1] != 14:
        raise ValueError(f"Expected original 14-class head, got {values.shape}")
    return values[..., 1:]
