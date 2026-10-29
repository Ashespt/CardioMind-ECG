# CardioMind
## Zhongshan 13-class inference

Output classes: `NORM, STTC, O, AA, IVB, MI, VA, VH, CD, AE, JA, QTIA, VP`. 

## Setup

```bash
pip install -r requirements.txt
```

Place the original `model.safetensors` in `models/C1/`. 

Prepare real records and existing tool outputs using this layout. See [data/README.md](data/README.md) and `examples/` for formats.

```text
data/
  dat/<record>.dat
  images/<record>.png
  labels.csv
  tools/measurement.jsonl
  tools/clear_predictions.jsonl
  tools/PULSE/predictions.jsonl
```
Use the original experiment PNGs when available. To render DAT files:

```bash
pip install -r requirements-render.txt
python render_dat.py
```

## Run

From this directory:

```bash
bash run_zhongshan.sh
# Multiple GPUs: NPROC_PER_NODE=8 bash run_zhongshan.sh
```

The script builds a manifest from DAT filenames and existing PNGs, then runs inference and evaluation. Outputs: `outputs/zs/predictions.{jsonl,npz}`, `outputs/zs/summary.json`, and `outputs/zhongshan_metrics.{json,xlsx}`. Public prediction label IDs are 0-12.

Evaluate existing predictions from this package: `python evaluate_zhongshan.py`.

Evaluation reports micro AUC, BAcc, Sensitivity, Specificity, F1, LR+/LR-, Exact Match, and per-class metrics. Prediction sample order must match the manifest.

