# Zhongshan input contract

Paths are relative to the release root. The `dat/` and `images/` directories are placeholders; no real patient records are included.

- `dat/<record>.dat`: original Zhongshan DAT; see the root README for the binary format.
- `images/<record>.png`: preferably the original experiment image, or an image produced by `render_dat.py`.
- `labels.csv`: columns `Filename,Diagnosis1`. Filename is `<record>.dat`; Diagnosis1 is a Python-style list of English labels from the original 14-class ontology. SR is ignored during 13-class evaluation. Known spelling aliases are accepted; unknown labels are rejected. See `examples/labels.csv`.
- `tools/measurement.jsonl`: sample_id or id is `zs/<record>`, status is `ok`, and Measurement is a nonempty object. The example heart_rate field illustrates the format only; preserve original measurements for reproduction.
- `tools/PULSE/*.jsonl`: sample_id/id is `zs/<record>`, subset or dataset is `zs`, and diagnosis and/or report contains the original tool evidence.
- `tools/clear_predictions.jsonl`: sample_id is `zs/<record>.dat` (the suffix is optional), status is `ok`, and probabilities contains the original 14-class probability dictionary. Use English keys as in `examples/clear_predictions.jsonl`. Legacy Chinese keys remain supported internally for compatibility.

`prepare_manifest.py` builds the manifest without reading labels. Tool evidence must not contain test ground truth. When renaming records, update DAT names, image names, labels, and all tool IDs together. Examples are synthetic format demonstrations, not reproduction data.
