# Processed Multimodal NMR/MS Spectroscopic Dataset

This archive contains the processed dataset used for the experiments in the paper "Set-Structured Multimodal Spectral Representation Learning" (submitted to *Digital Discovery*).

## Source
- Raw data: Alberts et al. (2024) multimodal spectroscopic dataset (¹H-NMR, ¹³C-NMR, MS).
- Processing: Converted to set-structured representation (unordered peaks).
- Each peak: 24-dimensional feature vector (includes position, intensity, and derived attributes for each modality).
- Per-sample: padded/truncated to max 256 peaks.
- Splits: 80% train / 10% val / 10% test, random seed 42.
- Used under the **strict protocol**: H-branch receives only ¹H peaks; C-branch receives only ¹³C peaks (no cross-modal information leakage at encoding time).

## Files in this deposit (typical layout)
- `processed_aligned_chunk_*.pt` (or equivalent tar.gz / shards): the tensor shards containing the peak sets + labels for pretraining and downstream tasks.
- `splits/` or index files (if separate): train/val/test indices.
- Any global stats or vocab files used during preprocessing.

## Loading
See the accompanying code release (https://github.com/RichardHuang0001/ms-nmr-representation-release , Zenodo 10.5281/zenodo.20519353) and `src/data/` + `downstream/task_dataset.py` for the exact `Dataset` classes and loading logic used in all paper experiments.

## Reproducibility note
This dataset + the pinned code release (v1.0.1) allow full reproduction of:
- Masked pretraining
- Strict H/C cross-modal alignment (main positive result: +15.6 pp R@1)
- Downstream classification (functional group, element presence, molwt bin) under random + scaffold splits
- Few-shot and unfreezing ablations

All random seeds, splits, and feature dimensions match the manuscript exactly.

## License
CC-BY-4.0 (same as the code release).

## Citation
Please cite both the original Alberts et al. 2024 source and this processed deposit + the code Zenodo DOI when using.

