# Set-Structured Multimodal Spectral Representation Learning

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20519353.svg)](https://doi.org/10.5281/zenodo.20519353)

**Code release accompanying the paper submitted to *Digital Discovery* (Royal Society of Chemistry).**

This repository provides a clean, self-contained snapshot of the code used for large-scale pretraining of a Set Transformer on multimodal NMR and MS spectra, along with the strict-protocol experiments reported in the paper.

## Highlights

- Permutation-invariant modeling of spectral peaks using Set Transformer
- Masked spectroscopic modeling pretraining
- Strong emphasis on reproducibility: strict cross-modal leakage-free protocol for H/C alignment
- Comprehensive ablations (encoder unfreezing depth, scratch baselines, scaffold splits, few-shot regimes)

## Repository Contents

- `train.py` — Main pretraining entry point
- `src/` — Core model, training, and data handling code
- `downstream/` — Downstream classification and H/C cross-modal alignment code (strict protocol)
- `scripts/` — Runner scripts used to produce the main results in the paper
- `configs/` — Example configuration files
- `evaluate_reconstruction.py` — Reconstruction quality evaluation

## Installation

```bash
bash install.sh
```

Or manually:

```bash
pip install -r requirements.txt
```

## Reproducing the Experiments

See `reproduce.sh` for a high-level overview.

Detailed instructions for pretraining and the main H/C alignment experiments (including strict protocol, unfreezing ablations, and scratch baselines) are provided in the `docs/` folder and inline in the scripts.

## Data

The processed multimodal spectroscopic dataset (shard-based 24-dim peak features, max 256 peaks, 80/10/10 splits seed 42, strict-protocol ready) is archived separately as a Zenodo **Dataset** record. It will be linked here and in the paper once published. See the data record for the exact files used to reproduce all pretraining + H/C alignment experiments.

## Citation

This code release is archived on Zenodo:

> RichardHuang0001/ms-nmr-representation-release: v1.0.1 - Publication Release (Zenodo DOI). (2026). Zenodo. https://doi.org/10.5281/zenodo.20519353

If you use this code, please cite the associated paper (full citation will be added upon publication of the *Digital Discovery* article) + the above Zenodo DOI for the exact code snapshot.

## License

MIT License

## Contact

For questions regarding the code or the paper, please open an issue or contact the corresponding author.