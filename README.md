# Set-Structured Multimodal Spectral Representation Learning

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

The processed multimodal spectroscopic dataset used in this work will be made publicly available via Zenodo (link will be added upon archiving).

## Citation

If you use this code, please cite the associated paper (citation details will be added upon publication).

## License

MIT License

## Contact

For questions regarding the code or the paper, please open an issue or contact the corresponding author.