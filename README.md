# Set-Structured Multimodal Spectral Representation Learning

**Publication Release** for the paper submitted to *Digital Discovery* (Royal Society of Chemistry).

This repository contains the clean, minimal code required to reproduce the main experimental results reported in the paper.

## Paper Summary (Key Finding)

This work investigates the conditions under which a pretrained Set Transformer encoder for multimodal NMR spectra provides value.

**Main Result**:
- On standard downstream classification tasks, models trained from scratch often outperform or match pretrained transfer.
- On a strict H/C cross-modal retrieval task (aligning ¹H and ¹³C NMR), pretrained representations + contrastive alignment yield substantial, consistent improvements over strong scratch baselines.

The repository focuses on the **strict protocol** experiments (H-branch and C-branch see modality-specific peaks only).

## Repository Structure

```
.
├── src/
│   ├── models/               # Set Transformer implementation
│   └── data/                 # Data loading and preprocessing
├── downstream/               # Strict H/C alignment + probe training code
├── scripts/                  # Reproducible runner scripts
├── configs/                  # Configuration files used in reported experiments
└── README.md
```

## Requirements

Python 3.9+ recommended.

Core dependencies:
- PyTorch
- NumPy
- scikit-learn
- tqdm

Install via:
```bash
pip install torch numpy scikit-learn tqdm
```

(Exact versions used in the paper are documented in the experimental section of the manuscript.)

## Reproducing Main Results

### H/C Cross-Modal Alignment (Strict Protocol)

The key experiments use the strict H-only / C-only protocol.

Example commands (see individual scripts for full arguments):

```bash
# Full finetune from pretrained
bash scripts/run_hc_alignment_strict_full_finetune.sh

# Full finetune from scratch (baseline)
bash scripts/run_hc_alignment_strict_scratch_full_finetune.sh

# Unfreezing depth ablations
bash scripts/run_hc_alignment_strict_unfreeze_last3.sh

# Few-shot scaling
bash scripts/run_hc_alignment_strict_fewshot.sh
```

### Downstream Classification Probes

See scripts in `downstream/` for probe training (frozen / finetune_fast / scratch) under both random and scaffold splits.

## Data Availability

The processed dataset used in this work is archived at:

[Zenodo DOI will be inserted here upon archiving]

## Citation

If you use this code, please cite the associated paper (citation details will be updated upon publication).

## License

MIT License (to be confirmed before final release).

## Contact

For questions regarding the paper or this release, please refer to the corresponding author information in the manuscript.
