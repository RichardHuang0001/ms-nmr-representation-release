# Set-Structured Multimodal Spectral Representation Learning

Code release for the paper submitted to *Digital Discovery*.

This repository contains the code used for the large-scale pretraining of the Set Transformer and the subsequent strict-protocol experiments.

## What is included

- Pretraining code (`train.py` + training framework)
- Core model implementation
- Data loading and preprocessing
- Strict H/C cross-modal alignment training and evaluation (main results)
- Downstream probe training
- Reconstruction evaluation of the pretrained model

## Installation

```bash
pip install torch numpy scikit-learn tqdm pyyaml
```

Or run:
```bash
bash install.sh
```

## Running Pretraining

The main entry point is `train.py`. Example usage:

```bash
python train.py configs/pretrain_set_transformer.yaml
```

## Running Main Experiments (H/C Alignment)

See `scripts/` for the runner scripts used in the reported results:

- Full finetuning from pretrained / scratch
- Unfreezing depth ablations
- Few-shot scaling
- Scaffold split controls

## Evaluation

- `scripts/evaluate_reconstruction.py`: Assesses pretrained model reconstruction quality.
- Downstream evaluation code lives in `downstream/`.

## Data

Processed dataset will be available via Zenodo (link to be added).

## Citation

Please cite the paper when using this code.

## License

MIT
