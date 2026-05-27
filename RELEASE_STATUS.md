# Release Repository Status

**Repository Purpose**  
Clean publication release version of the code for the paper submitted to *Digital Discovery* (Royal Society of Chemistry).

**Current State (as of late May 2026)**

- This is a fresh Git repository with no development history from the original `ms-nmr-representation` repo.
- Only files necessary for reproducing the main pretraining and strict-protocol H/C alignment + downstream experiments have been included.
- Hardcoded server paths in runner scripts have been removed/minimized (scripts now use relative paths and environment variables).
- Core components included:
  - Pretraining entry point (`train.py`)
  - Training framework (`src/training/`)
  - Model implementation
  - Data handling used in experiments
  - Strict H/C alignment training and evaluation code
  - Reconstruction and cross-modal evaluation scripts
  - Key configuration files and runner scripts

**What has been intentionally excluded**
- Experimental logs, checkpoints, and intermediate results
- Internal paper writing materials
- Abandoned or exploratory code
- Environment-specific paths and server configurations

**Next Work Needed (not yet done)**
- Further improvement of documentation and reproducibility instructions
- Adding Data DOIs (Zenodo) once datasets are archived
- Final polishing of README and any additional usage guides
- Tagging a release version and uploading to Zenodo

This repository is in a clean, reviewable state for the code portion of the submission.
