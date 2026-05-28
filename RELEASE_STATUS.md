# Release Repository Status

**Repository Purpose**  
Clean, publication-ready release of the code for the paper submitted to *Digital Discovery* (Royal Society of Chemistry).

**Publication Date of this Release**  
May 2026

**Current State**

- This is a fresh Git repository with **no development history** from the original messy `ms-nmr-representation` repo.
- Only files required to reproduce the main pretraining and the strict-protocol H/C cross-modal alignment + downstream classification experiments are included.
- Hardcoded paths have been cleaned (scripts use relative paths where possible).
- Core components included:
  - Pretraining (`train.py` + full training framework)
  - Model architecture
  - Strict H/C alignment training and evaluation (main positive results)
  - Downstream probe training
  - Reconstruction evaluation
  - Key runner scripts and configurations used in the paper

**What was deliberately excluded**
- Training logs, checkpoints, and intermediate outputs
- Abandoned/exploratory code
- Internal paper-writing materials and notes
- Environment-specific or server-specific configurations

**Documentation**
- `README.md` provides a high-level overview.
- `reproduce.sh` and scripts contain usage examples.
- Further reproducibility improvements and Zenodo DOIs for code + data will be added in subsequent updates.

This repository represents the exact code snapshot corresponding to the methods and results described in the submitted manuscript.

**Next steps after initial publication**
- Archive to Zenodo for permanent DOI
- Add data DOI once the processed dataset is uploaded
- Expand documentation as needed for reviewers