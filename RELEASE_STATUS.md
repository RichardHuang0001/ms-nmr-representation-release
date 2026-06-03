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
- Code Zenodo DOI obtained (10.5281/zenodo.20519353). Data DOI in progress.

This repository represents the exact code snapshot corresponding to the methods and results described in the submitted manuscript.

**Zenodo Archiving Status (as of 2026-06-03)**
- **Code (Software record)**: Published via GitHub integration. DOI: **10.5281/zenodo.20519353**
  - Record: https://zenodo.org/records/20519353
  - View on GitHub release: the badge should now appear on v1.0.1
- **Data (Dataset record)**: In progress — user has New upload page open. Will use separate Dataset record + bidirectional Related identifiers.
- After data publish: edit both records to cross-link, then update paper DAS and release docs.
## Data Upload Instructions (for Zenodo)

The processed dataset should be uploaded separately as a "Dataset" record on Zenodo.

Recommended tools (run directly on the remote server where data lives):
- `scripts/zenodo_upload_dataset.py` (Python, more robust for very large files)
- `scripts/zenodo_upload_curl.sh` (pure curl + bash, minimal dependencies)

Steps:
1. On Zenodo web (or Sandbox first), create a new draft upload, set Resource type = Dataset, save as draft. Note the deposition ID.
2. Generate a Personal access token (deposit:write + deposit:actions scopes).
3. Export ZENODO_TOKEN on the remote server.
4. Run one of the scripts above with the deposition ID and file paths.
5. After upload, edit the Zenodo record metadata and publish.
6. Link the new data DOI to the code record (and vice versa) using Related identifiers.

See the scripts for detailed usage. Always test with https://sandbox.zenodo.org first.
