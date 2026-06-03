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

**Zenodo Archiving Status (as of 2026-06-03, updated)**
- **Code (Software record)**: Published via GitHub integration. DOI: **10.5281/zenodo.20519353**
  - Record: https://zenodo.org/records/20519353
  - View on GitHub release: the badge should now appear on v1.0.1
- **Data (Dataset record draft)**: Draft created and partially populated (ID 20519555, https://zenodo.org/uploads/20519555). Core metadata pre-filled via OpenCLI (title, description with strict protocol/reproducibility details, Related identifiers to code DOI + GitHub, Resource type=Dataset, etc.). Draft is persistent.
- **Server access issue**: Remote server (116.169.116.30:42594) currently unreachable (connection refused/closed during kex; likely unstable reverse tunnel + rate limiting + client/server algo mismatch). Direct server upload plan on hold.
- **Current plan (user decision)**: Fully local workflow using this repo's reproducibility code.
  - Download raw (~10GB) locally via `download_dataset.sh` (or direct curl to Zenodo 11611178; produces aligned_chunk parquets).
  - Process locally: `python src/data/preprocess.py --config configs/pretrain_set_transformer.yaml` (from main workspace) → exact `data/processed/processed_aligned_chunk_*.pt` shards (24-dim, max 256 peaks, seed 42, strict H/C).
  - Upload to draft 20519555 from Mac using `scripts/zenodo_upload_dataset.py --deposition_id 20519555` (or curl) + `docs/DATASET.md`. Helper scripts added in main workspace: local_rebuild_processed_for_zenodo.sh (Mac-adapted full pipeline) and local_upload_to_zenodo_draft.sh. Recommend tarring processed/ for fewer files before upload. Background download started in agent session (monitor logs/download_raw.log; use caffeinate on Mac).
  - Post-upload: Polish in browser (license CC-BY-4.0, keywords, communities, funding, ORCID), Publish. Then cross-link records + update paper DAS.
- See: server_data_location_and_upload_guide.md (in this repo), main workspace local_*.sh, and Submission_Progress_and_Remaining_Tasks.md for details. Once published, update release docs/README with final data DOI.
## Data Upload Instructions (for Zenodo)

The processed dataset should be uploaded separately as a "Dataset" record on Zenodo.

**Current status (server access broken):** Direct remote upload on hold. Using local Mac workflow instead (download raw from public Zenodo 11611178 via code's download_dataset.sh, preprocess locally to generate exact shards, upload to draft 20519555 from local using the scripts below + helpers in main workspace). See updated Zenodo Archiving Status above and server_data_location_and_upload_guide.md.

Recommended tools (scripts work from anywhere with the files + token; prefer local when server unavailable):
- `scripts/zenodo_upload_dataset.py` (Python, more robust for very large files; requires `requests`)
- `scripts/zenodo_upload_curl.sh` (pure curl + bash, minimal dependencies)

**For local/Mac execution (current plan):**
1. Download + process raw locally (see main workspace `local_rebuild_processed_for_zenodo.sh` or manual: `bash download_dataset.sh`, move to data/raw/nips_2024_dataset/multimodal_spectroscopic_dataset, `python src/data/preprocess.py --config configs/pretrain_set_transformer.yaml` from main repo root in spectra env; produces data/processed/processed_aligned_chunk_*.pt).
2. On Zenodo web, use/edit the existing draft (ID 20519555) or create new if needed. Note the deposition ID.
3. Generate a Personal access token (deposit:write + deposit:actions scopes).
4. Export ZENODO_TOKEN locally.
5. Run one of the scripts above with the deposition ID and paths to processed chunks + docs/DATASET.md (recommend `tar` the processed/ dir first for fewer objects due to ~245 files + local bandwidth; upload tar(s) + md).
6. After upload, edit the Zenodo record metadata in browser (add keywords, communities, license=CC-BY-4.0, funding/ORCID if applicable, verify description with strict protocol details) and publish.
7. Link the new data DOI to the code record (and vice versa) using Related identifiers.

See the scripts for detailed usage. Always test with https://sandbox.zenodo.org first. Local helpers (in main workspace): local_rebuild_processed_for_zenodo.sh and local_upload_to_zenodo_draft.sh. Background download was started via agent (check logs/download_raw.log).
