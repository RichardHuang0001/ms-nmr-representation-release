# CURRENT WORK HANDOFF - COMPLETE CONTEXT
# For new agent to continue Zenodo data upload and submission prep
# Date: 2026-06-03
# User: huangwei (RichardHuang0001 on GitHub)

## 1. PROJECT OVERVIEW
- **Repo**: ms-nmr-representation (main dev repo on eval branch)
- **Paper Topic**: Set-Structured Multimodal Spectral Representation Learning using Set Transformer for NMR (¹H, ¹³C) and MS data.
  - Pretraining: Masked spectroscopic modeling.
  - Key finding: Pretraining value is **task-dependent** — strong on strict cross-modal H/C retrieval (with leakage-free protocol), but scratch often beats pretrained on standard downstream classification (functional group, element presence, molwt bin).
  - Strict protocol emphasized: H-branch sees only ¹H peaks, C-branch only ¹³C (no cross-modal leakage during encoding).
  - Ablations: unfreezing depth, scratch baselines, scaffold splits, few-shot.
- **Target Journal**: *Digital Discovery* (RSC, Gold OA). Very high bar on **reproducibility**, open code/data, FAIR principles. Has dedicated data reviewers who will test code + data.
- **Paper Draft Location**: paper_writing/Digital_Discovery/rsc_draft/main_rsc_submission.tex (RSC LaTeX template, ~10 pages, cleaned layout: images single-column 0.48\textwidth, tables table*, "Fig." captions).
- **Graphical Abstract**: exists in paper_writing/figures/graphical_abstract.png

## 2. CURRENT BLOCKING TASK (HIGHEST PRIORITY)
**Obtain real Zenodo DOIs for Code + Data** so Data Availability Statement in paper can be updated from placeholder to citable DOIs.

- Code: Already "associated" by user on Zenodo (via GitHub release integration).
- **Data**: In progress. User has **Zenodo "New upload" page open in browser** (production, not sandbox). Wants to fill metadata and upload the processed dataset directly. "不想走沙盒这么麻烦了，直接上传，不行就重开一份upload嘛" (direct upload; if fails, just delete draft and start new).

**Why separate records?** Best practice for Zenodo (code = Software record, data = Dataset record). Must link them bidirectionally via "Related identifiers". Allows proper licensing, versioning, citability.

**Journal requirement** (from Digital_Discovery_Submission_Guide.md):
- DAS must be prominent.
- Code and processed dataset must have persistent DOIs (Zenodo recommended).
- Reviewers must be able to access during review (can use embargo or provide access link via confidential comments).
- "Data available upon request" is NOT acceptable.

Current placeholder in tex (paper_writing/Digital_Discovery/rsc_draft/main_rsc_submission.tex lines ~289-290):
```
\section*{Data availability}
The code used in this study is available in the associated GitHub repository (to be archived on Zenodo upon acceptance). The processed spectroscopic dataset is available via Zenodo (DOI to be added). Additional results and training logs are provided in the Supplementary Information.
```

## 3. KEY ARTIFACTS & LOCATIONS

### Release Repository (Critical for reproducibility)
- **Local path**: /Users/huangwei/Desktop/PythonProjects/ms-nmr-representation-release/
- **GitHub (published)**: https://github.com/RichardHuang0001/ms-nmr-representation-release.git
  - SSH remote: git@github.com:RichardHuang0001/ms-nmr-representation-release.git
  - Clean git history (no dev mess).
  - Branch: main
  - Note: Latest upload scripts and metadata guide are **local only** (uncommitted/unpushed as of 2026-06-03). See git status below.

**What it contains** (whitelisted, minimal for reproduction):
- train.py + src/ (pretraining + Set Transformer)
- downstream/ (strict H/C alignment + classification)
- scripts/ (runners for ablations, full finetune, etc.)
- configs/, reproduce.sh, install.sh
- Cleaned of /root/huangwei hardcoded paths.

**Prepared for Zenodo**:
- README.md (updated)
- RELEASE_STATUS.md (updated with data upload instructions)
- scripts/zenodo_upload_dataset.py (Python, robust for large files)
- scripts/zenodo_upload_curl.sh (bash/curl, minimal deps)
- zenodo_dataset_metadata.txt (detailed copy-paste guide for the form the user has open)

**Git status (as of now)**:
M RELEASE_STATUS.md
?? scripts/zenodo_upload_curl.sh
?? scripts/zenodo_upload_dataset.py
?? zenodo_dataset_metadata.txt

Recent commits include docs improvements for publication.

### Paper Writing Dir
- paper_writing/Digital_Discovery/
  - rsc_draft/main_rsc_submission.tex (active RSC draft)
  - rsc_draft/main_rsc_submission.pdf (10-page cleaned version)
  - Submission_Progress_and_Remaining_Tasks.md (this handoff doc's source)
  - Digital_Discovery_Submission_Guide.md (full journal prep guide)
  - Paper_Rewriting_Directions.md (original minimal change plan)
  - Publication_Release_Execution_Plan.md (how the clean release was created)
  - templates/ and rsc_draft/ have official RSC template assets.

### Figures
- paper_writing/figures/ (unified, renamed):
  - graphical_abstract.png
  - arch_pretraining.png, arch_hc_alignment.png
  - results_classification.png, results_hc_ablation.png, results_hc_pretrained_vs_scratch.png

## 4. PROGRESS HISTORY (RELEVANT TO CURRENT TASK)
- Created clean independent release repo per Publication_Release_Execution_Plan.md (whitelist only, fresh history, docs).
- Pushed to GitHub.
- Code linked on Zenodo (user did GitHub integration).
- Multiple iterations on paper LaTeX for RSC compliance (single-col images, double-col tables, caption style, junk cleanup).
- Prepared upload scripts because data lives on remote server (large processed dataset — do NOT download to local Mac first; use API from remote).
- User opened Zenodo New Upload page (production).
- I prepared zenodo_dataset_metadata.txt with exact text for every field (title, description, keywords, related identifiers linking to code GitHub + code Zenodo DOI, etc.).
- User asked for help "taking over" the form ("用OpenCLI接管，帮我填好对应的信息").

**Data specifics** (from manuscript):
- Derived from Alberts et al. 2024 multimodal dataset.
- Processed to shard-based tensors, 24-dim peak features, max 256 peaks per sample.
- Splits: 80/10/10 random seed 42.
- Used in strict protocol experiments.

## 5. EXACT CURRENT STATE & USER'S LAST REQUEST
- User has https://zenodo.org (or /uploads/new) "New upload" form open in browser.
- Code Zenodo record exists; user needs to link the new data record to it (and vice versa) using Related identifiers (e.g., GitHub URL + code DOI with relations like "Is supplement to", "References", "Compiles").
- Wants **direct production upload** (no sandbox testing this time; "if wrong, just delete draft and start new upload").
- Data files are (presumably) accessible or need to be uploaded from remote server.

**Immediate need**: Exact text/values for every field in the open form:
- Resource type: Dataset
- Title, description, creators, publication date, license, keywords, related identifiers, etc.
- Then upload the actual data files (recommend the prepared scripts if large).

I created /Users/huangwei/Desktop/PythonProjects/ms-nmr-representation-release/zenodo_dataset_metadata.txt with full instructions + copy-paste text. It uses:
- Title: "Processed Multimodal NMR and MS Spectroscopic Dataset for Set-Structured Representation Learning"
- Description: detailed, reproducibility-focused paragraph (see file).
- Creators: Huang, Wei (update affiliation/ORCID as needed)
- Date: 2026-06-03 (update)
- License: CC-BY-4.0
- Related identifiers: placeholders for code DOI + GitHub (user must replace [YOUR_CODE_DOI])

**Open variables the new agent must clarify/fill**:
- Exact author full name, affiliation, ORCID.
- The actual Zenodo DOI of the code record (for linking).
- Exact filenames/paths of the dataset files to upload (and whether they are local or need remote API upload).
- Preferred license for data.
- Any funding/grants to list.
- Communities to join.

## 6. PREPARED SCRIPTS & COMMANDS (RUN ON REMOTE IF DATA IS LARGE)
See release repo scripts/ (transfer via scp or commit/push first):
- zenodo_upload_dataset.py (preferred for large files; needs `pip install requests`)
- zenodo_upload_curl.sh (no Python dep)

Typical flow (from remote server):
1. Create draft on Zenodo web → note deposition_id.
2. export ZENODO_TOKEN=...
3. python scripts/zenodo_upload_dataset.py --deposition_id XXX --token $ZENODO_TOKEN --files /path/to/processed_data*.tar.gz --sandbox (test first) or without for prod.
4. Go back to web, edit metadata (use the .txt guide), publish.
5. Add bidirectional links between code Zenodo record and this new data record.

## 7. FULL CHECKLIST FOR NEW AGENT (FROM Submission_Progress_and_Remaining_Tasks.md)
Highest priority right now: Finish this Zenodo data record → get DOI → update DAS in tex → commit/push paper draft + release repo updates.

Then tackle:
- Cover Letter
- CRediT Author contributions (expand "W.H." section)
- Funding in Acknowledgements
- Run https://submission-checker.rsc.org
- Final TOC graphic specs
- etc.

See the full Submission_Progress_and_Remaining_Tasks.md for phased plan.

## 8. USEFUL COMMANDS FOR NEW AGENT
```bash
# On local Mac
cd /Users/huangwei/Desktop/PythonProjects/ms-nmr-representation-release
git status
git add scripts/ zenodo_dataset_metadata.txt RELEASE_STATUS.md
git commit -m "docs: Add Zenodo upload scripts and dataset metadata guide for direct production upload"
git push origin main

# View prepared metadata
cat zenodo_dataset_metadata.txt

# On remote server (after scp or git clone the release)
export ZENODO_TOKEN="..."
python scripts/zenodo_upload_dataset.py --deposition_id <ID_FROM_WEB> --token $ZENODO_TOKEN --files /path/to/your/data/files...
```

## 9. NOTES / WARNINGS
- **No credentials**: No remote server address/password has ever been provided to any agent. Do not ask user for passwords. Guide user to run commands themselves after they SSH in.
- User prefers no sandbox this time for the final upload.
- After publish, update the paper tex DAS with the real DOIs.
- Link the two Zenodo records (code + data) + GitHub + (future) paper.
- This is for *Digital Discovery* — emphasize strict protocol and reproducibility in all descriptions.
- If form has issues, user is OK deleting draft and restarting.

## 10. HANDOFF NEXT STEPS FOR NEW AGENT
1. Read this file + zenodo_dataset_metadata.txt + RELEASE_STATUS.md + the two scripts.
2. Ask user for: exact code Zenodo DOI, author details (affiliation/ORCID), list of exact data files to upload, any specific tweaks to title/description.
3. Guide user field-by-field on the open browser page using the texts from zenodo_dataset_metadata.txt (update the file live if needed and re-cat it).
4. Once draft saved, help with file upload (browser or API script).
5. After publish: get the new data DOI, help user add Related identifier links on BOTH records.
6. Update main_rsc_submission.tex Data availability section with real DOIs.
7. Commit/push changes to both paper_writing and the release repo.
8. Update Submission_Progress_and_Remaining_Tasks.md to mark this item complete.
9. Move to next item (Cover Letter?).

**Contact the user for any missing details (e.g., "What is your exact affiliation and ORCID? What is the Zenodo DOI of the code record? What are the exact filenames of the dataset archives?").**

This context should allow seamless continuation without re-explaining the entire history.
