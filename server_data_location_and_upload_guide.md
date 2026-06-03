# Server Data Location & Zenodo Upload Guide (for 20519555 draft)

**2026-06-03 UPDATE (pivot to local):** Remote server (116.169.116.30:42594) currently unreachable from user's Mac (alternating "Connection refused" / "closed by remote during kex" — consistent with agent tests; likely unstable reverse tunnel/port forward + rate limiting + OpenSSH client/server mismatch). Direct server upload on hold until access recovers.

**Current plan:** Fully local on Mac using the code's reproducibility instructions (raw download from public Zenodo 11611178, local preprocess to generate exact processed shards, upload to this draft from local). See main workspace helper scripts `local_rebuild_processed_for_zenodo.sh` and `local_upload_to_zenodo_draft.sh` (created/copied by agent; download started in bg via agent, ~350kB/s, monitor logs/download_raw.log; use `caffeinate -i` on Mac). Recommend tarring processed/ for upload practicality. Draft 20519555 already has core metadata pre-filled (title, description, related IDs to code 20519353 + GitHub). After upload files via script, polish in browser and Publish. Then cross-link + update paper/release docs.

(The original server instructions below are kept for when access returns.)

## 1. Login with tmux (prevent disconnection)
On your local machine, run:

```bash
ssh -p 42594 root@116.169.116.30
# enter password: 3XViVNnZqIW=Sn_8
```

Once logged in, **immediately**:

```bash
tmux new -s zenodo_upload
# If session exists: tmux attach -t zenodo_upload
```

All exploration and upload commands should be run **inside this tmux**.

If your SSH drops, re-ssh and `tmux attach -t zenodo_upload` to resume/monitor.

## 2. Exploration commands (run inside tmux)
Paste these one by one or in blocks and send the full output back.

```bash
echo "=== SERVER INFO ==="
whoami
hostname
date
df -h
echo "=== /root ls and du ==="
ls -lah /root/
du -sh /root/* 2>/dev/null | sort -h | tail -30
echo "=== SEARCH FOR DATA DIRS ==="
find /root -maxdepth 5 -type d \( -iname '*data*' -o -iname '*processed*' -o -iname '*nmr*' -o -iname '*spectra*' -o -iname '*shard*' -o -iname '*alberts*' -o -iname '*multimodal*' \) 2>/dev/null | head -40
echo "=== FIND PROCESSED .pt CHUNKS ==="
find /root -maxdepth 6 \( -name '*processed*chunk*.pt' -o -name 'processed_aligned*.pt' \) 2>/dev/null | head -30
echo "=== CHECK OTHER MOUNTS ==="
du -sh /data /mnt /home /opt 2>/dev/null | sort -h
ls /data 2>/dev/null || true
echo "=== CHECK FOR RELEASE REPO / SCRIPTS ==="
find /root -name 'zenodo_upload_dataset.py' 2>/dev/null | head -5
ls -d /root/*release* /root/ms-nmr* 2>/dev/null || echo "no release dir found"
echo "=== TMUX STATUS ==="
tmux list-sessions 2>/dev/null || echo "no sessions listed"
echo "=== EXPLORATION END ==="
```

## 3. Best upload method
**Use the prepared script from the release repo directly on the server inside tmux.**

Why best:
- Data stays on server (no download to Mac).
- Uses Zenodo bucket API, good for large files.
- Runs in tmux → survives SSH disconnect.
- We already have the draft at deposition_id=20519555 (basic metadata + related identifiers to code are set).
- You can upload files now, then go back to browser (https://zenodo.org/uploads/20519555) to review/edit metadata, then Publish.

### Steps inside tmux on server

```bash
# 1. Get the release repo (contains the upload scripts + DATASET.md)
git clone https://github.com/RichardHuang0001/ms-nmr-representation-release.git /root/ms-nmr-representation-release || (cd /root/ms-nmr-representation-release && git pull)
cd /root/ms-nmr-representation-release

# 2. Create Zenodo token (do this in your browser at https://zenodo.org/account/settings/applications/tokens/new )
# Scopes: deposit:write + deposit:actions
# Then:
export ZENODO_TOKEN="paste_your_token_here"

# 3. Upload (example - replace paths after you find the data)
python scripts/zenodo_upload_dataset.py \
  --deposition_id 20519555 \
  --token $ZENODO_TOKEN \
  --files \
    /path/to/your/processed_aligned_chunk_*.pt \
    docs/DATASET.md \
    # add any other index files, metadata, small samples etc.
```

The script will upload the files to the existing draft.

Monitor with `tmux attach` if needed.

After upload completes:
- Go to https://zenodo.org/uploads/20519555 in browser.
- Review / edit metadata if needed (add keywords, communities, funding, verify description etc.).
- Upload any small remaining files via browser if wanted.
- Publish.

## 4. After publish
Tell me the final published DOI of the Dataset record.
I will:
- Add the back-link on the code record (10.5281/zenodo.20519353).
- Update the paper tex Data availability statement with both DOIs.
- Update all docs.

## Notes
- Include `docs/DATASET.md` (we prepared it with reproducibility details, strict protocol, splits seed 42, 24-dim, 256 peaks).
- If files are thousands of small .pt, consider tarring groups first on server for fewer API calls, but the script can handle lists.
- The curl version `scripts/zenodo_upload_curl.sh 20519555 /path1 /path2 ...` is alternative if no python deps.

Run the exploration commands in tmux and paste the full output here so I can give you the exact --files paths and any tweaks.
