#!/bin/bash
# Simple curl-based uploader for Zenodo (run this on your remote server)
#
# Prerequisites:
# 1. Create draft Dataset deposit on Zenodo web, note the DEPOSITION_ID.
# 2. Generate Personal access token at https://zenodo.org/account/settings/applications/tokens/new/
#    (scopes: deposit:write, deposit:actions)
# 3. Export token: export ZENODO_TOKEN="your_token_here"
#
# Usage:
#   ./scripts/zenodo_upload_curl.sh 1234567 /path/to/bigfile1.tar.gz /path/to/bigfile2.tar.gz
#
# For Sandbox testing: change ZENODO_BASE to https://sandbox.zenodo.org

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <deposition_id> <file1> [file2 ...]"
    echo "Example: $0 1234567 /data/processed_spectra.tar.gz"
    exit 1
fi

DEPOSITION_ID=$1
shift
FILES=("$@")

ZENODO_BASE="${ZENODO_BASE:-https://zenodo.org}"
TOKEN="${ZENODO_TOKEN:-}"

if [ -z "$TOKEN" ]; then
    echo "Error: Set ZENODO_TOKEN environment variable or export it."
    exit 1
fi

echo "Getting bucket for deposition $DEPOSITION_ID ..."
BUCKET_URL=$(curl -s -H "Authorization: Bearer $TOKEN" \
    "$ZENODO_BASE/api/deposit/depositions/$DEPOSITION_ID" | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['links']['bucket'])")

echo "Bucket: $BUCKET_URL"

for FILE in "${FILES[@]}"; do
    BASENAME=$(basename "$FILE")
    echo "Uploading $FILE as $BASENAME ..."
    curl -H "Authorization: Bearer $TOKEN" \
         --upload-file "$FILE" \
         "$BUCKET_URL/$BASENAME"
    echo "Done with $BASENAME"
done

echo ""
echo "Uploads complete. Now go to $ZENODO_BASE/deposit/$DEPOSITION_ID"
echo "Edit metadata (set Resource type=Dataset, add description, license, link to your code DOI), then Publish."
