#!/usr/bin/env python3
"""
Zenodo Dataset Upload Script (for large data on remote servers)
Usage:
  1. Create a draft Dataset deposit on Zenodo web first (get the deposition ID).
  2. Get your personal access token from Zenodo (https://zenodo.org/account/settings/applications/tokens/new/)
  3. Run this on your remote server (where the data lives).

Example:
  python scripts/zenodo_upload_dataset.py \
    --deposition_id 1234567 \
    --token YOUR_ZENODO_TOKEN \
    --files /path/to/data1.tar.gz /path/to/data2.tar.gz \
    --sandbox   # optional, for testing

After upload, go back to the Zenodo web page to add metadata and publish.
"""

import argparse
import os
import requests
from pathlib import Path

def get_bucket_url(deposition_id: int, token: str, sandbox: bool = False) -> str:
    """Get the bucket URL for a draft deposition."""
    base = "https://sandbox.zenodo.org" if sandbox else "https://zenodo.org"
    url = f"{base}/api/deposit/depositions/{deposition_id}"
    headers = {"Authorization": f"Bearer {token}"}
    
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    
    data = resp.json()
    bucket_url = data["links"]["bucket"]
    print(f"Bucket URL: {bucket_url}")
    return bucket_url

def upload_file(bucket_url: str, file_path: Path, token: str):
    """Upload a single file to the Zenodo bucket using the modern API."""
    headers = {"Authorization": f"Bearer {token}"}
    dest_url = f"{bucket_url}/{file_path.name}"
    
    print(f"Uploading {file_path} ({file_path.stat().st_size / 1e9:.2f} GB) ...")
    
    with open(file_path, "rb") as fp:
        resp = requests.put(dest_url, data=fp, headers=headers)
    
    if resp.status_code in (200, 201):
        print(f"  ✓ Success: {resp.json().get('key')}")
    else:
        print(f"  ✗ Failed ({resp.status_code}): {resp.text}")
        resp.raise_for_status()

def main():
    parser = argparse.ArgumentParser(description="Upload large dataset directly from server to Zenodo.")
    parser.add_argument("--deposition_id", type=int, required=True, help="The ID of the draft deposition you created on Zenodo web.")
    parser.add_argument("--token", required=True, help="Your Zenodo personal access token (or set ZENODO_TOKEN env var).")
    parser.add_argument("--files", nargs="+", required=True, help="Paths to files to upload (on the remote server).")
    parser.add_argument("--sandbox", action="store_true", help="Use Zenodo Sandbox for testing (recommended first).")
    
    args = parser.parse_args()
    
    token = args.token or os.environ.get("ZENODO_TOKEN")
    if not token:
        raise ValueError("Please provide --token or set ZENODO_TOKEN environment variable.")
    
    bucket_url = get_bucket_url(args.deposition_id, token, sandbox=args.sandbox)
    
    for f in args.files:
        path = Path(f).expanduser().resolve()
        if not path.exists():
            print(f"File not found: {path}")
            continue
        upload_file(bucket_url, path, token)
    
    print("\nAll uploads attempted. Go to the Zenodo web interface to review, add full metadata (title, description, license, related identifiers for the code DOI), and publish the record.")

if __name__ == "__main__":
    main()
