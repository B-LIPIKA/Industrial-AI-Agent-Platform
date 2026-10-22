"""
Reproducible data ingestion script for NASA C-MAPSS Turbofan Engine Degradation Simulation.
Supports:
1. Kaggle API (if configured via ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KEY)
2. Manual placement in data/raw/
3. Direct HTTPS download from verified public mirrors with SHA-256 verification

Usage:
    python scripts/download_data.py [--subset FD001] [--force]
"""

import os
import sys
import argparse
import urllib.request
import json
import time
from datetime import datetime, timezone
import shutil
import zipfile

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.loader import check_dataset_availability, DEFAULT_RAW_DIR, compute_file_sha256, prepare_processed_data

# Verified mirrors for NASA C-MAPSS FD001 raw text files
MIRROR_SOURCES = {
    "FD001": {
        "train_FD001.txt": [
            "https://raw.githubusercontent.com/mapr-demos/predictive-maintenance/master/notebooks/jupyter/Dataset/CMAPSSData/train_FD001.txt",
            "https://huggingface.co/datasets/DeveloperMindset123/CMAPSS_Jet_Engine_Simulated_Data/resolve/main/train_FD001.txt"
        ],
        "test_FD001.txt": [
            "https://raw.githubusercontent.com/mapr-demos/predictive-maintenance/master/notebooks/jupyter/Dataset/CMAPSSData/test_FD001.txt",
            "https://huggingface.co/datasets/DeveloperMindset123/CMAPSS_Jet_Engine_Simulated_Data/resolve/main/test_FD001.txt"
        ],
        "RUL_FD001.txt": [
            "https://raw.githubusercontent.com/mapr-demos/predictive-maintenance/master/notebooks/jupyter/Dataset/CMAPSSData/RUL_FD001.txt",
            "https://huggingface.co/datasets/DeveloperMindset123/CMAPSS_Jet_Engine_Simulated_Data/resolve/main/RUL_FD001.txt"
        ]
    }
}


def download_file_with_fallback(filename: str, urls: list, dest_path: str) -> str:
    """Attempts to download a file from a list of URLs with fallback."""
    for idx, url in enumerate(urls, start=1):
        try:
            print(f"  -> Attempting download of {filename} from source {idx}: {url} ...")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "EngineMind-AI/1.0 (NASA-CMAPSS-Ingestion)"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                if len(data) < 100:
                    raise ValueError(f"Downloaded payload too small: {len(data)} bytes")
                with open(dest_path, "wb") as f:
                    f.write(data)
            print(f"     [OK] Downloaded {filename} ({len(data)} bytes).")
            return url
        except Exception as e:
            print(f"     [WARN] Source {idx} failed: {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
    raise RuntimeError(f"All download mirrors failed for {filename}")


def check_kaggle_cli() -> bool:
    """Checks if kaggle CLI is available and authenticated."""
    try:
        import subprocess
        res = subprocess.run(["kaggle", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return res.returncode == 0
    except Exception:
        return False


def download_via_kaggle(raw_dir: str) -> bool:
    """Attempts download via Kaggle API if available."""
    if not check_kaggle_cli():
        return False
    print("Kaggle CLI detected. Attempting to download predictive-maintenance competition data...")
    import subprocess
    try:
        cmd = ["kaggle", "competitions", "download", "-c", "predictive-maintenance", "-p", raw_dir]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            # Unzip if downloaded as zip
            for item in os.listdir(raw_dir):
                if item.endswith(".zip"):
                    zip_path = os.path.join(raw_dir, item)
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(raw_dir)
                    os.remove(zip_path)
            return True
        else:
            print(f"Kaggle download failed or terms not accepted: {res.stderr}")
            return False
    except Exception as e:
        print(f"Kaggle download exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download and prepare NASA C-MAPSS dataset.")
    parser.add_argument("--subset", default="FD001", help="Dataset subset (FD001, FD002, FD003, FD004)")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, help="Target directory for raw data")
    args = parser.parse_args()

    os.makedirs(args.raw_dir, exist_ok=True)
    status = check_dataset_availability(args.raw_dir, args.subset)

    if status["is_ready"] and not args.force:
        print(f"NASA C-MAPSS {args.subset} raw files already exist and validated in: {args.raw_dir}")
        for fname, info in status["files"].items():
            print(f"  - {fname}: {info['size_bytes']} bytes | sha256: {info['sha256'][:16]}...")
    else:
        print(f"Downloading NASA C-MAPSS {args.subset} dataset...")
        kaggle_success = download_via_kaggle(args.raw_dir)
        
        used_sources = {}
        if not kaggle_success:
            print("Using direct HTTPS download from verified public mirrors...")
            subset_mirrors = MIRROR_SOURCES.get(args.subset)
            if not subset_mirrors:
                print(f"ERROR: No known public mirrors for subset {args.subset}. Only FD001 is automated.")
                sys.exit(1)
            
            for filename, mirrors in subset_mirrors.items():
                dest = os.path.join(args.raw_dir, filename)
                if not os.path.exists(dest) or args.force:
                    used_url = download_file_with_fallback(filename, mirrors, dest)
                    used_sources[filename] = used_url
                else:
                    print(f"  - {filename} already present, skipping download.")
                    used_sources[filename] = "local-cache"

        # Validate availability after download
        status = check_dataset_availability(args.raw_dir, args.subset)
        if not status["is_ready"]:
            print(f"ERROR: Dataset validation failed after download. Status: {status}")
            sys.exit(1)

        # Write provenance metadata
        provenance = {
            "dataset_name": "NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset",
            "subset": args.subset,
            "retrieval_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "sources": used_sources,
            "kaggle_competition_url": "https://www.kaggle.com/competitions/predictive-maintenance/data",
            "nasa_pcoe_url": "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/",
            "license_notes": "NASA open science / public domain simulation data published for prognostics research. Commercial/benchmark competition usage terms apply on Kaggle.",
            "files": {
                k: {
                    "size_bytes": v["size_bytes"],
                    "sha256": v["sha256"]
                }
                for k, v in status["files"].items()
            }
        }
        prov_path = os.path.join(args.raw_dir, f"provenance_{args.subset}.json")
        with open(prov_path, "w", encoding="utf-8") as f:
            json.dump(provenance, f, indent=2)
        print(f"[OK] Saved provenance metadata to {prov_path}")

    # Process and build RUL targets
    print("Preparing processed dataset with RUL targets...")
    meta = prepare_processed_data(raw_dir=args.raw_dir, subset=args.subset)
    print(f"[OK] Processed {meta['train_engines_count']} training engines ({meta['train_rows_count']} cycles) and {meta['test_engines_count']} test engines.")
    print("Data preparation complete!")


if __name__ == "__main__":
    main()
