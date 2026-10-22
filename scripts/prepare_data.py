"""
Data preparation script for NASA C-MAPSS dataset.
Converts raw files to processed training and testing frames with RUL targets.

Usage:
    python scripts/prepare_data.py [--subset FD001] [--max-rul-clip 125]
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.loader import prepare_processed_data, DEFAULT_RAW_DIR, DEFAULT_PROCESSED_DIR


def main():
    parser = argparse.ArgumentParser(description="Prepare C-MAPSS dataset with RUL targets.")
    parser.add_argument("--subset", default="FD001", help="Dataset subset")
    parser.add_argument("--max-rul-clip", type=int, default=125, help="Piecewise RUL clipping upper bound")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, help="Raw data directory")
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR, help="Processed data directory")
    args = parser.parse_args()

    print(f"Preparing {args.subset} data with max RUL clip = {args.max_rul_clip} cycles...")
    meta = prepare_processed_data(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        subset=args.subset,
        max_rul_clip=args.max_rul_clip
    )
    print("Dataset prepared successfully:")
    for k, v in meta.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
