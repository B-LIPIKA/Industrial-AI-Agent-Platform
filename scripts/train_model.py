"""
Training and evaluation script for NASA C-MAPSS Remaining Useful Life (RUL) models.
Trains baseline and machine learning models (Random Forest, Gradient Boosting),
evaluates on test engines at their last observed cycle against ground-truth RUL,
and persists model artifacts and evaluation metrics.

Usage:
    python scripts/train_model.py [--subset FD001] [--save-dir models/saved]
"""

import os
import sys
import argparse
import json

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.loader import load_processed_data, DEFAULT_PROCESSED_DIR
from models.rul_model import train_and_evaluate_rul_pipeline, DEFAULT_MODELS_DIR


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate RUL prediction models.")
    parser.add_argument("--subset", default="FD001", help="Dataset subset (default: FD001)")
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR, help="Path to processed data directory")
    parser.add_argument("--save-dir", default=DEFAULT_MODELS_DIR, help="Directory to save model artifacts")
    args = parser.parse_args()

    print(f"Loading processed data for {args.subset} from {args.processed_dir}...")
    train_df, test_df, truth_df = load_processed_data(args.processed_dir, subset=args.subset)

    print(f"Loaded: {len(train_df)} training rows ({train_df['unit_number'].nunique()} engines), "
          f"{len(test_df)} test rows ({test_df['unit_number'].nunique()} engines), "
          f"{len(truth_df)} ground-truth RUL targets.")

    print(f"\nStarting training & evaluation pipeline (saving artifacts to {args.save_dir})...")
    results = train_and_evaluate_rul_pipeline(
        train_df=train_df,
        test_df=test_df,
        truth_df=truth_df,
        save_dir=args.save_dir
    )

    print("\n" + "=" * 60)
    print("TRAINING AND EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Selected Best Model: {results.get('selected_best_model')}")
    print("\nBaseline Metrics:")
    for k, v in results.get("baseline", {}).items():
        print(f"  {k}: {v}")

    print("\nModel Metrics:")
    for model_name, m_results in results.get("models", {}).items():
        print(f"  [{model_name}]")
        for k, v in m_results.items():
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
