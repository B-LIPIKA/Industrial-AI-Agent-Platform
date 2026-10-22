"""
Data loading, validation, and schema definitions for NASA C-MAPSS Turbofan Engine Degradation Simulation.
Supports FD001 initially with extensible architecture for FD002, FD003, and FD004.
"""

import os
import json
import hashlib
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

# Column specifications for C-MAPSS
INDEX_COLUMNS = ["unit_number", "time_cycles"]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"s_{i}" for i in range(1, 22)]
ALL_COLUMNS = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS

# Informative engineering descriptions for the 21 sensors
SENSOR_METADATA = {
    "s_1": {"name": "T2", "description": "Total temperature at fan inlet", "unit": "°R"},
    "s_2": {"name": "T24", "description": "Total temperature at LPC outlet", "unit": "°R"},
    "s_3": {"name": "T30", "description": "Total temperature at HPC outlet", "unit": "°R"},
    "s_4": {"name": "T50", "description": "Total temperature at LPT outlet", "unit": "°R"},
    "s_5": {"name": "P2", "description": "Pressure at fan inlet", "unit": "psia"},
    "s_6": {"name": "P15", "description": "Total pressure in bypass-duct", "unit": "psia"},
    "s_7": {"name": "P30", "description": "Total pressure at HPC outlet", "unit": "psia"},
    "s_8": {"name": "Nf", "description": "Physical fan speed", "unit": "rpm"},
    "s_9": {"name": "Nc", "description": "Physical core speed", "unit": "rpm"},
    "s_10": {"name": "epr", "description": "Engine pressure ratio (P50/P2)", "unit": "--"},
    "s_11": {"name": "Ps30", "description": "Static pressure at HPC outlet", "unit": "psia"},
    "s_12": {"name": "phi", "description": "Ratio of fuel flow to Ps30", "unit": "pps/psi"},
    "s_13": {"name": "NRf", "description": "Corrected fan speed", "unit": "rpm"},
    "s_14": {"name": "NRc", "description": "Corrected core speed", "unit": "rpm"},
    "s_15": {"name": "BPR", "description": "Bypass Ratio", "unit": "--"},
    "s_16": {"name": "farB", "description": "Burner fuel-air ratio", "unit": "--"},
    "s_17": {"name": "htBleed", "description": "Bleed Enthalpy", "unit": "--"},
    "s_18": {"name": "Nf_dmd", "description": "Demanded fan speed", "unit": "rpm"},
    "s_19": {"name": "PCNfR_dmd", "description": "Demanded corrected fan speed", "unit": "rpm"},
    "s_20": {"name": "W31", "description": "HPT coolant bleed", "unit": "lbm/s"},
    "s_21": {"name": "W32", "description": "LPT coolant bleed", "unit": "lbm/s"},
}

DEFAULT_RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
DEFAULT_PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file for data provenance validation."""
    if not os.path.isfile(filepath):
        return ""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_dataset_availability(raw_dir: str = DEFAULT_RAW_DIR, subset: str = "FD001") -> Dict[str, Any]:
    """
    Checks if raw NASA C-MAPSS files exist for the specified subset.
    Returns status dictionary with file paths, sizes, and readiness.
    """
    train_file = os.path.join(raw_dir, f"train_{subset}.txt")
    test_file = os.path.join(raw_dir, f"test_{subset}.txt")
    rul_file = os.path.join(raw_dir, f"RUL_{subset}.txt")

    train_exists = os.path.isfile(train_file)
    test_exists = os.path.isfile(test_file)
    rul_exists = os.path.isfile(rul_file)
    ready = train_exists and test_exists and rul_exists

    return {
        "subset": subset,
        "is_ready": ready,
        "raw_dir": raw_dir,
        "files": {
            "train": {
                "path": train_file,
                "exists": train_exists,
                "size_bytes": os.path.getsize(train_file) if train_exists else 0,
                "sha256": compute_file_sha256(train_file) if train_exists else None,
            },
            "test": {
                "path": test_file,
                "exists": test_exists,
                "size_bytes": os.path.getsize(test_file) if test_exists else 0,
                "sha256": compute_file_sha256(test_file) if test_exists else None,
            },
            "rul": {
                "path": rul_file,
                "exists": rul_exists,
                "size_bytes": os.path.getsize(rul_file) if rul_exists else 0,
                "sha256": compute_file_sha256(rul_file) if rul_exists else None,
            },
        },
    }


def load_raw_dataset(raw_dir: str = DEFAULT_RAW_DIR, subset: str = "FD001") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Loads raw C-MAPSS text files into pandas DataFrames.
    Raises FileNotFoundError if files are missing.
    """
    status = check_dataset_availability(raw_dir, subset)
    if not status["is_ready"]:
        missing = [k for k, v in status["files"].items() if not v["exists"]]
        raise FileNotFoundError(
            f"Missing required NASA C-MAPSS {subset} raw files: {missing} in '{raw_dir}'. "
            f"Run 'python scripts/download_data.py' or provide the files manually."
        )

    train_path = status["files"]["train"]["path"]
    test_path = status["files"]["test"]["path"]
    rul_path = status["files"]["rul"]["path"]

    # Space-delimited without header
    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=ALL_COLUMNS)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=ALL_COLUMNS)
    rul_series = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["ground_truth_rul"])["ground_truth_rul"]

    validate_raw_dataframe(train_df, "train", subset)
    validate_raw_dataframe(test_df, "test", subset)
    if len(rul_series) != test_df["unit_number"].nunique():
        raise ValueError(
            f"RUL labels count ({len(rul_series)}) does not match unique test engines ({test_df['unit_number'].nunique()})"
        )

    return train_df, test_df, rul_series


def validate_raw_dataframe(df: pd.DataFrame, split_name: str, subset: str) -> None:
    """Validates dataframe schema, column types, and checks for missing values."""
    if len(df.columns) != 26:
        raise ValueError(f"{split_name} DataFrame has {len(df.columns)} columns, expected 26.")
    null_counts = df.isnull().sum().sum()
    if null_counts > 0:
        raise ValueError(f"{split_name} DataFrame contains {null_counts} unexpected null values.")
    if df["unit_number"].nunique() == 0:
        raise ValueError(f"{split_name} DataFrame contains zero engines.")


def prepare_processed_data(
    raw_dir: str = DEFAULT_RAW_DIR,
    processed_dir: str = DEFAULT_PROCESSED_DIR,
    subset: str = "FD001",
    max_rul_clip: Optional[int] = 125,
) -> Dict[str, Any]:
    """
    Constructs remaining useful life targets for training engines:
    RUL = max_cycle_per_unit - current_cycle.
    Optionally applies standard piecewise-linear RUL clipping (max_rul_clip=125)
    widely established in turbofan literature to prevent early-life overestimation.
    Saves processed CSVs and provenance metadata.
    """
    train_df, test_df, rul_series = load_raw_dataset(raw_dir, subset)
    os.makedirs(processed_dir, exist_ok=True)

    # Compute training RUL targets
    # Group by unit_number to find max cycle for each run-to-failure trajectory
    max_cycles = train_df.groupby("unit_number")["time_cycles"].max().reset_index()
    max_cycles.columns = ["unit_number", "max_cycle"]
    train_merged = train_df.merge(max_cycles, on="unit_number", how="left")
    train_merged["rul_raw"] = train_merged["max_cycle"] - train_merged["time_cycles"]

    if max_rul_clip is not None and max_rul_clip > 0:
        train_merged["rul"] = train_merged["rul_raw"].clip(upper=max_rul_clip)
    else:
        train_merged["rul"] = train_merged["rul_raw"]

    train_processed = train_merged.drop(columns=["max_cycle"])

    # Test set: record last observed cycle per unit
    test_last_cycles = test_df.groupby("unit_number")["time_cycles"].max().reset_index()
    test_last_cycles["ground_truth_rul"] = rul_series.values

    # Persist processed datasets
    train_out = os.path.join(processed_dir, f"train_{subset}_processed.csv")
    test_out = os.path.join(processed_dir, f"test_{subset}_processed.csv")
    test_truth_out = os.path.join(processed_dir, f"test_{subset}_truth.csv")

    train_processed.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    test_last_cycles.to_csv(test_truth_out, index=False)

    metadata = {
        "subset": subset,
        "max_rul_clip": max_rul_clip,
        "train_engines_count": int(train_df["unit_number"].nunique()),
        "train_rows_count": int(len(train_df)),
        "test_engines_count": int(test_df["unit_number"].nunique()),
        "test_rows_count": int(len(test_df)),
        "features_count": 21,
        "settings_count": 3,
        "output_files": {
            "train_processed": train_out,
            "test_processed": test_out,
            "test_truth": test_truth_out,
        },
    }

    meta_out = os.path.join(processed_dir, f"metadata_{subset}.json")
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def load_processed_data(
    processed_dir: str = DEFAULT_PROCESSED_DIR, subset: str = "FD001"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads processed CSVs for model training and evaluation."""
    train_path = os.path.join(processed_dir, f"train_{subset}_processed.csv")
    test_path = os.path.join(processed_dir, f"test_{subset}_processed.csv")
    test_truth_path = os.path.join(processed_dir, f"test_{subset}_truth.csv")

    if not (os.path.isfile(train_path) and os.path.isfile(test_path) and os.path.isfile(test_truth_path)):
        raise FileNotFoundError(
            f"Processed data for {subset} not found in '{processed_dir}'. "
            f"Run 'python scripts/prepare_data.py' first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    truth_df = pd.read_csv(test_truth_path)
    return train_df, test_df, truth_df
