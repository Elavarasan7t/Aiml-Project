"""
Data Cleaning Module for Business Entity Resolution.

Cleans and normalizes training source datasets:
- dataset/train/train_source1.tsv
- dataset/train/train_source2.tsv
- dataset/train/train_source3.tsv

Outputs cleaned datasets with normalized fields and missing-value indicators to:
- dataset/cleaned/train_source1_clean.tsv
- dataset/cleaned/train_source2_clean.tsv
- dataset/cleaned/train_source3_clean.tsv
"""

import os
import re
import sys
import unicodedata
from pathlib import Path
import pandas as pd


def get_project_root() -> Path:
    """
    Locates the project root directory containing the 'dataset' folder.
    Supports running from project root, repository root, or src directory.
    """
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    if (parent_dir / "dataset" / "train").exists():
        return parent_dir

    cwd = Path.cwd()
    if (cwd / "dataset" / "train").exists():
        return cwd

    nested = cwd / "synthetic_entity_resolution_dataset" / "synthetic_entity_resolution_dataset"
    if (nested / "dataset" / "train").exists():
        return nested

    nested_single = cwd / "synthetic_entity_resolution_dataset"
    if (nested_single / "dataset" / "train").exists():
        return nested_single

    return parent_dir


def normalize_text(text: object) -> str:
    """
    Normalize text following these rules:
    - Handle missing values safely (None, NaN, empty strings).
    - Convert values to strings.
    - Apply Unicode normalization using NFKD and strip combining diacritics.
    - Convert text to lowercase.
    - Replace punctuation and special characters with spaces.
    - Normalize multiple spaces into one space.
    - Strip leading and trailing spaces.

    Preserves numbers, words, abbreviations (e.g. Pvt, Ltd, LLC, Inc, Co),
    country names, and address numbers without stemming or lemmatization.
    """
    if text is None:
        return ""
    if isinstance(text, float) and pd.isna(text):
        return ""
    text_str = str(text)
    if not text_str.strip():
        return ""

    # Replace German sharp s (ß -> ss)
    text_str = text_str.replace("ß", "ss")

    # Apply Unicode normalization using NFKD and remove combining characters (diacritics)
    normalized = unicodedata.normalize("NFKD", text_str)
    decomposed = "".join(c for c in normalized if not unicodedata.combining(c))

    # Convert text to lowercase
    lowered = decomposed.lower()

    # Replace punctuation and special characters with spaces (preserve alphanumeric and whitespace)
    no_punct = re.sub(r"[^a-z0-9\s]", " ", lowered)

    # Normalize multiple spaces into one space
    single_spaced = re.sub(r"\s+", " ", no_punct)

    # Strip leading and trailing spaces
    return single_spaced.strip()


def clean_source_file(input_path: Path, output_path: Path) -> dict:
    """
    Cleans a single TSV source file according to data-cleaning rules.
    """
    # 1. Load file with sep="\t", all columns as strings, preserve empty values as empty strings
    df = pd.read_csv(input_path, sep="\t", dtype=str, keep_default_na=False)

    orig_rows = len(df)

    # 2. Check and remove duplicate entity_id values only if they exist
    dup_mask = df["entity_id"].duplicated(keep="first")
    dup_found = int(dup_mask.sum())
    dup_removed = 0
    if dup_found > 0:
        df = df.drop_duplicates(subset=["entity_id"], keep="first").reset_index(drop=True)
        dup_removed = dup_found

    # 3. Create normalized columns without modifying original columns
    df["name_clean"] = df["business_name"].apply(normalize_text)
    df["address_clean"] = df["business_address"].apply(normalize_text)
    df["country_clean"] = df["country"].apply(normalize_text)

    # 4. Create missing-value indicator columns (0 = exists, 1 = missing or empty after normalization)
    df["name_missing"] = (df["name_clean"] == "").astype(int)
    df["address_missing"] = (df["address_clean"] == "").astype(int)
    df["country_missing"] = (df["country_clean"] == "").astype(int)

    final_rows = len(df)

    # 5. Ensure output directory exists and save cleaned TSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")

    stats = {
        "filename": input_path.name,
        "original_rows": orig_rows,
        "duplicate_ids_found": dup_found,
        "duplicate_ids_removed": dup_removed,
        "missing_names": int(df["name_missing"].sum()),
        "missing_addresses": int(df["address_missing"].sum()),
        "missing_countries": int(df["country_missing"].sum()),
        "final_rows": final_rows,
        "output_path": output_path,
    }

    return stats


def main():
    project_root = get_project_root()
    train_dir = project_root / "dataset" / "train"
    cleaned_dir = project_root / "dataset" / "cleaned"

    source_files = [
        ("train_source1.tsv", "train_source1_clean.tsv"),
        ("train_source2.tsv", "train_source2_clean.tsv"),
        ("train_source3.tsv", "train_source3_clean.tsv"),
    ]

    for in_name, out_name in source_files:
        in_path = train_dir / in_name
        out_path = cleaned_dir / out_name

        if not in_path.exists():
            print(f"Error: Source file not found: {in_path}", file=sys.stderr)
            continue

        stats = clean_source_file(in_path, out_path)

        # Print data quality report
        print(f"Processing: {stats['filename']}")
        print(f"Original rows: {stats['original_rows']}")
        print(f"Duplicate entity IDs found: {stats['duplicate_ids_found']}")
        print(f"Duplicate IDs removed: {stats['duplicate_ids_removed']}")
        print(f"Missing names: {stats['missing_names']}")
        print(f"Missing addresses: {stats['missing_addresses']}")
        print(f"Missing countries: {stats['missing_countries']}")
        print(f"Final rows: {stats['final_rows']}")
        print(f"Saved: dataset/cleaned/{out_name}")
        print()


if __name__ == "__main__":
    main()
