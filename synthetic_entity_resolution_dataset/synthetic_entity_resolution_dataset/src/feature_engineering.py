"""
Feature Engineering Module for Business Entity Resolution.

Reads:
- dataset/cleaned/train_source1_clean.tsv
- dataset/cleaned/train_source2_clean.tsv
- dataset/cleaned/train_source3_clean.tsv
- output/candidate_pairs.tsv
- dataset/train/train_ground_truth.tsv (for training label assignment only)

Computes pairwise features for all candidate pairs:
- Business name similarity features (exact, ratio, partial ratio, token ratio, Jaccard, token counts, length diff)
- Business address similarity features (exact, ratio, partial ratio, Jaccard, token counts, length diff)
- Country match feature
- Missing value indicators

Outputs:
- output/candidate_features.tsv (unlabeled candidate feature vectors)
- output/training_features.tsv (labeled feature dataset with ground-truth binary label)
"""

import sys
from pathlib import Path
import pandas as pd
from rapidfuzz import fuzz


def get_project_root() -> Path:
    """
    Locates the project root directory containing the dataset and output folders.
    Supports running from project root, repository root, or src directory.
    """
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    if (parent_dir / "dataset" / "cleaned").exists():
        return parent_dir

    cwd = Path.cwd()
    if (cwd / "dataset" / "cleaned").exists():
        return cwd

    nested = cwd / "synthetic_entity_resolution_dataset" / "synthetic_entity_resolution_dataset"
    if (nested / "dataset" / "cleaned").exists():
        return nested

    nested_single = cwd / "synthetic_entity_resolution_dataset"
    if (nested_single / "dataset" / "cleaned").exists():
        return nested_single

    return parent_dir


def load_cleaned_records(tsv_path: Path) -> dict[str, dict]:
    """
    Loads a cleaned source TSV file and converts it into a dictionary keyed by entity_id
    with pre-tokenized sets for fast O(1) attribute lookup.
    """
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False)
    records = {}
    for _, row in df.iterrows():
        eid = row["entity_id"]
        nm = row["name_clean"]
        addr = row["address_clean"]
        cntry = row["country_clean"]

        records[eid] = {
            "name": nm,
            "addr": addr,
            "cntry": cntry,
            "name_tokens": set(nm.split()) if nm else set(),
            "addr_tokens": set(addr.split()) if addr else set(),
            "name_missing": 1 if not nm else 0,
            "addr_missing": 1 if not addr else 0,
            "cntry_missing": 1 if not cntry else 0,
        }
    return records


def load_ground_truth_pairs(gt_path: Path) -> set[tuple[str, str]]:
    """
    Loads the training ground truth file and extracts all true matching pairs:
    (source1_entity_id, candidate_entity_id).
    """
    if not gt_path.exists():
        return set()

    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    true_pairs = set()
    for _, row in gt_df.iterrows():
        s1_id = row["source1_entity_id"]
        matches = [m.strip() for m in row["matched_entity_ids"].split(",") if m.strip()]
        for m in matches:
            true_pairs.add((s1_id, m))
    return true_pairs


def compute_features_for_pair(
    s1_rec: dict, cand_rec: dict
) -> dict:
    """
    Computes all pairwise similarity and missingness features between a Source 1 record
    and a candidate record.
    """
    nm1, nm2 = s1_rec["name"], cand_rec["name"]
    ad1, ad2 = s1_rec["addr"], cand_rec["addr"]
    c1, c2 = s1_rec["cntry"], cand_rec["cntry"]

    tok1, tok2 = s1_rec["name_tokens"], cand_rec["name_tokens"]
    atok1, atok2 = s1_rec["addr_tokens"], cand_rec["addr_tokens"]

    # 1. Name Features
    name_exact = 1 if (nm1 and nm1 == nm2) else 0
    name_ratio = float(fuzz.ratio(nm1, nm2)) if (nm1 and nm2) else 0.0
    name_partial_ratio = float(fuzz.partial_ratio(nm1, nm2)) if (nm1 and nm2) else 0.0
    name_token_ratio = float(fuzz.token_sort_ratio(nm1, nm2)) if (nm1 and nm2) else 0.0

    union_name = tok1 | tok2
    name_jaccard = (len(tok1 & tok2) / len(union_name)) if union_name else 0.0
    common_name_tokens = len(tok1 & tok2)
    name_length_difference = abs(len(nm1) - len(nm2))

    # 2. Address Features (safe defaults when either address is missing)
    if ad1 and ad2:
        address_exact = 1 if ad1 == ad2 else 0
        address_ratio = float(fuzz.ratio(ad1, ad2))
        address_partial_ratio = float(fuzz.partial_ratio(ad1, ad2))
        union_addr = atok1 | atok2
        address_jaccard = (len(atok1 & atok2) / len(union_addr)) if union_addr else 0.0
        common_address_tokens = len(atok1 & atok2)
    else:
        address_exact = 0
        address_ratio = 0.0
        address_partial_ratio = 0.0
        address_jaccard = 0.0
        common_address_tokens = 0

    address_length_difference = abs(len(ad1) - len(ad2))

    # 3. Country Feature (dynamic country match without hard-coded names)
    country_match = 1 if (c1 and c2 and c1 == c2) else 0

    # 4. Missing Value Indicators
    s1_name_missing = s1_rec["name_missing"]
    cand_name_missing = cand_rec["name_missing"]
    s1_addr_missing = s1_rec["addr_missing"]
    cand_addr_missing = cand_rec["addr_missing"]
    s1_cntry_missing = s1_rec["cntry_missing"]
    cand_cntry_missing = cand_rec["cntry_missing"]

    return {
        "name_exact": name_exact,
        "name_ratio": name_ratio,
        "name_partial_ratio": name_partial_ratio,
        "name_token_ratio": name_token_ratio,
        "name_jaccard": name_jaccard,
        "address_exact": address_exact,
        "address_ratio": address_ratio,
        "address_partial_ratio": address_partial_ratio,
        "address_jaccard": address_jaccard,
        "country_match": country_match,
        "name_length_difference": name_length_difference,
        "address_length_difference": address_length_difference,
        "common_name_tokens": common_name_tokens,
        "common_address_tokens": common_address_tokens,
        "source1_name_missing": s1_name_missing,
        "candidate_name_missing": cand_name_missing,
        "source1_address_missing": s1_addr_missing,
        "candidate_address_missing": cand_addr_missing,
        "source1_country_missing": s1_cntry_missing,
        "candidate_country_missing": cand_cntry_missing,
    }


def generate_features(
    candidate_pairs_path: Path,
    s1_records: dict,
    s2_records: dict,
    s3_records: dict,
    true_pairs: set[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Iterates over candidate_pairs.tsv and generates feature vectors for every candidate pair.
    Returns:
      (df_candidate_features, df_training_features)
    """
    cands_df = pd.read_csv(candidate_pairs_path, sep="\t", dtype=str, keep_default_na=False)

    s1_ids = cands_df["source1_entity_id"].tolist()
    cand_ids = cands_df["candidate_entity_id"].tolist()
    sources = cands_df["candidate_source"].tolist()

    feature_rows = []
    labels = []

    for s1_id, cid, src in zip(s1_ids, cand_ids, sources):
        s1_rec = s1_records[s1_id]
        cand_rec = s2_records[cid] if src == "S2" else s3_records[cid]

        feat_dict = compute_features_for_pair(s1_rec, cand_rec)
        feat_dict["source1_entity_id"] = s1_id
        feat_dict["candidate_entity_id"] = cid
        feat_dict["candidate_source"] = src

        feature_rows.append(feat_dict)
        labels.append(1 if (s1_id, cid) in true_pairs else 0)

    # Column ordering specified in Task 8
    feature_cols = [
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
        "name_exact",
        "name_ratio",
        "name_partial_ratio",
        "name_token_ratio",
        "name_jaccard",
        "address_exact",
        "address_ratio",
        "address_partial_ratio",
        "address_jaccard",
        "country_match",
        "name_length_difference",
        "address_length_difference",
        "common_name_tokens",
        "common_address_tokens",
        "source1_name_missing",
        "candidate_name_missing",
        "source1_address_missing",
        "candidate_address_missing",
        "source1_country_missing",
        "candidate_country_missing",
    ]

    df_cand_features = pd.DataFrame(feature_rows)[feature_cols]

    # Assign proper numeric dtypes
    int_cols = [
        "name_exact", "address_exact", "country_match",
        "name_length_difference", "address_length_difference",
        "common_name_tokens", "common_address_tokens",
        "source1_name_missing", "candidate_name_missing",
        "source1_address_missing", "candidate_address_missing",
        "source1_country_missing", "candidate_country_missing"
    ]
    float_cols = [
        "name_ratio", "name_partial_ratio", "name_token_ratio", "name_jaccard",
        "address_ratio", "address_partial_ratio", "address_jaccard"
    ]

    for col in int_cols:
        df_cand_features[col] = df_cand_features[col].astype(int)
    for col in float_cols:
        df_cand_features[col] = df_cand_features[col].astype(float)

    # Training features dataframe includes ground-truth label
    df_train_features = df_cand_features.copy()
    df_train_features["label"] = labels
    df_train_features["label"] = df_train_features["label"].astype(int)

    return df_cand_features, df_train_features


def validate_features(df_train: pd.DataFrame) -> dict:
    """
    Validates feature distributions, ranges, and data integrity according to Task 12.
    """
    missing_vals = int(df_train.isnull().sum().sum())
    dup_pairs = int(df_train.duplicated(subset=["source1_entity_id", "candidate_entity_id", "candidate_source"]).sum())

    validation = {
        "candidate_pairs": len(df_train),
        "feature_rows": len(df_train),
        "positive_labels": int((df_train["label"] == 1).sum()),
        "negative_labels": int((df_train["label"] == 0).sum()),
        "s2_candidates": int((df_train["candidate_source"] == "S2").sum()),
        "s3_candidates": int((df_train["candidate_source"] == "S3").sum()),
        "duplicate_pairs": dup_pairs,
        "missing_values": missing_vals,
        "name_ratio_min": float(df_train["name_ratio"].min()),
        "name_ratio_max": float(df_train["name_ratio"].max()),
        "address_ratio_min": float(df_train["address_ratio"].min()),
        "address_ratio_max": float(df_train["address_ratio"].max()),
        "country_matches": int(df_train["country_match"].sum()),
    }

    assert missing_vals == 0, f"Found {missing_vals} missing values in feature dataframe!"
    assert dup_pairs == 0, f"Found {dup_pairs} duplicate pairs in feature dataframe!"

    return validation


def inspect_samples(
    df_train: pd.DataFrame,
    s1_records: dict,
    s2_records: dict,
    s3_records: dict,
    n_samples: int = 3,
):
    """
    Prints sample positive and negative examples for visual label and similarity verification.
    """
    pos_df = df_train[df_train["label"] == 1].head(n_samples)
    neg_df = df_train[df_train["label"] == 0].head(n_samples)

    print("SAMPLE POSITIVE EXAMPLES (label = 1):")
    print("-------------------------------------")
    for _, r in pos_df.iterrows():
        s1_id = r["source1_entity_id"]
        cid = r["candidate_entity_id"]
        src = r["candidate_source"]
        nm1 = s1_records[s1_id]["name"]
        nm2 = s2_records[cid]["name"] if src == "S2" else s3_records[cid]["name"]

        print(f"Source 1 ID: {s1_id}")
        print(f"Candidate ID: {cid}")
        print(f"Candidate source: {src}")
        print(f"Source 1 business name: {nm1}")
        print(f"Candidate business name: {nm2}")
        print(f"Name similarity: {r['name_ratio']:.2f}")
        print(f"Address similarity: {r['address_ratio']:.2f}")
        print(f"Country match: {int(r['country_match'])}")
        print(f"Label: {int(r['label'])}")
        print()

    print("SAMPLE NEGATIVE EXAMPLES (label = 0):")
    print("-------------------------------------")
    for _, r in neg_df.iterrows():
        s1_id = r["source1_entity_id"]
        cid = r["candidate_entity_id"]
        src = r["candidate_source"]
        nm1 = s1_records[s1_id]["name"]
        nm2 = s2_records[cid]["name"] if src == "S2" else s3_records[cid]["name"]

        print(f"Source 1 ID: {s1_id}")
        print(f"Candidate ID: {cid}")
        print(f"Candidate source: {src}")
        print(f"Source 1 business name: {nm1}")
        print(f"Candidate business name: {nm2}")
        print(f"Name similarity: {r['name_ratio']:.2f}")
        print(f"Address similarity: {r['address_ratio']:.2f}")
        print(f"Country match: {int(r['country_match'])}")
        print(f"Label: {int(r['label'])}")
        print()


def main():
    project_root = get_project_root()
    cleaned_dir = project_root / "dataset" / "cleaned"
    train_dir = project_root / "dataset" / "train"
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. File paths
    s1_path = cleaned_dir / "train_source1_clean.tsv"
    s2_path = cleaned_dir / "train_source2_clean.tsv"
    s3_path = cleaned_dir / "train_source3_clean.tsv"
    cand_path = output_dir / "candidate_pairs.tsv"
    gt_path = train_dir / "train_ground_truth.tsv"

    for p in [s1_path, s2_path, s3_path, cand_path]:
        if not p.exists():
            print(f"Error: Missing required input file: {p}", file=sys.stderr)
            sys.exit(1)

    # 2. Load cleaned data and candidate pairs
    s1_records = load_cleaned_records(s1_path)
    s2_records = load_cleaned_records(s2_path)
    s3_records = load_cleaned_records(s3_path)
    true_pairs = load_ground_truth_pairs(gt_path)

    # 3. Compute pairwise features
    df_cand_features, df_train_features = generate_features(
        cand_path, s1_records, s2_records, s3_records, true_pairs
    )

    # 4. Save feature datasets
    out_cand_features_path = output_dir / "candidate_features.tsv"
    out_train_features_path = output_dir / "training_features.tsv"

    df_cand_features.to_csv(out_cand_features_path, sep="\t", index=False)
    df_train_features.to_csv(out_train_features_path, sep="\t", index=False)

    # 5. Validation and Reporting
    val = validate_features(df_train_features)

    print("FEATURE ENGINEERING REPORT")
    print("--------------------------------")
    print(f"Candidate pairs:\n{val['candidate_pairs']}\n")
    print(f"Training feature rows:\n{val['feature_rows']}\n")
    print(f"Positive matches:\n{val['positive_labels']}\n")
    print(f"Negative matches:\n{val['negative_labels']}\n")
    print(f"S2 candidates:\n{val['s2_candidates']}\n")
    print(f"S3 candidates:\n{val['s3_candidates']}\n")
    print(f"Duplicate pairs:\n{val['duplicate_pairs']}\n")
    print(f"Missing feature values:\n{val['missing_values']}\n")
    print(f"Name ratio:\nmin = {val['name_ratio_min']:.2f}\nmax = {val['name_ratio_max']:.2f}\n")
    print(f"Address ratio:\nmin = {val['address_ratio_min']:.2f}\nmax = {val['address_ratio_max']:.2f}\n")
    print(f"Country matches:\n{val['country_matches']}\n")

    # 6. Sample inspection
    inspect_samples(df_train_features, s1_records, s2_records, s3_records, n_samples=3)

    print("OUTPUT FILES")
    print("--------------------------------")
    print(f"Saved: output/candidate_features.tsv ({out_cand_features_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Saved: output/training_features.tsv ({out_train_features_path.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
