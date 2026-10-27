"""
Candidate Generation and Blocking Module for Business Entity Resolution.

Generates candidate pairs between Source 1 and Source 2 / Source 3:
  Source 1 -> Inverted Indexes / Multi-strategy Blocking -> (S1, S2) and (S1, S3) pairs

Strategies implemented:
1. Country-guided blocking (Source 1 country == candidate country with initial tokens)
2. Meaningful name token blocking (inverted index on significant name tokens)
3. Name prefix blocking (multi-token prefix e.g. "lakshmi electricals")
4. Address token blocking (postal / PIN codes, street number + street name, distinctive address bigrams)
5. Multiple blocking rules union (high recall retention)

Outputs:
  output/candidate_pairs.tsv
"""

import sys
from collections import defaultdict
from pathlib import Path
import pandas as pd

# Generic legal entity words that are too non-distinctive to block on alone
GENERIC_LEGAL = {
    "ltd", "limited", "pvt", "private", "inc", "incorporated",
    "llc", "gmbh", "sarl", "co", "corp", "corporation",
    "and", "the", "of", "sa", "sas", "pte", "pty", "llp", "company"
}

# Generic address descriptors
GENERIC_ADDR = {
    "street", "road", "lane", "drive", "avenue", "boulevard",
    "suite", "floor", "apartment", "building", "tower", "block",
    "apt", "fl", "ste", "st", "rd", "ave", "dr", "blvd",
    "sl", "tce", "terrace", "cross", "main", "salai", "no",
    "near", "close", "to", "a", "b", "c", "d", "e"
}


def get_project_root() -> Path:
    """
    Locates the project root directory containing the dataset folder.
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


def get_name_tokens(name: str) -> list[str]:
    """Extract meaningful name tokens (excluding generic legal keywords)."""
    return [t for t in name.split() if t not in GENERIC_LEGAL and len(t) >= 2]


def get_name_prefix(name: str) -> str:
    """Extract 2-word name prefix for prefix blocking."""
    tokens = name.split()
    if len(tokens) >= 2:
        return f"{tokens[0]} {tokens[1]}"
    elif tokens:
        return tokens[0]
    return ""


def get_addr_tokens(addr: str) -> list[str]:
    """Extract address tokens excluding generic street designators."""
    return [t for t in addr.split() if t not in GENERIC_ADDR]


def compute_distinctive_address_bigrams(
    s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame
) -> set[str]:
    """
    Identifies distinctive, rare address bigrams across all sources
    to capture location matches without exploding block size.
    """
    all_addrs = (
        list(s1_df["address_clean"])
        + list(s2_df["address_clean"])
        + list(s3_df["address_clean"])
    )
    bg_counts = defaultdict(int)
    for a in all_addrs:
        words = [w for w in a.split() if w not in GENERIC_ADDR and len(w) >= 3]
        for i in range(len(words) - 1):
            bg_counts[f"{words[i]}_{words[i+1]}"] += 1

    return {bg for bg, cnt in bg_counts.items() if 1 < cnt <= 40}


def build_candidate_index(df_cand: pd.DataFrame, distinctive_bgs: set[str]) -> dict:
    """
    Builds inverted indexes for a candidate dataset (S2 or S3).
    Returns a dictionary of inverted indexes for each blocking strategy.
    """
    idx_name_token = defaultdict(set)
    idx_name_prefix = defaultdict(set)
    idx_name_prefix4 = defaultdict(set)
    idx_country_first = defaultdict(set)
    idx_addr_pin = defaultdict(set)
    idx_addr_num_word = defaultdict(set)
    idx_addr_bg = defaultdict(set)

    for _, row in df_cand.iterrows():
        cid = row["entity_id"]
        nm = row["name_clean"]
        addr = row["address_clean"]
        cntry = row["country_clean"]

        # 1. Meaningful name tokens
        ntoks = get_name_tokens(nm)
        for t in ntoks:
            idx_name_token[t].add(cid)

        # 2. 2-word name prefix
        np = get_name_prefix(nm)
        if np:
            idx_name_prefix[np].add(cid)

        # 3. First word 4-char prefix
        if ntoks:
            first_w = ntoks[0]
            if len(first_w) >= 4:
                idx_name_prefix4[first_w[:4]].add(cid)
            # 4. Country + first word 3-char prefix (exact country blocking signal)
            if cntry and len(first_w) >= 3:
                idx_country_first[(cntry, first_w[:3])].add(cid)

        # 5. Address tokens: PIN / postal codes / alphanumeric units
        atoks = get_addr_tokens(addr)
        for t in atoks:
            if any(c.isdigit() for c in t) and len(t) >= 3:
                if t.isdigit() and len(t) < 4:
                    continue  # skip small pure integers like 1, 2, 10
                idx_addr_pin[t].add(cid)

        # 6. Address number + street word & distinctive bigrams
        for i in range(len(atoks) - 1):
            w1, w2 = atoks[i], atoks[i + 1]
            if any(c.isdigit() for c in w1) and len(w2) >= 3 and not any(c.isdigit() for c in w2):
                idx_addr_num_word[f"{w1}_{w2}"].add(cid)
            bg = f"{w1}_{w2}"
            if bg in distinctive_bgs:
                idx_addr_bg[bg].add(cid)

    return {
        "name_token": idx_name_token,
        "name_prefix": idx_name_prefix,
        "name_prefix4": idx_name_prefix4,
        "country_first": idx_country_first,
        "addr_pin": idx_addr_pin,
        "addr_num_word": idx_addr_num_word,
        "addr_bg": idx_addr_bg,
    }


def query_candidates_for_s1(
    s1_row: pd.Series, indexes: dict, distinctive_bgs: set[str]
) -> set[str]:
    """
    Applies multiple blocking strategies to retrieve candidate IDs for a single Source 1 record.
    A candidate is retained if found through ANY blocking rule (union for high recall).
    """
    nm = s1_row["name_clean"]
    addr = s1_row["address_clean"]
    cntry = s1_row["country_clean"]

    cands = set()
    ntoks = get_name_tokens(nm)

    # Rule 1: Meaningful name tokens
    for t in ntoks:
        if t in indexes["name_token"]:
            cands.update(indexes["name_token"][t])

    # Rule 2: 2-word name prefix
    np = get_name_prefix(nm)
    if np and np in indexes["name_prefix"]:
        cands.update(indexes["name_prefix"][np])

    # Rule 3: First word 4-char prefix
    if ntoks and len(ntoks[0]) >= 4:
        p4 = ntoks[0][:4]
        if p4 in indexes["name_prefix4"]:
            cands.update(indexes["name_prefix4"][p4])

    # Rule 4: Country + first word 3-char prefix (country-guided blocking)
    if cntry and ntoks and len(ntoks[0]) >= 3:
        k = (cntry, ntoks[0][:3])
        if k in indexes["country_first"]:
            cands.update(indexes["country_first"][k])

    # Rule 5: Address postal / PIN code
    atoks = get_addr_tokens(addr)
    for t in atoks:
        if any(c.isdigit() for c in t) and len(t) >= 3:
            if t.isdigit() and len(t) < 4:
                continue
            if t in indexes["addr_pin"]:
                cands.update(indexes["addr_pin"][t])

    # Rule 6: Address number + street word & distinctive bigrams
    for i in range(len(atoks) - 1):
        w1, w2 = atoks[i], atoks[i + 1]
        if any(c.isdigit() for c in w1) and len(w2) >= 3 and not any(c.isdigit() for c in w2):
            nw = f"{w1}_{w2}"
            if nw in indexes["addr_num_word"]:
                cands.update(indexes["addr_num_word"][nw])
        bg = f"{w1}_{w2}"
        if bg in distinctive_bgs and bg in indexes["addr_bg"]:
            cands.update(indexes["addr_bg"][bg])

    return cands


def generate_candidate_pairs(
    s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generates candidate pairs between Source 1 and both Source 2 and Source 3.
    Returns DataFrame with columns: source1_entity_id, candidate_entity_id, candidate_source
    """
    distinctive_bgs = compute_distinctive_address_bigrams(s1_df, s2_df, s3_df)

    candidate_records = []
    seen_pairs = set()

    for src_name, cand_df in [("S2", s2_df), ("S3", s3_df)]:
        indexes = build_candidate_index(cand_df, distinctive_bgs)

        for _, s1_row in s1_df.iterrows():
            s1_id = s1_row["entity_id"]
            matched_cands = query_candidates_for_s1(s1_row, indexes, distinctive_bgs)

            for cid in matched_cands:
                pair = (s1_id, cid, src_name)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    candidate_records.append(pair)

    df_pairs = pd.DataFrame(
        candidate_records,
        columns=["source1_entity_id", "candidate_entity_id", "candidate_source"],
    )

    # Ensure no duplicates
    df_pairs = df_pairs.drop_duplicates(
        subset=["source1_entity_id", "candidate_entity_id", "candidate_source"]
    ).reset_index(drop=True)

    return df_pairs


def run_safety_checks(
    df_pairs: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
) -> dict:
    """
    Verifies all safety and integrity requirements from Task 14.
    """
    s1_ids = set(s1_df["entity_id"])
    s2_ids = set(s2_df["entity_id"])
    s3_ids = set(s3_df["entity_id"])

    s2_pairs = df_pairs[df_pairs["candidate_source"] == "S2"]
    s3_pairs = df_pairs[df_pairs["candidate_source"] == "S3"]

    all_s1_valid = bool(df_pairs["source1_entity_id"].isin(s1_ids).all())
    all_s2_valid = bool(s2_pairs["candidate_entity_id"].isin(s2_ids).all())
    all_s3_valid = bool(s3_pairs["candidate_entity_id"].isin(s3_ids).all())
    no_s1_in_cands = bool((~df_pairs["candidate_entity_id"].isin(s1_ids)).all())
    no_duplicate_pairs = bool(
        df_pairs.duplicated(
            subset=["source1_entity_id", "candidate_entity_id", "candidate_source"]
        ).sum() == 0
    )
    valid_sources = bool(set(df_pairs["candidate_source"].unique()) <= {"S2", "S3"})

    checks = {
        "all_s1_valid": all_s1_valid,
        "all_s2_valid": all_s2_valid,
        "all_s3_valid": all_s3_valid,
        "no_s1_in_cands": no_s1_in_cands,
        "no_duplicate_pairs": no_duplicate_pairs,
        "valid_sources": valid_sources,
    }

    assert all(checks.values()), f"Safety checks failed: {checks}"
    return checks


def evaluate_ground_truth(
    df_pairs: pd.DataFrame, gt_df: pd.DataFrame, s1_df: pd.DataFrame
) -> dict:
    """
    Evaluates candidate generation recall against training ground truth.
    Computes true matching pairs captured, missed, and singleton statistics.
    """
    all_s1_ids = set(s1_df["entity_id"])
    true_pairs = set()
    s1_with_gt = set()

    for _, r in gt_df.iterrows():
        s1_id = r["source1_entity_id"]
        matches = [m.strip() for m in r["matched_entity_ids"].split(",") if m.strip()]
        if matches:
            s1_with_gt.add(s1_id)
            for m in matches:
                true_pairs.add((s1_id, m))

    cand_pair_set = set(
        zip(df_pairs["source1_entity_id"], df_pairs["candidate_entity_id"])
    )

    captured = true_pairs.intersection(cand_pair_set)
    missed = true_pairs - cand_pair_set
    recall = len(captured) / len(true_pairs) if true_pairs else 0.0

    s1_ids_with_cands = set(df_pairs["source1_entity_id"])
    s1_singletons = all_s1_ids - s1_with_gt
    singletons_with_cands = s1_singletons.intersection(s1_ids_with_cands)
    singletons_zero_cands = s1_singletons - s1_ids_with_cands

    return {
        "total_true_pairs": len(true_pairs),
        "captured_true_pairs": len(captured),
        "missed_true_pairs": len(missed),
        "candidate_recall": recall,
        "s1_with_gt_matches": len(s1_with_gt),
        "s1_singletons": len(s1_singletons),
        "singletons_with_cands": len(singletons_with_cands),
        "singletons_zero_cands": len(singletons_zero_cands),
    }


def main():
    project_root = get_project_root()
    cleaned_dir = project_root / "dataset" / "cleaned"
    train_dir = project_root / "dataset" / "train"
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load cleaned datasets
    s1_path = cleaned_dir / "train_source1_clean.tsv"
    s2_path = cleaned_dir / "train_source2_clean.tsv"
    s3_path = cleaned_dir / "train_source3_clean.tsv"
    gt_path = train_dir / "train_ground_truth.tsv"

    for p in [s1_path, s2_path, s3_path]:
        if not p.exists():
            print(f"Error: Missing cleaned file: {p}", file=sys.stderr)
            sys.exit(1)

    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    s2_df = pd.read_csv(s2_path, sep="\t", dtype=str, keep_default_na=False)
    s3_df = pd.read_csv(s3_path, sep="\t", dtype=str, keep_default_na=False)

    # 2. Generate candidate pairs
    df_pairs = generate_candidate_pairs(s1_df, s2_df, s3_df)

    # 3. Safety checks
    run_safety_checks(df_pairs, s1_df, s2_df, s3_df)

    # 4. Save output
    output_file = output_dir / "candidate_pairs.tsv"
    df_pairs.to_csv(output_file, sep="\t", index=False)

    # 5. Compute quality reporting metrics
    num_s1 = len(s1_df)
    num_s2 = len(s2_df)
    num_s3 = len(s3_df)
    total_pairs = len(df_pairs)
    s1_s2_count = int((df_pairs["candidate_source"] == "S2").sum())
    s1_s3_count = int((df_pairs["candidate_source"] == "S3").sum())

    cands_per_s1 = df_pairs.groupby("source1_entity_id").size()
    all_s1_ids = set(s1_df["entity_id"])
    s1_with_cands = len(cands_per_s1)
    s1_zero_cands = len(all_s1_ids - set(cands_per_s1.index))

    avg_cands = float(cands_per_s1.mean()) if len(cands_per_s1) > 0 else 0.0
    max_cands = int(cands_per_s1.max()) if len(cands_per_s1) > 0 else 0

    print("Candidate Generation Report")
    print("--------------------------------")
    print(f"Source 1 entities: {num_s1}")
    print(f"Source 2 entities: {num_s2}")
    print(f"Source 3 entities: {num_s3}")
    print()
    print(f"S1-S2 candidates: {s1_s2_count}")
    print(f"S1-S3 candidates: {s1_s3_count}")
    print()
    print(f"Total candidate pairs: {total_pairs}")
    print()
    print(f"S1 entities with zero candidates: {s1_zero_cands}")
    print(f"S1 entities with candidates: {s1_with_cands}")
    print()
    print(f"Average candidates per S1: {avg_cands:.2f}")
    print(f"Maximum candidates for one S1: {max_cands}")
    print()

    # 6. Evaluate against ground truth if available
    if gt_path.exists():
        gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
        eval_stats = evaluate_ground_truth(df_pairs, gt_df, s1_df)

        print("Ground Truth Evaluation")
        print("--------------------------------")
        print(f"Total true matching pairs: {eval_stats['total_true_pairs']}")
        print(f"True matching pairs captured: {eval_stats['captured_true_pairs']}")
        print(f"True matching pairs missed: {eval_stats['missed_true_pairs']}")
        print(f"Candidate Recall: {eval_stats['candidate_recall']:.2%}")
        print()

        print("Singleton Analysis")
        print("--------------------------------")
        print(f"Source 1 entities with true matches: {eval_stats['s1_with_gt_matches']}")
        print(f"Source 1 singleton entities: {eval_stats['s1_singletons']}")
        print(f"Singleton entities with candidates: {eval_stats['singletons_with_cands']}")
        print(f"Singleton entities with zero candidates: {eval_stats['singletons_zero_cands']}")
        print()

    print("Safety Checks")
    print("--------------------------------")
    print("All Source 1 IDs exist in train_source1: True")
    print("All S2 candidate IDs exist in train_source2: True")
    print("All S3 candidate IDs exist in train_source3: True")
    print("No Source 1 ID appears as a candidate: True")
    print("No duplicate candidate pairs: True")
    print("Candidate sources correctly set to S2 / S3: True")
    print()
    print(f"Saved candidate pairs to: output/candidate_pairs.tsv")


if __name__ == "__main__":
    main()
