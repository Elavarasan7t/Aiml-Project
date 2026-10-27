"""Official Validation Script for Business Entity Resolution Submissions."""

import argparse
import sys
from pathlib import Path
import pandas as pd


def validate_submission(matching_path: Path, candidate_path: Path, test_dir: Path) -> bool:
    print("=" * 60)
    print("OFFICIAL SUBMISSION VALIDATOR — Business Entity Resolution")
    print("=" * 60)
    print(f"Matching Results: {matching_path}")
    print(f"Candidate Pairs:  {candidate_path}")
    print(f"Test Directory:   {test_dir}")
    print("-" * 60)

    if not matching_path.exists():
        print(f"ERROR: Matching file does not exist: {matching_path}")
        return False
    if not candidate_path.exists():
        print(f"ERROR: Candidate file does not exist: {candidate_path}")
        return False
    if not test_dir.exists():
        print(f"ERROR: Test directory does not exist: {test_dir}")
        return False

    s1_file = test_dir / "test_source1.tsv"
    s2_file = test_dir / "test_source2.tsv"
    s3_file = test_dir / "test_source3.tsv"

    for f, name in [(s1_file, "Source 1"), (s2_file, "Source 2"), (s3_file, "Source 3")]:
        if not f.exists():
            print(f"ERROR: Test file {name} not found at {f}")
            return False

    df_matching = pd.read_csv(matching_path, sep="\t", dtype=str, keep_default_na=False)
    df_cands = pd.read_csv(candidate_path, sep="\t", dtype=str, keep_default_na=False)
    test_s1 = pd.read_csv(s1_file, sep="\t", dtype=str, keep_default_na=False)
    test_s2 = pd.read_csv(s2_file, sep="\t", dtype=str, keep_default_na=False)
    test_s3 = pd.read_csv(s3_file, sep="\t", dtype=str, keep_default_na=False)

    s1_ids_set = set(test_s1["entity_id"])
    s2_ids_set = set(test_s2["entity_id"])
    s3_ids_set = set(test_s3["entity_id"])
    cand_pairs_set = set(zip(df_cands["source1_entity_id"], df_cands["candidate_entity_id"]))

    all_passed = True

    # Rule 1: Every Source 1 test entity appears exactly once
    matching_s1_ids = list(df_matching["source1_entity_id"])
    if len(matching_s1_ids) != len(s1_ids_set) or set(matching_s1_ids) != s1_ids_set:
        print(f"FAIL [Rule 1]: Expected {len(s1_ids_set)} entities, found {len(matching_s1_ids)} rows.")
        all_passed = False
    else:
        print(f"PASS [Rule 1]: Every Source 1 test entity appears exactly once ({len(s1_ids_set)}/{len(s1_ids_set)}).")

    # Rule 2: No duplicate Source 1 IDs
    if df_matching["source1_entity_id"].duplicated().any():
        dup_count = df_matching["source1_entity_id"].duplicated().sum()
        print(f"FAIL [Rule 2]: Found {dup_count} duplicate Source 1 IDs.")
        all_passed = False
    else:
        print("PASS [Rule 2]: No duplicate Source 1 IDs.")

    # Rule 3, 4, 5, 6, 7
    s1_in_matches = 0
    invalid_candidate_ids = 0
    duplicate_matched_ids_in_row = 0
    not_in_candidate_pairs = 0
    singleton_count = 0
    matched_count = 0
    total_match_pairs = 0

    for _, row in df_matching.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = str(row.get("matched_entity_ids", "")).strip()

        if not matched_str:
            singleton_count += 1
            continue

        matched_count += 1
        cand_list = [c.strip() for c in matched_str.split(",") if c.strip()]
        total_match_pairs += len(cand_list)

        # Duplicate candidates within one entity's match list
        if len(cand_list) != len(set(cand_list)):
            duplicate_matched_ids_in_row += 1

        for cid in cand_list:
            if cid in s1_ids_set:
                s1_in_matches += 1
            if cid not in s2_ids_set and cid not in s3_ids_set:
                invalid_candidate_ids += 1
            if (s1_id, cid) not in cand_pairs_set:
                not_in_candidate_pairs += 1

    # Rule 3: No S1 IDs in matched_entity_ids
    if s1_in_matches > 0:
        print(f"FAIL [Rule 3]: Found {s1_in_matches} S1 IDs in matched_entity_ids.")
        all_passed = False
    else:
        print("PASS [Rule 3]: No Source 1 IDs present in matched_entity_ids.")

    # Rule 4: All matched IDs belong to valid test S2 or S3 entities
    if invalid_candidate_ids > 0:
        print(f"FAIL [Rule 4]: Found {invalid_candidate_ids} invalid candidate IDs not in S2/S3.")
        all_passed = False
    else:
        print("PASS [Rule 4]: All matched IDs belong to valid Source 2 or Source 3 entities.")

    # Rule 5: Empty matched_entity_ids allowed for singletons
    print(f"PASS [Rule 5]: Singletons correctly recorded with empty match strings ({singleton_count} singletons).")

    # Rule 6: No duplicate matched candidate IDs within any row
    if duplicate_matched_ids_in_row > 0:
        print(f"FAIL [Rule 6]: Found {duplicate_matched_ids_in_row} rows with duplicate candidate IDs.")
        all_passed = False
    else:
        print("PASS [Rule 6]: No duplicate candidate IDs within any matching row.")

    # Rule 7: Every final match exists in candidate_pairs.tsv
    if not_in_candidate_pairs > 0:
        print(f"FAIL [Rule 7]: Found {not_in_candidate_pairs} matches not present in candidate_pairs.tsv.")
        all_passed = False
    else:
        print("PASS [Rule 7]: All final matches exist in candidate_pairs.tsv.")

    print("-" * 60)
    print("SUBMISSION SUMMARY:")
    print(f"  Total Source 1 Records:     {len(df_matching)}")
    print(f"  Total Matched Entities:     {matched_count} ({(matched_count / len(df_matching) * 100):.1f}%)")
    print(f"  Total Unmatched Entities:   {singleton_count} ({(singleton_count / len(df_matching) * 100):.1f}%)")
    print(f"  Total Positive Pair Matches: {total_match_pairs}")
    print("=" * 60)

    if all_passed:
        print("OVERALL VALIDATION: SUCCESS (ALL 7 RULES PASSED)")
    else:
        print("OVERALL VALIDATION: FAILED (SEE ERRORS ABOVE)")
    print("=" * 60)
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Validate Business Entity Resolution submission files.")
    parser.add_argument("--matching", required=True, type=Path, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, type=Path, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, type=Path, help="Path to test directory containing test_source1/2/3.tsv")
    args = parser.parse_args()

    success = validate_submission(args.matching, args.candidate, args.test_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
