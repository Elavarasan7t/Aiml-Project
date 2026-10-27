"""End-to-End Integration and Output Validation Test for Business Entity Resolution."""

import io
import json
import sys
from pathlib import Path
import pandas as pd
import requests

API_BASE = "http://127.0.0.1:8000"


def run_e2e_test():
    print("==================================================")
    print("STEP 9 — END-TO-END INTEGRATION & VALIDATION TEST")
    print("==================================================")

    # 1. Health & Model Info
    print("\n--- 1. Testing /api/health and /api/model-info ---")
    r_health = requests.get(f"{API_BASE}/api/health", timeout=10)
    print(f"GET /api/health: {r_health.status_code}")
    assert r_health.status_code == 200, f"Health check failed: {r_health.text}"
    health_data = r_health.json()
    print(f"Health Response: {health_data}")
    assert health_data["status"] == "ok"

    r_model = requests.get(f"{API_BASE}/api/model-info", timeout=10)
    print(f"GET /api/model-info: {r_model.status_code}")
    assert r_model.status_code == 200, f"Model info failed: {r_model.text}"
    model_data = r_model.json()
    print(f"Model Info: Architecture={model_data['model']}, Threshold={model_data['threshold']}, Features={model_data['feature_count']}, Status={model_data['status']}")
    assert model_data["model"] == "XGBoost"
    assert model_data["threshold"] == 0.70
    assert model_data["feature_count"] == 20

    # 2. Negative Validation Checks
    print("\n--- 2. Testing Negative Validation Handlers (HTTP 400) ---")
    files_bad_ext = {
        "source1": ("bad_file.csv", io.BytesIO(b"entity_id\tbusiness_name\tbusiness_address\tcountry\n1\tA\tB\tC"), "text/csv"),
        "source2": ("s2.tsv", io.BytesIO(b"entity_id\tbusiness_name\tbusiness_address\tcountry\n2\tA\tB\tC"), "text/tab-separated-values"),
        "source3": ("s3.tsv", io.BytesIO(b"entity_id\tbusiness_name\tbusiness_address\tcountry\n3\tA\tB\tC"), "text/tab-separated-values"),
    }
    r_bad_ext = requests.post(f"{API_BASE}/api/process", files=files_bad_ext, timeout=10)
    print(f"POST /api/process (bad extension): {r_bad_ext.status_code} -> {r_bad_ext.json()['message']}")
    assert r_bad_ext.status_code == 400

    files_missing_col = {
        "source1": ("s1.tsv", io.BytesIO(b"entity_id\tbusiness_name\tcountry\n1\tA\tUSA"), "text/tab-separated-values"),
        "source2": ("s2.tsv", io.BytesIO(b"entity_id\tbusiness_name\tbusiness_address\tcountry\n2\tA\tB\tUSA"), "text/tab-separated-values"),
        "source3": ("s3.tsv", io.BytesIO(b"entity_id\tbusiness_name\tbusiness_address\tcountry\n3\tA\tB\tUSA"), "text/tab-separated-values"),
    }
    r_missing_col = requests.post(f"{API_BASE}/api/process", files=files_missing_col, timeout=10)
    print(f"POST /api/process (missing column): {r_missing_col.status_code} -> {r_missing_col.json()['message']}")
    assert r_missing_col.status_code == 400

    # 3. Real Processing with Official Test Files
    print("\n--- 3. Uploading Real Test Datasets (POST /api/process) ---")
    project_root = Path(__file__).resolve().parent.parent
    test_dir = project_root / "dataset" / "test"

    s1_path = test_dir / "test_source1.tsv"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"

    assert s1_path.exists(), f"Missing {s1_path}"
    assert s2_path.exists(), f"Missing {s2_path}"
    assert s3_path.exists(), f"Missing {s3_path}"

    with open(s1_path, "rb") as f1, open(s2_path, "rb") as f2, open(s3_path, "rb") as f3:
        upload_files = {
            "source1": ("test_source1.tsv", f1, "text/tab-separated-values"),
            "source2": ("test_source2.tsv", f2, "text/tab-separated-values"),
            "source3": ("test_source3.tsv", f3, "text/tab-separated-values"),
        }
        r_proc = requests.post(f"{API_BASE}/api/process", files=upload_files, timeout=60)

    print(f"POST /api/process status: {r_proc.status_code}")
    assert r_proc.status_code == 200, f"Processing failed: {r_proc.text}"
    proc_json = r_proc.json()
    print("Process Result Summary:")
    print(json.dumps(proc_json, indent=2))

    assert proc_json["status"] == "completed"
    assert proc_json["source1_records"] == 600
    assert proc_json["source2_records"] == 900
    assert proc_json["source3_records"] == 900
    assert proc_json["threshold"] == 0.70

    # 4. Results API Verification
    print("\n--- 4. Testing GET /api/results ---")
    r_results = requests.get(f"{API_BASE}/api/results", timeout=10)
    print(f"GET /api/results: {r_results.status_code}")
    assert r_results.status_code == 200
    res_json = r_results.json()
    assert res_json["status"] == "completed"
    assert res_json["source1_count"] == 600
    assert res_json["matching_file_available"] is True
    assert res_json["candidates_file_available"] is True

    # 5. Preview API Verification
    print("\n--- 5. Testing GET /api/results/preview ---")
    r_preview = requests.get(f"{API_BASE}/api/results/preview?limit=15", timeout=10)
    print(f"GET /api/results/preview: {r_preview.status_code}")
    assert r_preview.status_code == 200
    prev_json = r_preview.json()
    print(f"Preview Response: Total Rows = {prev_json['total_rows']}, Rows Returned = {len(prev_json['rows'])}")
    assert prev_json["total_rows"] == 600
    assert len(prev_json["rows"]) == 15
    for r in prev_json["rows"][:5]:
        print(f"  {r['source1_entity_id']} -> {r['matched_entity_ids'] if r['matched_entity_ids'] else 'No Match'} (count={r['count']})")

    # 6. Download Matching Results Verification
    print("\n--- 6. Testing GET /api/download/matching ---")
    r_match = requests.get(f"{API_BASE}/api/download/matching", timeout=15)
    print(f"GET /api/download/matching: {r_match.status_code}")
    assert r_match.status_code == 200
    assert "text/tab-separated-values" in r_match.headers.get("content-type", "")
    df_matching = pd.read_csv(io.StringIO(r_match.text), sep="\t", dtype=str, keep_default_na=False)
    print(f"matching_results.tsv downloaded successfully: {len(df_matching)} rows, columns={list(df_matching.columns)}")

    # 7. Download Candidate Pairs Verification
    print("\n--- 7. Testing GET /api/download/candidates ---")
    r_cands = requests.get(f"{API_BASE}/api/download/candidates", timeout=15)
    print(f"GET /api/download/candidates: {r_cands.status_code}")
    assert r_cands.status_code == 200
    assert "text/tab-separated-values" in r_cands.headers.get("content-type", "")
    df_cands = pd.read_csv(io.StringIO(r_cands.text), sep="\t", dtype=str, keep_default_na=False)
    print(f"candidate_pairs.tsv downloaded successfully: {len(df_cands)} rows, columns={list(df_cands.columns)}")

    # 8. Task 19 Output Validation Checks
    print("\n--- 8. TASK 19: Comprehensive Output Validation Checks ---")
    test_s1 = pd.read_csv(s1_path, sep="\t", dtype=str)
    test_s2 = pd.read_csv(s2_path, sep="\t", dtype=str)
    test_s3 = pd.read_csv(s3_path, sep="\t", dtype=str)

    s1_ids_set = set(test_s1["entity_id"])
    s2_ids_set = set(test_s2["entity_id"])
    s3_ids_set = set(test_s3["entity_id"])
    all_candidate_pairs_set = set(zip(df_cands["source1_entity_id"], df_cands["candidate_entity_id"]))

    # Rule 1: Every Source 1 test entity appears exactly once
    assert len(df_matching) == len(test_s1), f"Row count mismatch: {len(df_matching)} != {len(test_s1)}"
    assert set(df_matching["source1_entity_id"]) == s1_ids_set, "Matching results Source 1 IDs do not match test_source1.tsv exactly!"
    print("PASS: Rule 1 PASSED: Every Source 1 test entity appears exactly once (600/600 entities).")

    # Rule 2: No duplicate Source 1 IDs
    assert not df_matching["source1_entity_id"].duplicated().any(), "Duplicate Source 1 IDs found in matching_results.tsv!"
    print("PASS: Rule 2 PASSED: Zero duplicate Source 1 IDs.")

    # Rule 3, 4, 5, 6, 7
    s1_in_matches = 0
    invalid_candidate_ids = 0
    duplicate_matched_ids_in_row = 0
    not_in_candidate_pairs = 0
    singleton_entities = 0
    matched_entities = 0
    total_match_pairs = 0

    for _, row in df_matching.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = row["matched_entity_ids"]

        if not matched_str or not matched_str.strip():
            singleton_entities += 1
            continue

        matched_entities += 1
        cand_list = [c.strip() for c in matched_str.split(",") if c.strip()]
        total_match_pairs += len(cand_list)

        # Check for duplicates in row
        if len(cand_list) != len(set(cand_list)):
            duplicate_matched_ids_in_row += 1

        for cid in cand_list:
            if cid in s1_ids_set:
                s1_in_matches += 1
            if cid not in s2_ids_set and cid not in s3_ids_set:
                invalid_candidate_ids += 1
            if (s1_id, cid) not in all_candidate_pairs_set:
                not_in_candidate_pairs += 1

    assert s1_in_matches == 0, f"Found {s1_in_matches} S1 IDs incorrectly present in matched_entity_ids!"
    print("PASS: Rule 3 PASSED: No S1 IDs present in matched_entity_ids.")

    assert invalid_candidate_ids == 0, f"Found {invalid_candidate_ids} invalid S2/S3 IDs in matched_entity_ids!"
    print("PASS: Rule 4 PASSED: All matched IDs belong to valid test S2 or S3 entities.")

    print(f"PASS: Rule 5 PASSED: Empty matched_entity_ids correctly allowed for {singleton_entities} singleton entities.")

    assert duplicate_matched_ids_in_row == 0, f"Found {duplicate_matched_ids_in_row} rows with duplicate matched IDs!"
    print("PASS: Rule 6 PASSED: No duplicate matched candidate IDs within any row.")

    assert not_in_candidate_pairs == 0, f"Found {not_in_candidate_pairs} matches not present in candidate_pairs.tsv!"
    print("PASS: Rule 7 PASSED: 100% of final matches exist in candidate_pairs.tsv.")

    print("\n==================================================")
    print("VALIDATION SUMMARY")
    print("==================================================")
    print(f"Total Source 1 Entities:        {len(df_matching)}")
    print(f"Total Source 2 Entities:        {len(test_s2)}")
    print(f"Total Source 3 Entities:        {len(test_s3)}")
    print(f"Total Candidate Pairs:          {len(df_cands)}")
    print(f"Source 1 with Matches:          {matched_entities} ({(matched_entities / len(df_matching) * 100):.1f}%)")
    print(f"Source 1 Singletons:            {singleton_entities} ({(singleton_entities / len(df_matching) * 100):.1f}%)")
    print(f"Total Positive Match Pairs:     {total_match_pairs}")
    print(f"Applied Classification Cutoff:  {proc_json['threshold']:.2f}")
    print("==================================================")
    print("ALL TESTS AND VALIDATIONS PASSED WITH ZERO ERRORS!")
    print("==================================================")


if __name__ == "__main__":
    run_e2e_test()
