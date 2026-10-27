"""Pipeline execution service integrating existing ML modules for Business Entity Resolution."""

import json
import sys
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import xgboost as xgb

from .file_service import get_project_root


class PipelineExecutionError(Exception):
    """Exception raised when ML pipeline fails during execution."""
    def __init__(self, error: str, message: str):
        super().__init__(message)
        self.error = error
        self.message = message


class PipelineService:
    """Service to coordinate data cleaning, candidate blocking, feature extraction, and ML prediction."""

    def __init__(self):
        self.project_root = get_project_root()
        self.output_dir = self.project_root / "output"
        self.models_dir = self.project_root / "models"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._ensure_sys_path()

        # In-memory lock and job state
        self._lock = threading.Lock()
        self.booster: Optional[xgb.Booster] = None
        self.feature_columns: Optional[list] = None
        self.threshold: Optional[float] = None

        self.last_job_state: Dict[str, Any] = {
            "status": "idle",
            "message": "No processing jobs have been run yet.",
            "source1_count": 0,
            "source2_count": 0,
            "source3_count": 0,
            "candidate_pair_count": 0,
            "matched_entity_count": 0,
            "unmatched_entity_count": 0,
            "threshold": None,
            "processed_at": None,
            "matching_file_available": False,
            "candidates_file_available": False,
        }

    def _ensure_sys_path(self):
        """Ensures project root is on sys.path so src.* modules can be imported."""
        root_str = str(self.project_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

    def load_model_and_threshold(self) -> Dict[str, Any]:
        """Loads and caches XGBoost model, feature list, and selected threshold."""
        model_path = self.models_dir / "entity_resolution_xgboost.json"
        features_path = self.models_dir / "feature_columns.json"
        threshold_path = self.models_dir / "threshold.json"

        if not model_path.exists():
            raise PipelineExecutionError(
                error="Model File Missing",
                message=f"XGBoost model file not found at: {model_path}"
            )

        if not features_path.exists():
            raise PipelineExecutionError(
                error="Feature Columns File Missing",
                message=f"Feature columns file not found at: {features_path}"
            )

        if not threshold_path.exists():
            raise PipelineExecutionError(
                error="Threshold File Missing",
                message=f"Threshold configuration file not found at: {threshold_path}"
            )

        try:
            with open(features_path, "r", encoding="utf-8") as f:
                self.feature_columns = json.load(f)

            with open(threshold_path, "r", encoding="utf-8") as f:
                threshold_info = json.load(f)
                self.threshold = float(threshold_info["threshold"])

            if self.booster is None:
                booster = xgb.Booster()
                booster.load_model(str(model_path))
                self.booster = booster

            return {
                "model": "XGBoost",
                "task": "Binary Entity Classification",
                "classes": "Different Business / Same Business",
                "threshold": self.threshold,
                "feature_count": len(self.feature_columns),
                "status": "loaded",
                "metric": threshold_info.get("metric", "macro_entity_f0_5"),
                "features": self.feature_columns,
            }
        except Exception as e:
            raise PipelineExecutionError(
                error="Model Loading Error",
                message=f"Failed to load ML artifacts: {str(e)}"
            )

    def get_model_info(self) -> Dict[str, Any]:
        """Returns loaded model metadata."""
        return self.load_model_and_threshold()

    def get_latest_results(self) -> Dict[str, Any]:
        """Returns in-memory state of the most recent processing job."""
        matching_path = self.output_dir / "matching_results.tsv"
        candidates_path = self.output_dir / "candidate_pairs.tsv"

        state = dict(self.last_job_state)
        state["matching_file_available"] = matching_path.exists()
        state["candidates_file_available"] = candidates_path.exists()
        state["success"] = (state.get("status") == "completed")

        results_list = []
        if matching_path.exists():
            try:
                df_match = pd.read_csv(matching_path, sep="\t", dtype=str, keep_default_na=False)
                for _, r in df_match.iterrows():
                    s1_id = r["source1_entity_id"]
                    m_str = r.get("matched_entity_ids", "")
                    m_list = [x.strip() for x in m_str.split(",") if x.strip()] if m_str else []
                    results_list.append({
                        "source1_entity_id": s1_id,
                        "matched_entity_ids": m_list,
                        "match_count": len(m_list),
                    })
            except Exception:
                pass

        state["results"] = results_list
        state["statistics"] = {
            "source1_count": state.get("source1_count", 0),
            "source2_count": state.get("source2_count", 0),
            "source3_count": state.get("source3_count", 0),
            "candidate_count": state.get("candidate_pair_count", 0),
            "matched_entities": state.get("matched_entity_count", 0),
            "unmatched_entities": state.get("unmatched_entity_count", 0),
            "threshold": state.get("threshold"),
        }
        return state

    def get_matching_preview(self, limit: int = 50) -> Dict[str, Any]:
        """Returns top matching result rows for client-side preview."""
        matching_path = self.output_dir / "matching_results.tsv"
        if not matching_path.exists():
            return {
                "status": "idle",
                "total_rows": 0,
                "preview_limit": limit,
                "rows": [],
            }

        try:
            df = pd.read_csv(matching_path, sep="\t", dtype=str, keep_default_na=False)
            total_rows = len(df)
            preview_subset = df.head(limit)
            rows = []
            for _, r in preview_subset.iterrows():
                s1_id = r["source1_entity_id"]
                m_str = r.get("matched_entity_ids", "")
                m_list = [x.strip() for x in m_str.split(",") if x.strip()] if m_str else []
                rows.append({
                    "source1_entity_id": s1_id,
                    "matched_entity_ids": m_str,
                    "count": len(m_list),
                })

            return {
                "status": "completed",
                "total_rows": total_rows,
                "preview_limit": limit,
                "rows": rows,
            }
        except Exception as e:
            return {
                "status": "error",
                "total_rows": 0,
                "preview_limit": limit,
                "rows": [],
            }

    def run_pipeline(
        self,
        df_source1: pd.DataFrame,
        df_source2: pd.DataFrame,
        df_source3: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Executes the full Entity Resolution pipeline:
        1. Normalization & Data Cleaning
        2. Candidate Generation / Blocking
        3. Feature Engineering
        4. XGBoost Inference using selected threshold
        5. Generating matching_results.tsv and candidate_pairs.tsv
        6. In-memory state update
        """
        with self._lock:
            # Step 0: Ensure model and threshold are loaded
            self.load_model_and_threshold()

            self._ensure_sys_path()
            try:
                from src.clean_data import normalize_text
                from src.candidate_generation import generate_candidate_pairs
                from src.feature_engineering import compute_features_for_pair
            except ImportError as e:
                raise PipelineExecutionError(
                    error="Module Import Error",
                    message=f"Failed to import ML pipeline modules from src: {str(e)}"
                )

            # Step 1: Normalization & Cleaning
            def clean_df(df: pd.DataFrame) -> pd.DataFrame:
                clean = df.copy()
                # Deduplicate entity_id if any duplicates exist
                dup_mask = clean["entity_id"].duplicated(keep="first")
                if dup_mask.sum() > 0:
                    clean = clean.drop_duplicates(subset=["entity_id"], keep="first").reset_index(drop=True)

                clean["name_clean"] = clean["business_name"].apply(normalize_text)
                clean["address_clean"] = clean["business_address"].apply(normalize_text)
                clean["country_clean"] = clean["country"].apply(normalize_text)

                clean["name_missing"] = (clean["name_clean"] == "").astype(int)
                clean["address_missing"] = (clean["address_clean"] == "").astype(int)
                clean["country_missing"] = (clean["country_clean"] == "").astype(int)
                return clean

            s1_clean = clean_df(df_source1)
            s2_clean = clean_df(df_source2)
            s3_clean = clean_df(df_source3)

            # Step 2: Candidate Generation / Blocking
            cand_pairs_df = generate_candidate_pairs(s1_clean, s2_clean, s3_clean)

            # Step 3: Feature Engineering
            def df_to_records(df: pd.DataFrame) -> dict:
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
                        "name_missing": int(row["name_missing"]),
                        "addr_missing": int(row["address_missing"]),
                        "cntry_missing": int(row["country_missing"]),
                    }
                return records

            s1_records = df_to_records(s1_clean)
            s2_records = df_to_records(s2_clean)
            s3_records = df_to_records(s3_clean)

            total_pairs = len(cand_pairs_df)
            matched_pairs_by_s1 = defaultdict(list)

            if total_pairs > 0:
                s1_ids = cand_pairs_df["source1_entity_id"].tolist()
                cand_ids = cand_pairs_df["candidate_entity_id"].tolist()
                sources = cand_pairs_df["candidate_source"].tolist()

                feature_rows = []
                for s1_id, cid, src in zip(s1_ids, cand_ids, sources):
                    s1_rec = s1_records.get(s1_id)
                    cand_rec = s2_records.get(cid) if src == "S2" else s3_records.get(cid)

                    if s1_rec is None or cand_rec is None:
                        continue

                    feat = compute_features_for_pair(s1_rec, cand_rec)
                    feature_rows.append(feat)

                feat_df = pd.DataFrame(feature_rows)

                # Ensure exact column ordering matching trained XGBoost model
                dmatrix_features = feat_df[self.feature_columns]
                dmat = xgb.DMatrix(dmatrix_features)

                # Step 4: Inference
                probs = self.booster.predict(dmat)
                cand_pairs_df["probability"] = probs

                # Step 5: Filter matches using selected validation threshold
                matched_mask = cand_pairs_df["probability"] >= self.threshold
                matched_subset = cand_pairs_df[matched_mask]

                for _, row in matched_subset.iterrows():
                    matched_pairs_by_s1[row["source1_entity_id"]].append(row["candidate_entity_id"])

            # Step 6: Construct entity-level matching results
            all_s1_ids = s1_clean["entity_id"].tolist()
            matching_rows = []
            matched_entity_count = 0
            unmatched_entity_count = 0

            for s1_id in all_s1_ids:
                matches = matched_pairs_by_s1.get(s1_id, [])
                if matches:
                    matched_entity_count += 1
                    # Comma-separated unique candidate IDs
                    unique_matches = sorted(set(matches))
                    matching_rows.append({
                        "source1_entity_id": s1_id,
                        "matched_entity_ids": ",".join(unique_matches),
                    })
                else:
                    unmatched_entity_count += 1
                    matching_rows.append({
                        "source1_entity_id": s1_id,
                        "matched_entity_ids": "",
                    })

            matching_results_df = pd.DataFrame(matching_rows)

            # Step 7: Save output files
            matching_results_path = self.output_dir / "matching_results.tsv"
            matching_results_df.to_csv(
                matching_results_path,
                sep="\t",
                index=False,
                encoding="utf-8"
            )

            candidate_pairs_path = self.output_dir / "candidate_pairs.tsv"
            # Keep standard columns
            out_cand_cols = ["source1_entity_id", "candidate_entity_id", "candidate_source"]
            if all(col in cand_pairs_df.columns for col in out_cand_cols):
                cand_pairs_df[out_cand_cols].to_csv(
                    candidate_pairs_path,
                    sep="\t",
                    index=False,
                    encoding="utf-8"
                )
            else:
                pd.DataFrame(columns=out_cand_cols).to_csv(
                    candidate_pairs_path,
                    sep="\t",
                    index=False,
                    encoding="utf-8"
                )

            # Step 8: Update in-memory job state
            result_data = {
                "success": True,
                "message": "Entity matching completed successfully",
                "status": "completed",
                "source1_records": len(s1_clean),
                "source2_records": len(s2_clean),
                "source3_records": len(s3_clean),
                "candidate_pairs": total_pairs,
                "matched_entities": matched_entity_count,
                "unmatched_entities": unmatched_entity_count,
                "threshold": self.threshold,
                "statistics": {
                    "source1_count": len(s1_clean),
                    "source2_count": len(s2_clean),
                    "source3_count": len(s3_clean),
                    "candidate_count": total_pairs,
                    "matched_entities": matched_entity_count,
                    "unmatched_entities": unmatched_entity_count,
                    "threshold": self.threshold,
                },
            }

            self.last_job_state = {
                "status": "completed",
                "message": "Processing completed successfully.",
                "source1_count": len(s1_clean),
                "source2_count": len(s2_clean),
                "source3_count": len(s3_clean),
                "candidate_pair_count": total_pairs,
                "matched_entity_count": matched_entity_count,
                "unmatched_entity_count": unmatched_entity_count,
                "threshold": self.threshold,
                "processed_at": datetime.now().isoformat(),
                "matching_file_available": True,
                "candidates_file_available": True,
            }

            return result_data


# Global singleton pipeline service instance
pipeline_service = PipelineService()
