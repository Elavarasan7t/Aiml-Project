"""
Machine Learning Model Training Module for Business Entity Resolution.

Trains a baseline XGBoost binary classification model to predict whether
a candidate pair represents the same business entity.

Inputs:
- output/training_features.tsv
- dataset/train/train_ground_truth.tsv (for singleton validation analysis only)

Outputs:
- models/entity_resolution_xgboost.json (trained model artifact)
- models/feature_columns.json (ordered list of feature names)
- output/validation_predictions.tsv (validation predictions with probabilities)
- output/feature_importance.tsv (feature importance rankings)
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def get_project_root() -> Path:
    """
    Locates the project root directory containing the dataset and output folders.
    Supports running from project root, repository root, or src directory.
    """
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    if (parent_dir / "output" / "training_features.tsv").exists():
        return parent_dir

    cwd = Path.cwd()
    if (cwd / "output" / "training_features.tsv").exists():
        return cwd

    nested = cwd / "synthetic_entity_resolution_dataset" / "synthetic_entity_resolution_dataset"
    if (nested / "output" / "training_features.tsv").exists():
        return nested

    nested_single = cwd / "synthetic_entity_resolution_dataset"
    if (nested_single / "output" / "training_features.tsv").exists():
        return nested_single

    return parent_dir


def load_and_check_data(
    features_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str]]:
    """
    Loads training features, separates identifier columns and target,
    validates data integrity, and prints summary checks.
    """
    df = pd.read_csv(features_path, sep="\t")

    id_cols = ["source1_entity_id", "candidate_entity_id", "candidate_source"]
    target_col = "label"

    feature_cols = [
        col for col in df.columns if col not in id_cols and col != target_col
    ]

    total_rows = len(df)
    n_features = len(feature_cols)
    pos_count = int((df[target_col] == 1).sum())
    neg_count = int((df[target_col] == 0).sum())
    pos_pct = (pos_count / total_rows) * 100 if total_rows > 0 else 0.0
    neg_pct = (neg_count / total_rows) * 100 if total_rows > 0 else 0.0

    missing_per_feature = df[feature_cols].isnull().sum().to_dict()
    total_missing = sum(missing_per_feature.values())
    dup_rows = int(df.duplicated(subset=id_cols).sum())

    print("DATA CHECK")
    print("--------------------------------")
    print(f"Number of rows: {total_rows}")
    print(f"Number of features: {n_features}")
    print(f"Number of positive examples: {pos_count}")
    print(f"Number of negative examples: {neg_count}")
    print(f"Positive percentage: {pos_pct:.2f}%")
    print(f"Negative percentage: {neg_pct:.2f}%")
    print(f"Missing values across features: {total_missing}")
    print(f"Duplicate rows: {dup_rows}")
    print()

    assert total_missing == 0, f"Found {total_missing} missing values in feature set!"
    assert dup_rows == 0, f"Found {dup_rows} duplicate rows in feature set!"

    X = df[feature_cols]
    y = df[target_col]
    identifiers = df[id_cols]

    return identifiers, X, y, feature_cols


def compute_f05_score(precision: float, recall: float) -> float:
    """
    Calculates the F0.5 score:
      F0.5 = (1 + 0.5^2) * Precision * Recall / (0.5^2 * Precision + Recall)
    F0.5 places twice as much weight on precision as recall.
    """
    beta_sq = 0.5 ** 2
    denominator = (beta_sq * precision) + recall
    if denominator == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denominator


def evaluate_singletons_and_entities(
    val_df: pd.DataFrame, ground_truth_path: Path
) -> tuple[dict, dict]:
    """
    Performs entity-level and singleton evaluation on the validation set.
    """
    # 1. Group predictions by source1_entity_id
    val_groups = val_df.groupby("source1_entity_id")
    no_pred_s1 = 0
    one_pred_s1 = 0
    multi_pred_s1 = 0

    for _, grp in val_groups:
        pred_cnt = int(grp["prediction_0_50"].sum())
        if pred_cnt == 0:
            no_pred_s1 += 1
        elif pred_cnt == 1:
            one_pred_s1 += 1
        else:
            multi_pred_s1 += 1

    entity_stats = {
        "val_entities_total": len(val_groups),
        "entities_no_preds": no_pred_s1,
        "entities_one_pred": one_pred_s1,
        "entities_multi_preds": multi_pred_s1,
    }

    # 2. Singleton analysis against training ground truth
    singleton_stats = {
        "singleton_entities": 0,
        "singletons_correctly_unmatched": 0,
        "singletons_incorrectly_matched": 0,
    }

    if ground_truth_path.exists():
        gt_df = pd.read_csv(ground_truth_path, sep="\t", dtype=str, keep_default_na=False)
        all_s1 = set(gt_df["source1_entity_id"])
        s1_with_gt = set()
        for _, r in gt_df.iterrows():
            if r["matched_entity_ids"].strip():
                s1_with_gt.add(r["source1_entity_id"])

        true_singletons = all_s1 - s1_with_gt
        val_singletons = set(val_df["source1_entity_id"]).intersection(true_singletons)

        corr_unmatched = 0
        incorr_matched = 0

        for s1_id in val_singletons:
            grp = val_groups.get_group(s1_id)
            if int(grp["prediction_0_50"].sum()) == 0:
                corr_unmatched += 1
            else:
                incorr_matched += 1

        singleton_stats = {
            "singleton_entities": len(val_singletons),
            "singletons_correctly_unmatched": corr_unmatched,
            "singletons_incorrectly_matched": incorr_matched,
        }

    return entity_stats, singleton_stats


def main():
    project_root = get_project_root()
    output_dir = project_root / "output"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    features_path = output_dir / "training_features.tsv"
    gt_path = project_root / "dataset" / "train" / "train_ground_truth.tsv"

    if not features_path.exists():
        print(f"Error: Missing training features: {features_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Load data and verify
    identifiers, X, y, feature_cols = load_and_check_data(features_path)

    # 2. Stratified 80/20 train/validation split
    train_idx, val_idx = train_test_split(
        X.index, test_size=0.20, random_state=42, stratify=y
    )

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_val, y_val = X.loc[val_idx], y.loc[val_idx]
    val_identifiers = identifiers.loc[val_idx].copy()

    # 3. Class imbalance check
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    imbalance_ratio = n_neg / n_pos if n_pos > 0 else 1.0

    # For F0.5 entity resolution (which heavily penalizes false positives),
    # unweighted training (scale_pos_weight=1.0) produces well-calibrated
    # probabilities and optimal precision without inflating false positives.
    scale_pos_weight_used = 1.0

    print(f"Calculated negative/positive imbalance ratio: {imbalance_ratio:.2f}")
    print(f"Using scale_pos_weight: {scale_pos_weight_used:.1f} (baseline calibrated)\n")

    # 4. Initialize and train baseline XGBoost model
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight_used,
        n_jobs=-1,
    )

    print("Training XGBoost binary classification model...")
    model.fit(X_train, y_train)
    print("Model training completed successfully.\n")

    # 5. Predictions on validation set
    y_val_prob = model.predict_proba(X_val)[:, 1]
    y_val_pred_050 = (y_val_prob >= 0.50).astype(int)

    # Metrics calculation
    val_precision = float(precision_score(y_val, y_val_pred_050, zero_division=0))
    val_recall = float(recall_score(y_val, y_val_pred_050, zero_division=0))
    val_f1 = float(f1_score(y_val, y_val_pred_050, zero_division=0))
    val_f05 = compute_f05_score(val_precision, val_recall)
    val_accuracy = float(accuracy_score(y_val, y_val_pred_050))
    val_roc_auc = float(roc_auc_score(y_val, y_val_prob))

    # 6. Entity-level and singleton validation
    val_predictions_df = val_identifiers.copy()
    val_predictions_df["true_label"] = y_val.values
    val_predictions_df["prediction_probability"] = y_val_prob
    val_predictions_df["prediction_0_50"] = y_val_pred_050

    entity_stats, singleton_stats = evaluate_singletons_and_entities(
        val_predictions_df, gt_path
    )

    # 7. Save trained model and artifacts
    model_save_path = models_dir / "entity_resolution_xgboost.json"
    feature_cols_save_path = models_dir / "feature_columns.json"
    val_preds_save_path = output_dir / "validation_predictions.tsv"
    feat_imp_save_path = output_dir / "feature_importance.tsv"

    model.save_model(model_save_path)

    with open(feature_cols_save_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    val_predictions_df.to_csv(val_preds_save_path, sep="\t", index=False)

    # Feature importance
    feature_importances = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    feature_importances.to_csv(feat_imp_save_path, sep="\t", index=False)

    # 8. Print formatted final reports (Task 17)
    print("MODEL TRAINING REPORT")
    print("--------------------------------")
    print()
    print("Training rows:")
    print(len(X_train))
    print()
    print("Validation rows:")
    print(len(X_val))
    print()
    print("Number of features:")
    print(len(feature_cols))
    print()
    print("Positive examples:")
    print(int((y_train == 1).sum()))
    print()
    print("Negative examples:")
    print(int((y_train == 0).sum()))
    print()
    print("Scale positive weight:")
    print(f"{scale_pos_weight_used:.1f}")
    print()
    print("Model:")
    print("XGBoost")
    print()
    print("Validation Precision:")
    print(f"{val_precision:.4f}")
    print()
    print("Validation Recall:")
    print(f"{val_recall:.4f}")
    print()
    print("Validation F1:")
    print(f"{val_f1:.4f}")
    print()
    print("Validation F0.5:")
    print(f"{val_f05:.4f}")
    print()
    print("Validation Accuracy:")
    print(f"{val_accuracy:.4f}")
    print()
    print("Validation ROC-AUC:")
    print(f"{val_roc_auc:.4f}")
    print()
    print("Singleton Source 1 entities:")
    print(singleton_stats["singleton_entities"])
    print()
    print("Singletons correctly left unmatched:")
    print(singleton_stats["singletons_correctly_unmatched"])
    print()
    print("Singletons incorrectly matched:")
    print(singleton_stats["singletons_incorrectly_matched"])
    print()
    print("ENTITY-LEVEL VALIDATION SUMMARY")
    print("--------------------------------")
    print(f"Validation Source 1 entities: {entity_stats['val_entities_total']}")
    print(f"Entities with 0 predicted matches: {entity_stats['entities_no_preds']}")
    print(f"Entities with 1 predicted match:   {entity_stats['entities_one_pred']}")
    print(f"Entities with >1 predicted match:  {entity_stats['entities_multi_preds']}")
    print()
    print("SAVED ARTIFACTS")
    print("--------------------------------")
    print(f"Model saved to: {model_save_path.relative_to(project_root)}")
    print(f"Feature columns saved to: {feature_cols_save_path.relative_to(project_root)}")
    print(f"Validation predictions saved to: {val_preds_save_path.relative_to(project_root)}")
    print(f"Feature importance saved to: {feat_imp_save_path.relative_to(project_root)}")


if __name__ == "__main__":
    main()
