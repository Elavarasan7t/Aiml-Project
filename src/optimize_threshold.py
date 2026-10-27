"""
Threshold Optimization Module for Business Entity Resolution.

Evaluates candidate pair prediction probabilities across thresholds from 0.05 to 0.95:
- Computes global pair-level metrics (Precision, Recall, F1, F0.5, Predicted Matches)
- Computes entity-level macro metrics (Macro Precision, Macro Recall, Macro F0.5)
- Evaluates singleton handling (accuracy on entities with zero true matches)
- Identifies the optimal threshold maximizing Macro Entity F0.5

Inputs:
- output/validation_predictions.tsv
- dataset/train/train_ground_truth.tsv

Outputs:
- output/threshold_results.tsv (pair-level sweep results)
- output/entity_threshold_results.tsv (entity-level sweep results)
- models/threshold.json (selected threshold parameters)
- output/validation_summary.txt (comprehensive validation report)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def get_project_root() -> Path:
    """
    Locates the project root directory containing the dataset and output folders.
    Supports running from project root, repository root, or src directory.
    """
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    if (parent_dir / "output" / "validation_predictions.tsv").exists():
        return parent_dir

    cwd = Path.cwd()
    if (cwd / "output" / "validation_predictions.tsv").exists():
        return cwd

    nested = cwd / "synthetic_entity_resolution_dataset" / "synthetic_entity_resolution_dataset"
    if (nested / "output" / "validation_predictions.tsv").exists():
        return nested

    nested_single = cwd / "synthetic_entity_resolution_dataset"
    if (nested_single / "output" / "validation_predictions.tsv").exists():
        return nested_single

    return parent_dir


def compute_f05(precision: float, recall: float) -> float:
    """
    Calculates F0.5 score:
      F0.5 = (1 + 0.5^2) * Precision * Recall / (0.5^2 * Precision + Recall)
           = 1.25 * Precision * Recall / (0.25 * Precision + Recall)
    Safely handles zero denominator.
    """
    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1.25 * precision * recall) / denom


def evaluate_pair_level_thresholds(
    y_true: np.ndarray, y_prob: np.ndarray
) -> pd.DataFrame:
    """
    Evaluates thresholds from 0.05 to 0.95 (step 0.01) at the pair level.
    Returns DataFrame sorted by f0_5 descending.
    """
    records = []
    for t_int in range(5, 96):
        t = round(t_int / 100.0, 2)
        preds = (y_prob >= t).astype(int)
        p = float(precision_score(y_true, preds, zero_division=0))
        r = float(recall_score(y_true, preds, zero_division=0))
        f1 = float(f1_score(y_true, preds, zero_division=0))
        f05 = compute_f05(p, r)
        matches = int(preds.sum())

        records.append({
            "threshold": t,
            "precision": p,
            "recall": r,
            "f1": f1,
            "f0_5": f05,
            "predicted_matches": matches,
        })

    df = pd.DataFrame(records)
    return df


def evaluate_entity_level_thresholds(
    val_df: pd.DataFrame, gt_dict: dict[str, set[str]]
) -> pd.DataFrame:
    """
    Evaluates thresholds from 0.05 to 0.95 (step 0.01) at the Source 1 entity level.
    Returns DataFrame sorted by macro_f0_5 descending.
    """
    # Pre-group validation pairs by Source 1 entity for high-speed evaluation
    # Each entry: list of (candidate_id, probability, true_label)
    s1_pairs = defaultdict(list)
    for _, r in val_df.iterrows():
        s1_pairs[r["source1_entity_id"]].append(
            (r["candidate_entity_id"], float(r["prediction_probability"]), int(r["true_label"]))
        )

    all_s1_ids = sorted(gt_dict.keys())
    records = []

    for t_int in range(5, 96):
        t = round(t_int / 100.0, 2)
        entity_prec = []
        entity_rec = []
        entity_f05 = []

        with_preds = 0
        without_preds = 0
        singleton_entities = 0
        singleton_errors = 0

        for s1_id in all_s1_ids:
            pairs = s1_pairs.get(s1_id, [])
            pred_ids = {cid for cid, prob, _ in pairs if prob >= t}
            true_ids = {cid for cid, _, is_true in pairs if is_true == 1}

            raw_gt = gt_dict[s1_id]
            is_singleton = (len(raw_gt) == 0)

            if is_singleton:
                singleton_entities += 1
                if len(pred_ids) == 0:
                    p, r, f = 1.0, 1.0, 1.0
                else:
                    singleton_errors += 1
                    p, r, f = 0.0, 0.0, 0.0
            else:
                # Non-singleton entity
                if len(true_ids) == 0:
                    # Non-singleton whose true matches were in train split
                    if len(pred_ids) == 0:
                        p, r, f = 1.0, 1.0, 1.0
                    else:
                        p, r, f = 0.0, 0.0, 0.0
                else:
                    tp = len(pred_ids & true_ids)
                    fp = len(pred_ids - true_ids)
                    fn = len(true_ids - pred_ids)

                    p = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                    r = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                    f = compute_f05(p, r)

            if len(pred_ids) > 0:
                with_preds += 1
            else:
                without_preds += 1

            entity_prec.append(p)
            entity_rec.append(r)
            entity_f05.append(f)

        macro_p = float(sum(entity_prec) / len(entity_prec))
        macro_r = float(sum(entity_rec) / len(entity_rec))
        macro_f05 = float(sum(entity_f05) / len(entity_f05))

        records.append({
            "threshold": t,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f0_5": macro_f05,
            "entities_with_predictions": with_preds,
            "entities_without_predictions": without_preds,
            "singleton_entities": singleton_entities,
            "singleton_errors": singleton_errors,
        })

    df = pd.DataFrame(records)
    return df


def get_detailed_stats_at_threshold(
    val_df: pd.DataFrame, gt_dict: dict[str, set[str]], s1_pairs: dict[str, list], threshold: float
) -> dict:
    """
    Computes all detailed metrics and singleton breakdowns at a specific threshold.
    """
    # 1. Pair-level metrics
    y_true = val_df["true_label"].values
    y_prob = val_df["prediction_probability"].values
    preds = (y_prob >= threshold).astype(int)

    pair_prec = float(precision_score(y_true, preds, zero_division=0))
    pair_rec = float(recall_score(y_true, preds, zero_division=0))
    pair_f1 = float(f1_score(y_true, preds, zero_division=0))
    pair_f05 = compute_f05(pair_prec, pair_rec)
    pred_matches = int(preds.sum())

    # 2. Entity-level metrics
    all_s1_ids = sorted(gt_dict.keys())
    entity_prec = []
    entity_rec = []
    entity_f05 = []

    singletons_correct = 0
    singletons_incorrect = 0
    non_sing_correct_match = 0
    non_sing_no_match = 0

    total_singletons = 0
    total_non_singletons = 0

    for s1_id in all_s1_ids:
        pairs = s1_pairs.get(s1_id, [])
        pred_ids = {cid for cid, prob, _ in pairs if prob >= threshold}
        true_ids = {cid for cid, _, is_true in pairs if is_true == 1}

        raw_gt = gt_dict[s1_id]
        is_singleton = (len(raw_gt) == 0)

        if is_singleton:
            total_singletons += 1
            if len(pred_ids) == 0:
                singletons_correct += 1
                p, r, f = 1.0, 1.0, 1.0
            else:
                singletons_incorrect += 1
                p, r, f = 0.0, 0.0, 0.0
        else:
            total_non_singletons += 1
            if len(pred_ids & true_ids) > 0:
                non_sing_correct_match += 1
            else:
                non_sing_no_match += 1

            if len(true_ids) == 0:
                if len(pred_ids) == 0:
                    p, r, f = 1.0, 1.0, 1.0
                else:
                    p, r, f = 0.0, 0.0, 0.0
            else:
                tp = len(pred_ids & true_ids)
                fp = len(pred_ids - true_ids)
                fn = len(true_ids - pred_ids)
                p = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                r = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                f = compute_f05(p, r)

        entity_prec.append(p)
        entity_rec.append(r)
        entity_f05.append(f)

    macro_p = float(sum(entity_prec) / len(entity_prec))
    macro_r = float(sum(entity_rec) / len(entity_rec))
    macro_f05 = float(sum(entity_f05) / len(entity_f05))

    return {
        "threshold": threshold,
        "pair_precision": pair_prec,
        "pair_recall": pair_rec,
        "pair_f1": pair_f1,
        "pair_f05": pair_f05,
        "predicted_matches": pred_matches,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f05": macro_f05,
        "total_singletons": total_singletons,
        "singletons_correct": singletons_correct,
        "singletons_incorrect": singletons_incorrect,
        "total_non_singletons": total_non_singletons,
        "non_singletons_correct_match": non_sing_correct_match,
        "non_singletons_no_match": non_sing_no_match,
    }


def main():
    project_root = get_project_root()
    output_dir = project_root / "output"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    val_preds_path = output_dir / "validation_predictions.tsv"
    gt_path = project_root / "dataset" / "train" / "train_ground_truth.tsv"

    if not val_preds_path.exists():
        print(f"Error: Missing validation predictions file: {val_preds_path}", file=sys.stderr)
        sys.exit(1)

    if not gt_path.exists():
        print(f"Error: Missing ground truth file: {gt_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Load validation predictions and ground truth
    val_df = pd.read_csv(val_preds_path, sep="\t")
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)

    gt_dict = {}
    for _, r in gt_df.iterrows():
        s1_id = r["source1_entity_id"]
        ms = set(m.strip() for m in r["matched_entity_ids"].split(",") if m.strip())
        gt_dict[s1_id] = ms

    y_true = val_df["true_label"].values
    y_prob = val_df["prediction_probability"].values

    # 2. Pair-level evaluation across thresholds
    df_pair_results = evaluate_pair_level_thresholds(y_true, y_prob)
    df_pair_sorted = df_pair_results.sort_values("f0_5", ascending=False).reset_index(drop=True)

    threshold_results_path = output_dir / "threshold_results.tsv"
    df_pair_sorted.to_csv(threshold_results_path, sep="\t", index=False)

    best_pair_row = df_pair_sorted.iloc[0]

    # 3. Entity-level evaluation across thresholds
    df_entity_results = evaluate_entity_level_thresholds(val_df, gt_dict)
    df_entity_sorted = df_entity_results.sort_values("macro_f0_5", ascending=False).reset_index(drop=True)

    entity_threshold_results_path = output_dir / "entity_threshold_results.tsv"
    df_entity_sorted.to_csv(entity_threshold_results_path, sep="\t", index=False)

    best_entity_row = df_entity_sorted.iloc[0]
    selected_threshold = float(best_entity_row["threshold"])

    # Pre-group validation pairs by Source 1 entity for fast evaluation
    s1_pairs = defaultdict(list)
    for _, r in val_df.iterrows():
        s1_pairs[r["source1_entity_id"]].append(
            (r["candidate_entity_id"], float(r["prediction_probability"]), int(r["true_label"]))
        )

    # 4. Detailed stats at selected threshold
    stats = get_detailed_stats_at_threshold(val_df, gt_dict, s1_pairs, selected_threshold)

    # 5. Save threshold.json (Task 12)
    threshold_json_path = models_dir / "threshold.json"
    threshold_data = {
        "threshold": selected_threshold,
        "metric": "macro_entity_f0_5",
        "validation_macro_f0_5": round(stats["macro_f05"], 4),
        "pair_f0_5": round(stats["pair_f05"], 4),
        "pair_precision": round(stats["pair_precision"], 4),
        "pair_recall": round(stats["pair_recall"], 4),
    }

    with open(threshold_json_path, "w", encoding="utf-8") as f:
        json.dump(threshold_data, f, indent=4)

    # 6. Save validation summary (Task 15)
    summary_path = output_dir / "validation_summary.txt"
    summary_content = f"""VALIDATION SUMMARY
==================================================
Selected Threshold:              {selected_threshold:.2f}
Optimization Metric:             macro_entity_f0_5

Pair-level Precision:            {stats['pair_precision']:.4f}
Pair-level Recall:               {stats['pair_recall']:.4f}
Pair-level F1:                   {stats['pair_f1']:.4f}
Pair-level F0.5:                 {stats['pair_f05']:.4f}

Entity-level Macro Precision:    {stats['macro_precision']:.4f}
Entity-level Macro Recall:       {stats['macro_recall']:.4f}
Entity-level Macro F0.5:         {stats['macro_f05']:.4f}

Number of Singleton Entities:    {stats['total_singletons']}
Singleton False-Match Count:     {stats['singletons_incorrect']}
Singletons Correctly Unmatched:  {stats['singletons_correct']}

Number of Predicted Matches:     {stats['predicted_matches']}
Total Non-singleton Entities:    {stats['total_non_singletons']}
Non-singletons with >=1 Match:   {stats['non_singletons_correct_match']}
==================================================
"""

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_content)

    # 7. Print Output Reports
    print("BEST PAIR-LEVEL THRESHOLD")
    print("-------------------------")
    print(f"Threshold: {best_pair_row['threshold']:.2f}")
    print(f"Precision: {best_pair_row['precision']:.4f}")
    print(f"Recall: {best_pair_row['recall']:.4f}")
    print(f"F1: {best_pair_row['f1']:.4f}")
    print(f"F0.5: {best_pair_row['f0_5']:.4f}")
    print(f"Predicted matches: {int(best_pair_row['predicted_matches'])}\n")

    print("ENTITY-LEVEL BEST THRESHOLD")
    print("--------------------------------")
    print(f"Threshold: {selected_threshold:.2f}")
    print(f"Macro Precision: {stats['macro_precision']:.4f}")
    print(f"Macro Recall: {stats['macro_recall']:.4f}")
    print(f"Macro F0.5: {stats['macro_f05']:.4f}\n")

    print("PAIR-LEVEL METRICS AT SELECTED THRESHOLD")
    print("----------------------------------------")
    print(f"Precision: {stats['pair_precision']:.4f}")
    print(f"Recall: {stats['pair_recall']:.4f}")
    print(f"F1: {stats['pair_f1']:.4f}")
    print(f"F0.5: {stats['pair_f05']:.4f}")
    print(f"Predicted matches: {stats['predicted_matches']}\n")

    # Task 13: Precision / Recall Tradeoff Table around selected threshold
    print("PRECISION / RECALL TRADEOFF AROUND SELECTED THRESHOLD")
    print("-----------------------------------------------------")
    print(f"{'Threshold':<12}{'Precision':<14}{'Recall':<14}{'Macro F0.5':<12}")
    tradeoff_thresholds = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    for t_val in tradeoff_thresholds:
        row_stat = get_detailed_stats_at_threshold(val_df, gt_dict, s1_pairs, t_val)
        print(f"{t_val:<12.2f}{row_stat['pair_precision']:<14.4f}{row_stat['pair_recall']:<14.4f}{row_stat['macro_f05']:<12.4f}")
    print()

    # Task 14: Singleton and Non-singleton verification
    print("SINGLETON PERFORMANCE VERIFICATION")
    print("----------------------------------")
    print(f"Total singleton Source 1 entities: {stats['total_singletons']}")
    print(f"Singletons correctly predicted with no matches: {stats['singletons_correct']}")
    print(f"Singletons incorrectly assigned matches: {stats['singletons_incorrect']}\n")

    print("NON-SINGLETON PERFORMANCE VERIFICATION")
    print("--------------------------------------")
    print(f"Total non-singleton Source 1 entities: {stats['total_non_singletons']}")
    print(f"Non-singletons with at least one correct match: {stats['non_singletons_correct_match']}")
    print(f"Non-singletons with no correct match: {stats['non_singletons_no_match']}\n")

    print("SAVED ARTIFACTS")
    print("----------------------------------")
    print(f"Saved: models/threshold.json")
    print(f"Saved: output/threshold_results.tsv")
    print(f"Saved: output/entity_threshold_results.tsv")
    print(f"Saved: output/validation_summary.txt")


if __name__ == "__main__":
    main()
