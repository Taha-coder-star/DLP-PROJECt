"""Phase 4 -- Idea 13 & 14: Stacked meta-model and 3-way voting ensemble.

Idea 13 -- Stacked meta-model:
  A regularised Logistic Regression is trained on all user-level normalised
  signals using CERT ground-truth labels.  This replaces fixed domain-knowledge
  weights with a data-driven combination learned from the actual dataset.
  Class-balanced training handles the ~7 % insider prevalence.
  5-fold cross-validation on train-split users reports out-of-fold ROC-AUC.

Idea 14 -- 3-way majority voting ensemble:
  A user is flagged when they appear in the top-K of at least 2 out of 3
  independent rankers: LSTM score_p95 | IF score (inverted) | risk_score.
  Majority voting raises recall without proportionally degrading precision.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

REPO_DIR = Path(os.environ.get("DLP_REPO", str(Path(__file__).resolve().parent.parent)))
sys.path.insert(0, str(REPO_DIR))
from config import CLEANED_DIR, MODELS_DIR  # noqa: E402
from ground_truth import describe_selection, select_ground_truth_release  # noqa: E402
from risk_scorer import (  # noqa: E402
    compute_behavioral_signals,
    compute_risk_scores,
    load_sensitivity_signals,
)
from user_level_eval import compute_user_scores  # noqa: E402

IFOREST_CSV      = CLEANED_DIR / "email_user_daily_scored.csv"
LSTM_CSV         = CLEANED_DIR / "email_user_daily_lstm_scored.csv"
META_SCORES_CSV  = CLEANED_DIR / "user_meta_scores.csv"
META_MODEL_PKL   = MODELS_DIR  / "meta_model.pkl"
META_REPORT_JSON = MODELS_DIR  / "meta_model_report.json"

K_VALUES = [10, 20, 50]

# All normalised signals produced by compute_risk_scores()
META_FEATURE_COLS = [
    "lstm_p95_norm",
    "lstm_max_norm",
    "flagged_day_norm",
    "latent_dist_norm",
    "after_hours_norm",
    "bcc_usage_norm",
    "file_exfil_norm",
    "usb_activity_norm",
    "multi_pc_norm",
    "content_sensitivity_norm",
    "max_file_exfil_norm",
    "max_usb_norm",
    "if_inverted_norm",
]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _topk_metrics(ranked_df: pd.DataFrame, score_col: str,
                   insider_users: set[str], k: int) -> dict:
    actual_k = min(k, len(ranked_df))
    top_k = set(ranked_df.nlargest(actual_k, score_col)["user"])
    tp = len(top_k & insider_users)
    fp = actual_k - tp
    fn = len(insider_users) - tp
    prec   = tp / actual_k if actual_k > 0 else 0.0
    recall = tp / len(insider_users) if insider_users else 0.0
    f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
    return {"k": k, "actual_k": actual_k, "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": recall, "f1": f1}


def _print_topk_table(risk_df: pd.DataFrame, score_col: str,
                       insider_users: set[str], title: str) -> list[dict]:
    rows = []
    w = 68
    print("\n" + "=" * w)
    print(f"  {title}")
    print("=" * w)
    print(f"  {'K':>5}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}"
          f"  {'TP':>4}  {'FP':>5}  {'FN':>4}")
    print("  " + "-" * (w - 2))
    for k in K_VALUES:
        m = _topk_metrics(risk_df, score_col, insider_users, k)
        star = "*" if m["actual_k"] < m["k"] else ""
        print(f"  {m['k']}{star:<1}  {m['precision']:>10.4f}  {m['recall']:>8.4f}"
              f"  {m['f1']:>8.4f}  {m['tp']:>4}  {m['fp']:>5}  {m['fn']:>4}")
        rows.append(m)
    print("=" * w)
    return rows


def evaluate_3way_voting(
    lstm_user_df: pd.DataFrame,
    if_user_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    insider_users: set[str],
    k: int,
    lstm_agg: str = "score_p95",
) -> dict:
    """Idea 14: flag users who appear in ≥2/3 independent top-K rankers.

    Rankers:
      1. LSTM  -- top-K by score_p95
      2. IF    -- bottom-K by score_p95 (inverted: low IF score = insider-like)
      3. Risk  -- top-K by risk_score (risk_df already sorted descending)
    """
    top_lstm = set(lstm_user_df.nlargest(k, lstm_agg)["user"])
    bot_if   = set(if_user_df.nsmallest(k, "score_p95")["user"])
    top_risk = set(risk_df.head(k)["user"])

    majority = {
        u for u in (top_lstm | bot_if | top_risk)
        if (u in top_lstm) + (u in bot_if) + (u in top_risk) >= 2
    }
    tp = len(majority & insider_users)
    fp = len(majority) - tp
    fn = len(insider_users) - tp
    prec   = tp / len(majority)      if majority      else 0.0
    recall = tp / len(insider_users) if insider_users else 0.0
    f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
    return {
        "k": k, "flagged": len(majority),
        "lstm_pool": len(top_lstm), "if_pool": len(bot_if), "risk_pool": len(top_risk),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": prec, "recall": recall, "f1": f1,
    }


def print_3way_table(
    lstm_user_df: pd.DataFrame,
    if_user_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    insider_users: set[str],
) -> list[dict]:
    w = 84
    print("\n" + "=" * w)
    print("  Idea 14 — 3-Way Majority Voting Ensemble")
    print("  LSTM top-K  ∪  IF bottom-K  ∪  Risk top-K   (≥2/3 votes = flagged)")
    print("=" * w)
    print(f"  {'K':>5}  {'Flagged':>8}  {'LSTM':>6}  {'IF':>6}  {'Risk':>6}"
          f"  {'TP':>4}  {'FP':>5}  {'FN':>4}"
          f"  {'Prec':>8}  {'Recall':>8}  {'F1':>8}")
    print("  " + "-" * (w - 2))
    rows = []
    for k in K_VALUES:
        m = evaluate_3way_voting(lstm_user_df, if_user_df, risk_df, insider_users, k)
        print(
            f"  {k:>5}  {m['flagged']:>8}  {m['lstm_pool']:>6}  {m['if_pool']:>6}"
            f"  {m['risk_pool']:>6}"
            f"  {m['tp']:>4}  {m['fp']:>5}  {m['fn']:>4}"
            f"  {m['precision']:>8.4f}  {m['recall']:>8.4f}  {m['f1']:>8.4f}"
        )
        rows.append(m)
    print("=" * w + "\n")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    gt = select_ground_truth_release([IFOREST_CSV, LSTM_CSV])
    insider_users = gt.matching_users
    print("Ground truth:")
    print(f"  {describe_selection(gt)}\n")

    # ── Load scored CSVs ──────────────────────────────────────────────────────
    if not IFOREST_CSV.exists():
        raise FileNotFoundError(f"IF scored CSV not found: {IFOREST_CSV}")
    if not LSTM_CSV.exists():
        raise FileNotFoundError(f"LSTM scored CSV not found: {LSTM_CSV}")

    idf = pd.read_csv(IFOREST_CSV, usecols=["user", "iforest_score", "dataset_split"])
    if_user_df = compute_user_scores(idf, "iforest_score", insider_users)

    _lstm_cols = ["user", "lstm_score", "lstm_risk_severity", "dataset_split"]
    _avail = pd.read_csv(LSTM_CSV, nrows=0).columns.tolist()
    if "latent_distance" in _avail:
        _lstm_cols.append("latent_distance")
    ldf = pd.read_csv(LSTM_CSV, usecols=_lstm_cols)
    ldf = ldf[ldf["lstm_risk_severity"] != "undetermined"]
    lstm_user_df = compute_user_scores(ldf, "lstm_score", insider_users)

    idf_full      = pd.read_csv(IFOREST_CSV)
    behavioral_df = compute_behavioral_signals(idf_full)
    sensitivity_df = load_sensitivity_signals()

    risk_df = compute_risk_scores(
        lstm_user_df, behavioral_df,
        insider_users=insider_users,
        sensitivity_df=sensitivity_df,
        if_user_df=if_user_df,
    )

    # ── Build feature matrix ──────────────────────────────────────────────────
    avail_feats = [c for c in META_FEATURE_COLS if c in risk_df.columns]
    missing     = [c for c in META_FEATURE_COLS if c not in risk_df.columns]
    if missing:
        print(f"  [INFO] Features not yet available (will be 0): {missing}")

    X = risk_df[avail_feats].fillna(0).values
    y = risk_df["is_insider"].values if "is_insider" in risk_df.columns else \
        risk_df["user"].isin(insider_users).astype(int).values

    print(f"Feature matrix  : {X.shape[0]} users × {X.shape[1]} signals")
    print(f"Insiders labelled: {y.sum()} / {len(y)}")

    # ── Idea 13: train Logistic Regression meta-model ────────────────────────
    train_mask = (risk_df["dataset_split"] == "train").values
    X_tr, y_tr = X[train_mask], y[train_mask]
    print(f"\nTraining meta-model on {train_mask.sum()} train-split users "
          f"({y_tr.sum()} insiders)...")

    clf = LogisticRegression(
        C=0.1, class_weight="balanced",
        max_iter=1000, random_state=42, solver="lbfgs",
    )

    cv_auc = cv_ap = 0.0
    if y_tr.sum() >= 5:
        cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_prob = cross_val_predict(clf, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        cv_auc  = float(roc_auc_score(y_tr, cv_prob))
        cv_ap   = float(average_precision_score(y_tr, cv_prob))
        print(f"  5-fold CV (train users) → ROC-AUC={cv_auc:.4f}  AP={cv_ap:.4f}")
    else:
        print("  [WARN] Fewer than 5 insiders in train split — skipping CV.")

    clf.fit(X_tr, y_tr)
    risk_df["meta_score"] = clf.predict_proba(X)[:, 1]

    # ── Print top-5 feature importances ──────────────────────────────────────
    print("\n  Top meta-model signal weights (LR coefficients):")
    for feat, coef in sorted(zip(avail_feats, clf.coef_[0]),
                              key=lambda x: -abs(x[1]))[:8]:
        direction = "↑" if coef > 0 else "↓"
        print(f"    {feat:<30}  {coef:+.4f}  {direction}")

    # ── Meta-score top-K evaluation ───────────────────────────────────────────
    meta_rows = _print_topk_table(
        risk_df, "meta_score", insider_users,
        "Idea 13 — Meta-model Top-K (all users, meta_score ranking)",
    )

    # ── Idea 14: 3-way voting ensemble ────────────────────────────────────────
    voting_rows = print_3way_table(lstm_user_df, if_user_df, risk_df, insider_users)

    # ── Best configuration summary ─────────────────────────────────────────────
    best_meta   = max(meta_rows,   key=lambda r: r["f1"])
    best_voting = max(voting_rows, key=lambda r: r["f1"])
    n_ins = len(insider_users)
    print("=" * 68)
    print("  Best meta-model  : "
          f"K={best_meta['k']}  F1={best_meta['f1']:.4f}  "
          f"TP={best_meta['tp']}/{n_ins}  Prec={best_meta['precision']:.4f}")
    print("  Best 3-way vote  : "
          f"K={best_voting['k']}  F1={best_voting['f1']:.4f}  "
          f"TP={best_voting['tp']}/{n_ins}  Prec={best_voting['precision']:.4f}")
    print("=" * 68 + "\n")

    # ── Save outputs ──────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    save_cols = ["user", "meta_score", "risk_score", "dataset_split"]
    if "is_insider" in risk_df.columns:
        save_cols.append("is_insider")
    risk_df[save_cols].sort_values("meta_score", ascending=False).to_csv(
        META_SCORES_CSV, index=False
    )
    print(f"Saved meta-scores  -> {META_SCORES_CSV}")

    with open(META_MODEL_PKL, "wb") as fh:
        pickle.dump({"model": clf, "feature_cols": avail_feats,
                     "cv_roc_auc": cv_auc, "cv_avg_precision": cv_ap}, fh)
    print(f"Saved meta-model   -> {META_MODEL_PKL}")

    report = {
        "cv_roc_auc":       round(cv_auc, 4),
        "cv_avg_precision": round(cv_ap, 4),
        "n_users":          int(len(risk_df)),
        "n_insiders":       int(y.sum()),
        "n_train_users":    int(train_mask.sum()),
        "feature_cols":     avail_feats,
        "lr_coef":          dict(zip(avail_feats, [round(c, 6) for c in clf.coef_[0]])),
        "meta_topk":        meta_rows,
        "voting_topk":      voting_rows,
    }
    META_REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved meta-report  -> {META_REPORT_JSON}")


if __name__ == "__main__":
    main()
