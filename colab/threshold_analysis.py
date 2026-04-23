"""Threshold analysis — no retraining.

Loads saved anomaly scores (Isolation Forest + LSTM), merges with CERT r4.2
ground truth, and evaluates Precision / Recall / F1 at percentile thresholds
90, 95, 97, 99.  Results are printed at both day level (test split) and
user level (max score per user, all splits).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

REPO_DIR = Path(os.environ.get("DLP_REPO", "/content/dlp-project"))
sys.path.insert(0, str(REPO_DIR))
from config import CLEANED_DIR, MODELS_DIR  # noqa: E402

INSIDERS_CSV = REPO_DIR / "archive" / "answers" / "answers" / "insiders.csv"
IFOREST_CSV  = CLEANED_DIR / "email_user_daily_scored.csv"
LSTM_CSV     = CLEANED_DIR / "email_user_daily_lstm_scored.csv"

PERCENTILES  = [90, 95, 97, 99]


# ── Ground truth ─────────────────────────────────────────────────────────────

def load_day_labels() -> pd.DataFrame:
    ins = pd.read_csv(INSIDERS_CSV)
    ins = ins[ins["dataset"] == 4.2].copy()
    ins["start"] = pd.to_datetime(ins["start"], errors="coerce").dt.normalize()
    ins["end"]   = pd.to_datetime(ins["end"],   errors="coerce").dt.normalize()
    rows = []
    for _, r in ins.iterrows():
        if pd.isna(r["start"]) or pd.isna(r["end"]):
            continue
        for d in pd.date_range(r["start"], r["end"], freq="D"):
            rows.append({"user": r["user"], "email_day": d})
    labels = pd.DataFrame(rows).drop_duplicates()
    labels["is_insider"] = 1
    return labels


# ── Metrics helper ────────────────────────────────────────────────────────────

def metrics_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    cutoff: float,
) -> tuple[float, float, float]:
    y_pred = (y_score >= cutoff).astype(int)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    return p, r, f


def print_table(rows: list[dict], title: str) -> None:
    header = f"{'Pct':>5}  {'Cutoff':>8}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'TP':>5}  {'FP':>6}  {'FN':>5}"
    print(f"\n{'─'*len(header)}")
    print(f"  {title}")
    print(f"{'─'*len(header)}")
    print(header)
    print(f"{'─'*len(header)}")
    for r in rows:
        y_pred = (r["y_score"] >= r["cutoff"]).astype(int)
        tp = int(((r["y_true"] == 1) & (y_pred == 1)).sum())
        fp = int(((r["y_true"] == 0) & (y_pred == 1)).sum())
        fn = int(((r["y_true"] == 1) & (y_pred == 0)).sum())
        print(
            f"  {r['pct']:>3}%  {r['cutoff']:>8.4f}"
            f"  {r['precision']:>10.4f}  {r['recall']:>8.4f}  {r['f1']:>8.4f}"
            f"  {tp:>5}  {fp:>6}  {fn:>5}"
        )
    print(f"{'─'*len(header)}\n")


# ── Per-model evaluation ──────────────────────────────────────────────────────

def analyse(
    df: pd.DataFrame,
    score_col: str,
    day_labels: pd.DataFrame,
    model_name: str,
) -> None:
    # Thresholds derived from the train split only (no leakage)
    train_scores = df.loc[df["dataset_split"] == "train", score_col].dropna()
    cutoffs = {p: float(np.percentile(train_scores, p)) for p in PERCENTILES}

    insider_users = set(day_labels["user"].unique())

    # ── Day level — test split ────────────────────────────────────────────────
    test = df[df["dataset_split"] == "test"].copy()
    test = test.merge(day_labels, on=["user", "email_day"], how="left")
    test["is_insider"] = test["is_insider"].fillna(0).astype(int)

    y_true  = test["is_insider"].values
    y_score = test[score_col].fillna(0).values

    rows = []
    for pct in PERCENTILES:
        cut = cutoffs[pct]
        p, r, f = metrics_at_threshold(y_true, y_score, cut)
        rows.append({"pct": pct, "cutoff": cut, "precision": p, "recall": r, "f1": f,
                     "y_true": y_true, "y_score": y_score})

    print_table(rows, f"{model_name} — Day Level  (test split, {int(y_true.sum())} insider-days / {len(y_true):,} total)")

    # ── User level — all splits, max score per user ───────────────────────────
    user_max = df.groupby("user")[score_col].max().reset_index()
    user_max["is_insider"] = user_max["user"].isin(insider_users).astype(int)

    y_true_u  = user_max["is_insider"].values
    y_score_u = user_max[score_col].fillna(0).values

    rows_u = []
    for pct in PERCENTILES:
        cut = cutoffs[pct]
        p, r, f = metrics_at_threshold(y_true_u, y_score_u, cut)
        rows_u.append({"pct": pct, "cutoff": cut, "precision": p, "recall": r, "f1": f,
                       "y_true": y_true_u, "y_score": y_score_u})

    print_table(rows_u, f"{model_name} — User Level  (max score per user, {int(y_true_u.sum())} insiders / {len(y_true_u)} users)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not INSIDERS_CSV.exists():
        raise FileNotFoundError(
            f"insiders.csv not found at {INSIDERS_CSV}\n"
            "Ensure the repo is cloned to /content/dlp-project or set DLP_REPO."
        )

    day_labels = load_day_labels()
    print(f"Ground truth loaded: {day_labels['user'].nunique()} insider users, "
          f"{len(day_labels):,} insider-days")

    if IFOREST_CSV.exists():
        idf = pd.read_csv(IFOREST_CSV, usecols=["user", "email_day", "iforest_score", "dataset_split"])
        idf["email_day"] = pd.to_datetime(idf["email_day"], errors="coerce").dt.normalize()
        analyse(idf, "iforest_score", day_labels, "Isolation Forest")
    else:
        print(f"[SKIP] {IFOREST_CSV} not found")

    if LSTM_CSV.exists():
        ldf = pd.read_csv(LSTM_CSV, usecols=["user", "email_day", "lstm_score", "lstm_risk_severity", "dataset_split"])
        ldf["email_day"] = pd.to_datetime(ldf["email_day"], errors="coerce").dt.normalize()
        ldf = ldf[ldf["lstm_risk_severity"] != "undetermined"]
        analyse(ldf, "lstm_score", day_labels, "LSTM Autoencoder")
    else:
        print(f"[SKIP] {LSTM_CSV} not found")


if __name__ == "__main__":
    main()
