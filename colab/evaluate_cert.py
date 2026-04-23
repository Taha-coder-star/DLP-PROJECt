"""Generate evaluation_report.json for the monitoring app.

Reads insiders.csv from the cloned repo (REPO_DIR/archive/answers/answers/insiders.csv),
evaluates IF and LSTM against CERT r4.2 ground truth, and saves the report with the
exact key structure the monitoring app expects.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

# ── Path setup ───────────────────────────────────────────────────────────────
REPO_DIR  = Path(os.environ.get("DLP_REPO",  "/content/dlp-project"))
sys.path.insert(0, str(REPO_DIR))
from config import CLEANED_DIR, MODELS_DIR  # noqa: E402

INSIDERS_CSV  = REPO_DIR / "archive" / "answers" / "answers" / "insiders.csv"
IFOREST_CSV   = CLEANED_DIR / "email_user_daily_scored.csv"
LSTM_CSV      = CLEANED_DIR / "email_user_daily_lstm_scored.csv"
REPORT_PATH   = MODELS_DIR / "evaluation_report.json"


# ── Ground truth loader ───────────────────────────────────────────────────────

def load_day_labels() -> pd.DataFrame:
    """Expand CERT r4.2 insider windows to one row per (user, date)."""
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


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> dict:
    if y_true.sum() == 0:
        return {}
    roc  = roc_auc_score(y_true, y_score)
    ap   = average_precision_score(y_true, y_score)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "roc_auc":       round(roc,  4),
        "avg_precision": round(ap,   4),
        "precision":     round(prec, 4),
        "recall":        round(rec,  4),
        "f1":            round(f1,   4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "n_insiders": int(y_true.sum()),
        "n_total":    int(len(y_true)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading ground truth from: {INSIDERS_CSV}")
    if not INSIDERS_CSV.exists():
        raise FileNotFoundError(
            f"insiders.csv not found at {INSIDERS_CSV}\n"
            "Make sure the repo was cloned to /content/dlp-project "
            "or set the DLP_REPO env var."
        )
    day_labels = load_day_labels()
    insider_users = day_labels["user"].unique()
    print(f"  Insider users: {len(insider_users)}  |  Insider user-days: {len(day_labels)}")

    report: dict = {}

    # ── Isolation Forest ──────────────────────────────────────────────────────
    if IFOREST_CSV.exists():
        print(f"Loading IF scored: {IFOREST_CSV}")
        idf = pd.read_csv(IFOREST_CSV, usecols=["user", "email_day", "iforest_score", "risk_severity", "dataset_split"])
        idf["email_day"] = pd.to_datetime(idf["email_day"], errors="coerce").dt.normalize()

        # Day-level — test split only
        test_idf = idf[idf["dataset_split"] == "test"].merge(day_labels, on=["user", "email_day"], how="left")
        test_idf["is_insider"] = test_idf["is_insider"].fillna(0).astype(int)
        y_true  = test_idf["is_insider"].values
        y_score = test_idf["iforest_score"].values
        y_pred  = test_idf["risk_severity"].isin(["suspicious", "high"]).astype(int).values
        report["if_day_test"] = evaluate(y_true, y_score, y_pred)
        print(f"  IF day test  : ROC={report['if_day_test'].get('roc_auc', 'n/a')}  positives={report['if_day_test'].get('n_insiders', 0)}")

        # User-level — all splits (max score per user)
        user_max  = idf.groupby("user")["iforest_score"].max().reset_index()
        ul = day_labels[["user"]].drop_duplicates().assign(is_insider=1)
        user_max = user_max.merge(ul, on="user", how="left")
        user_max["is_insider"] = user_max["is_insider"].fillna(0).astype(int)
        threshold = idf["iforest_score"].quantile(0.93)
        y_true  = user_max["is_insider"].values
        y_score = user_max["iforest_score"].values
        y_pred  = (y_score >= threshold).astype(int)
        report["if_user_all"] = evaluate(y_true, y_score, y_pred)
        print(f"  IF user all  : ROC={report['if_user_all'].get('roc_auc', 'n/a')}  positives={report['if_user_all'].get('n_insiders', 0)}")
    else:
        print(f"  [SKIP] IF scored file not found: {IFOREST_CSV}")

    # ── LSTM Autoencoder ──────────────────────────────────────────────────────
    if LSTM_CSV.exists():
        print(f"Loading LSTM scored: {LSTM_CSV}")
        ldf = pd.read_csv(LSTM_CSV, usecols=["user", "email_day", "lstm_score", "lstm_risk_severity", "dataset_split"])
        ldf["email_day"] = pd.to_datetime(ldf["email_day"], errors="coerce").dt.normalize()
        ldf = ldf[ldf["lstm_risk_severity"] != "undetermined"]

        # Day-level — test split only
        test_ldf = ldf[ldf["dataset_split"] == "test"].merge(day_labels, on=["user", "email_day"], how="left")
        test_ldf["is_insider"] = test_ldf["is_insider"].fillna(0).astype(int)
        y_true  = test_ldf["is_insider"].values
        y_score = test_ldf["lstm_score"].fillna(0).values
        y_pred  = test_ldf["lstm_risk_severity"].isin(["suspicious", "high"]).astype(int).values
        report["lstm_day_test"] = evaluate(y_true, y_score, y_pred)
        print(f"  LSTM day test: ROC={report['lstm_day_test'].get('roc_auc', 'n/a')}  positives={report['lstm_day_test'].get('n_insiders', 0)}")

        # User-level — all splits (max score per user)
        user_max = ldf.groupby("user")["lstm_score"].max().reset_index()
        ul = day_labels[["user"]].drop_duplicates().assign(is_insider=1)
        user_max = user_max.merge(ul, on="user", how="left")
        user_max["is_insider"] = user_max["is_insider"].fillna(0).astype(int)
        threshold = ldf["lstm_score"].quantile(0.93)
        y_true  = user_max["is_insider"].values
        y_score = user_max["lstm_score"].fillna(0).values
        y_pred  = (y_score >= threshold).astype(int)
        report["lstm_user_all"] = evaluate(y_true, y_score, y_pred)
        print(f"  LSTM user all: ROC={report['lstm_user_all'].get('roc_auc', 'n/a')}  positives={report['lstm_user_all'].get('n_insiders', 0)}")
    else:
        print(f"  [SKIP] LSTM scored file not found: {LSTM_CSV}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved evaluation report to: {REPORT_PATH}")

    # Quick summary
    print("\nSUMMARY")
    print(f"{'Metric':<18} {'IF Day':>10} {'IF User':>10} {'LSTM Day':>10} {'LSTM User':>10}")
    print("-" * 60)
    for k in ("roc_auc", "avg_precision", "precision", "recall", "f1"):
        vals = [report.get(s, {}).get(k, "—") for s in ("if_day_test", "if_user_all", "lstm_day_test", "lstm_user_all")]
        print(f"  {k:<16} " + "  ".join(f"{str(v):>10}" for v in vals))


if __name__ == "__main__":
    main()
