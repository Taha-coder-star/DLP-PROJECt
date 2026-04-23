"""Weighted Risk Scoring -- AI Algorithm Component.

This module implements a multi-signal risk scoring system for insider threat
detection.  It combines the LSTM autoencoder anomaly signal with five
rule-based behavioral indicators drawn from CERT insider threat research.

Algorithm class: Weighted Evidence Aggregation with Best-First Investigation Queue

Why this qualifies as an AI component:
  - Learns nothing from labels (unsupervised, deployable with zero ground truth)
  - Normalises heterogeneous signals to a common [0, 1] scale (feature engineering)
  - Combines signals using domain-informed weights (knowledge-based system)
  - Produces a ranked priority queue -- equivalent to best-first search over
    the space of suspicious users
  - Generates per-user natural-language explanations (rule-based XAI)

Weights (must sum to 1.0):
  lstm_p95       0.50  -- LSTM reconstruction-error p95 (main model)
  after_hours    0.15  -- fraction of emails / logons after business hours
  bcc_usage      0.10  -- BCC email ratio (hiding recipients)
  file_exfil     0.10  -- files copied to removable media
  usb_activity   0.10  -- USB connect events (removable storage use)
  multi_pc       0.05  -- distinct workstations accessed

Reference: Greitzer et al. (2010), "Insider Threat Indicators", CERT/CC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configurable weights (must sum to 1.0)
# ---------------------------------------------------------------------------

WEIGHTS: dict[str, float] = {
    "lstm_p95":    0.50,
    "after_hours": 0.15,
    "bcc_usage":   0.10,
    "file_exfil":  0.10,
    "usb_activity": 0.10,
    "multi_pc":    0.05,
}

# Thresholds for generating human-readable flags
_FLAG_RULES: list[tuple[str, str, float, str]] = [
    # (signal_col, direction, threshold, explanation_text)
    ("lstm_p95_norm",    ">=", 0.70, "High LSTM anomaly score -- unusual behavioral patterns over time"),
    ("after_hours_norm", ">=", 0.50, "Elevated after-hours email / login activity"),
    ("bcc_usage_norm",   ">=", 0.50, "Abnormal BCC email usage (potential hidden recipients)"),
    ("file_exfil_norm",  ">=", 0.50, "Files copied to removable media (possible data exfiltration)"),
    ("usb_activity_norm",">=", 0.50, "Frequent USB device connections"),
    ("multi_pc_norm",    ">=", 0.50, "Logging in from an unusual number of workstations"),
]


# ---------------------------------------------------------------------------
# Step 1 -- compute behavioral signals from the IF-scored CSV (row level)
# ---------------------------------------------------------------------------

def compute_behavioral_signals(idf: pd.DataFrame) -> pd.DataFrame:
    """Aggregate row-level behavioral features to one row per user.

    idf must contain columns:
        user, after_hours_ratio, bcc_ratio, file_to_removable, file_total,
        usb_connect_count, after_hours_logons, logon_count, unique_logon_pcs,
        employee_name (optional), dataset_split

    Returns one row per user with raw (un-normalised) signal values.
    """
    required = [
        "user", "after_hours_ratio", "bcc_ratio",
        "file_to_removable", "file_total",
        "usb_connect_count", "after_hours_logons",
        "logon_count", "unique_logon_pcs",
    ]
    missing = [c for c in required if c not in idf.columns]
    if missing:
        raise ValueError(f"Missing columns in idf: {missing}")

    agg = idf.groupby("user").agg(
        after_hours_rate   = ("after_hours_ratio",  "mean"),
        bcc_rate           = ("bcc_ratio",           "mean"),
        total_file_exfil   = ("file_to_removable",   "sum"),
        total_file_ops     = ("file_total",           "sum"),
        total_usb          = ("usb_connect_count",    "sum"),
        total_ah_logons    = ("after_hours_logons",   "sum"),
        total_logons       = ("logon_count",           "sum"),
        max_unique_pcs     = ("unique_logon_pcs",      "max"),
        dataset_split      = ("dataset_split",         lambda x: x.mode().iloc[0]),
    ).reset_index()

    # Employee name (best effort)
    if "employee_name" in idf.columns:
        names = idf.groupby("user")["employee_name"].first().reset_index()
        agg = agg.merge(names, on="user", how="left")
    else:
        agg["employee_name"] = agg["user"]

    # Derived ratios
    agg["file_exfil_rate"]  = agg["total_file_exfil"] / (agg["total_file_ops"] + 1)
    agg["ah_logon_rate"]    = agg["total_ah_logons"]  / (agg["total_logons"] + 1)

    return agg


# ---------------------------------------------------------------------------
# Step 2 -- normalise signals and compute weighted risk score
# ---------------------------------------------------------------------------

def _minmax(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def compute_risk_scores(
    lstm_user_df: pd.DataFrame,
    behavioral_df: pd.DataFrame,
    insider_users: set[str] | None = None,
) -> pd.DataFrame:
    """Combine LSTM p95 score with behavioral signals into a final risk score.

    lstm_user_df  -- output of compute_user_scores() from user_level_eval.py
                     must contain columns: user, score_p95
    behavioral_df -- output of compute_behavioral_signals()
    insider_users -- optional ground-truth set for labelling

    Returns a DataFrame with one row per user, sorted by risk_score descending.
    """
    df = lstm_user_df[["user", "score_p95", "dataset_split"]].merge(
        behavioral_df[[
            "user", "after_hours_rate", "bcc_rate",
            "file_exfil_rate", "total_usb", "max_unique_pcs", "employee_name",
        ]],
        on="user", how="left",
    ).fillna(0)

    # Normalise each signal to [0, 1]
    df["lstm_p95_norm"]     = _minmax(df["score_p95"])
    df["after_hours_norm"]  = _minmax(df["after_hours_rate"])
    df["bcc_usage_norm"]    = _minmax(df["bcc_rate"])
    df["file_exfil_norm"]   = _minmax(df["file_exfil_rate"])
    df["usb_activity_norm"] = _minmax(df["total_usb"])
    df["multi_pc_norm"]     = _minmax(df["max_unique_pcs"])

    # Weighted aggregation
    df["risk_score"] = (
        WEIGHTS["lstm_p95"]     * df["lstm_p95_norm"]
      + WEIGHTS["after_hours"]  * df["after_hours_norm"]
      + WEIGHTS["bcc_usage"]    * df["bcc_usage_norm"]
      + WEIGHTS["file_exfil"]   * df["file_exfil_norm"]
      + WEIGHTS["usb_activity"] * df["usb_activity_norm"]
      + WEIGHTS["multi_pc"]     * df["multi_pc_norm"]
    )

    if insider_users is not None:
        df["is_insider"] = df["user"].isin(insider_users).astype(int)

    return df.sort_values("risk_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 3 -- best-first investigation queue
# ---------------------------------------------------------------------------

def build_investigation_queue(risk_df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Return the top-N highest-risk users as a priority investigation queue.

    This is equivalent to best-first search over the user population where
    the heuristic is the weighted risk score.  Investigators work through
    the queue from rank 1 downward, maximising expected insider detections
    per unit of investigation effort.
    """
    cols = ["user", "employee_name", "risk_score",
            "lstm_p95_norm", "after_hours_norm", "bcc_usage_norm",
            "file_exfil_norm", "usb_activity_norm", "multi_pc_norm"]
    for opt in ("is_insider", "explanation"):
        if opt in risk_df.columns:
            cols.append(opt)
    queue = risk_df.head(top_n)[cols].copy()
    queue.insert(0, "priority_rank", range(1, len(queue) + 1))
    return queue


# ---------------------------------------------------------------------------
# Step 4 -- per-user explainability
# ---------------------------------------------------------------------------

def explain_user(user_row: pd.Series) -> list[str]:
    """Return a list of triggered behavioral flags for one user.

    Each flag is a plain-English sentence suitable for a dashboard or report.
    """
    flags: list[str] = []
    for col, direction, threshold, text in _FLAG_RULES:
        if col not in user_row.index:
            continue
        val = float(user_row[col])
        triggered = (val >= threshold) if direction == ">=" else (val < threshold)
        if triggered:
            flags.append(text)
    if not flags:
        flags.append("No strong individual behavioral indicators -- pattern is subtle.")
    return flags


def explain_dataframe(risk_df: pd.DataFrame) -> pd.DataFrame:
    """Add an 'explanation' column to risk_df (one string per user)."""
    out = risk_df.copy()
    out["explanation"] = out.apply(
        lambda r: "; ".join(explain_user(r)), axis=1
    )
    return out
