"""Train an app-ready Isolation Forest on cleaned CERT daily email features."""

from __future__ import annotations

import sys
from pathlib import Path
import json
import pickle

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLEANED_DIR, MODELS_DIR  # noqa: E402

INPUT_PATH   = CLEANED_DIR / "email_user_daily_with_psychometric.csv"
MODEL_PATH   = MODELS_DIR  / "isolation_forest_cert.pkl"
OUTPUT_PATH  = CLEANED_DIR / "email_user_daily_scored.csv"
METRICS_PATH = MODELS_DIR  / "isolation_forest_summary.json"

FEATURE_COLUMNS = [
    "email_count",
    "unique_pcs",
    "total_size",
    "avg_size",
    "total_attachments",
    "emails_with_attachments",
    "after_hours_emails",
    "avg_recipients",
    "max_recipients",
    "avg_content_words",
    "max_content_words",
    "bcc_email_count",
    "cc_email_count",
    "attachment_email_ratio",
    "after_hours_ratio",
    "bcc_ratio",
    # logon features
    "logon_count",
    "logoff_count",
    "after_hours_logons",
    "unique_logon_pcs",
    # device/USB features
    "usb_connect_count",
    "usb_disconnect_count",
    # file access features
    "file_total",
    "file_to_removable",
    "file_from_removable",
    "file_write_count",
    "file_after_hours",
    # psychometric scores
    "o",
    "c",
    "e",
    "a",
    "n",
]

def load_feature_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH)
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    df["email_day"] = pd.to_datetime(df["email_day"], errors="coerce")
    df = df.dropna(subset=["email_day"]).sort_values(["email_day", "user"]).reset_index(drop=True)
    if "dataset_split" not in df.columns:
        raise ValueError("dataset_split column missing — re-run clean_cert_email_data.py first.")
    return df


def split_from_column(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["dataset_split"] == "train"].copy()
    val_df   = df[df["dataset_split"] == "val"].copy()
    test_df  = df[df["dataset_split"] == "test"].copy()
    return train_df, val_df, test_df


def normalize_scores(raw_scores: pd.Series, scaler: MinMaxScaler) -> pd.Series:
    normalized = scaler.transform(raw_scores.to_numpy().reshape(-1, 1)).ravel()
    normalized = normalized.clip(0.0, 1.0)
    return pd.Series(normalized, index=raw_scores.index)


def assign_severity(score: float, suspicious_threshold: float, high_threshold: float) -> str:
    if score >= high_threshold:
        return "high"
    if score >= suspicious_threshold:
        return "suspicious"
    return "normal"


def train_isolation_forest(
    train_df: pd.DataFrame,
    full_df: pd.DataFrame,
    contamination: float = 0.03,
    random_state: int = 42,
) -> tuple[dict, pd.DataFrame]:
    train_features = train_df[FEATURE_COLUMNS].copy().fillna(0)
    full_features = full_df[FEATURE_COLUMNS].copy().fillna(0)

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(train_features)

    train_raw_scores = pd.Series(-model.score_samples(train_features), index=train_df.index)
    full_raw_scores = pd.Series(-model.score_samples(full_features), index=full_df.index)

    scaler = MinMaxScaler()
    scaler.fit(train_raw_scores.to_numpy().reshape(-1, 1))

    train_normalized = normalize_scores(train_raw_scores, scaler)
    full_normalized = normalize_scores(full_raw_scores, scaler)

    suspicious_threshold = float(train_normalized.quantile(0.95))
    high_threshold = float(train_normalized.quantile(0.99))

    scored_df = full_df.copy()
    scored_df["iforest_raw_score"] = full_raw_scores
    scored_df["iforest_score"] = full_normalized
    scored_df["iforest_flag"] = (full_normalized >= suspicious_threshold).astype(int)
    scored_df["risk_severity"] = scored_df["iforest_score"].apply(
        lambda value: assign_severity(float(value), suspicious_threshold, high_threshold)
    )

    artifacts = {
        "model": model,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "suspicious_threshold": suspicious_threshold,
        "high_threshold": high_threshold,

    }
    return artifacts, scored_df


def build_summary(
    scored_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    artifacts: dict,
) -> dict:
    top_anomalies = (
        scored_df.sort_values("iforest_score", ascending=False)
        .head(20)[["user", "email_day", "iforest_score", "risk_severity"]]
        .assign(email_day=lambda frame: frame["email_day"].astype("string"))
        .to_dict(orient="records")
    )
    return {
        "rows": int(len(scored_df)),
        "users": int(scored_df["user"].nunique()),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_start_day": str(train_df["email_day"].min().date()),
        "train_end_day": str(train_df["email_day"].max().date()),
        "val_start_day": str(val_df["email_day"].min().date()) if not val_df.empty else None,
        "val_end_day": str(val_df["email_day"].max().date()) if not val_df.empty else None,
        "test_start_day": str(test_df["email_day"].min().date()) if not test_df.empty else None,
        "test_end_day": str(test_df["email_day"].max().date()) if not test_df.empty else None,
        "suspicious_threshold": float(artifacts["suspicious_threshold"]),
        "high_threshold": float(artifacts["high_threshold"]),
        "suspicious_rows": int((scored_df["risk_severity"] == "suspicious").sum()),
        "high_rows": int((scored_df["risk_severity"] == "high").sum()),
        "score_min": float(scored_df["iforest_score"].min()),
        "score_max": float(scored_df["iforest_score"].max()),
        "top_feature_columns": FEATURE_COLUMNS,
        "top_anomalies": top_anomalies,
    }


def save_outputs(artifacts: dict, scored_df: pd.DataFrame, summary: dict) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as file_obj:
        pickle.dump(artifacts, file_obj)

    output_df = scored_df.copy()
    output_df["email_day"] = output_df["email_day"].dt.strftime("%Y-%m-%d")
    output_df.to_csv(OUTPUT_PATH, index=False)
    METRICS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    full_df = load_feature_data()
    train_df, val_df, test_df = split_from_column(full_df)
    artifacts, scored_df = train_isolation_forest(train_df, full_df)
    summary = build_summary(scored_df, train_df, val_df, test_df, artifacts)
    save_outputs(artifacts, scored_df, summary)

    print(f"Saved scored dataset to {OUTPUT_PATH}")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved summary to {METRICS_PATH}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(
        scored_df[["user", "email_day", "dataset_split", "iforest_score", "risk_severity"]].head().to_string(index=False)
    )


if __name__ == "__main__":
    main()

