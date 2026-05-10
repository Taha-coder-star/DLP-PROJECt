"""
Global LSTM autoencoder for insider threat anomaly detection on CERT daily email features.

Single model trained on all users' training windows combined — much faster than
per-user training, fills the GPU properly, and learns cross-user behavioral norms.
Reconstruction error on a 7-day sliding window is the anomaly score.

Per-user z-score normalization is applied before the global MinMaxScaler so that
each user's features are measured against their own baseline rather than the global
population. This significantly improves recall for gradual and short-window insiders.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLEANED_DIR, MODELS_DIR  # noqa: E402

INPUT_PATH   = CLEANED_DIR / "email_user_daily_with_psychometric.csv"
MODEL_PATH   = MODELS_DIR  / "lstm_autoencoder_cert.pkl"
OUTPUT_PATH  = CLEANED_DIR / "email_user_daily_lstm_scored.csv"
METRICS_PATH = MODELS_DIR  / "lstm_autoencoder_summary.json"

BEHAVIORAL_FEATURES = [
    "email_count", "unique_pcs", "total_size", "avg_size",
    "total_attachments", "emails_with_attachments", "after_hours_emails",
    "avg_recipients", "max_recipients", "avg_content_words", "max_content_words",
    "bcc_email_count", "cc_email_count", "attachment_email_ratio",
    "after_hours_ratio", "bcc_ratio",
    "logon_count", "logoff_count", "after_hours_logons", "unique_logon_pcs",
    "usb_connect_count", "usb_disconnect_count",
    "file_total", "file_to_removable", "file_from_removable",
    "file_write_count", "file_after_hours",
    # Idea 9: communication graph features
    "n_external_recipients", "bcc_external_count", "external_email_ratio",
    # Idea 10: cyclical time encoding
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    # Idea 8: velocity/delta — deviation from 30-day rolling baseline
    "file_to_removable_delta", "usb_connect_count_delta",
    "bcc_ratio_delta", "after_hours_ratio_delta",
]

WINDOW_SIZE    = 7
HIDDEN_DIM     = 64
LATENT_DIM     = 32
EPOCHS         = 20
BATCH_SIZE     = 256
LEARNING_RATE  = 1e-3
MIN_TRAIN_DAYS = 3     # users below this fall back to global z-score stats
ZSCORE_CLIP    = 5.0   # clip z-scores to [-5, 5] to suppress extreme outliers

# Type alias for per-user normalization stats
UserStats = dict[str, tuple[np.ndarray, np.ndarray]]  # user -> (mean, std)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder_lstm = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.enc_fc       = nn.Linear(hidden_dim, latent_dim)
        self.decoder_lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.out_fc       = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        _, (h_n, _) = self.encoder_lstm(x)
        latent  = self.enc_fc(h_n.squeeze(0))                    # (batch, latent_dim)
        dec_in  = latent.unsqueeze(1).repeat(1, x.size(1), 1)    # (batch, seq_len, latent_dim)
        dec_out, _ = self.decoder_lstm(dec_in)
        return self.out_fc(dec_out)                               # (batch, seq_len, n_features)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def make_windows(arr: np.ndarray, window_size: int) -> np.ndarray:
    """Slide a fixed-length window over a 2-D array (n_days, n_features)."""
    n = len(arr)
    if n < window_size:
        return np.empty((0, window_size, arr.shape[1]), dtype=np.float32)
    return np.stack(
        [arr[i : i + window_size] for i in range(n - window_size + 1)]
    ).astype(np.float32)


def compute_user_stats(
    df: pd.DataFrame,
) -> tuple[UserStats, np.ndarray, np.ndarray]:
    """
    Compute per-user mean and std from training rows only (no leakage).

    Users with fewer than MIN_TRAIN_DAYS training days fall back to global stats.
    Returns (user_stats_dict, global_mean, global_std).
    """
    train_df    = df[df["dataset_split"] == "train"]
    train_vals  = train_df[BEHAVIORAL_FEATURES].fillna(0).values
    global_mean = train_vals.mean(axis=0)
    global_std  = train_vals.std(axis=0) + 1e-8

    stats: UserStats = {}
    for user, grp in train_df.groupby("user"):
        vals = grp[BEHAVIORAL_FEATURES].fillna(0).values
        if len(vals) >= MIN_TRAIN_DAYS:
            stats[user] = (vals.mean(axis=0), vals.std(axis=0) + 1e-8)
        else:
            stats[user] = (global_mean, global_std)

    return stats, global_mean, global_std


def zscore_user(
    vals: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Z-score normalize and clip to ZSCORE_CLIP standard deviations."""
    return np.clip((vals - mean) / std, -ZSCORE_CLIP, ZSCORE_CLIP).astype(np.float32)


def normalize(
    vals: np.ndarray,
    user: str,
    user_stats: UserStats,
    global_stats: tuple[np.ndarray, np.ndarray],
    scaler: MinMaxScaler,
) -> np.ndarray:
    """Apply per-user z-score then global MinMaxScaler."""
    mean, std = user_stats.get(user, global_stats)
    return scaler.transform(zscore_user(vals, mean, std)).astype(np.float32)


def batch_reconstruction_errors(
    model: LSTMAutoencoder,
    windows: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Compute per-window MSE reconstruction error without storing gradients."""
    model.eval()
    errors = []
    with torch.no_grad():
        for start in range(0, len(windows), BATCH_SIZE):
            batch = torch.tensor(
                windows[start : start + BATCH_SIZE], dtype=torch.float32, device=device
            )
            recon = model(batch)
            mse   = ((batch - recon) ** 2).mean(dim=(1, 2)).cpu().numpy()
            errors.append(mse)
    return np.concatenate(errors) if errors else np.array([])


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------

def load_feature_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH)
    missing = [c for c in BEHAVIORAL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if "dataset_split" not in df.columns:
        raise ValueError("dataset_split column missing — re-run clean_cert_email_data.py first.")
    df["email_day"] = pd.to_datetime(df["email_day"], errors="coerce")
    df = df.dropna(subset=["email_day"]).sort_values(["user", "email_day"]).reset_index(drop=True)
    return df


def build_global_training_windows(
    df: pd.DataFrame,
    scaler: MinMaxScaler,
    user_stats: UserStats,
    global_stats: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """
    Build all 7-day windows from every user's train-split rows.
    Features are per-user z-scored then globally scaled before windowing.
    """
    all_windows = []
    train_df = df[df["dataset_split"] == "train"]

    for user, user_rows in train_df.groupby("user", sort=False):
        user_rows = user_rows.sort_values("email_day")
        raw    = user_rows[BEHAVIORAL_FEATURES].fillna(0).values
        scaled = normalize(raw, user, user_stats, global_stats, scaler)
        windows = make_windows(scaled, WINDOW_SIZE)
        if len(windows) > 0:
            all_windows.append(windows)

    if not all_windows:
        raise ValueError("No training windows found — check dataset_split labels.")

    combined = np.concatenate(all_windows, axis=0)
    print(f"  Total training windows: {len(combined):,} from {len(all_windows)} users")
    return combined


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_global_model(
    train_windows: np.ndarray,
    device: torch.device,
) -> LSTMAutoencoder:
    """Train a single LSTM autoencoder on all users' training windows."""
    model     = LSTMAutoencoder(len(BEHAVIORAL_FEATURES), HIDDEN_DIM, LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn   = nn.MSELoss()

    dataset = TensorDataset(torch.tensor(train_windows))
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            recon = model(batch)
            loss  = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 5 == 0 or epoch == EPOCHS:
            avg = epoch_loss / len(loader)
            print(f"  Epoch {epoch:02d}/{EPOCHS} — avg loss: {avg:.6f}")

    return model


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_all_users(
    df: pd.DataFrame,
    model: LSTMAutoencoder,
    scaler: MinMaxScaler,
    suspicious_threshold: float,
    high_threshold: float,
    device: torch.device,
    user_stats: UserStats,
    global_stats: tuple[np.ndarray, np.ndarray],
) -> pd.DataFrame:
    """
    Score every user-day using the global model with per-user z-score normalization.
    Each user's features are measured against their own training-period baseline,
    so gradual drift and burst anomalies stand out relative to personal norms.
    """
    scored_chunks = []

    for user, user_rows in df.groupby("user", sort=False):
        user_rows = user_rows.sort_values("email_day").reset_index(drop=True)
        n_days    = len(user_rows)

        raw     = user_rows[BEHAVIORAL_FEATURES].fillna(0).values
        scaled  = normalize(raw, user, user_stats, global_stats, scaler)
        windows = make_windows(scaled, WINDOW_SIZE)

        day_errors = np.full(n_days, np.nan)
        if len(windows) > 0:
            errors = batch_reconstruction_errors(model, windows, device)
            for i, err in enumerate(errors):
                day_errors[i + WINDOW_SIZE - 1] = float(err)

        day_scores = np.clip(
            (day_errors - suspicious_threshold * 0.5) /
            (high_threshold - suspicious_threshold * 0.5 + 1e-9),
            0.0, 1.0,
        )

        user_rows["lstm_raw_error"]     = day_errors
        user_rows["lstm_score"]         = day_scores
        user_rows["lstm_flag"]          = (
            (~np.isnan(day_scores)) & (day_scores >= 0.5)
        ).astype(int)
        user_rows["lstm_risk_severity"] = [
            _severity(s, suspicious_threshold, high_threshold, day_errors[i])
            for i, s in enumerate(day_scores)
        ]
        scored_chunks.append(user_rows)

    return pd.concat(scored_chunks, ignore_index=True)


def _severity(score: float, susp_raw: float, high_raw: float, raw_err: float) -> str:
    if np.isnan(raw_err):
        return "undetermined"
    if raw_err >= high_raw:
        return "high"
    if raw_err >= susp_raw:
        return "suspicious"
    return "normal"


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def build_summary(scored_df: pd.DataFrame) -> dict:
    valid = scored_df.dropna(subset=["lstm_score"])
    top   = (
        valid.sort_values("lstm_score", ascending=False)
        .head(20)[["user", "email_day", "dataset_split", "lstm_score", "lstm_risk_severity"]]
        .assign(email_day=lambda f: f["email_day"].astype(str))
        .to_dict(orient="records")
    )
    return {
        "model_type": "global_per_user_zscore",
        "rows": int(len(scored_df)),
        "users": int(scored_df["user"].nunique()),
        "window_size": WINDOW_SIZE,
        "hidden_dim": HIDDEN_DIM,
        "latent_dim": LATENT_DIM,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "min_train_days": MIN_TRAIN_DAYS,
        "zscore_clip": ZSCORE_CLIP,
        "suspicious_rows":   int((scored_df["lstm_risk_severity"] == "suspicious").sum()),
        "high_rows":         int((scored_df["lstm_risk_severity"] == "high").sum()),
        "undetermined_rows": int((scored_df["lstm_risk_severity"] == "undetermined").sum()),
        "top_anomalies": top,
    }


def save_outputs(
    model: LSTMAutoencoder,
    scaler: MinMaxScaler,
    suspicious_threshold: float,
    high_threshold: float,
    scored_df: pd.DataFrame,
    summary: dict,
    user_stats: UserStats,
    global_stats: tuple[np.ndarray, np.ndarray],
) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "model_state":        {k: v.cpu() for k, v in model.state_dict().items()},
        "scaler":             scaler,
        "feature_columns":    BEHAVIORAL_FEATURES,
        "window_size":        WINDOW_SIZE,
        "hidden_dim":         HIDDEN_DIM,
        "latent_dim":         LATENT_DIM,
        "suspicious_threshold": suspicious_threshold,
        "high_threshold":     high_threshold,
        "model_type":         "global_per_user_zscore",
        "user_stats":         user_stats,
        "global_stats":       global_stats,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifacts, f)

    out = scored_df.copy()
    out["email_day"] = out["email_day"].dt.strftime("%Y-%m-%d")
    out.to_csv(OUTPUT_PATH, index=False)
    METRICS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading feature data...")
    df = load_feature_data()
    print(f"Loaded {len(df):,} rows, {df['user'].nunique()} users")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {(df['dataset_split'] == split).sum():,} rows")

    # Compute per-user z-score stats from training data only (no leakage)
    print("\nComputing per-user z-score statistics from training data...")
    user_stats, global_mean, global_std = compute_user_stats(df)
    global_stats = (global_mean, global_std)
    n_fallback = sum(
        1 for user in df["user"].unique()
        if user not in user_stats
    )
    print(f"  Per-user stats: {len(user_stats)} users")
    print(f"  Global fallback applied to: {n_fallback} users with no training rows")

    # Fit global scaler on z-scored training data only (no leakage)
    print("\nFitting global MinMaxScaler on z-scored training data...")
    train_df = df[df["dataset_split"] == "train"]
    zscored_train_blocks = []
    for user, grp in train_df.groupby("user"):
        raw  = grp[BEHAVIORAL_FEATURES].fillna(0).values
        mean, std = user_stats.get(user, global_stats)
        zscored_train_blocks.append(zscore_user(raw, mean, std))
    train_zscored = np.vstack(zscored_train_blocks)
    scaler = MinMaxScaler()
    scaler.fit(train_zscored)
    print(f"  Scaler fitted on {len(train_zscored):,} z-scored training rows")

    # Build all training windows
    print("\nBuilding training windows...")
    train_windows = build_global_training_windows(df, scaler, user_stats, global_stats)

    # Train single global model
    print("\nTraining global LSTM autoencoder...")
    model = train_global_model(train_windows, device)

    # Compute global thresholds from training reconstruction errors
    print("\nComputing global anomaly thresholds...")
    train_errors         = batch_reconstruction_errors(model, train_windows, device)
    suspicious_threshold = float(np.percentile(train_errors, 95))
    high_threshold       = float(np.percentile(train_errors, 99))
    print(f"  Suspicious threshold (p95): {suspicious_threshold:.6f}")
    print(f"  High threshold       (p99): {high_threshold:.6f}")

    # Score every user-day
    print("\nScoring all users...")
    scored_df = score_all_users(
        df, model, scaler, suspicious_threshold, high_threshold, device,
        user_stats, global_stats,
    )

    summary = build_summary(scored_df)
    save_outputs(
        model, scaler, suspicious_threshold, high_threshold,
        scored_df, summary, user_stats, global_stats,
    )

    print(f"\nSaved scored dataset  -> {OUTPUT_PATH}")
    print(f"Saved model artifacts -> {MODEL_PATH}")
    print(f"Saved summary         -> {METRICS_PATH}")
    print(f"High-risk rows: {summary['high_rows']} | Suspicious: {summary['suspicious_rows']}")

    preview = (
        scored_df.dropna(subset=["lstm_score"])
        .sort_values("lstm_score", ascending=False)
        .head(5)[["user", "email_day", "dataset_split", "lstm_score", "lstm_risk_severity"]]
    )
    print("\nTop 5 anomalies:")
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
