"""Reusable inference utilities for the global CERT LSTM autoencoder model."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODELS_DIR  # noqa: E402
from train_lstm_autoencoder_cert import (
    BEHAVIORAL_FEATURES,
    WINDOW_SIZE,
    HIDDEN_DIM,
    LATENT_DIM,
    LSTMAutoencoder,
    make_windows,
    batch_reconstruction_errors,
    _severity,
)

MODEL_PATH = MODELS_DIR / "lstm_autoencoder_cert.pkl"


def load_artifacts(model_path: str | Path = MODEL_PATH) -> dict:
    with open(model_path, "rb") as f:
        return pickle.load(f)


def _rebuild_model(artifacts: dict, device: torch.device) -> LSTMAutoencoder:
    model = LSTMAutoencoder(len(BEHAVIORAL_FEATURES), HIDDEN_DIM, LATENT_DIM).to(device)
    model.load_state_dict(artifacts["model_state"])
    model.eval()
    return model


def score_user_days(
    user_df: pd.DataFrame,
    artifacts: dict,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Score a sorted DataFrame of daily rows for one user using the global model."""
    device = device or torch.device("cpu")
    result = user_df.copy().reset_index(drop=True)
    n_days = len(result)

    scaler               = artifacts["scaler"]
    suspicious_threshold = artifacts["suspicious_threshold"]
    high_threshold       = artifacts["high_threshold"]

    scaled  = scaler.transform(result[BEHAVIORAL_FEATURES].fillna(0).values).astype(np.float32)
    windows = make_windows(scaled, WINDOW_SIZE)

    model      = _rebuild_model(artifacts, device)
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

    result["lstm_raw_error"]     = day_errors
    result["lstm_score"]         = day_scores
    result["lstm_flag"]          = ((~np.isnan(day_scores)) & (day_scores >= 0.5)).astype(int)
    result["lstm_risk_severity"] = [
        _severity(s, suspicious_threshold, high_threshold, day_errors[i])
        for i, s in enumerate(day_scores)
    ]
    return result


def score_dataframe(
    df: pd.DataFrame,
    artifacts: dict | None = None,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Score a full multi-user DataFrame with the global model."""
    artifacts = artifacts or load_artifacts()
    device    = device or torch.device("cpu")

    scored_chunks = []
    for _, user_df in df.groupby("user", sort=False):
        user_df = user_df.sort_values("email_day").reset_index(drop=True)
        scored_chunks.append(score_user_days(user_df, artifacts, device))

    return pd.concat(scored_chunks, ignore_index=True)


def score_single_user_sequence(
    feature_rows: list[dict],
    artifacts: dict | None = None,
) -> list[dict]:
    """Score a chronologically-ordered list of daily feature dicts for one user."""
    artifacts = artifacts or load_artifacts()
    df        = pd.DataFrame(feature_rows)
    return score_user_days(df, artifacts).to_dict(orient="records")
