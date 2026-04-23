"""UEBA Dashboard -- Insider Threat Detection System.

Standalone Streamlit app.  Run with:
    streamlit run app/ueba_dashboard.py

Integrates:
  - Fixed user-level evaluation pipeline  (colab/user_level_eval.py)
  - Weighted risk scoring AI component    (colab/risk_scorer.py)
  - Presentation-ready charts
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
for _p in [str(_ROOT), str(_ROOT / "colab")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import CLEANED_DIR, MODELS_DIR  # noqa: E402
from user_level_eval import (  # noqa: E402
    compute_user_scores, apply_user_threshold, evaluate_topk_users,
)
from risk_scorer import (  # noqa: E402
    compute_behavioral_signals, compute_risk_scores,
    build_investigation_queue, explain_user, explain_dataframe, WEIGHTS,
)

INSIDERS_CSV = _ROOT / "archive" / "answers" / "answers" / "insiders.csv"
IFOREST_CSV  = CLEANED_DIR / "email_user_daily_scored.csv"
LSTM_CSV     = CLEANED_DIR / "email_user_daily_lstm_scored.csv"

INSIDER_COLOR = "#E85454"
NORMAL_COLOR  = "#4C9BE8"

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UEBA — Insider Threat Detection",
    page_icon="🔒",
    layout="wide",
)


# ── cached loaders ────────────────────────────────────────────────────────────

@st.cache_data
def load_insider_users() -> set[str]:
    if not INSIDERS_CSV.exists():
        return set()
    ins = pd.read_csv(INSIDERS_CSV)
    return set(ins.loc[ins["dataset"] == 4.2, "user"].unique())


@st.cache_data
def load_lstm_user_df() -> pd.DataFrame:
    ldf = pd.read_csv(LSTM_CSV,
                      usecols=["user", "lstm_score", "lstm_risk_severity", "dataset_split"])
    ldf = ldf[ldf["lstm_risk_severity"] != "undetermined"]
    iu  = load_insider_users()
    return compute_user_scores(ldf, "lstm_score", iu)


@st.cache_data
def load_if_user_df() -> pd.DataFrame:
    idf = pd.read_csv(IFOREST_CSV,
                      usecols=["user", "iforest_score", "dataset_split"])
    iu  = load_insider_users()
    return compute_user_scores(idf, "iforest_score", iu)


@st.cache_data
def load_behavioral_df() -> pd.DataFrame:
    cols = [
        "user", "email_day", "after_hours_ratio", "bcc_ratio",
        "file_to_removable", "file_total", "usb_connect_count",
        "after_hours_logons", "logon_count", "unique_logon_pcs",
        "employee_name", "dataset_split",
    ]
    idf = pd.read_csv(IFOREST_CSV, usecols=cols)
    return compute_behavioral_signals(idf)


@st.cache_data
def load_risk_df() -> pd.DataFrame:
    iu    = load_insider_users()
    lu    = load_lstm_user_df()
    behav = load_behavioral_df()
    return compute_risk_scores(lu, behav, iu)


# ── chart helpers (return fig, do NOT call st.pyplot inside) ──────────────────

def fig_prf1_vs_k(user_df, agg, thresh_pct, insider_users, k_values):
    rows = []
    for k in k_values:
        filtered, _ = apply_user_threshold(user_df, agg, thresh_pct)
        m = evaluate_topk_users(filtered, insider_users, k)
        rows.append({"K": k, "Precision": m["precision"],
                     "Recall": m["recall"], "F1": m["f1"]})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(df["K"], df["Precision"], "o-", color="#E85454", linewidth=2, label="Precision")
    ax.plot(df["K"], df["Recall"],    "s-", color=NORMAL_COLOR, linewidth=2, label="Recall")
    ax.plot(df["K"], df["F1"],        "^-", color="#5CB85C", linewidth=2.5, label="F1")
    ax.set_xlabel("K (top users flagged)")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs K")
    ax.set_ylim(0, 1)
    ax.set_xticks(k_values)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_score_distribution(user_df, insider_users):
    insiders = user_df[user_df["is_insider"] == 1]["score_p95"].dropna()
    normals  = user_df[user_df["is_insider"] == 0]["score_p95"].dropna()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(normals,  bins=40, alpha=0.6, color=NORMAL_COLOR,
            label=f"Normal (n={len(normals):,})", density=True)
    ax.hist(insiders, bins=40, alpha=0.75, color=INSIDER_COLOR,
            label=f"Insider (n={len(insiders):,})", density=True)
    ax.axvline(normals.mean(),  color=NORMAL_COLOR,  linestyle="--", linewidth=1.5,
               label=f"Normal mean = {normals.mean():.3f}")
    ax.axvline(insiders.mean(), color=INSIDER_COLOR, linestyle="--", linewidth=1.5,
               label=f"Insider mean = {insiders.mean():.3f}")
    ax.set_xlabel("LSTM score_p95")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution: Insiders vs Normal Users")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_top_users_bar(risk_df, insider_users, top_n=20):
    top = risk_df.head(top_n).copy()
    top["label"] = top["user"].apply(
        lambda u: f"★ {u}" if u in insider_users else u
    )
    colors = [INSIDER_COLOR if u in insider_users else NORMAL_COLOR
              for u in top["user"]]
    fig, ax = plt.subplots(figsize=(7, max(4, top_n * 0.28)))
    ax.barh(top["label"][::-1], top["risk_score"][::-1],
            color=colors[::-1], alpha=0.85)
    ax.set_xlabel("Weighted Risk Score")
    ax.set_title(f"Top {top_n} Suspicious Users")
    ins_patch = mpatches.Patch(color=INSIDER_COLOR, label="Confirmed insider (★)")
    nor_patch  = mpatches.Patch(color=NORMAL_COLOR,  label="Normal user")
    ax.legend(handles=[ins_patch, nor_patch], fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_tp_fp_fn(user_df, agg, thresh_pct, insider_users):
    k_vals = [10, 20, 50, 100]
    tps, fps, fns = [], [], []
    for k in k_vals:
        filtered, _ = apply_user_threshold(user_df, agg, thresh_pct)
        m = evaluate_topk_users(filtered, insider_users, k)
        tps.append(m["tp"]); fps.append(m["fp"]); fns.append(m["fn"])
    x = np.arange(len(k_vals))
    w = 0.25
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(x - w, tps, w, label="TP (caught)",     color="#5CB85C", alpha=0.85)
    ax.bar(x,     fps, w, label="FP (false alarm)", color=INSIDER_COLOR, alpha=0.85)
    ax.bar(x + w, fns, w, label="FN (missed)",      color="#AAAAAA", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in k_vals])
    ax.set_ylabel("Count")
    ax.set_title("Detection Outcomes (TP / FP / FN)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_weights_pie():
    fig, ax = plt.subplots(figsize=(4, 4))
    labels = [k.replace("_", " ").title() for k in WEIGHTS]
    sizes  = list(WEIGHTS.values())
    colors = ["#5CB85C", "#4C9BE8", "#F4A83A", "#E85454", "#9B59B6", "#95A5A6"]
    ax.pie(sizes, labels=labels, autopct="%1.0f%%",
           colors=colors, startangle=140, textprops={"fontsize": 9})
    ax.set_title("Risk Score Weight Distribution", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Dashboard layout
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:

    # ── Header ───────────────────────────────────────────────────────────────
    st.title("🔒 UEBA — Insider Threat Detection System")
    st.markdown(
        "**User and Entity Behaviour Analytics** pipeline built on the CERT r4.2 "
        "insider threat dataset.  Detects suspicious users using an LSTM Autoencoder "
        "combined with a weighted behavioural risk scoring algorithm."
    )
    st.markdown("---")

    # ── Sidebar controls ─────────────────────────────────────────────────────
    st.sidebar.header("Analysis Settings")

    model_choice = st.sidebar.selectbox(
        "Model", ["LSTM Autoencoder", "Isolation Forest"]
    )
    agg_choice = st.sidebar.selectbox(
        "Score Aggregation",
        ["score_p95", "score_mean", "score_max"],
        help="How to collapse a user's daily scores into one value.\n"
             "score_p95 gives best results for LSTM.",
    )
    thresh_pct = st.sidebar.select_slider(
        "Threshold Percentile",
        options=[80, 85, 90, 95, 97, 99],
        value=90,
        help="Pre-filter: keep only users whose score exceeds this percentile "
             "of train-user scores.",
    )
    k_choice = st.sidebar.selectbox(
        "Top-K (users to flag)",
        [10, 20, 30, 50, 75, 100],
        index=3,
    )
    show_ground_truth = st.sidebar.checkbox(
        "Show ground-truth labels", value=INSIDERS_CSV.exists()
    )
    run_btn = st.sidebar.button("▶  Run Analysis", type="primary")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Weight breakdown:**")
    for k, v in WEIGHTS.items():
        st.sidebar.progress(v, text=f"{k.replace('_', ' ').title()}: {v:.0%}")

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading scored data…"):
        insider_users = load_insider_users() if show_ground_truth else set()
        lstm_user_df  = load_lstm_user_df()
        if_user_df    = load_if_user_df()
        risk_df       = load_risk_df()

    user_df   = lstm_user_df if model_choice == "LSTM Autoencoder" else if_user_df
    score_col = agg_choice

    if not run_btn and "ran" not in st.session_state:
        st.info("Configure settings in the sidebar and click **Run Analysis**.")
        st.stop()

    st.session_state["ran"] = True

    # ── Compute results ───────────────────────────────────────────────────────
    filtered_df, cutoff = apply_user_threshold(user_df, score_col, thresh_pct)
    metrics = evaluate_topk_users(filtered_df, insider_users, k_choice)

    # ── Section 1: Metrics ────────────────────────────────────────────────────
    st.header("📊 Detection Metrics")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Precision",  f"{metrics['precision']:.3f}")
    c2.metric("Recall",     f"{metrics['recall']:.3f}")
    c3.metric("F1",         f"{metrics['f1']:.3f}")
    c4.metric("TP",         metrics["tp"])
    c5.metric("FP",         metrics["fp"])
    c6.metric("FN",         metrics["fn"])

    pool = len(filtered_df)
    st.caption(
        f"Threshold (train {thresh_pct}th pct of user {score_col}) = **{cutoff:.4f}**  |  "
        f"Pool after filter: **{pool}** users  |  "
        f"Flagging top **{k_choice}** as suspicious"
    )
    st.markdown("---")

    # ── Section 2: Top suspicious users table ────────────────────────────────
    st.header("🚨 Top Suspicious Users")
    top_k_users = set(filtered_df.head(k_choice)["user"])
    display_df  = risk_df[risk_df["user"].isin(top_k_users)].copy()
    display_df["rank"] = display_df["user"].apply(
        lambda u: list(filtered_df.head(k_choice)["user"]).index(u) + 1
        if u in list(filtered_df.head(k_choice)["user"]) else "-"
    )

    show_cols = ["rank", "user", "employee_name", "risk_score",
                 "lstm_p95_norm", "after_hours_norm", "bcc_usage_norm",
                 "file_exfil_norm", "usb_activity_norm"]
    if show_ground_truth and "is_insider" in display_df.columns:
        show_cols.append("is_insider")
    display_df = display_df.sort_values("risk_score", ascending=False)

    col_rename = {
        "rank": "Rank", "user": "User ID", "employee_name": "Name",
        "risk_score": "Risk Score", "lstm_p95_norm": "LSTM P95",
        "after_hours_norm": "After Hours", "bcc_usage_norm": "BCC Usage",
        "file_exfil_norm": "File Exfil", "usb_activity_norm": "USB Activity",
        "is_insider": "Insider?",
    }
    st.dataframe(
        display_df[show_cols].rename(columns=col_rename)
                             .set_index("Rank")
                             .style.format({
                                 "Risk Score":   "{:.4f}",
                                 "LSTM P95":     "{:.3f}",
                                 "After Hours":  "{:.3f}",
                                 "BCC Usage":    "{:.3f}",
                                 "File Exfil":   "{:.3f}",
                                 "USB Activity": "{:.3f}",
                             }),
        use_container_width=True,
    )
    st.markdown("---")

    # ── Section 3: Charts ─────────────────────────────────────────────────────
    st.header("📈 Visual Analysis")
    tab1, tab2, tab3, tab4 = st.tabs([
        "P/R/F1 vs K", "Score Distribution",
        "Top Users Risk", "TP/FP/FN Summary"
    ])

    k_sweep = [5, 10, 20, 30, 50, 75, 100]
    with tab1:
        st.pyplot(fig_prf1_vs_k(user_df, score_col, thresh_pct, insider_users, k_sweep))
    with tab2:
        st.pyplot(fig_score_distribution(lstm_user_df, insider_users))
    with tab3:
        top_n = min(k_choice, 20)
        st.pyplot(fig_top_users_bar(risk_df, insider_users, top_n=top_n))
    with tab4:
        st.pyplot(fig_tp_fp_fn(user_df, score_col, thresh_pct, insider_users))

    st.markdown("---")

    # ── Section 4: User explainability ───────────────────────────────────────
    st.header("🔍 User Investigation")
    st.markdown("Select a flagged user to see why they were flagged.")

    flagged_users = list(filtered_df.head(k_choice)["user"])
    selected_user = st.selectbox("Select user", flagged_users)

    if selected_user:
        row = risk_df[risk_df["user"] == selected_user].iloc[0]
        emp = row.get("employee_name", selected_user)
        st.subheader(f"{emp}  ({selected_user})")

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Risk Score",  f"{row['risk_score']:.4f}")
        mc2.metric("LSTM P95",    f"{row['score_p95']:.4f}")
        if show_ground_truth and "is_insider" in row.index:
            mc3.metric("Ground Truth", "Insider" if row["is_insider"] else "Normal")

        st.markdown("**Triggered behavioral flags:**")
        flags = explain_user(row)
        for flag in flags:
            st.markdown(f"- {flag}")

        # Signal breakdown bar
        signals = {
            "LSTM P95":     float(row["lstm_p95_norm"]),
            "After Hours":  float(row["after_hours_norm"]),
            "BCC Usage":    float(row["bcc_usage_norm"]),
            "File Exfil":   float(row["file_exfil_norm"]),
            "USB Activity": float(row["usb_activity_norm"]),
            "Multi PC":     float(row["multi_pc_norm"]),
        }
        sig_df = pd.DataFrame({"Signal": list(signals.keys()),
                               "Value":  list(signals.values())})
        fig, ax = plt.subplots(figsize=(6, 2.8))
        colors_sig = ["#E85454" if v >= 0.5 else "#4C9BE8" for v in sig_df["Value"]]
        ax.barh(sig_df["Signal"], sig_df["Value"], color=colors_sig, alpha=0.85)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="Flag threshold (0.5)")
        ax.set_xlabel("Normalised Signal Value")
        ax.set_title(f"Signal Breakdown — {selected_user}")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # ── Section 5: AI component explanation ──────────────────────────────────
    st.header("🤖 AI Algorithm Component")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("""
**Weighted Risk Scoring with Best-First Investigation Queue**

This system implements a *multi-signal evidence aggregation* algorithm:

1. **LSTM Autoencoder** (deep learning) — detects temporal behavioural anomalies
   by measuring reconstruction error on sequences of daily activity features.

2. **Behavioural Signal Extraction** — five rule-based indicators derived from
   CERT insider threat research (after-hours activity, BCC email usage,
   removable media file transfers, USB events, multi-workstation access).

3. **Normalised Weighted Aggregation** — each signal is scaled to [0, 1] and
   combined using domain-informed weights, producing a single *risk score*.

4. **Best-First Investigation Queue** — users are ranked by risk score
   (highest first), forming a priority queue that guides investigators to
   the highest-risk cases first — equivalent to best-first search over
   the suspicious-user population.

5. **Rule-Based Explainability (XAI)** — each flagged user receives a
   natural-language explanation identifying which signals triggered the alert.
        """)
    with col_b:
        st.pyplot(fig_weights_pie())
        plt.close()

    # ── Section 6: Best settings summary ─────────────────────────────────────
    st.markdown("---")
    st.header("✅ Best Configuration Summary")
    st.success(
        "**Best overall setup (from user-level evaluation):**  \n"
        "Model: **LSTM Autoencoder** | Aggregation: **score_p95** | "
        "Threshold: **90th percentile** | K = **50**  \n"
        "→ Precision = 0.32 | Recall = 0.23 | F1 = 0.27 | **16 / 70 insiders detected**  \n\n"
        "At K=20, Precision rises to **0.50** — half of the flagged users are real insiders.  \n"
        "Isolation Forest is not recommended: it scores insiders *lower* than normal users (ROC AUC < 0.5)."
    )


if __name__ == "__main__":
    main()
