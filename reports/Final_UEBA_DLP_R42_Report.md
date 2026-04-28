z# AI-Driven UEBA + DLP Risk Scoring System Using LSTM, Isolation Forest, Content Sensitivity, and Genetic Algorithm

Spring 2026

## Abstract

This project implements an AI-driven User and Entity Behavior Analytics (UEBA) plus Data Loss Prevention (DLP) system for insider threat detection. The system combines behavioral anomaly detection, content sensitivity scoring, composite risk scoring, and explainable analyst-facing outputs. The final pipeline was evaluated on the CERT r4.2 insider threat dataset after resolving a ground-truth release mismatch. The corrected r4.2 run aligned 70 insider users with 1,000 scored users and produced a working end-to-end dashboard for investigation and model interpretation.

## 1. Project Goal

Organizations face insider threats when trusted users misuse access to sensitive data. Traditional DLP systems are often rule-based and do not adapt well to changing behavior. This project addresses that gap by combining:

- UEBA behavioral anomaly detection.
- DLP content sensitivity scoring.
- LSTM temporal anomaly detection.
- Isolation Forest anomaly detection.
- Composite risk scoring.
- Genetic Algorithm (GA) weight optimization.
- Analyst-facing explanations and dashboard visualizations.

The goal is not only to flag suspicious users, but also to explain why a user is risky and help analysts prioritize investigations.

## 2. Dataset And Ground Truth

The final run used the CERT r4.2 dataset.

Ground truth alignment:

| Item | Result |
|---|---:|
| Selected CERT release | 4.2 |
| Matching insider users | 70 / 70 |
| Scored users | 1,000 |
| Insider user-days | 1,892 |

This was an important correction. An earlier Kaggle mirror matched CERT 6.2 and contained only 5 insider users. The final pipeline now automatically selects the matching CERT answer-key release by comparing scored user IDs against all available ground-truth releases.

## 3. Data Processing Summary

The cleaning pipeline processed email, logon, device, and file activity logs. Optional metadata files such as `users.csv`, `ldap.csv`, and `decoy_file.csv` were not available in the r4.2 Kaggle mirror, so the pipeline safely skipped them.

| Source | Cleaned Rows |
|---|---:|
| Email | 2,629,979 |
| Logon | 854,859 |
| Device | 405,380 |
| File access | 445,581 |
| Daily user feature rows | 326,985 |
| Users | 1,000 |

Dataset split:

| Split | Rows |
|---|---:|
| Train | 228,435 |
| Validation | 49,150 |
| Test | 49,400 |

The project uses chronological user-level splitting to avoid random leakage across each user's timeline.

## 4. Implemented System Components

### 4.1 Isolation Forest

Isolation Forest was trained on daily user behavior features such as email volume, attachments, after-hours activity, BCC usage, logon behavior, USB activity, file access behavior, and multi-PC access.

Final model artifacts:

- `models/isolation_forest_cert.pkl`
- `cleaned/email_user_daily_scored.csv`
- `models/isolation_forest_summary.json`

### 4.2 LSTM Autoencoder

The LSTM Autoencoder learned temporal patterns in user-day sequences. It was trained on GPU using 222,435 training windows from 1,000 users.

Training summary:

| Item | Result |
|---|---:|
| Device | CUDA |
| Epochs | 20 |
| Final average loss | 0.000718 |
| Suspicious threshold | 0.002034 |
| High threshold | 0.003932 |
| High-risk rows | 3,529 |
| Suspicious rows | 12,890 |

Final model artifacts:

- `models/lstm_autoencoder_cert.pkl`
- `cleaned/email_user_daily_lstm_scored.csv`
- `models/lstm_autoencoder_summary.json`

### 4.3 DLP Content Sensitivity

The DLP module scores email and file content using keyword patterns, file extension/magic-byte signals, and removable-media activity. It classifies activity into sensitivity levels: Internal, Sensitive, and Restricted.

Final DLP sensitivity results:

| Label | Rows | Percentage |
|---|---:|---:|
| Restricted | 4,167 | 3.0% |
| Sensitive | 127,149 | 91.0% |
| Internal | 8,361 | 6.0% |
| Public | 0 | 0.0% |

Additional DLP totals:

| Item | Result |
|---|---:|
| Content sensitivity daily rows | 139,677 |
| Users with sensitivity signals | 998 |
| Sensitive events | 593,510 |
| Restricted events | 4,418 |
| Runtime | 350.8 seconds |

Output artifact:

- `cleaned/content_sensitivity_daily.csv`

### 4.4 Composite Risk Scoring

The composite risk scoring engine combines behavioral and content-based signals into one investigation score.

Signals include:

- LSTM p95 anomaly score.
- After-hours behavior.
- BCC usage.
- File exfiltration.
- USB activity.
- Multi-PC access.
- DLP content sensitivity.

The resulting score is used to rank users into a best-first investigation queue.

### 4.5 Explainability

The dashboard explains each flagged user using triggered risk signals. Instead of hiding decisions inside a black box, the system shows the contribution of key signals such as LSTM anomaly score, file exfiltration, USB activity, and content sensitivity.

This provides practical explainability for analysts. Full SHAP/LIME integration remains future work, but the current weighted signal attribution is transparent and directly interpretable.

## 5. Evaluation Results

### 5.1 Day-Level And User-Level Evaluation

Evaluation used the corrected CERT r4.2 answer key with 70 matching insider users.

| Metric | IF Day | IF User | LSTM Day | LSTM User |
|---|---:|---:|---:|---:|
| ROC AUC | 0.7379 | 0.8496 | 0.8021 | 0.8245 |
| Average Precision | 0.0465 | 0.2563 | 0.1071 | 0.1899 |
| Precision | 0.0325 | 0.1899 | 0.1572 | 0.1408 |
| Recall | 0.0732 | 0.7000 | 0.4092 | 0.9714 |
| F1 | 0.0450 | 0.2988 | 0.2272 | 0.2459 |

Interpretation:

- LSTM performed better at day-level detection with ROC AUC 0.8021.
- Isolation Forest performed strongly at user-level ROC AUC 0.8496.
- LSTM user-level recall was high at 0.9714, meaning it covered most insider users when thresholded broadly.
- Precision remained challenging because insider threat detection is highly imbalanced.

### 5.2 User-Level Top-K Evaluation

The best top-K configuration before GA was:

| Item | Result |
|---|---:|
| Model | Isolation Forest |
| Aggregation | score_max |
| Threshold | 95th percentile |
| K | 50 |
| Precision | 0.3200 |
| Recall | 0.2286 |
| F1 | 0.2667 |
| True positives | 16 / 70 |

This means 16 of the 70 insider users appeared in the top 50 investigation queue.

## 6. Genetic Algorithm Optimization

The Genetic Algorithm does not retrain the LSTM or Isolation Forest. Instead, it optimizes how the final risk score is calculated.

What GA does:

1. Creates candidate weight and threshold configurations.
2. Applies those weights to normalized risk signals.
3. Ranks users by the resulting risk score.
4. Scores each candidate using F1@K and flag coverage.
5. Selects, crosses over, and mutates candidates over generations.
6. Saves the best configuration for the dashboard and risk scorer.

In plain terms, GA searches for better investigation-queue settings after the models have already produced their scores.

GA quick-run settings:

| Parameter | Value |
|---|---:|
| Population | 60 |
| Generations | 20 |
| Elitism | 4 |
| Fitness | 0.8 x F1@K + 0.2 x flag coverage |
| K | 50 |

Optimized GA weights from the quick run:

| Signal | Weight |
|---|---:|
| LSTM p95 | 0.0471 |
| After hours | 0.2982 |
| BCC usage | 0.0579 |
| File exfiltration | 0.0671 |
| USB activity | 0.2161 |
| Multi-PC access | 0.3136 |

GA result at K=50:

| Metric | GA | Baseline | Improvement |
|---|---:|---:|---:|
| Precision | 0.3000 | 0.1800 | +0.1200 |
| Recall | 0.2143 | 0.1286 | +0.0857 |
| F1 | 0.2500 | 0.1500 | +0.1000 |
| True positives | 15 | 9 | +6 |
| Flag coverage | 1.0000 | - | - |

Important note: this GA run optimized six UEBA behavioral signals. The DLP content sensitivity signal is included in the composite risk scorer and dashboard, but this specific quick GA run did not optimize content sensitivity as a GA gene. Extending GA to seven signals is a recommended future improvement.

## 7. Important Figures

The following figures should be included in the final submission. They are generated by the pipeline.

### Figure 1. User-Level Model Comparison

![Model comparison](../plots/user_level/model_comparison.png)

### Figure 2. Aggregation Comparison

![Aggregation comparison](../plots/user_level/aggregation_comparison.png)

### Figure 3. Precision, Recall, and F1 vs K

![Precision Recall F1 vs K](../plots/user_level/pr_f1_vs_k.png)

### Figure 4. LSTM Score Distribution

![Score distribution](../plots/user_level/score_distribution.png)

### Figure 5. Top Users By Risk Score

![Top users risk](../plots/user_level/top_users_risk.png)

### Figure 6. TP, FP, and FN Summary

![TP FP FN summary](../plots/user_level/tp_fp_fn_summary.png)

### Figure 7. Isolation Forest Score Distribution

![Isolation Forest score distribution](../plots/iforest_score_distribution.png)

### Figure 8. LSTM Score Distribution

![LSTM score distribution](../plots/lstm_score_distribution.png)

### Figure 9. LSTM Top User Timelines

![LSTM top user timelines](../plots/lstm_top5_user_timelines.png)

## 8. Dashboard

The project includes a tabbed Streamlit dashboard with separate pages for:

- Overview.
- Isolation Forest.
- LSTM Autoencoder.
- Risk Queue.
- User Investigation.
- GA Explained.
- Evaluation.

The dashboard allows analysts to inspect model outputs, ranked suspicious users, per-user explanations, and GA optimization behavior.

Dashboard command:

```bash
streamlit run app/ueba_dashboard_tabs.py
```

## 9. Strengths

- End-to-end UEBA + DLP pipeline runs successfully on CERT r4.2.
- Corrected ground-truth matching now uses 70 insider users instead of the earlier 5-user mismatch.
- LSTM captures temporal behavior patterns.
- Isolation Forest provides a strong user-level anomaly baseline.
- DLP content sensitivity adds data-awareness to behavioral risk.
- GA improves risk ranking over baseline weights in the quick run.
- Dashboard supports analyst investigation and explainability.

## 10. Limitations

- Precision remains limited because insider threat detection is highly imbalanced.
- LSTM maximum score saturates for many users, making `score_max` less informative in some cases.
- Current DLP sensitivity uses scalable keyword/rule-based classification rather than BERT.
- GA quick run used only six behavioral signals; content sensitivity should be added as a seventh GA gene in future work.
- SHAP/LIME were not fully integrated; current explainability uses transparent weighted signal attribution.

## 11. Future Work

Recommended improvements:

- Add BERT or a zero-shot transformer classifier for high-risk email/file content.
- Extend GA to optimize the content sensitivity weight directly.
- Add SHAP/LIME-style explanations for model-level feature attribution.
- Tune LSTM thresholds to reduce score saturation.
- Add analyst feedback loops so confirmed alerts can update risk scoring over time.
- Evaluate on additional CERT releases if matching raw logs and answer keys are available.

## 12. Conclusion

The final system successfully implements an AI-driven UEBA + DLP insider threat detection pipeline. After correcting the CERT release mismatch and moving to r4.2, the system aligned with 70 insider users and produced meaningful evaluation results. The strongest user-level configuration detected 16 of 70 insiders in the top 50 investigation queue before GA, while GA improved the baseline risk-scoring configuration from 9 to 15 true positives at K=50.

Overall, the project demonstrates a practical insider threat detection workflow that combines anomaly detection, content sensitivity, adaptive risk scoring, and explainable dashboard-based investigation.
