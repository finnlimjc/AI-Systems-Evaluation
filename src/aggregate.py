import numpy as np
import pandas as pd

from src.config import MODEL_ORDER
from src.metrics import compute_bert_f1, normalize_metric

_METRIC_AGGREGATIONS = {
    "Base WER (%)": "mean",
    "Prompted WER (%)": "mean",
    "Base SER (%)": "mean",
    "Prompted SER (%)": "mean",
    "Base DER (%)": "mean",
    "Prompted DER (%)": "mean",
    "Base IER (%)": "mean",
    "Prompted IER (%)": "mean",
    "SER Reduction (Δ%)": "mean"
}


def summarise_by_system(df_results:pd.DataFrame) -> pd.DataFrame:
    """Mean accuracy and speed metrics per system, in order of appearance."""
    df_summary = (
        df_results.groupby("System", sort=False)
        .agg(**{
            "WER (%)": ("WER (%)", "mean"),
            "CER (%)": ("CER (%)", "mean"),
            "Precision (%)": ("Precision (%)", "mean"),
            "Recall (%)": ("Recall (%)", "mean"),
            "F1 Score (%)": ("F1 Score (%)", "mean"),
            "Mean Latency (s)": ("Latency (s)", "mean"),
            "RTF": ("RTF", "mean")
        })
        .reset_index()
    )
    return df_summary


def cost_efficiency_summary(df_results:pd.DataFrame, hourly_cost:dict[str, float]) -> pd.DataFrame:
    """Latency percentiles, mean RTF and cost per 1,000 audio hours from the per-clip measurements."""
    rows = []
    for system, group in df_results.groupby("System", sort=False):
        latency_ms = group["Latency (s)"] * 1000.0
        total_audio_hours = group["Duration (s)"].sum() / 3600.0
        total_compute_cost = (group["Latency (s)"].sum() / 3600.0) * hourly_cost[system]
        
        rows.append({
            "System": system,
            "p50 Latency (ms)": np.percentile(latency_ms, 50),
            "p95 Latency (ms)": np.percentile(latency_ms, 95),
            "p99 Latency (ms)": np.percentile(latency_ms, 99),
            "Mean RTF": group["RTF"].mean(),
            "Cost / 1,000 Audio Hours ($)": (total_compute_cost / total_audio_hours) * 1000.0
        })
    
    column_dtypes = {
        "System": "object",
        "p50 Latency (ms)": "float64",
        "p95 Latency (ms)": "float64",
        "p99 Latency (ms)": "float64",
        "Mean RTF": "float64",
        "Cost / 1,000 Audio Hours ($)": "float64"
    }
    summary_df = pd.DataFrame(rows).astype(column_dtypes)
    return summary_df


def build_bert_table(df_accent:pd.DataFrame) -> pd.DataFrame:
    """Mean BERT-F1 of base and prompted hypotheses per (model, accent), computed from the stored Part 2 hypotheses."""
    rows = []
    for model_name in df_accent["Model"].unique():
        df_model = df_accent[df_accent["Model"] == model_name]
        references = df_model["Reference"].tolist()
        base_bert_f1 = compute_bert_f1(df_model["Base Hypothesis"].tolist(), references)
        prompted_bert_f1 = compute_bert_f1(df_model["Prompted Hypothesis"].tolist(), references)
        
        for accent in df_model["Accent"].unique():
            accent_mask = (df_model["Accent"] == accent).to_numpy()
            base_f1_mean = float(base_bert_f1[accent_mask].mean())
            prompted_f1_mean = float(prompted_bert_f1[accent_mask].mean())
            
            rows.append({
                "Model": model_name,
                "Accent": accent,
                "Base BERT-F1": round(base_f1_mean, 4),
                "Prompted BERT-F1": round(prompted_f1_mean, 4),
                "F1 Improvement (Δ)": round(prompted_f1_mean - base_f1_mean, 4)
            })
    
    column_dtypes = {
        "Model": "object",
        "Accent": "object",
        "Base BERT-F1": "float64",
        "Prompted BERT-F1": "float64",
        "F1 Improvement (Δ)": "float64"
    }
    df_bert = pd.DataFrame(rows).astype(column_dtypes)
    return df_bert


def build_metrics_matrix(df_accent:pd.DataFrame, df_bert:pd.DataFrame) -> pd.DataFrame:
    """Per-model mean error rates and BERT scores, one row per model in MODEL_ORDER."""
    error_metrics = df_accent.groupby("Model").agg(_METRIC_AGGREGATIONS).round(2)
    bert_metrics = df_bert.groupby("Model").agg({
        "Base BERT-F1": "mean",
        "Prompted BERT-F1": "mean",
        "F1 Improvement (Δ)": "mean"
    }).round(2)
    
    metrics_matrix = error_metrics.join(bert_metrics).reindex(MODEL_ORDER)
    return metrics_matrix


def build_heatmap_data(metrics_matrix:pd.DataFrame) -> pd.DataFrame:
    """Normalise every metric column to a 0-100 score where 100 is best."""
    scores = {}
    for column in metrics_matrix.columns:
        is_gain = "Reduction" in column or "Improvement" in column
        scores[column] = normalize_metric(metrics_matrix[column], lower_is_better=not is_gain)
    
    heatmap_data = pd.DataFrame(scores, dtype="float64")
    return heatmap_data


def build_summary_report(df_accent:pd.DataFrame, df_bert:pd.DataFrame) -> pd.DataFrame:
    """Final per-model comparison of base vs prompted WER, SER reduction and BERT-F1."""
    accent_means = df_accent.groupby("Model")[["Base WER (%)", "Prompted WER (%)", "SER Reduction (Δ%)"]].mean().reindex(MODEL_ORDER)
    bert_means = df_bert.groupby("Model")[["Base BERT-F1", "Prompted BERT-F1"]].mean().reindex(MODEL_ORDER)
    wer_improvement = accent_means["Base WER (%)"] - accent_means["Prompted WER (%)"]
    
    summary_report = pd.DataFrame({
        "Model": MODEL_ORDER,
        "Base WER (%)": accent_means["Base WER (%)"].round(2).to_numpy(),
        "Prompted WER (%)": accent_means["Prompted WER (%)"].round(2).to_numpy(),
        "WER Improvement (Δ%)": wer_improvement.round(2).to_numpy(),
        "SER Reduction (Δ%)": accent_means["SER Reduction (Δ%)"].round(2).to_numpy(),
        "Base BERT-F1": bert_means["Base BERT-F1"].round(2).to_numpy(),
        "Prompted BERT-F1": bert_means["Prompted BERT-F1"].round(2).to_numpy()
    }).astype({
        "Model": "object",
        "Base WER (%)": "float64",
        "Prompted WER (%)": "float64",
        "WER Improvement (Δ%)": "float64",
        "SER Reduction (Δ%)": "float64",
        "Base BERT-F1": "float64",
        "Prompted BERT-F1": "float64"
    })
    return summary_report
