import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from src.config import RESULTS_DIR, MODEL_ORDER

# Plot config
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.size": 9})

DPI = 300
BAR_WIDTH = 0.35
GRID_ALPHA = 0.3
BAR_ALPHA = 0.8
GAIN_COLOR = "#55A868"
LOSS_COLOR = "#C44E52"
LATENCY_PERCENTILE_COLORS = ("#4C72B0", "#C44E52")
COST_COLORS = ["#55A868", "#4C72B0", "#C44E52"]
WER_CONDITION_PALETTE = ["#4C72B0", "#55A868"]
DASHBOARD_WER_COLORS = ("#FF6B6B", "#4ECDC4")
DASHBOARD_BERT_COLORS = ("#FFE66D", "#95E1D3")
DASHBOARD_ACCENT_COLORS = ("#9B59B6", "#3498DB")
DASHBOARD_LATENCY_COLORS = ["#E74C3C", "#F39C12", "#F1C40F"]
DASHBOARD_RTF_COLORS = ["#2ECC71", "#27AE60", "#229954"]
DASHBOARD_ERROR_COLORS = ("#E74C3C", "#F39C12", "#3498DB")
DASHBOARD_IMPROVEMENT_COLORS = ["#E74C3C", "#F39C12", "#2ECC71"]
RADAR_COLORS = ["#FF6B6B", "#4ECDC4", "#95E1D3"]
RADAR_CATEGORIES = ["WER\n(Lower)", "SER\n(Lower)", "DER\n(Lower)", "BERT-F1\n(Higher)", "Speed\n(RTF)", "Reliability\n(Consistency)"]
RADAR_SCALES = {"wer": 30, "ser": 20, "der": 15, "rtf": 2, "consistency": 15} #value at which each axis reads 0
WATERFALL_COLORS = ["#95A5A6", "#2ECC71", "#3498DB"]
WATERFALL_STAGES = ["Baseline\n(Part 1)", "Prompt\nEngineering\n(Part 2)", "Semantic\nEvaluation\n(Part 3)"]
TABLE_HEADER_COLOR = "#34495E"
TABLE_ROW_COLORS = ("#FFFFFF", "#ECF0F1")


def save_figure(fig:Figure, filename:str) -> None:
    """Save a figure to RESULTS_DIR."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / filename, dpi=DPI, bbox_inches="tight")


def _short(names:list[str]|pd.Index) -> list[str]:
    short_names = [name.replace("Whisper ", "") for name in names]
    return short_names


def _by_model(df:pd.DataFrame, columns:list[str], model_column:str="Model") -> pd.DataFrame:
    means = df.groupby(model_column)[columns].mean().reindex(MODEL_ORDER)
    return means


def plot_accuracy_latency(df_results:pd.DataFrame) -> Figure:
    """Part 1: WER and per-clip latency by category and system (bars show mean +/- CI)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.barplot(data=df_results, x="Category", y="WER (%)", hue="System", hue_order=MODEL_ORDER, ax=axes[0])
    axes[0].set_title("Word Error Rate (WER %) by Category")
    axes[0].set_ylabel("WER (%) - Lower is Better")
    
    sns.barplot(data=df_results, x="Category", y="Latency (s)", hue="System", hue_order=MODEL_ORDER, ax=axes[1])
    axes[1].set_title("Per-Clip Processing Latency (s) by Category")
    axes[1].set_ylabel("Latency (s) - Lower is Better")
    
    plt.tight_layout()
    
    return fig


def plot_cost_efficiency(summary_df:pd.DataFrame) -> Figure:
    """Part 1: p50/p95 latency and compute cost per 1,000 audio hours for each model."""
    models = _short(summary_df["System"])
    x = np.arange(len(models))
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    axes[0].bar(x - BAR_WIDTH / 2, summary_df["p50 Latency (ms)"], BAR_WIDTH, label="p50 Median", color=LATENCY_PERCENTILE_COLORS[0])
    axes[0].bar(x + BAR_WIDTH / 2, summary_df["p95 Latency (ms)"], BAR_WIDTH, label="p95 Tail Latency", color=LATENCY_PERCENTILE_COLORS[1])
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Per-Clip Latency Comparison Across Whisper Models")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)
    
    axes[1].bar(models, summary_df["Cost / 1,000 Audio Hours ($)"], color=COST_COLORS[:len(models)], width=0.4)
    axes[1].set_ylabel("Compute Cost per 1,000 Audio Hours ($)")
    axes[1].set_title("Infrastructure Cost Comparison")
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    
    return fig


def plot_wer_comparison(df_accent:pd.DataFrame) -> Figure:
    """Part 2: base vs prompted WER by accent, faceted by model."""
    df_wer = df_accent.melt(
        id_vars=["Model", "Accent"],
        value_vars=["Base WER (%)", "Prompted WER (%)"],
        var_name="Condition",
        value_name="WER (%)"
    )
    df_wer["Condition"] = df_wer["Condition"].str.replace(" WER (%)", "", regex=False)
    
    grid = sns.catplot(
        data=df_wer,
        x="Accent",
        y="WER (%)",
        hue="Condition",
        col="Model",
        col_order=MODEL_ORDER,
        kind="bar",
        palette=WER_CONDITION_PALETTE,
        height=4,
        aspect=1.0
    )
    fig = grid.figure
    
    fig.subplots_adjust(top=0.82)
    fig.suptitle("Word Error Rate (WER): Base vs. Prompted Across Whisper Models", fontsize=14, fontweight="bold")
    grid.set_axis_labels("Accent", "WER (%)", fontweight="bold")
    grid.set_titles(col_template="{col_name}", weight="bold")
    
    for ax in grid.axes.flat:
        for patch in ax.patches:
            height = patch.get_height()
            if not np.isnan(height) and height > 0:
                ax.annotate(f"{height:.1f}%", (patch.get_x() + patch.get_width() / 2., height),
                            ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
    
    plt.tight_layout()
    
    return fig


def plot_ser_reduction_heatmap(df_accent:pd.DataFrame) -> Figure:
    """Part 2: mean relative SER reduction per model and accent."""
    pivot_ser = df_accent.pivot_table(index="Model", columns="Accent", values="SER Reduction (Δ%)", aggfunc="mean")
    pivot_ser = pivot_ser.reindex(MODEL_ORDER)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot_ser,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        cbar_kws={"label": "SER Relative Reduction Δ (%)"},
        linewidths=1.5,
        annot_kws={"size": 11, "weight": "bold"},
        ax=ax
    )
    ax.set_title("Substitution Error Rate (SER) Reduction (Δ%)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Accent", fontweight="bold")
    ax.set_ylabel("Model Size", fontweight="bold")
    plt.tight_layout()
    
    return fig


def plot_dashboard(df_latency:pd.DataFrame, df_accent:pd.DataFrame, df_bert:pd.DataFrame) -> Figure:
    """Part 4: nine-panel overview of accuracy, semantics, speed and trade-offs."""
    fig = plt.figure(figsize=(16, 12))
    grid = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    _panel_wer_by_model(fig.add_subplot(grid[0, 0]), df_accent)
    _panel_ser_reduction(fig.add_subplot(grid[0, 1]), df_accent)
    _panel_bert_by_model(fig.add_subplot(grid[0, 2]), df_bert)
    _panel_wer_by_accent(fig.add_subplot(grid[1, 0]), df_accent)
    _panel_latency(fig.add_subplot(grid[1, 1]), df_latency)
    _panel_rtf(fig.add_subplot(grid[1, 2]), df_latency)
    _panel_error_types(fig.add_subplot(grid[2, 0]), df_accent)
    _panel_improvements(fig.add_subplot(grid[2, 1]), df_accent, df_bert)
    _panel_tradeoff(fig.add_subplot(grid[2, 2]), df_latency)
    
    # No tight_layout: it would override the gridspec hspace/wspace
    fig.suptitle("COMPREHENSIVE WHISPER STT BENCHMARK DASHBOARD\nAll Methods & Metrics Comparison", fontsize=16, fontweight="bold", y=0.995)
    
    return fig


def plot_metrics_heatmap(heatmap_data:pd.DataFrame) -> Figure:
    """Part 4: normalised 0-100 performance score of every metric per model."""
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        heatmap_data.T,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=50,
        cbar_kws={"label": "Performance Score (0=Worst, 100=Best)"},
        linewidths=1,
        annot_kws={"size": 8, "weight": "bold"},
        ax=ax
    )
    ax.set_title("Normalized Performance Matrix: All Metrics Comparison\n(0=Worst, 100=Best)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Model", fontweight="bold")
    ax.set_ylabel("Metric", fontweight="bold")
    plt.tight_layout()
    
    return fig


def plot_radar_profiles(df_latency:pd.DataFrame, df_accent:pd.DataFrame, df_bert:pd.DataFrame) -> Figure:
    """Part 4: six-dimensional capability profile of each model under the prompted condition."""
    n_axes = len(RADAR_CATEGORIES)
    angles = np.empty(n_axes + 1, dtype=np.float64)
    angles[:n_axes] = np.arange(n_axes) / n_axes * 2 * np.pi
    angles[n_axes] = angles[0]
    
    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(16, 5), subplot_kw=dict(projection="polar"))
    fig.suptitle("Model Capability Profiles (Prompted Condition)\n6-Dimensional Comparison", fontsize=14, fontweight="bold", y=1.02)
    
    for ax, model, color in zip(axes, MODEL_ORDER, RADAR_COLORS):
        model_data = df_accent[df_accent["Model"] == model]
        model_bert = df_bert[df_bert["Model"] == model]
        model_latency = df_latency[df_latency["System"] == model]
        
        values = np.empty(n_axes + 1, dtype=np.float64)
        values[0] = 100 - (model_data["Prompted WER (%)"].mean() / RADAR_SCALES["wer"] * 100)
        values[1] = 100 - (model_data["Prompted SER (%)"].mean() / RADAR_SCALES["ser"] * 100)
        values[2] = 100 - (model_data["Prompted DER (%)"].mean() / RADAR_SCALES["der"] * 100)
        values[3] = model_bert["Prompted BERT-F1"].mean()
        values[4] = 100 - (model_latency["RTF"].mean() / RADAR_SCALES["rtf"] * 100)
        values[5] = 100 - (model_data["Prompted WER (%)"].std() / RADAR_SCALES["consistency"] * 100)
        values[n_axes] = values[0]
        
        ax.plot(angles, values, "o-", linewidth=2, color=color, label=_short([model])[0])
        ax.fill(angles, values, alpha=0.25, color=color)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(RADAR_CATEGORIES, size=9)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"], size=8)
        ax.set_title(_short([model])[0], size=11, fontweight="bold", pad=20)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    
    return fig


def plot_waterfall(df_accent:pd.DataFrame, df_bert:pd.DataFrame) -> Figure:
    """Part 4: baseline WER, WER after prompting and the prompted BERT-F1."""
    baseline_wer = df_accent["Base WER (%)"].mean()
    after_prompt = df_accent["Prompted WER (%)"].mean()
    prompted_bert = df_bert["Prompted BERT-F1"].mean()
    final_semantic = prompted_bert / 100 * 25 #scaled into WER space for visualization
    wer_change = -(baseline_wer - after_prompt)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    edge_style = dict(alpha=BAR_ALPHA, edgecolor="black", linewidth=2)
    
    ax.bar(0, baseline_wer, color=WATERFALL_COLORS[0], label="WER-based metrics", **edge_style)
    ax.text(0, baseline_wer / 2, f"{baseline_wer:.1f}%", ha="center", va="center", fontweight="bold", fontsize=11, color="white")
    
    ax.bar(1, abs(wer_change), bottom=baseline_wer + wer_change, color=WATERFALL_COLORS[1], label="WER Improvement", **edge_style)
    ax.text(1, (baseline_wer + wer_change) / 2, f"{after_prompt:.1f}%", ha="center", va="center", fontweight="bold", fontsize=11, color="white")
    
    ax.text(2, after_prompt + 3, f"BERT-F1:\n{final_semantic:.1f}%", ha="center", fontweight="bold", fontsize=10,
            bbox=dict(boxstyle="round", facecolor=WATERFALL_COLORS[2], alpha=0.7))
    
    ax.plot([0.4, 0.6], [baseline_wer, baseline_wer], "k--", linewidth=1.5, alpha=0.5)
    
    ax.set_xticks(np.arange(len(WATERFALL_STAGES)))
    ax.set_xticklabels(WATERFALL_STAGES, fontweight="bold")
    ax.set_ylabel("WER (%) / Improvement", fontweight="bold", fontsize=11)
    ax.set_title("Evaluation Method Impact Summary\nCumulative Improvement Across All Methods", fontsize=13, fontweight="bold", pad=15)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.legend(fontsize=10, loc="upper right")
    
    summary_text = "\n".join([
        "",
        f"Baseline WER: {baseline_wer:.2f}%",
        f"After Prompting: {after_prompt:.2f}%",
        f"Improvement: {baseline_wer - after_prompt:.2f}% absolute",
        f"Semantic Score: {prompted_bert:.1f}%",
        ""
    ])
    ax.text(0.98, 0.97, summary_text, transform=ax.transAxes, fontsize=10, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    
    plt.tight_layout()
    
    return fig


def plot_summary_table(summary_report:pd.DataFrame) -> Figure:
    """Part 4: the final summary report rendered as a table."""
    header = ["Model"] + summary_report.columns[1:].tolist()
    body = [[row["Model"].replace("Whisper ", "")] + [str(value) for value in row.iloc[1:].to_numpy()] for _, row in summary_report.iterrows()]
    table_data = [header] + body
    
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axis("tight")
    ax.axis("off")
    
    table = ax.table(cellText=table_data, cellLoc="center", loc="center", colWidths=[0.12] + [0.10] * (len(header) - 1))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    for col_idx in range(len(header)):
        table[(0, col_idx)].set_facecolor(TABLE_HEADER_COLOR)
        table[(0, col_idx)].set_text_props(weight="bold", color="white")
    
    for row_idx in range(1, len(table_data)):
        row_color = TABLE_ROW_COLORS[row_idx % 2 == 0]
        for col_idx in range(len(header)):
            table[(row_idx, col_idx)].set_facecolor(row_color)
    
    ax.set_title("FINAL COMPREHENSIVE COMPARISON TABLE\nAll Methods, Models & Metrics", fontsize=14, fontweight="bold", pad=20)
    
    return fig


def _style_axis(ax, ylabel:str, title:str, grid_axis:str="y") -> None:
    ax.set_title(title, fontweight="bold")
    ax.grid(axis=grid_axis, alpha=GRID_ALPHA)
    if grid_axis == "y":
        ax.set_ylabel(ylabel, fontweight="bold")
    else:
        ax.set_xlabel(ylabel, fontweight="bold")


def _paired_bars(ax, labels:list[str], base:pd.Series, prompted:pd.Series, colors:tuple) -> None:
    x = np.arange(len(labels))
    ax.bar(x - BAR_WIDTH / 2, base, BAR_WIDTH, label="Base", color=colors[0], alpha=BAR_ALPHA)
    ax.bar(x + BAR_WIDTH / 2, prompted, BAR_WIDTH, label="Prompted", color=colors[1], alpha=BAR_ALPHA)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()


def _panel_wer_by_model(ax, df_accent:pd.DataFrame) -> None:
    model_perf = _by_model(df_accent, ["Base WER (%)", "Prompted WER (%)"])
    _paired_bars(ax, _short(model_perf.index), model_perf["Base WER (%)"], model_perf["Prompted WER (%)"], DASHBOARD_WER_COLORS)
    _style_axis(ax, "WER (%)", "WER: Base vs. Prompted")


def _panel_ser_reduction(ax, df_accent:pd.DataFrame) -> None:
    ser_by_model = _by_model(df_accent, ["SER Reduction (Δ%)"])["SER Reduction (Δ%)"]
    colors = [GAIN_COLOR if value > 0 else LOSS_COLOR for value in ser_by_model.to_numpy()]
    ax.barh(_short(ser_by_model.index), ser_by_model.to_numpy(), color=colors, alpha=BAR_ALPHA)
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
    _style_axis(ax, "SER Reduction (Δ%)", "Substitution Error Improvement", grid_axis="x")


def _panel_bert_by_model(ax, df_bert:pd.DataFrame) -> None:
    bert_by_model = _by_model(df_bert, ["Base BERT-F1", "Prompted BERT-F1"])
    _paired_bars(ax, _short(bert_by_model.index), bert_by_model["Base BERT-F1"], bert_by_model["Prompted BERT-F1"], DASHBOARD_BERT_COLORS)
    _style_axis(ax, "BERT-F1 (%)", "Semantic Similarity (BERT-F1)")


def _panel_wer_by_accent(ax, df_accent:pd.DataFrame) -> None:
    accent_perf = df_accent.groupby("Accent")[["Base WER (%)", "Prompted WER (%)"]].mean()
    _paired_bars(ax, accent_perf.index.tolist(), accent_perf["Base WER (%)"], accent_perf["Prompted WER (%)"], DASHBOARD_ACCENT_COLORS)
    _style_axis(ax, "WER (%)", "Accent Robustness (WER by Accent)")


def _panel_latency(ax, df_latency:pd.DataFrame) -> None:
    latency_by_system = _by_model(df_latency, ["Latency (s)"], model_column="System")["Latency (s)"] * 1000 #seconds to ms
    ax.barh(_short(latency_by_system.index), latency_by_system.to_numpy(), color=DASHBOARD_LATENCY_COLORS, alpha=BAR_ALPHA)
    _style_axis(ax, "Latency (ms)", "Processing Speed (Lower is Better)", grid_axis="x")


def _panel_rtf(ax, df_latency:pd.DataFrame) -> None:
    rtf_by_system = _by_model(df_latency, ["RTF"], model_column="System")["RTF"]
    ax.bar(_short(rtf_by_system.index), rtf_by_system.to_numpy(), color=DASHBOARD_RTF_COLORS, alpha=BAR_ALPHA)
    ax.axhline(y=1, color="red", linestyle="--", linewidth=1.5, label="Real-time threshold")
    ax.legend()
    _style_axis(ax, "RTF", "Real-Time Factor (Lower is Better)")


def _panel_error_types(ax, df_accent:pd.DataFrame) -> None:
    error_breakdown = _by_model(df_accent, ["Base SER (%)", "Base DER (%)", "Base IER (%)"])
    labels = ["SER (Base)", "DER (Base)", "IER (Base)"]
    width = 0.25
    x = np.arange(len(error_breakdown))
    for offset, column, label, color in zip((-width, 0, width), error_breakdown.columns, labels, DASHBOARD_ERROR_COLORS):
        ax.bar(x + offset, error_breakdown[column], width, label=label, color=color, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(_short(error_breakdown.index))
    ax.legend(fontsize=8)
    _style_axis(ax, "Error Rate (%)", "Error Type Breakdown (Base)")


def _panel_improvements(ax, df_accent:pd.DataFrame, df_bert:pd.DataFrame) -> None:
    wer_means = _by_model(df_accent, ["Base WER (%)", "Prompted WER (%)", "SER Reduction (Δ%)"])
    bert_means = _by_model(df_bert, ["F1 Improvement (Δ)"])
    improvements = pd.DataFrame({
        "WER": wer_means["Base WER (%)"] - wer_means["Prompted WER (%)"],
        "SER": wer_means["SER Reduction (Δ%)"],
        "BERT-F1": bert_means["F1 Improvement (Δ)"]
    }, dtype="float64")
    improvements.plot(kind="bar", ax=ax, color=DASHBOARD_IMPROVEMENT_COLORS, alpha=BAR_ALPHA)
    ax.set_xticklabels(_short(improvements.index), rotation=45)
    ax.legend(loc="upper left", fontsize=8)
    _style_axis(ax, "Improvement (%)", "Multi-Metric Improvements with Prompting")


def _panel_tradeoff(ax, df_latency:pd.DataFrame) -> None:
    tradeoff = _by_model(df_latency, ["WER (%)", "Latency (s)"], model_column="System")
    latency_ms = tradeoff["Latency (s)"].to_numpy() * 1000
    wer_values = tradeoff["WER (%)"].to_numpy()
    ax.scatter(latency_ms, wer_values, s=300, c=np.arange(len(tradeoff)), cmap="viridis", alpha=0.7, edgecolors="black", linewidth=2)
    for label, latency, wer_value in zip(_short(tradeoff.index), latency_ms, wer_values):
        ax.annotate(label, (latency, wer_value), fontsize=9, fontweight="bold", ha="center", va="center")
    ax.set_xlabel("Latency (ms)", fontweight="bold")
    ax.set_ylabel("WER (%)", fontweight="bold")
    ax.set_title("Accuracy vs. Speed Trade-off", fontweight="bold")
    ax.grid(True, alpha=GRID_ALPHA)
