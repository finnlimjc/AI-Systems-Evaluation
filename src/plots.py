import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src.config import FIGURES_DIR, MODEL_ORDER

# Plot config
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "figure.titleweight": "bold"
})

DPI = 200
BAR_WIDTH = 0.38
GRID_ALPHA = 0.3
BAR_ALPHA = 0.9
VALUE_LABEL_SIZE = 9
HEADROOM = 0.18 #fractional axis margin so value labels stay inside the axes
LOOP_WER_THRESHOLD = 100 #a WER above this means the output is longer than the reference, i.e. looping or hallucinating
CONDITION_COLORS = {"Base": "#4C72B0", "Prompted": "#DD8452"}
MODEL_COLORS = ["#55A868", "#4C72B0", "#C44E52"]
ACCENT_COLORS = ["#8172B3", "#CCB974", "#64B5CD"]
ERROR_TYPE_COLORS = ["#C44E52", "#DD8452", "#4C72B0"]
GAIN_COLOR = "#55A868"
LOSS_COLOR = "#C44E52"
LATENCY_PERCENTILE_COLORS = ("#4C72B0", "#C44E52")
OUTCOME_COLORS = {"Improved": "#55A868", "Unchanged": "#BBBBBB", "Worse": "#C44E52"}
RADAR_CATEGORIES = ["WER\n(Lower)", "SER\n(Lower)", "DER\n(Lower)", "BERT-F1\n(Higher)", "Speed\n(RTF)", "Reliability\n(WER spread)"]
RADAR_SCALES = {"wer": 30, "ser": 20, "der": 15, "rtf": 2, "consistency": 60} #value at which each axis reads 0
RADAR_LINE_STYLES = {"Base": "-", "Prompted": "--"}
TABLE_HEADER_COLOR = "#34495E"
TABLE_ROW_COLORS = ("#FFFFFF", "#ECF0F1")


def save_figure(fig:Figure, filename:str) -> None:
    """Save a figure to FIGURES_DIR."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, dpi=DPI, bbox_inches="tight")


def _short(names:list[str]|pd.Index) -> list[str]:
    short_names = [name.replace("Whisper ", "") for name in names]
    return short_names


def _median_by_model(df:pd.DataFrame, columns:list[str], model_column:str="Model") -> pd.DataFrame:
    medians = df.groupby(model_column)[columns].median().reindex(MODEL_ORDER)
    return medians


def _mean_by_model(df:pd.DataFrame, columns:list[str], model_column:str="Model") -> pd.DataFrame:
    means = df.groupby(model_column)[columns].mean().reindex(MODEL_ORDER)
    return means


def _label_bars(ax, fmt:str) -> None:
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, padding=2, fontsize=VALUE_LABEL_SIZE)


def _paired_bars(ax, labels:list[str], base:pd.Series, prompted:pd.Series, ylabel:str, title:str, fmt:str, legend_loc:str="upper right") -> None:
    x = np.arange(len(labels))
    ax.bar(x - BAR_WIDTH / 2, base, BAR_WIDTH, label="Base", color=CONDITION_COLORS["Base"], alpha=BAR_ALPHA)
    ax.bar(x + BAR_WIDTH / 2, prompted, BAR_WIDTH, label="Prompted", color=CONDITION_COLORS["Prompted"], alpha=BAR_ALPHA)
    _label_bars(ax, fmt)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.margins(y=HEADROOM)
    ax.grid(axis="x", visible=False)
    ax.legend(loc=legend_loc)


def plot_accuracy_latency(df_results:pd.DataFrame) -> Figure:
    """Part 1: WER and per-clip latency by category and system (bars show mean +/- CI)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    sns.barplot(data=df_results, x="Category", y="WER (%)", hue="System", hue_order=MODEL_ORDER, palette=MODEL_COLORS, ax=axes[0])
    axes[0].set_title("Word Error Rate by Category")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("WER (%), lower is better")
    
    sns.barplot(data=df_results, x="Category", y="Latency (s)", hue="System", hue_order=MODEL_ORDER, palette=MODEL_COLORS, ax=axes[1])
    axes[1].set_title("Per-Clip Latency by Category")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Latency (s), lower is better")
    
    for ax in axes:
        ax.margins(y=HEADROOM)
        ax.grid(axis="x", visible=False)
        ax.legend(title="", loc="upper left")
    
    plt.tight_layout()
    
    return fig


def plot_speed_overview(df_results:pd.DataFrame, summary_df:pd.DataFrame) -> Figure:
    """Part 1: latency percentiles, real-time factor, cost per 1,000 audio hours and the accuracy/speed trade-off."""
    models = _short(summary_df["System"])
    x = np.arange(len(models))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    ax = axes[0, 0]
    ax.bar(x - BAR_WIDTH / 2, summary_df["p50 Latency (ms)"], BAR_WIDTH, label="p50 (median)", color=LATENCY_PERCENTILE_COLORS[0], alpha=BAR_ALPHA)
    ax.bar(x + BAR_WIDTH / 2, summary_df["p95 Latency (ms)"], BAR_WIDTH, label="p95 (tail)", color=LATENCY_PERCENTILE_COLORS[1], alpha=BAR_ALPHA)
    _label_bars(ax, "%.0f")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Latency per clip (ms)")
    ax.set_title("Latency")
    ax.margins(y=HEADROOM)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    
    ax = axes[0, 1]
    ax.bar(models, summary_df["Mean RTF"], width=0.5, color=MODEL_COLORS[:len(models)], alpha=BAR_ALPHA)
    _label_bars(ax, "%.3f")
    ax.set_ylabel("RTF (processing time / audio time)")
    ax.set_title("Real-Time Factor (lower is faster)")
    ax.margins(y=HEADROOM)
    ax.grid(axis="x", visible=False)
    
    ax = axes[1, 0]
    ax.bar(models, summary_df["Cost / 1,000 Audio Hours ($)"], width=0.5, color=MODEL_COLORS[:len(models)], alpha=BAR_ALPHA)
    _label_bars(ax, "$%.0f")
    ax.set_ylabel("Compute cost per 1,000 audio hours ($)")
    ax.set_title("Infrastructure Cost")
    ax.margins(y=HEADROOM)
    ax.grid(axis="x", visible=False)
    
    ax = axes[1, 1]
    wer_by_model = _mean_by_model(df_results, ["WER (%)"], model_column="System")["WER (%)"]
    latency_p50 = summary_df.set_index("System")["p50 Latency (ms)"].reindex(MODEL_ORDER)
    ax.scatter(latency_p50, wer_by_model, s=260, c=MODEL_COLORS, edgecolors="black", linewidth=1.5, zorder=3)
    for label, latency, wer_value in zip(_short(MODEL_ORDER), latency_p50, wer_by_model):
        ax.annotate(label, (latency, wer_value), xytext=(0, 16), textcoords="offset points", ha="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("p50 latency (ms)")
    ax.set_ylabel("Mean WER (%)")
    ax.set_title("Accuracy vs. Speed (bottom-left is best)")
    ax.margins(x=0.15, y=0.2)
    
    fig.suptitle("Speed and Cost (LibriSpeech, sequential requests)")
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    
    return fig


def plot_wer_comparison(df_accent:pd.DataFrame) -> Figure:
    """Part 2: median base vs prompted WER by accent, faceted by model."""
    df_wer = df_accent.melt(id_vars=["Model", "Accent"], value_vars=["Base WER (%)", "Prompted WER (%)"], var_name="Condition", value_name="WER (%)")
    df_wer["Condition"] = df_wer["Condition"].str.replace(" WER (%)", "", regex=False)
    
    grid = sns.catplot(
        data=df_wer,
        x="Accent",
        y="WER (%)",
        hue="Condition",
        col="Model",
        col_order=MODEL_ORDER,
        kind="bar",
        estimator="median",
        errorbar=None,
        palette=CONDITION_COLORS,
        height=4.2,
        aspect=1.0
    )
    fig = grid.figure
    grid.set_axis_labels("Accent", "Median WER (%)")
    grid.set_titles(col_template="{col_name}")
    
    for ax in grid.axes.flat:
        _label_bars(ax, "%.1f")
        ax.margins(y=HEADROOM)
    
    fig.suptitle("Median Word Error Rate by Accent: Base vs. Prompted")
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    
    return fig


def plot_accuracy_overview(df_accent:pd.DataFrame) -> Figure:
    """Part 2: how prompting changes typical WER, how often outputs run away, SER and the error types."""
    medians = _median_by_model(df_accent, ["Base WER (%)", "Prompted WER (%)"])
    means = _mean_by_model(df_accent, ["Base SER (%)", "Base DER (%)", "Base IER (%)"])
    models = _short(MODEL_ORDER)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    _paired_bars(axes[0, 0], models, medians["Base WER (%)"], medians["Prompted WER (%)"], "Median WER (%)", "Typical WER", "%.1f")
    
    loop_share = pd.DataFrame({
        "Base": (df_accent["Base WER (%)"] > LOOP_WER_THRESHOLD).groupby(df_accent["Model"]).mean().reindex(MODEL_ORDER) * 100,
        "Prompted": (df_accent["Prompted WER (%)"] > LOOP_WER_THRESHOLD).groupby(df_accent["Model"]).mean().reindex(MODEL_ORDER) * 100
    }, dtype="float64")
    _paired_bars(axes[0, 1], models, loop_share["Base"], loop_share["Prompted"], f"Samples with WER > {LOOP_WER_THRESHOLD}% (%)", "Runaway Outputs", "%.1f")
    
    ax = axes[1, 0]
    wer_change = df_accent["Prompted WER (%)"] - df_accent["Base WER (%)"]
    outcomes = pd.DataFrame({
        "Improved": (wer_change < 0).groupby(df_accent["Model"]).mean().reindex(MODEL_ORDER) * 100,
        "Unchanged": (wer_change == 0).groupby(df_accent["Model"]).mean().reindex(MODEL_ORDER) * 100,
        "Worse": (wer_change > 0).groupby(df_accent["Model"]).mean().reindex(MODEL_ORDER) * 100
    }, dtype="float64")
    bottom = np.zeros(len(models), dtype=np.float64)
    for outcome, color in OUTCOME_COLORS.items():
        ax.bar(models, outcomes[outcome], bottom=bottom, label=outcome, color=color, alpha=BAR_ALPHA)
        bottom += outcomes[outcome].to_numpy()
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f%%", label_type="center", fontsize=VALUE_LABEL_SIZE, color="black")
    ax.set_ylabel("Share of samples (%)")
    ax.set_title("Per-Sample Effect of Prompting on WER")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
    
    ax = axes[1, 1]
    x = np.arange(len(models))
    width = 0.26
    error_columns = ["Base SER (%)", "Base DER (%)", "Base IER (%)"]
    for offset, column, label, color in zip((-width, 0, width), error_columns, ("Substitutions", "Deletions", "Insertions"), ERROR_TYPE_COLORS):
        ax.bar(x + offset, means[column], width, label=label, color=color, alpha=BAR_ALPHA)
    _label_bars(ax, "%.1f")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Mean error rate (%)")
    ax.set_title("Error Types (Base)")
    ax.margins(y=HEADROOM)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    
    fig.suptitle("Accent Evaluation: Accuracy (all accents pooled)")
    plt.tight_layout(rect=(0, 0.02, 1, 0.95))
    
    return fig


def plot_semantic_overview(df_bert:pd.DataFrame) -> Figure:
    """Part 3: BERT-F1 on its own 0-1 scale, level and change with prompting."""
    means = _mean_by_model(df_bert, ["Base BERT-F1", "Prompted BERT-F1"])
    models = _short(MODEL_ORDER)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    
    _paired_bars(axes[0], models, means["Base BERT-F1"], means["Prompted BERT-F1"], "BERT-F1 (0-1, higher is better)", "Semantic Similarity to Reference", "%.3f", "upper left")
    lower_limit = float(np.floor((means.min().min() - 0.05) * 20) / 20)
    axes[0].set_ylim(lower_limit, 1.0)
    
    ax = axes[1]
    change = df_bert.pivot_table(index="Model", columns="Accent", values="F1 Improvement (Δ)").reindex(MODEL_ORDER)
    x = np.arange(len(models))
    width = 0.26
    for offset, accent, color in zip((-width, 0, width), change.columns, ACCENT_COLORS):
        ax.bar(x + offset, change[accent], width, label=accent, color=color, alpha=BAR_ALPHA)
    _label_bars(ax, "%.3f")
    ax.axhline(y=0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Prompted minus base BERT-F1")
    ax.set_title("Change in Similarity with Prompting")
    ax.margins(y=HEADROOM)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right")
    
    fig.suptitle("Semantic Evaluation (BERT-F1)")
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    
    return fig


def plot_prompt_shift(df_accent:pd.DataFrame) -> Figure:
    """Part 4: per model and accent, how far prompting moves the typical WER and how often it makes a sample worse."""
    models = _short(MODEL_ORDER)
    x = np.arange(len(models))
    width = 0.26
    df_shift = df_accent.assign(**{"WER Change": df_accent["Prompted WER (%)"] - df_accent["Base WER (%)"]})
    df_shift["Worse"] = (df_shift["WER Change"] > 0) * 100.0
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    
    panels = (("WER Change", "median", "Median WER change (points)", "Typical Change in WER (negative = prompt helps)"),
              ("Worse", "mean", "Samples where prompting raised WER (%)", "Share of Samples Made Worse"))
    for ax, (column, statistic, ylabel, title) in zip(axes, panels):
        table = df_shift.pivot_table(index="Model", columns="Accent", values=column, aggfunc=statistic).reindex(MODEL_ORDER)
        for offset, accent, color in zip((-width, 0, width), table.columns, ACCENT_COLORS):
            ax.bar(x + offset, table[accent], width, label=accent, color=color, alpha=BAR_ALPHA)
        _label_bars(ax, "%.0f")
        ax.axhline(y=0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.margins(y=HEADROOM)
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper right")
    
    fig.suptitle("Effect of the Accent Prompt by Model and Accent")
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    
    return fig


def _radar_score(value:float, scale:float) -> float:
    score = float(np.clip(100 - value / scale * 100, 0, 100))
    return score


def _radar_values(condition:str, model_data:pd.DataFrame, model_bert:pd.DataFrame, rtf:float) -> np.ndarray:
    n_axes = len(RADAR_CATEGORIES)
    wer = model_data[f"{condition} WER (%)"]
    values = np.empty(n_axes + 1, dtype=np.float64)
    values[0] = _radar_score(wer.median(), RADAR_SCALES["wer"])
    values[1] = _radar_score(model_data[f"{condition} SER (%)"].median(), RADAR_SCALES["ser"])
    values[2] = _radar_score(model_data[f"{condition} DER (%)"].median(), RADAR_SCALES["der"])
    values[3] = float(np.clip(model_bert[f"{condition} BERT-F1"].mean() * 100, 0, 100))
    values[4] = _radar_score(rtf, RADAR_SCALES["rtf"])
    values[5] = _radar_score(wer.quantile(0.75) - wer.quantile(0.25), RADAR_SCALES["consistency"])
    values[n_axes] = values[0]
    return values


def plot_radar_profiles(df_latency:pd.DataFrame, df_accent:pd.DataFrame, df_bert:pd.DataFrame) -> Figure:
    """Part 4: six-dimensional capability profile of each model, base (solid) against prompted (dashed)."""
    n_axes = len(RADAR_CATEGORIES)
    angles = np.empty(n_axes + 1, dtype=np.float64)
    angles[:n_axes] = np.arange(n_axes) / n_axes * 2 * np.pi
    angles[n_axes] = angles[0]
    
    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(17, 6.5), subplot_kw=dict(projection="polar"))
    
    for ax, model, color in zip(axes, MODEL_ORDER, MODEL_COLORS):
        model_data = df_accent[df_accent["Model"] == model]
        model_bert = df_bert[df_bert["Model"] == model]
        rtf = df_latency[df_latency["System"] == model]["RTF"].mean()
        
        for condition, line_style in RADAR_LINE_STYLES.items():
            values = _radar_values(condition, model_data, model_bert, rtf)
            ax.plot(angles, values, line_style, marker="o", linewidth=2.2, color=color)
            ax.fill(angles, values, alpha=0.30 if condition == "Base" else 0.08, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(RADAR_CATEGORIES, size=10)
        ax.tick_params(axis="x", pad=14)
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(["25", "50", "75", "100"], size=9)
        ax.set_title(_short([model])[0], size=14, pad=30)
        ax.grid(True, linestyle="--", alpha=0.5)
    
    handles = [Line2D([0], [0], color="dimgrey", linestyle=line_style, linewidth=2.2, label=condition) for condition, line_style in RADAR_LINE_STYLES.items()]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=11)
    fig.suptitle("Model Capability Profiles: Base vs. Prompted (outer edge = best)")
    plt.tight_layout(rect=(0, 0.06, 1, 0.93))
    
    return fig


def plot_summary_table(summary_report:pd.DataFrame) -> Figure:
    """Part 4: the final summary report rendered as a table."""
    header = ["Model"] + [column.replace(" (", "\n(") if column.endswith(")") else column.replace(" BERT-F1", "\nBERT-F1") for column in summary_report.columns[1:]]
    body = [[row["Model"].replace("Whisper ", "")] + [f"{value:.2f}" for value in row.iloc[1:].to_numpy()] for _, row in summary_report.iterrows()]
    table_data = [header] + body
    
    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.axis("off")
    
    table = ax.table(cellText=table_data, cellLoc="center", loc="center", colWidths=[0.11] + [0.13] * (len(header) - 1))
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.6)
    
    for col_idx in range(len(header)):
        table[(0, col_idx)].set_facecolor(TABLE_HEADER_COLOR)
        table[(0, col_idx)].set_text_props(weight="bold", color="white")
        table[(0, col_idx)].set_height(table[(0, col_idx)].get_height() * 1.4)
    
    for row_idx in range(1, len(table_data)):
        row_color = TABLE_ROW_COLORS[row_idx % 2 == 0]
        for col_idx in range(len(header)):
            table[(row_idx, col_idx)].set_facecolor(row_color)
    
    ax.set_title("Summary: Base vs. Prompted (means over all accents)", pad=12)
    plt.tight_layout()
    
    return fig
