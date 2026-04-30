"""
Generate all figures for the research paper.
Uses your real benchmark CSV output from benchmark_v2.py

Usage:
    python generate_figures.py results/benchmark_v2_YYYYMMDD_HHMMSS.csv

Output (saved to figures/ folder):
    fig1_ttft_vs_prompt.png         - TTFT vs Prompt Length
    fig2_tps_vs_output.png          - TPS vs Output Length
    fig3_e2e_batch_scaling.png      - E2E Latency Batch Scaling
    fig4_itl_heatmap.png            - ITL Heatmap (prompt x output)
    fig5_percentiles.png            - TTFT Percentiles (P50/P95/P99)
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "figure.dpi":         150,
    "savefig.dpi":        200,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
})

COLORS  = ["#2563eb", "#16a34a", "#dc2626", "#d97706"]
MARKERS = ["o", "s", "^", "D"]
OUT_DIR = "figures"

def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Columns: {list(df.columns)}")
    return df

def save(fig, name: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════
# FIGURE 1 — TTFT vs Prompt Length  (one line per batch size)
# ══════════════════════════════════════════════════════════════
def fig_ttft_vs_prompt(df):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    batch_sizes = sorted(df["batch_size"].unique())
    for i, bs in enumerate(batch_sizes):
        sub = df[df["batch_size"] == bs].groupby("prompt_length").agg(
            mean=("ttft_mean_ms", "mean"),
            std=("ttft_std_ms",  "mean")
        ).reset_index()
        ax.errorbar(
            sub["prompt_length"], sub["mean"], yerr=sub["std"],
            label=f"Batch={bs}", color=COLORS[i], marker=MARKERS[i],
            linewidth=2, markersize=6, capsize=4
        )

    ax.set_xlabel("Prompt Length (tokens)")
    ax.set_ylabel("Time to First Token — TTFT (ms)")
    ax.set_title("TTFT vs. Prompt Length\n(LLaMA 3.2, Apple M5, Ollama)")
    ax.legend(title="Batch Size", framealpha=0.9)
    ax.set_xticks(sorted(df["prompt_length"].unique()))
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, "fig1_ttft_vs_prompt.png")


# ══════════════════════════════════════════════════════════════
# FIGURE 2 — TPS vs Output Length  (one line per batch size)
# ══════════════════════════════════════════════════════════════
def fig_tps_vs_output(df):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    batch_sizes = sorted(df["batch_size"].unique())
    for i, bs in enumerate(batch_sizes):
        sub = df[df["batch_size"] == bs].groupby("output_length").agg(
            mean=("tps_mean", "mean"),
            std=("tps_std",   "mean")
        ).reset_index()
        ax.errorbar(
            sub["output_length"], sub["mean"], yerr=sub["std"],
            label=f"Batch={bs}", color=COLORS[i], marker=MARKERS[i],
            linewidth=2, markersize=6, capsize=4
        )

    ax.set_xlabel("Output Length (tokens)")
    ax.set_ylabel("Throughput (Tokens per Second)")
    ax.set_title("TPS vs. Output Length\n(LLaMA 3.2, Apple M5, Ollama)")
    ax.legend(title="Batch Size", framealpha=0.9)
    ax.set_xticks(sorted(df["output_length"].unique()))
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, "fig2_tps_vs_output.png")


# ══════════════════════════════════════════════════════════════
# FIGURE 3 — E2E Latency Batch Scaling  (grouped bar)
# ══════════════════════════════════════════════════════════════
def fig_e2e_batch_scaling(df):
    # Use prompt=128, all output lengths for clean comparison
    configs = [(128, ol) for ol in sorted(df["output_length"].unique())]
    batch_sizes = sorted(df["batch_size"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(configs))
    w = 0.25
    offsets = [-w, 0, w]

    for i, bs in enumerate(batch_sizes):
        vals, errs = [], []
        for (pl, ol) in configs:
            row = df[(df["prompt_length"] == pl) &
                     (df["output_length"]  == ol) &
                     (df["batch_size"]     == bs)]
            vals.append(row["e2e_mean_ms"].values[0] / 1000 if len(row) else 0)
            errs.append(row["e2e_std_ms"].values[0]  / 1000 if len(row) else 0)

        ax.bar(x + offsets[i], vals, width=w, yerr=errs,
               label=f"Batch={bs}", color=COLORS[i], alpha=0.85,
               capsize=4, error_kw={"linewidth": 1.2})

    ax.set_xlabel("Output Length (tokens)  [Prompt fixed at 128]")
    ax.set_ylabel("End-to-End Latency (seconds)")
    ax.set_title("E2E Latency vs. Batch Size\n(LLaMA 3.2, Apple M5, Ollama)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ol}t" for (_, ol) in configs])
    ax.legend(title="Batch Size", framealpha=0.9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, "fig3_e2e_batch_scaling.png")


# ══════════════════════════════════════════════════════════════
# FIGURE 4 — ITL Heatmap (prompt x output, batch=1)
# ══════════════════════════════════════════════════════════════
def fig_itl_heatmap(df):
    sub = df[df["batch_size"] == 1].copy()
    pivot = sub.pivot_table(
        index="prompt_length", columns="output_length",
        values="itl_mean_ms", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c}t" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r}t" for r in pivot.index])
    ax.set_xlabel("Output Length (tokens)")
    ax.set_ylabel("Prompt Length (tokens)")
    ax.set_title("Inter-Token Latency — ITL (ms)\nHeatmap by Prompt × Output Length  [Batch=1]")

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color="black" if val < pivot.values.max() * 0.7 else "white")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("ITL (ms)")
    fig.tight_layout()
    save(fig, "fig4_itl_heatmap.png")


# ══════════════════════════════════════════════════════════════
# FIGURE 5 — TTFT Percentiles P50 / P95 / P99
# ══════════════════════════════════════════════════════════════
def fig_ttft_percentiles(df):
    # Group by prompt length (batch=1, averaged over output lengths)
    sub = df[df["batch_size"] == 1].groupby("prompt_length").agg(
        p50=("ttft_p50_ms", "mean"),
        p95=("ttft_p95_ms", "mean"),
        p99=("ttft_p99_ms", "mean"),
    ).reset_index()

    x  = np.arange(len(sub))
    w  = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.bar(x - w, sub["p50"], width=w, label="P50 (median)",
           color="#2563eb", alpha=0.85)
    ax.bar(x,     sub["p95"], width=w, label="P95",
           color="#d97706", alpha=0.85)
    ax.bar(x + w, sub["p99"], width=w, label="P99",
           color="#dc2626", alpha=0.85)

    ax.set_xlabel("Prompt Length (tokens)")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("TTFT Percentiles (P50 / P95 / P99) by Prompt Length\n(Batch=1, LLaMA 3.2, Apple M5)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}t" for p in sub["prompt_length"]])
    ax.legend(framealpha=0.9)
    ax.set_ylim(bottom=0)

    # Add value labels on bars
    for bar in ax.patches:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                    f"{h:.0f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    save(fig, "fig5_ttft_percentiles.png")


# ══════════════════════════════════════════════════════════════
# FIGURE 6 — TPS/item (normalized per-request throughput)
# ══════════════════════════════════════════════════════════════
def fig_tps_per_item(df):
    batch_sizes = sorted(df["batch_size"].unique())
    fig, ax = plt.subplots(figsize=(7, 4.5))

    output_lengths = sorted(df["output_length"].unique())
    x = np.arange(len(output_lengths))
    w = 0.25
    offsets = [-w, 0, w]

    for i, bs in enumerate(batch_sizes):
        vals = []
        for ol in output_lengths:
            sub = df[(df["batch_size"] == bs) & (df["output_length"] == ol)]
            vals.append(sub["tps_per_item_mean"].mean() if len(sub) else 0)
        ax.bar(x + offsets[i], vals, width=w,
               label=f"Batch={bs}", color=COLORS[i], alpha=0.85)

    ax.set_xlabel("Output Length (tokens)")
    ax.set_ylabel("TPS per Batch Item (tokens/sec/request)")
    ax.set_title("Per-Request Throughput vs. Output Length\n(LLaMA 3.2, Apple M5, Ollama)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ol}t" for ol in output_lengths])
    ax.legend(title="Batch Size", framealpha=0.9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, "fig6_tps_per_item.png")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    # Auto-find latest CSV if no argument given
    if len(sys.argv) < 2:
        results_dir = "results"
        if not os.path.isdir(results_dir):
            print("No 'results/' folder found. Run benchmark_v2.py first.")
            sys.exit(1)
        csvs = sorted([
            os.path.join(results_dir, f)
            for f in os.listdir(results_dir)
            if f.endswith(".csv") and "benchmark_v2" in f
        ])
        if not csvs:
            # Fall back to any CSV
            csvs = sorted([
                os.path.join(results_dir, f)
                for f in os.listdir(results_dir)
                if f.endswith(".csv")
            ])
        if not csvs:
            print("No CSV files found in results/. Run benchmark_v2.py first.")
            sys.exit(1)
        csv_path = csvs[-1]
        print(f"Auto-selected: {csv_path}")
    else:
        csv_path = sys.argv[1]

    df = load(csv_path)

    print("\nGenerating figures...")
    fig_ttft_vs_prompt(df)
    fig_tps_vs_output(df)
    fig_e2e_batch_scaling(df)
    fig_itl_heatmap(df)
    fig_ttft_percentiles(df)
    fig_tps_per_item(df)

    print(f"\n✓ All 6 figures saved to '{OUT_DIR}/' folder.")
    print("  Use these in your paper — replace the old figures.")
    print("\n  Paper figure mapping:")
    print("    fig1_ttft_vs_prompt.png    → Figure 1 (TTFT analysis)")
    print("    fig2_tps_vs_output.png     → Figure 2 (TPS analysis)")
    print("    fig3_e2e_batch_scaling.png → Figure 3 (E2E / batch scaling)")
    print("    fig4_itl_heatmap.png       → Figure 4 (ITL heatmap)")
    print("    fig5_ttft_percentiles.png  → Figure 5 (percentiles)")
    print("    fig6_tps_per_item.png      → Figure 6 (normalized TPS)")

if __name__ == "__main__":
    main()
