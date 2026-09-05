"""
Overlay two or more benchmark CSVs to compare a variable (e.g. fp16 vs fp32,
or MPS vs CPU). Produces a throughput comparison and a memory comparison.

Usage:
    python compare.py results/benchmark_..._mps_fp16.csv results/benchmark_..._mps_fp32.csv
"""
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt


def label_for(df):
    """A short legend label from whatever differs (dtype/device)."""
    return f"{df['device'].iloc[0]} {df['dtype'].iloc[0]}"


def main(csv_paths):
    frames = [(p, pd.read_csv(p)) for p in csv_paths]
    model = frames[0][1]["model"].iloc[0].split("/")[-1]
    os.makedirs("results", exist_ok=True)

    # Pick the longest generation length present, for a clean single comparison.
    longest = max(df["new_tokens"].max() for _, df in frames)

    # 1. Throughput vs batch size, one line per CSV (at the longest length).
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, df in frames:
        sub = df[df["new_tokens"] == longest].sort_values("batch_size")
        ax.plot(sub["batch_size"], sub["throughput_tokens_per_s"],
                marker="o", label=label_for(df))
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title(f"Throughput comparison — {model}, {longest} new tokens")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/compare_throughput.png", dpi=140)

    # 2. Peak memory per CSV (bar chart) — dtype's headline effect.
    fig, ax = plt.subplots(figsize=(6, 4.5))
    labels = [label_for(df) for _, df in frames]
    mems = [df["peak_mem_mb"].max() for _, df in frames]
    bars = ax.bar(labels, mems, color=["#2b8cbe", "#e34a33", "#31a354"][:len(mems)])
    ax.set_ylabel("Peak memory (MB)")
    ax.set_title(f"Memory footprint — {model}")
    for b, m in zip(bars, mems):
        ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.0f}",
                ha="center", va="bottom")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/compare_memory.png", dpi=140)

    print("Wrote results/compare_throughput.png and results/compare_memory.png")
    # Print a small comparison table to stdout.
    for p, df in frames:
        peak = df["throughput_tokens_per_s"].max()
        print(f"  {label_for(df):12}  peak throughput {peak:7.1f} tok/s  "
              f"peak mem {df['peak_mem_mb'].max():6.0f} MB")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare.py <csv1> <csv2> [<csv3> ...]")
        sys.exit(1)
    main(sys.argv[1:])
