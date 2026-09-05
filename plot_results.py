"""
Turn a benchmark CSV into figures for the README / a writeup.

Produces three plots from results/benchmark_<model>.csv:
  1. Latency vs batch size (per generation length)
  2. Throughput (tokens/sec) vs batch size
  3. Latency vs generation length (per batch size)

Usage:
    python plot_results.py results/benchmark_distilgpt2.csv
"""
import sys
import os

import pandas as pd
import matplotlib.pyplot as plt


def main(csv_path):
    df = pd.read_csv(csv_path)
    device = df["device"].iloc[0]
    model = df["model"].iloc[0]
    os.makedirs("results", exist_ok=True)
    stem = os.path.splitext(os.path.basename(csv_path))[0]

    # 1. Latency vs batch size, one line per generation length.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for nt, grp in df.groupby("new_tokens"):
        grp = grp.sort_values("batch_size")
        ax.plot(grp["batch_size"], grp["mean_latency_s"] * 1000,
                marker="o", label=f"{nt} new tokens")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Mean generation latency (ms)")
    ax.set_title(f"Inference latency vs batch size\n{model} on {device}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"results/{stem}_latency_vs_batch.png", dpi=140)

    # 2. Throughput vs batch size.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for nt, grp in df.groupby("new_tokens"):
        grp = grp.sort_values("batch_size")
        ax.plot(grp["batch_size"], grp["throughput_tokens_per_s"],
                marker="s", label=f"{nt} new tokens")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title(f"Decode throughput vs batch size\n{model} on {device}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"results/{stem}_throughput_vs_batch.png", dpi=140)

    # 3. Latency vs generation length.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for bs, grp in df.groupby("batch_size"):
        grp = grp.sort_values("new_tokens")
        ax.plot(grp["new_tokens"], grp["mean_latency_s"] * 1000,
                marker="^", label=f"batch {bs}")
    ax.set_xlabel("Generation length (new tokens)")
    ax.set_ylabel("Mean generation latency (ms)")
    ax.set_title(f"Inference latency vs generation length\n{model} on {device}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"results/{stem}_latency_vs_length.png", dpi=140)

    print(f"Wrote 3 figures to results/ for {stem}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_results.py results/benchmark_<model>.csv")
        sys.exit(1)
    main(sys.argv[1])
