"""Pareto plot (recall vs ms/image) across every Task B configuration:
single detectors, multi-scale variants, and detector unions. Reads only from
results/*.csv -- no hand-typed numbers."""
import os

import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results")


def main():
    ms = pd.read_csv(os.path.join(RESULTS_DIR, "improve_multiscale.csv"))
    union = pd.read_csv(os.path.join(RESULTS_DIR, "improve_union.csv"))

    points = []
    for _, r in ms.iterrows():
        points.append({"label": f"{r['detector']} ({r['config']})", "kind": "single/multi-scale",
                        "recall": r["recall_overall"], "ms": r["ms_per_image"]})
    for _, r in union.iterrows():
        points.append({"label": r["config"], "kind": "union",
                        "recall": r["recall_overall"], "ms": r["ms_per_image"]})

    df = pd.DataFrame(points)

    plt.figure(figsize=(9, 6))
    colors = {"single/multi-scale": "#2E8B74", "union": "#C9A227"}
    for kind, group in df.groupby("kind"):
        plt.scatter(group["ms"], group["recall"], label=kind, s=90, color=colors[kind], alpha=0.85)
        for _, r in group.iterrows():
            plt.annotate(r["label"], (r["ms"], r["recall"]), fontsize=7,
                         textcoords="offset points", xytext=(6, 4))

    plt.xscale("log")
    plt.xlabel("Latency (ms/image, log scale)")
    plt.ylabel("Overall recall")
    plt.title("Recall vs Latency across every Task B configuration (CPU-only)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "pareto_recall_vs_latency.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")

    # find the actual Pareto-optimal set (no other point is both faster AND higher recall)
    df_sorted = df.sort_values("ms")
    pareto = []
    best_recall = -1
    for _, r in df_sorted.iterrows():
        if r["recall"] > best_recall:
            pareto.append(r["label"])
            best_recall = r["recall"]
    print("Pareto-optimal configurations (faster options with equal/better recall don't exist for these):")
    for p in pareto:
        print(f"  {p}")

    df["recall_per_ms"] = df["recall"] / df["ms"]
    df.sort_values("recall_per_ms", ascending=False).to_csv(
        os.path.join(RESULTS_DIR, "recall_per_ms_ranking.csv"), index=False)
    best = df.sort_values("recall_per_ms", ascending=False).iloc[0]
    print(f"\nBest recall-per-millisecond configuration: {best['label']} "
          f"(recall={best['recall']:.4f}, ms/image={best['ms']:.1f}, recall/ms={best['recall_per_ms']:.5f})")


if __name__ == "__main__":
    main()
