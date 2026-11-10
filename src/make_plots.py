"""Combine benchmark.csv + evaluation.csv into the report-ready table and plots."""
import os

import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results")

bench = pd.read_csv(os.path.join(RESULTS_DIR, "benchmark.csv"))
eval_df = pd.read_csv(os.path.join(RESULTS_DIR, "evaluation.csv"))

combined = pd.merge(bench, eval_df, on="detector")
combined_path = os.path.join(RESULTS_DIR, "combined_results.csv")
combined.to_csv(combined_path, index=False)
print("Combined table:")
print(combined.to_string(index=False))
print(f"Saved -> {combined_path}")

# ---- Plot 1: Accuracy (F1) vs Speed (FPS) trade-off scatter ----
plt.figure(figsize=(6, 5))
for _, row in combined.iterrows():
    plt.scatter(row["fps"], row["f1"], s=120)
    plt.annotate(row["detector"], (row["fps"], row["f1"]),
                 textcoords="offset points", xytext=(8, 5), fontsize=9)
plt.xlabel("Speed (FPS, CPU-only)")
plt.ylabel("Accuracy (F1 score)")
plt.title("Accuracy vs Speed Trade-off (CPU-only inference)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "accuracy_vs_speed.png"), dpi=150)
plt.close()

# ---- Plot 2: CPU usage bar chart ----
plt.figure(figsize=(6, 4))
plt.bar(combined["detector"], combined["cpu_percent"], color="#4C72B0")
plt.ylabel("CPU usage (%)")
plt.title("CPU Usage per Detector")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "cpu_usage.png"), dpi=150)
plt.close()

# ---- Plot 3: Model size vs F1 ----
plt.figure(figsize=(6, 4))
plt.bar(combined["detector"], combined["model_size_mb"], color="#55A868")
plt.ylabel("Model size (MB)")
plt.title("Model Size per Detector")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "model_size.png"), dpi=150)
plt.close()

print("Saved plots: accuracy_vs_speed.png, cpu_usage.png, model_size.png")
