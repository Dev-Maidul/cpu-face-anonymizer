"""
Generates every Markdown table used in README.md directly from results/*.csv.
No number in the README should be hand-typed -- this script is the single
source of truth for all of them. Run after evaluate.py / improve.py /
privacy_check.py / benchmark.py have all produced their CSVs.
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")


def df_to_md(df, float_fmt="{:.3f}"):
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].map(lambda v: float_fmt.format(v) if pd.notna(v) else "n/a")
    return df.to_markdown(index=False)


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)

    # --- recall by size bucket (the headline table) ---
    bucket = pd.read_csv(os.path.join(RESULTS_DIR, "eval_by_bucket.csv"))
    recall_size = bucket[bucket["metric_type"] == "recall_by_size"].copy()
    pivot = recall_size.pivot(index="bucket", columns="detector", values="recall")
    pivot = pivot.reindex(["<20px", "20-50px", "50-100px", ">100px"])
    gt_counts = recall_size[recall_size["detector"] == recall_size["detector"].iloc[0]][["bucket", "gt_count"]]
    gt_counts = gt_counts.set_index("bucket").reindex(pivot.index)
    pivot.insert(0, "gt_faces_in_bucket", gt_counts["gt_count"].astype(int))
    with open(os.path.join(TABLES_DIR, "recall_by_size.md"), "w") as f:
        f.write(df_to_md(pivot.reset_index()))

    # --- overall precision/recall/F1 with CI ---
    overall = bucket[bucket["metric_type"] == "overall"][
        ["detector", "precision", "recall", "f1", "precision_ci_lo", "precision_ci_hi", "recall_ci_lo", "recall_ci_hi"]
    ]
    with open(os.path.join(TABLES_DIR, "overall_with_ci.md"), "w") as f:
        f.write(df_to_md(overall))

    # --- recall by blur level ---
    recall_blur = bucket[bucket["metric_type"] == "recall_by_blur"].copy()
    pivot_blur = recall_blur.pivot(index="bucket", columns="detector", values="recall")
    pivot_blur = pivot_blur.reindex(["clear", "normal", "heavy"])
    with open(os.path.join(TABLES_DIR, "recall_by_blur.md"), "w") as f:
        f.write(df_to_md(pivot_blur.reset_index()))

    # --- recall-at-fixed-precision ---
    rap_path = os.path.join(RESULTS_DIR, "recall_at_precision.csv")
    if os.path.isfile(rap_path):
        rap = pd.read_csv(rap_path)
        with open(os.path.join(TABLES_DIR, "recall_at_precision.md"), "w") as f:
            f.write(df_to_md(rap))

    # --- multi-scale improvement ---
    ms_path = os.path.join(RESULTS_DIR, "improve_multiscale.csv")
    if os.path.isfile(ms_path):
        ms = pd.read_csv(ms_path)
        with open(os.path.join(TABLES_DIR, "improve_multiscale.md"), "w") as f:
            f.write(df_to_md(ms))

    # --- detector union ---
    union_path = os.path.join(RESULTS_DIR, "improve_union.csv")
    if os.path.isfile(union_path):
        union = pd.read_csv(union_path)
        with open(os.path.join(TABLES_DIR, "improve_union.md"), "w") as f:
            f.write(df_to_md(union))

    # --- privacy check ---
    priv_path = os.path.join(RESULTS_DIR, "privacy_check.csv")
    if os.path.isfile(priv_path):
        priv = pd.read_csv(priv_path)
        pivot_priv = priv.pivot(index="setting", columns="detector", values="redetection_rate")
        distortion = priv.groupby("setting")["mean_distortion"].first()
        pivot_priv.insert(0, "mean_distortion", distortion)
        with open(os.path.join(TABLES_DIR, "privacy_check.md"), "w") as f:
            f.write(df_to_md(pivot_priv.reset_index()))

    # --- benchmark ---
    bench_path = os.path.join(RESULTS_DIR, "benchmark.csv")
    if os.path.isfile(bench_path):
        bench = pd.read_csv(bench_path)
        with open(os.path.join(TABLES_DIR, "benchmark.md"), "w") as f:
            f.write(df_to_md(bench))

    print(f"Tables written to {TABLES_DIR}/")
    for fn in sorted(os.listdir(TABLES_DIR)):
        print(f"  {fn}")


if __name__ == "__main__":
    main()
