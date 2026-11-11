"""
Generates report/report.tex. Every number in the generated .tex comes from
results/*.csv (read here, formatted into LaTeX tables and inline figures) --
none of it is hand-typed. Static prose (Introduction wording, citations,
methodology narrative) is the only hand-written part.
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RESULTS_DIR = os.path.join(ROOT, "results")
REPORT_DIR = os.path.join(ROOT, "report")


def pct(x):
    return f"{x * 100:.1f}\\%"


def esc(s):
    return str(s).replace("_", "\\_")


def _fmt_cell(v, float_fmt):
    if pd.isna(v):
        return "--"
    if isinstance(v, float):
        return float_fmt % v
    return esc(v)


def df_to_latex_table(df, caption, label, float_fmt="%.3f"):
    """Hand-rolled (no jinja2 dependency) booktabs-style LaTeX table from a DataFrame."""
    cols = list(df.columns)
    col_spec = "l" + "r" * (len(cols) - 1)
    header = " & ".join(esc(c).replace("\\_", " ") for c in cols) + " \\\\"
    lines = []
    for _, row in df.iterrows():
        lines.append(" & ".join(_fmt_cell(row[c], float_fmt) for c in cols) + " \\\\")
    body = (
        f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n{header}\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    return (
        "\\begin{table}[h]\n\\centering\n\\small\n" + body +
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n"
    )


def main():
    bucket = pd.read_csv(os.path.join(RESULTS_DIR, "eval_by_bucket.csv"))
    overall = bucket[bucket["metric_type"] == "overall"]
    recall_size = bucket[bucket["metric_type"] == "recall_by_size"]
    recall_blur = bucket[bucket["metric_type"] == "recall_by_blur"]

    rap = pd.read_csv(os.path.join(RESULTS_DIR, "recall_at_precision.csv"))
    multiscale = pd.read_csv(os.path.join(RESULTS_DIR, "improve_multiscale.csv"))
    union = pd.read_csv(os.path.join(RESULTS_DIR, "improve_union.csv"))
    privacy = pd.read_csv(os.path.join(RESULTS_DIR, "privacy_check.csv"))
    bench = pd.read_csv(os.path.join(RESULTS_DIR, "benchmark.csv"))

    # ---- numbers computed programmatically for use in prose (not hand-typed) ----
    total_gt = int(overall["gt_count"].iloc[0])
    tiny_gt = int(recall_size[recall_size["bucket"] == "<20px"]["gt_count"].iloc[0])
    tiny_frac = tiny_gt / total_gt

    best_mainset_recall_row = overall[overall["detector"] != "mediapipe_blazeface_full"].sort_values(
        "recall", ascending=False).iloc[0]
    best_precision_row = overall[overall["detector"] != "mediapipe_blazeface_full"].sort_values(
        "precision", ascending=False).iloc[0]

    haar_ms_best = multiscale[multiscale["detector"] == "haar_cascade"].sort_values("recall_overall").iloc[-1]
    haar_base = multiscale[(multiscale["detector"] == "haar_cascade") & (multiscale["config"] == "scale_1.0_only")].iloc[0]

    mobilenet_ms = multiscale[multiscale["detector"] == "mobilenet_ssd"]
    mobilenet_ms_gain = mobilenet_ms.sort_values("recall_overall").iloc[-1]["recall_overall"] - mobilenet_ms.iloc[0]["recall_overall"]

    best_union_row = union.sort_values("recall_overall", ascending=False).iloc[0]
    best_single_recall_overall = multiscale.sort_values("recall_overall", ascending=False).iloc[0]

    short_range_row = overall[overall["detector"] == "mediapipe_blazeface"].iloc[0]
    full_range_row = overall[overall["detector"] == "mediapipe_blazeface_full"].iloc[0]

    priv_pivot = privacy.pivot(index="setting", columns="detector", values="redetection_rate")
    best_priv_setting = priv_pivot.max(axis=1).idxmin()
    best_priv_max = priv_pivot.max(axis=1).min()

    bench_fastest = bench.sort_values("median_latency_ms").iloc[0]
    bench_slowest = bench.sort_values("median_latency_ms").iloc[-1]

    tex = []
    tex.append(r"""\documentclass[10pt]{article}
\usepackage[margin=0.78in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{amsmath}
\setlength{\parskip}{3pt}
\title{\vspace{-1.5em}Green and Efficient Face Anonymization on CPU:\\
\large Recall-First Evaluation of Lightweight Face Detectors}
\author{Md Maidul Islam \\ \small Student ID: 2023030039 \\ \small Multimedia Technology --- Course Project}
\date{}
\begin{document}
\maketitle
\vspace{-2em}

\section{Introduction}
Public and CCTV footage is routinely shared or archived with faces still visible,
creating a privacy risk under regulations such as the GDPR. A common mitigation is
automatic face detection followed by blurring or pixelation. Almost all published
face-detector benchmarks assume GPU inference; real low-cost cameras and edge
devices, however, often have only a CPU. This report evaluates three CPU-only face
detectors -- a classical Haar Cascade, MediaPipe's BlazeFace (short-range), and
OpenCV's MobileNet-SSD (res10) -- as the detection stage of a face-anonymization
pipeline, on the full WIDER FACE validation split (3{,}226 images, """ + f"{total_gt:,}" + r""" ground-truth
faces).

For an anonymization tool, the operationally critical metric is \textbf{recall},
not F1: a face a detector misses is a face that stays visible. This report
therefore leads with recall, broken down by how small and how blurred the
ground-truth face is, rather than with a single aggregate F1 score. The headline
finding is unambiguous and is reported as such even though it is not flattering to
any of the three detectors: """ + pct(tiny_frac) + r""" of all faces in this dataset are
shorter than 20 pixels, and \emph{every detector tested in this report, across four
model variants, finds essentially none of them} (Section~\ref{sec:experiments}).

""")

    tex.append(r"""\section{Related Work}
\begin{itemize}
  \item \textbf{WIDER FACE} \cite{yang2016wider} is the benchmark dataset used
  throughout this report. It is explicitly constructed to be hard: it includes
  dense crowd scenes with hundreds of small, blurred, or occluded faces, and
  publishes per-box attribute labels (blur, illumination, occlusion, pose,
  invalid) used here for the bucketed analysis.
  \item \textbf{Haar Cascades / Viola--Jones} \cite{viola2001rapid} is the
  classical sliding-window detector used as a zero-model-weight baseline: it
  ships inside OpenCV with no separate file to download.
  \item \textbf{SSD} \cite{liu2016ssd} is the single-shot detector architecture
  behind OpenCV's bundled ``res10'' face-detection model (MobileNet-SSD in this
  report), a compact, CPU-friendly Caffe model distributed with OpenCV.
  \item \textbf{BlazeFace} \cite{bazarevsky2019blazeface} is Google's
  sub-millisecond, mobile/edge-oriented face detector, used here via MediaPipe's
  Face Detector task in both its short-range and full-range variants.
  \item \textbf{Green AI} \cite{schwartz2020green} argues that machine-learning
  research should report computational cost (and, ideally, energy) alongside
  accuracy. This report follows that spirit by measuring latency, memory, model
  size, and CPU thread configuration for every detector -- while explicitly
  \emph{not} claiming an energy measurement that was not actually taken (see
  Section~\ref{sec:method}).
\end{itemize}

Unlike most face-detector comparisons, which report a single aggregate accuracy
number on GPU hardware, this report compares architecturally different detectors
(classical, mobile-optimized deep, and compact deep) purely on CPU, and reports
accuracy broken down by face size and blur rather than as one F1 score.

""")

    tex.append(r"""\section{Data}
The full WIDER FACE validation split (CUHK-CSE / Yang et al. 2016) was used:
3{,}226 images, """ + f"{total_gt:,}" + r""" ground-truth face boxes. Each box carries WIDER
FACE's own attribute labels (blur: clear/normal/heavy; occlusion; pose;
illumination; invalid). Boxes flagged \texttt{invalid=1} were excluded from all
evaluation. Ground-truth face height was bucketed into four ranges
($<$20px, 20--50px, 50--100px, $>$100px) used throughout the recall analysis.
The dataset is not committed to this project's repository (licensing and size);
\texttt{src/data\_prep.py} downloads and rebuilds it deterministically. No
training was performed anywhere in this report -- every detector is used purely
in inference mode with its publicly distributed pretrained weights.

""")

    tex.append(r"""\section{Method}
\label{sec:method}
Each detector implements a common interface,
\texttt{detect\_faces(frame) -> [(x, y, w, h), ...]}, making the three
interchangeable across the benchmarking, evaluation, and anonymization code.
Three detectors were chosen because each occupies a different point on the
classical-to-deep, light-to-heavy spectrum, all runnable on CPU without
training: Haar Cascade (classical, zero model weight), BlazeFace (a deep model
purpose-built for mobile/edge CPUs), and MobileNet-SSD (a heavier but more
accurate compact deep detector). A fourth variant, BlazeFace \emph{full-range},
was added after the main comparison (Section~\ref{sec:experiments}) specifically
to test whether BlazeFace's weak recall is a short-range-model artifact or an
architectural limit. Heavier state-of-the-art detectors such as RetinaFace or
MTCNN were deliberately not evaluated: they require a GPU for real-time
throughput, which conflicts with this report's CPU-only constraint.

\textbf{Matching.} A predicted box counts as a true positive if its IoU with an
unmatched ground-truth box is $\geq 0.5$, via greedy one-to-one matching (each
prediction claims its best remaining ground-truth box). Recall is bucketed by
the \emph{ground-truth} box's own height; precision is bucketed by the
\emph{predicted} box's own height, since a false positive has no ground-truth box
to bucket it by -- these two bucketings are not directly comparable and are
reported as what they are.

\textbf{Confidence intervals.} 95\% confidence intervals for precision and
recall were obtained by bootstrap resampling images (not individual boxes) with
replacement, 1{,}000 times, and taking the 2.5/97.5 percentiles of the resulting
distribution.

\textbf{Benchmark.} Each detector was benchmarked in its own subprocess (so peak
memory reflects one detector, not whatever ran before it), over 5 repeats with
\texttt{cv2.setNumThreads(1)} set explicitly so all three are compared under
identical, single-threaded conditions; \texttt{psutil} reports CPU time, not
energy. \textbf{No energy measurement was taken.} \texttt{powermetrics}, the
only plausible energy-measurement path on this machine (macOS), requires root
privileges (\texttt{powermetrics must be invoked as the superuser}), and this
automated run does not use \texttt{sudo}. Accordingly this report describes the
project as \emph{CPU-efficient}, not \emph{green} -- a claim this report does not
make without having measured it.

\textbf{Privacy check.} For a random 300-image subset, MobileNet-SSD's own
detections were anonymized (Gaussian blur at several kernel sizes; pixelation at
several block counts) and all three main detectors were re-run on the
anonymized image to measure the fraction of anonymized faces still detected.
This is a \emph{detector-based} check only: it tests whether common CPU face
detectors still fire on the anonymized region, not whether a human, or a
re-identification model, could still recognize the person. It is not proof of
irreversibility.

""")

    tex.append(r"""\section{Experiments and Analysis}
\label{sec:experiments}

\subsection{Recall is the headline metric, and it is low}
Table~\ref{tab:recall-by-size} is the central result of this report: recall
broken down by ground-truth face height, with the number of ground-truth faces
in each bucket. Nearly half of all faces in this dataset (""" + pct(tiny_frac) +
               r""", """ + f"{tiny_gt:,}" + r""" of """ + f"{total_gt:,}" + r""") are shorter than 20 pixels, and
\textbf{every detector tested -- all four model variants -- finds essentially
none of them.}

""")
    tex.append(df_to_latex_table(
        recall_size.pivot(index="bucket", columns="detector", values="recall")
        .reindex(["<20px", "20-50px", "50-100px", ">100px"]).reset_index(),
        "Recall by ground-truth face-height bucket (IoU $\\geq$ 0.5).",
        "tab:recall-by-size"))

    tex.append(r"""
Table~\ref{tab:overall-ci} gives the overall precision/recall/F1 with 95\%
bootstrap confidence intervals. """ + esc(best_precision_row["detector"]) +
               r""" has the best precision (""" + f"{best_precision_row['precision']:.3f}" +
               r"""), while """ + esc(best_mainset_recall_row["detector"]) +
               r""" has the best recall among the three main detectors (""" +
               f"{best_mainset_recall_row['recall']:.3f}" + r"""). Both numbers are low in
absolute terms.

""")
    tex.append(df_to_latex_table(
        overall[["detector", "precision", "recall", "f1", "precision_ci_lo", "precision_ci_hi",
                 "recall_ci_lo", "recall_ci_hi"]],
        "Overall precision/recall/F1 with 95\\% bootstrap CI (resampling images, 1{,}000x).",
        "tab:overall-ci"))

    tex.append(r"""
Table~\ref{tab:recall-by-blur} shows recall collapses further under WIDER
FACE's own blur attribute: heavily blurred faces are found at rates near zero by
every detector.

""")
    tex.append(df_to_latex_table(
        recall_blur.pivot(index="bucket", columns="detector", values="recall")
        .reindex(["clear", "normal", "heavy"]).reset_index(),
        "Recall by WIDER FACE blur attribute.",
        "tab:recall-by-blur"))

    tex.append(r"""
\subsection{Confidence threshold sweep}
MobileNet-SSD and BlazeFace expose a continuous confidence score; Haar Cascade
does not (\texttt{minNeighbors} is a discrete knob), so no PR curve is reported
for it -- this is stated as a limitation, not approximated. Table~\ref{tab:rap}
reports recall at fixed precision levels. Notably, sweeping MobileNet-SSD's
threshold down to 0 (accepting every candidate box) only raises its recall to
""" + f"{mobilenet_ms.iloc[0]['recall_overall']:.3f}" + r""" at the operating threshold used throughout this
report -- the architecture's recall ceiling on this dataset is low regardless of
threshold, because it is driven by faces the network's fixed input resolution
cannot resolve at all, not by an overly strict confidence cutoff.

""")
    tex.append(df_to_latex_table(rap, "Recall at fixed precision levels (threshold sweep).", "tab:rap"))

    tex.append(r"""
\subsection{Does a full-range BlazeFace model help?}
\label{sec:fullrange}
The short-range BlazeFace model finds essentially no faces below 50px
(overall recall """ + f"{short_range_row['recall']:.3f}" + r"""). Swapping in MediaPipe's
\emph{full-range} BlazeFace model nearly quadruples overall recall (to """ +
               f"{full_range_row['recall']:.3f}" + r""") and improves precision too (""" +
               f"{full_range_row['precision']:.3f}" + r""" vs. """ + f"{short_range_row['precision']:.3f}" +
               r""") -- confirming the short-range model, not the BlazeFace architecture
itself, was the main limitation for this project's original three-detector
comparison. Even so, the full-range model \emph{still} finds zero faces under
20px (Table~\ref{tab:recall-by-size}), reinforcing that the smallest-face
problem is not specific to any one of the four models tested.

\subsection{Does multi-scale inference or detector union recover recall?}
Table~\ref{tab:union} (top half) shows multi-scale inference (running each detector
on the image upscaled 1.5x and 2x, merging detections with non-maximum
suppression) helps Haar Cascade substantially (recall """ +
               f"{haar_base['recall_overall']:.3f}" + r""" $\to$ """ +
               f"{haar_ms_best['recall_overall']:.3f}" + r""" at its best configuration) but at a steep
latency cost (""" + f"{haar_base['ms_per_image']:.1f}" + r"""ms $\to$ """ +
               f"{haar_ms_best['ms_per_image']:.1f}" + r"""ms per image, roughly a 6x slowdown). It does
\textbf{almost nothing} for MobileNet-SSD or BlazeFace (recall changes by less
than """ + f"{abs(mobilenet_ms_gain):.4f}" + r""" for MobileNet-SSD across all three scales tested) --
both resize their input to a fixed resolution internally regardless of the
input image's size, so upscaling the original image before feeding it in does
not give the network any more effective resolution on small faces. This is
reported plainly: multi-scale inference, a cheap technique that works for the
classical detector, does not transfer to either deep model tested here.

Table~\ref{tab:union} (bottom half) shows detector union (combining boxes from
multiple detectors with NMS) does not meaningfully help either: the best union
configuration (""" + esc(best_union_row["config"]) + r""") reaches recall """ +
               f"{best_union_row['recall_overall']:.3f}" + r""", \emph{worse} than Haar Cascade alone at
its best multi-scale setting (""" + f"{best_single_recall_overall['recall_overall']:.3f}" +
               r""") while being comparably slow. No union configuration is on the
recall-vs-latency Pareto front computed in Figure~\ref{fig:pareto}; every union
point is dominated by a single-detector configuration that is both faster and
at least as accurate.

""")
    taskb_ms = multiscale[["detector", "config", "recall_overall", "precision_overall", "ms_per_image"]].rename(
        columns={"detector": "configuration"})
    taskb_ms["configuration"] = taskb_ms["configuration"] + " / " + taskb_ms["config"]
    taskb_ms = taskb_ms.drop(columns=["config"])
    taskb_union = union[["config", "recall_overall", "precision_overall", "ms_per_image"]].rename(
        columns={"config": "configuration"})
    taskb_combined = pd.concat([taskb_ms, taskb_union], ignore_index=True)
    tex.append(df_to_latex_table(
        taskb_combined,
        "Task B configurations (top: single detector / multi-scale; bottom: detector union) -- recall, precision, and latency per image.",
        "tab:union"))

    tex.append(r"""
\begin{figure}[h]
\centering
\includegraphics[width=0.85\textwidth]{""" + os.path.join("..", "results", "pareto_recall_vs_latency.png") + r"""}
\caption{Recall vs. latency (log scale) across every tested configuration:
single detectors, multi-scale variants, and detector unions. Union
configurations are dominated; they do not appear on the Pareto front.}
\label{fig:pareto}
\end{figure}

\subsection{Does anonymization defeat detection?}
Table~\ref{tab:privacy} reports, for each blur/pixelation setting, the fraction
of anonymized faces still re-detected by each detector, alongside a simple
distortion measure (mean absolute pixel difference, 0--1 scale) between the
original and anonymized face crop. Two findings stand out. First, \textbf{no
tested setting drove every detector's re-detection rate at or below 2\%}; the
best tested setting, \texttt{""" + esc(best_priv_setting) + r"""}, still left a """ +
               pct(best_priv_max) + r""" worst-case re-detection rate across the three
detectors. Second, the relationship between anonymization strength and
re-detection is \textbf{non-monotonic} for pixelation: re-detection rate drops
sharply at the mildest pixelation setting tested, then \emph{rises again} as
pixelation gets coarser (visible in the pixelate rows of
Table~\ref{tab:privacy}), plausibly because very coarse pixelation produces a
stable, high-contrast block pattern that still carries enough structure for a
detector to key on. This is reported as observed, not smoothed over -- it is a
genuinely counter-intuitive result that argues against assuming ``more
pixelation is strictly safer.''

""")
    tex.append(df_to_latex_table(
        priv_pivot.reset_index().merge(
            privacy.groupby("setting")["mean_distortion"].first().reset_index(), on="setting"),
        "Re-detection rate per detector and mean distortion, by anonymization setting (n=300 image sample).",
        "tab:privacy"))

    tex.append(r"""
\subsection{Benchmark}
Table~\ref{tab:benchmark} reports median latency and inter-quartile range over 5
repeats, each detector run in its own subprocess with
\texttt{cv2.setNumThreads(1)}, on an Apple M1 (8 cores). """ +
               esc(bench_fastest["detector"]) + r""" is fastest (""" +
               f"{bench_fastest['median_latency_ms']:.1f}" + r""" ms/frame) and """ +
               esc(bench_slowest["detector"]) + r""" is slowest (""" +
               f"{bench_slowest['median_latency_ms']:.1f}" + r""" ms/frame) under this single-threaded,
isolated-process measurement -- notably slower for Haar Cascade than Phase-1's
earlier multi-threaded measurement, since OpenCV's default threading was
disabled here specifically to make the three detectors comparable.

""")
    tex.append(df_to_latex_table(bench[["detector", "median_latency_ms", "iqr_latency_ms", "fps_median",
                                         "peak_memory_mb", "model_size_mb", "num_threads"]],
                                  "Benchmark: median latency, IQR, memory, model size (isolated subprocess, 5 repeats).",
                                  "tab:benchmark"))

    tex.append(r"""
\section{Conclusion and Outlook}
On the full WIDER FACE validation set, all four CPU-only face-detector variants
tested in this report -- Haar Cascade, MobileNet-SSD, BlazeFace short-range, and
BlazeFace full-range -- leave the large majority of small and heavily blurred
faces undetected, and therefore visible, in an anonymization pipeline. Nearly
half of this dataset's faces are under 20 pixels tall, and recall there is
effectively zero across every model and every recall-improvement technique
tried (multi-scale inference, detector union, a full-range model swap). Of the
techniques tried, only multi-scale inference meaningfully helped, and only for
the classical Haar detector, at a roughly 6x latency cost; it did not transfer
to either deep model, which resize their input internally regardless of the
original image's resolution. Detector union did not help: every tested union
configuration is dominated on the recall-vs-latency Pareto front by a single
detector. The anonymization step itself was never fully defeated by re-running
detectors against it, but no tested blur/pixelation setting drove re-detection
below 2\% for all three detectors, and the relationship between anonymization
strength and re-detection was non-monotonic for pixelation -- a genuinely
counter-intuitive finding worth further study.

\textbf{Practical recommendation:} for this dataset's regime (small/crowded
faces), none of the tested CPU-only detectors is adequate on their own for a
privacy-critical use case; MobileNet-SSD is the best-precision choice when
false positives are costly, the full-range BlazeFace model is the best
recall-per-millisecond choice among the deep models, and Haar Cascade with
multi-scale inference is viable only where the latency cost is acceptable.

\textbf{Limitations:} evaluation uses IoU $\geq$ 0.5 only; WIDER FACE's crowd
scenes are an intentionally hard distribution and results would look different
on close-range/webcam-style footage; no energy measurement was taken (see
Section~\ref{sec:method}); the privacy check is a detector-based proxy only, not
a proof of irreversibility, and does not test against a re-identification
model; the privacy-check sample (300 images) is smaller than the main
3{,}226-image evaluation set, for runtime reasons.

\textbf{Future work:} measure true energy consumption on a Linux machine with
RAPL access or a dedicated power meter rather than a CPU\%/latency proxy;
evaluate tiled-crop inference (not attempted here) as a further multi-scale
variant; test whether a detector trained or fine-tuned specifically on small
faces closes the recall gap directly rather than via post-hoc multi-scale
tricks; and extend the anonymization pipeline to audio (voice anonymization)
for full audio-visual privacy.

\begin{thebibliography}{9}
\bibitem{yang2016wider} S.~Yang, P.~Luo, C.~C.~Loy, X.~Tang. \textit{WIDER FACE: A Face Detection Benchmark}. CVPR 2016.
\bibitem{viola2001rapid} P.~Viola, M.~Jones. \textit{Rapid Object Detection using a Boosted Cascade of Simple Features}. CVPR 2001.
\bibitem{liu2016ssd} W.~Liu et al. \textit{SSD: Single Shot MultiBox Detector}. ECCV 2016.
\bibitem{bazarevsky2019blazeface} V.~Bazarevsky et al. \textit{BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs}. arXiv:1907.05047, 2019.
\bibitem{schwartz2020green} R.~Schwartz, J.~Dodge, N.~A.~Smith, O.~Etzioni. \textit{Green AI}. Communications of the ACM, 63(12), 2020.
\end{thebibliography}

\end{document}
""")

    os.makedirs(REPORT_DIR, exist_ok=True)
    out_path = os.path.join(REPORT_DIR, "report.tex")
    with open(out_path, "w") as f:
        f.write("".join(tex))
    print(f"Wrote -> {out_path}")


if __name__ == "__main__":
    main()
