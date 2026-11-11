# CPU-Efficient Face Anonymization: A Recall-First Evaluation

A rigorous, full-dataset benchmark of four CPU-only face-detector variants
(Haar Cascade, OpenCV MobileNet-SSD, MediaPipe BlazeFace short-range, MediaPipe
BlazeFace full-range) as the detection stage of a face-anonymization pipeline —
evaluated on the **full WIDER FACE validation split** (3,226 images, 39,123
ground-truth faces), with bootstrap confidence intervals, a confidence-threshold
sweep, multi-scale/detector-union recall-improvement experiments, and a
detector-based check of whether blurring/pixelating a face actually defeats
detection.

> Course project for **Multimedia Technology** (7th Semester), Applications
> Track. See [`CLAUDE.md`](CLAUDE.md) for the full build log (not pushed to
> GitHub — local project notes only).

## The headline result: recall, not F1

For an anonymization tool, the metric that matters is **recall** — a face a
detector misses stays visible. This project does not lead with F1 or precision;
it leads with recall, broken down by how small the ground-truth face is:

| Face height | GT faces in bucket | Haar Cascade | MobileNet-SSD | BlazeFace (short-range) | BlazeFace (full-range) |
|---|---|---|---|---|---|
| **< 20 px** | 19,197 (49.1% of all faces) | 0.000 | 0.000 | 0.000 | 0.000 |
| 20–50 px | 12,597 | 0.203 | 0.001 | 0.000 | 0.032 |
| 50–100 px | 4,595 | 0.425 | 0.273 | 0.000 | 0.403 |
| > 100 px | 2,734 | 0.537 | 0.762 | 0.417 | 0.729 |

**Nearly half of all faces in WIDER FACE's validation set are under 20 pixels
tall, and every detector tested here — four separate model variants — finds
essentially none of them.** This holds whether or not the detector has a
confidence threshold to tune (Section "Does sweeping the threshold help?"
below), whether it's upscaled 2x (Section "Recall-improvement experiments"), or
whether its detections are combined with another detector's (same section).
This is reported plainly because it is the actual finding, not a flattering one:
**on crowded, small-face scenes, these CPU-only detectors leave most faces
visible, and none of the cheap fixes tried here close that gap.**

Overall precision/recall/F1 (95% bootstrap CI, resampling images 1,000x):

| Detector | Precision | Recall | F1 | Recall 95% CI |
|---|---|---|---|---|
| Haar Cascade | 0.688 | 0.153 | 0.250 | [0.135, 0.172] |
| MobileNet-SSD | 0.973 | 0.086 | 0.158 | [0.076, 0.097] |
| BlazeFace (short-range) | 0.676 | 0.029 | 0.056 | [0.026, 0.033] |
| BlazeFace (full-range) | 0.847 | 0.108 | 0.192 | [0.096, 0.121] |

Full breakdown (also by WIDER FACE's own blur attribute) in
[`results/eval_by_bucket.csv`](results/eval_by_bucket.csv) /
[`results/tables/`](results/tables/).

## Does sweeping the confidence threshold help?

MobileNet-SSD and BlazeFace expose a continuous confidence score; **Haar
Cascade does not** (`minNeighbors` is a discrete knob, not a probability) — no
PR curve is reported for it, as a stated limitation rather than an approximation.
Sweeping MobileNet-SSD's threshold all the way to 0 (accepting every candidate
box) only reaches recall ≈0.24 at best; its recall ceiling on this dataset is
architectural (its fixed input resolution cannot resolve most small faces at
all), not a product of an overly strict threshold. Full curve:
[`results/pr_curves.csv`](results/pr_curves.csv),
recall-at-fixed-precision: [`results/recall_at_precision.csv`](results/recall_at_precision.csv).

## Recall-improvement experiments

Two cheap, CPU-only techniques were tried to recover recall, measured on the
full dataset:

1. **Multi-scale inference** (run at 1x/1.5x/2x, merge with NMS) — helps **Haar
   Cascade substantially** (recall 0.153 → 0.228) but at a ~6x latency cost
   (63ms → 387ms/image). It does **almost nothing** for MobileNet-SSD or
   BlazeFace (recall changes by <0.001), because both resize their input to a
   fixed resolution internally regardless of the source image's size — upscaling
   before feeding them in doesn't give the network any more effective
   resolution on small faces. This is reported as a genuine negative result for
   the two deep models, not hidden.
2. **Detector union** (combine boxes from multiple detectors + NMS) — does
   **not help either**: the best union (Haar + MobileNet-SSD + BlazeFace)
   reaches recall 0.178, *worse* than Haar alone with multi-scale (0.228),
   while being comparably slow. No union configuration sits on the
   recall-vs-latency Pareto front — see `results/pareto_recall_vs_latency.png`.

**Best recall-per-millisecond configuration tested: `mobilenet_ssd` at its base
single-scale operating point** (recall 0.086 at 15.7ms/image, recall/ms =
0.00546 — see [`results/recall_per_ms_ranking.csv`](results/recall_per_ms_ranking.csv)
for the full ranking). Multi-scale and union configurations buy higher absolute
recall only at a worse recall-per-millisecond ratio.

A fourth model variant — **MediaPipe's full-range BlazeFace** — was added after
the original three-detector comparison specifically to test whether
short-range BlazeFace's poor recall was a short-range-specific limitation.
It nearly quadruples BlazeFace's recall (0.029 → 0.108) and improves its
precision too, confirming the short-range model (not the BlazeFace
architecture) was the main limiter — but it **still finds zero faces under
20px**, reinforcing that this is not specific to any one of the four models.

Full tables: [`results/improve_multiscale.csv`](results/improve_multiscale.csv),
[`results/improve_union.csv`](results/improve_union.csv).

## Does anonymizing a face actually defeat detection?

For a random 300-image sample, MobileNet-SSD's own detections were blurred
(several Gaussian kernel sizes) or pixelated (several block counts), then all
three main detectors were re-run on the anonymized image to check whether the
anonymized region is still detected as a face. Two findings:

- **No tested setting drove every detector's re-detection rate at or below
  2%.** The best tested setting (`pixelate_4`) still left a 16.0% worst-case
  re-detection rate across the three detectors — this is a *detector-based*
  check only (does a common CPU face detector still fire on the region?), not
  proof that the anonymization is irreversible, and it does not test against a
  re-identification model.
- The relationship between anonymization strength and re-detection is
  **non-monotonic for pixelation**: re-detection drops sharply at the mildest
  pixelation tested, then *rises again* as pixelation gets coarser — plausibly
  because very coarse pixelation produces a stable, high-contrast block pattern
  that still carries detectable structure. This is reported as observed, since
  it argues against assuming "more pixelation is strictly safer."

Full table: [`results/privacy_check.csv`](results/privacy_check.csv),
recommendation: [`results/privacy_recommendation.txt`](results/privacy_recommendation.txt).

## Benchmark (latency, memory, model size)

Each detector is benchmarked in its **own subprocess** (so peak memory reflects
one detector, not whatever ran before it in a shared process — a bug in this
project's first version), over 5 repeats with `cv2.setNumThreads(1)` set
explicitly so all three are compared under identical, single-threaded
conditions. Median latency + IQR, not a single run:

| Detector | Median latency | IQR | FPS | Peak memory | Model size |
|---|---|---|---|---|---|
| Haar Cascade | 498.0 ms | 67.4 ms | 2.0 | 266 MB | 0 MB |
| MobileNet-SSD | 45.0 ms | 2.6 ms | 22.2 | 673 MB | 10.2 MB |
| BlazeFace (short-range) | 5.2 ms | 0.4 ms | 193.1 | 840 MB | 0.2 MB |

(Apple M1, 8 cores; full environment record in
[`results/environment.txt`](results/environment.txt).)

**No energy measurement was taken.** `powermetrics` — the only plausible
energy-measurement path on macOS — requires root
(`powermetrics must be invoked as the superuser`), and this project's automated
runs do not use `sudo`. CPU time and latency are reported as efficiency
*proxies* only; they are not energy, and this project is described as
**CPU-efficient**, not "green", specifically because of this. (If you run this
on a Linux machine with Intel RAPL access, `/sys/class/powercap` would be the
path to a real energy measurement — not attempted here.)

## Is this "deployable"?

**Not as-is, for a privacy-critical use case.** On WIDER FACE's crowded, small-face
distribution, none of the four tested CPU-only detectors reaches usable recall
(all under 0.23 overall, and ~0 on faces under 20px), and none of the
recall-improvement techniques tried closes that gap. MobileNet-SSD is the best
choice when precision matters more than recall (0.973 precision, at 45ms/frame);
the full-range BlazeFace model is the best recall-per-millisecond choice among
the deep models; Haar Cascade with multi-scale inference is viable only where
a ~6x latency cost is acceptable. This project is a measurement of that
trade-off space, not a claim that any one configuration is production-ready.

## Project structure

```
.
├── src/
│   ├── detectors/
│   │   ├── haar.py                       # Haar Cascade (Viola-Jones)
│   │   ├── mobilenet_ssd.py              # OpenCV DNN MobileNet-SSD (res10)
│   │   ├── mediapipe_blazeface.py        # MediaPipe BlazeFace, short-range
│   │   ├── mediapipe_blazeface_full.py   # MediaPipe BlazeFace, full-range
│   │   └── download_models.py            # fetches pretrained weights (gitignored)
│   ├── geometry.py          # iou / match_boxes / match_per_prediction / nms (unit-tested)
│   ├── gt_utils.py          # loads ground_truth_full.jsonl, size/blur bucketing
│   ├── data_prep.py         # downloads + parses the FULL WIDER FACE val split
│   ├── anonymize.py         # blur / pixelate given detected boxes
│   ├── evaluate.py          # bucketed recall/precision, bootstrap CI, PR curves
│   ├── eval_full_range_blazeface.py  # adds the 4th detector to the same tables
│   ├── improve.py           # Task B: multi-scale + detector-union experiments
│   ├── privacy_check.py     # Task C: re-detection after anonymization
│   ├── benchmark.py / _benchmark_worker.py  # Task D: isolated-subprocess timing
│   ├── make_pareto_plot.py  # recall-vs-latency Pareto plot + ranking
│   └── make_report_tables.py / make_report_tex.py  # generate every table/figure from CSVs
├── tests/                   # pytest: geometry + anonymize edge cases
├── results/                 # every CSV, table, and figure this README cites
├── report/                  # report.tex + slides (not pushed — course submission only)
├── run_all.sh               # single reproducibility entrypoint (see below)
└── requirements.txt         # pinned versions
```

## Setup & reproduce everything

```bash
./run_all.sh
```

This recreates the venv, downloads model weights + the full WIDER FACE
validation set (~363MB, not committed), runs the test suite, runs every
experiment, and regenerates every CSV/figure/table in `results/`. Safe to
re-run — each step skips work that's already cached. See the script for the
individual commands if you want to run one stage at a time.

To benchmark/evaluate your own video instead of the WIDER FACE images, drop an
`.mp4` at `data/own_video.mp4` (auto-detected by `benchmark.py`); to generate a
visual before/after demo, place a photo **you have the rights to use** at
`data/own_photo.jpg` and run `python src/anonymize.py` (see that file for why
it doesn't default to a dataset image).

## Method

Each detector implements the same interface —
`detect_faces(frame) -> [(x, y, w, h), ...]` — so all four are interchangeable
across benchmarking, evaluation, and anonymization. Three detectors were
originally chosen to span the classical-to-deep spectrum, all CPU-inference
only: Haar Cascade (classical, zero model weight), BlazeFace (deep, built for
mobile/edge CPUs), and MobileNet-SSD (heavier but more accurate compact deep
detector). Heavier state-of-the-art detectors (RetinaFace, MTCNN) were
deliberately excluded: they need a GPU for real-time throughput, conflicting
with the CPU-only constraint this project is measuring.

**Matching:** IoU ≥ 0.5, greedy one-to-one (each prediction claims its best
remaining unmatched ground-truth box). **Recall** is bucketed by the
ground-truth box's own height; **precision** is bucketed by the *predicted*
box's own height, since a false positive has no ground-truth box to bucket it
by — these two bucketings measure different things and are not directly
comparable (see `results/eval_by_bucket.csv`'s `metric_type` column).
**Confidence intervals:** bootstrap resampling of *images* (not individual
boxes), 1,000 iterations, 2.5/97.5 percentiles.

## Data

**WIDER FACE** validation split (Yang et al., CVPR 2016), full 3,226 images /
39,123 ground-truth boxes, via the CUHK-CSE mirror on HuggingFace. Every box
carries WIDER FACE's own attribute labels (blur, expression, illumination,
occlusion, pose, invalid); boxes flagged `invalid=1` are excluded from all
evaluation. The dataset is **not committed to this repository** — its license
and the ~363MB size make that inappropriate; `src/data_prep.py` downloads and
rebuilds `data/ground_truth_full.jsonl` deterministically. No training is
performed anywhere in this project — every detector runs purely in inference
mode on its publicly distributed pretrained weights.

## Limitations

- Evaluation uses IoU ≥ 0.5 only; results at other IoU thresholds are not reported.
- WIDER FACE's validation set is an intentionally hard, crowd-scene-heavy
  distribution; results would look different on close-range/webcam-style
  footage (which is exactly BlazeFace short-range's intended use case).
- **No energy measurement was taken** (see Benchmark section) — CPU time and
  latency are efficiency proxies, not energy.
- The privacy check (re-detection after anonymization) is a **detector-based
  proxy only**: it is not proof of irreversibility, and it does not test
  against a re-identification model or a human observer.
- The privacy-check sample (300 images) is smaller than the main 3,226-image
  evaluation set, for runtime reasons, and is stated as such.
- Tiled-crop inference (mentioned as an optional multi-scale variant) was not
  implemented.

## Development note

Parts of this project's code and documentation were written with the
assistance of an AI coding assistant (Claude Code). The author defined the
study design, reviewed all code and results, ran every experiment on his own
machine, and is responsible for the findings reported here.

## Acknowledgments & references

- Yang, S., Luo, P., Loy, C. C., & Tang, X. (2016). *WIDER FACE: A Face Detection Benchmark.* CVPR 2016.
- Viola, P., & Jones, M. (2001). *Rapid Object Detection using a Boosted Cascade of Simple Features.* CVPR 2001.
- Liu, W., et al. (2016). *SSD: Single Shot MultiBox Detector.* ECCV 2016.
- Bazarevsky, V., et al. (2019). *BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs.* arXiv:1907.05047.
- Schwartz, R., Dodge, J., Smith, N. A., & Etzioni, O. (2020). *Green AI.* Communications of the ACM, 63(12).
- [MediaPipe](https://github.com/google-ai-edge/mediapipe) Face Detector (Google) — [OpenCV](https://opencv.org/) Haar Cascade and DNN face-detector models

## License

Code in this repository is released under the MIT License (see `LICENSE`). The
WIDER FACE dataset and the MediaPipe/OpenCV pretrained models retain their own
original licenses and are not redistributed here.
