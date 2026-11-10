# Results Summary (for Section 6 — Experiments & Analysis)

Evaluated on 60 WIDER FACE validation images (463 ground-truth faces total),
IoU threshold = 0.5. Benchmarked on CPU-only inference over the same 60 images.

| Detector | Model Size (MB) | FPS | Latency (ms/frame) | CPU % | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Haar Cascade | 0.00 | 17.69 | 56.51 | 568.8 | 0.548 | 0.198 | 0.291 |
| MobileNet-SSD | 10.17 | 81.84 | 12.22 | 318.0 | 0.972 | 0.204 | 0.337 |
| MediaPipe BlazeFace | 0.22 | 475.18 | 2.10 | 125.5 | 0.769 | 0.058 | 0.108 |

## Key takeaways
- **MobileNet-SSD** gives the best overall accuracy (highest precision, best F1) at a
  moderate speed (~82 FPS) — best choice when accuracy matters more than raw speed.
- **MediaPipe BlazeFace** is ~27x faster than MobileNet-SSD and the smallest model
  (0.22 MB) but recall drops sharply on crowded/small-face scenes — it is tuned for
  close-range (selfie-style) faces, not dense crowds, illustrating a real speed-accuracy
  trade-off rather than a universal "best" detector.
- **Haar Cascade** needs no model file at all (ships with OpenCV) but is the slowest
  and least precise of the three, consistent with it being a classical (non-deep)
  method.
- All three run comfortably on CPU only — confirming the pipeline is deployable on
  resource-constrained / edge hardware, which was the project's core constraint.

Files:
- `combined_results.csv` — full numeric table
- `accuracy_vs_speed.png` — F1 vs FPS scatter (main trade-off figure)
- `cpu_usage.png`, `model_size.png` — supporting bar charts
- `anonymized_samples/` — before/after visual demo for each detector
