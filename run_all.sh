#!/usr/bin/env bash
# Single reproducibility entrypoint: recreates the venv, downloads models +
# data, runs the test suite, runs every experiment, and regenerates every
# figure/table in results/. Safe to re-run -- each step skips work that's
# already cached (model files, downloaded dataset, per-image detection cache).
#
# Usage: ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1/8: venv =="
if [ ! -d venv ]; then
  python3 -m venv venv
fi
source venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "== 2/8: download pretrained model weights =="
python3 src/detectors/download_models.py

echo "== 3/8: download + prepare full WIDER FACE validation set (~363MB) =="
python3 src/data_prep.py

echo "== 4/8: unit tests =="
python3 -m pytest tests/ -v

echo "== 5/8: evaluation (recall/precision by bucket, bootstrap CI, PR curves) =="
python3 src/evaluate.py
python3 src/eval_full_range_blazeface.py

echo "== 6/8: recall-improvement experiments (multi-scale, detector union) =="
python3 src/improve.py

echo "== 7/8: privacy check (re-detection after anonymization) =="
python3 src/privacy_check.py

echo "== 8/8: benchmark (latency/memory, isolated per detector) + figures/tables =="
python3 src/benchmark.py
python3 src/make_pareto_plot.py
python3 src/make_report_tables.py

echo "Done. See results/ for every CSV, figure, and generated table."
