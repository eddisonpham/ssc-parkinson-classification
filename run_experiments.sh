#!/usr/bin/env bash
# run_experiments.sh
# ===================
# Runs train_models.py for every dimensionality reduction method.
#
# Usage
# -----
#   bash run_experiments.sh                          # all methods, default n_trials
#   N_TRIALS=30 bash run_experiments.sh              # override trials
#   DR_METHODS="pca famd" bash run_experiments.sh    # subset of methods
#   SKIP_TUNING=1 bash run_experiments.sh            # skip Optuna (debug)
#   MODELS="logistic_regression lightgbm" bash run_experiments.sh
#   PARALLEL=1 bash run_experiments.sh               # run methods in parallel (requires GNU parallel)
#
# Environment variables
# ---------------------
#   DATA_DIR      Path to C-OPN CSV directory         (default: data/ssc_data)
#   DR_METHODS    Space-separated list of DR methods  (default: all 5)
#   N_TRIALS      Optuna trials per model             (default: per-model registry default)
#   SKIP_TUNING   Set to 1 to disable Optuna          (default: unset)
#   MODELS        Space-separated model names         (default: all 8)
#   PARALLEL      Set to 1 to use GNU parallel        (default: sequential)
#   LOG_DIR       Directory for log files             (default: logs)
#   PYTHON        Python executable                   (default: python)

set -euo pipefail

# ── Configurable defaults ────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-data/ssc_data}"
DR_METHODS="${DR_METHODS:-pca famd catpca hellinger famd_hellinger}"
LOG_DIR="${LOG_DIR:-logs}"
PYTHON="${PYTHON:-python}"
PARALLEL="${PARALLEL:-0}"

# Build optional flags
EXTRA_FLAGS=""
[[ -n "${N_TRIALS:-}" ]]   && EXTRA_FLAGS="$EXTRA_FLAGS --n-trials $N_TRIALS"
[[ "${SKIP_TUNING:-0}" == "1" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --skip-tuning"
[[ -n "${MODELS:-}" ]]     && EXTRA_FLAGS="$EXTRA_FLAGS --models $MODELS"

# ── Setup ────────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
START_ALL=$(date +%s)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "======================================================================"
echo "  C-OPN DR Experiment Suite"
echo "  Started:   $(date)"
echo "  Data dir:  $DATA_DIR"
echo "  Methods:   $DR_METHODS"
echo "  Log dir:   $LOG_DIR"
[[ -n "$EXTRA_FLAGS" ]] && echo "  Extra:     $EXTRA_FLAGS"
echo "======================================================================"
echo ""

# ── Per-method runner ─────────────────────────────────────────────────────────
run_method() {
    local method="$1"
    local log_file="$LOG_DIR/${method}_${TIMESTAMP}.log"

    echo ">>> Starting: $method  (log: $log_file)"

    # Ensure current working directory is the script location to fix Python import errors
    local script_dir
    script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

    (
      cd "$script_dir"
      if PYTHONPATH="$script_dir" $PYTHON scripts/train_models.py \
          --dr-method   "$method"   \
          --data-dir    "$DATA_DIR" \
          $EXTRA_FLAGS              \
          2>&1 | tee "$log_file"; then
          echo "    ✓  $method completed"
      else
          echo "    ✗  $method FAILED (exit $?). See $log_file"
          # Continue with remaining methods rather than aborting the suite
          return 0
      fi
      echo ""
    )
}

export -f run_method
export PYTHON DATA_DIR EXTRA_FLAGS LOG_DIR TIMESTAMP

# ── Dispatch ─────────────────────────────────────────────────────────────────
IFS=' ' read -r -a METHOD_ARRAY <<< "$DR_METHODS"

if [[ "$PARALLEL" == "1" ]]; then
    if ! command -v parallel &>/dev/null; then
        echo "GNU parallel not found; falling back to sequential execution."
        PARALLEL=0
    fi
fi

if [[ "$PARALLEL" == "1" ]]; then
    echo "Running methods in parallel (GNU parallel)..."
    printf '%s\n' "${METHOD_ARRAY[@]}" | parallel -j "${#METHOD_ARRAY[@]}" run_method {}
else
    for method in "${METHOD_ARRAY[@]}"; do
        run_method "$method"
    done
fi

# ── Summary ──────────────────────────────────────────────────────────────────
END_ALL=$(date +%s)
ELAPSED=$(( END_ALL - START_ALL ))
ELAPSED_FMT=$(printf '%02dh %02dm %02ds' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))

echo ""
echo "======================================================================"
echo "  All methods completed in $ELAPSED_FMT"
echo "  Results are in: results/"
echo ""

# Print a quick comparison table from model_results.csv files
echo "  Balanced accuracy summary (best model per DR method):"
echo "  -------------------------------------------------------"
printf "  %-22s  %s\n" "DR Method" "Best balanced_accuracy"
printf "  %-22s  %s\n" "----------" "----------------------"

for method in "${METHOD_ARRAY[@]}"; do
    csv="results/${method}/model_results.csv"
    if [[ -f "$csv" ]]; then
        # Extract the highest balanced_accuracy value (column index may vary)
        best=$(tail -n +2 "$csv" \
               | awk -F',' 'NR==1 {
                    for(i=1;i<=NF;i++) if($i=="balanced_accuracy") col=i
                 }
                 NR>1 { if($col+0 > max+0) max=$col }
                 END { print max }' 2>/dev/null || echo "N/A")
        printf "  %-22s  %s\n" "$method" "$best"
    else
        printf "  %-22s  %s\n" "$method" "(no results)"
    fi
done

echo "======================================================================"