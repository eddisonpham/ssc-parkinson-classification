#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh
# Full ablation sweep for the C-OPN Parkinson classification pipeline.
#
# Compatible with Bash 3.2+ (macOS default).
#
# WHAT IT SWEEPS
# --------------
#   DR methods   : none (raw), mca, famd, catpca, hellinger, tfidf_embedding,
#                  famd+hellinger (recommended)
#   Model sets   : baseline (5 models), sota (12 models), feat-selection
#   Missingness  : 0.50, 0.65, 0.80
#   MoE          : always runs once at threshold=0.95 (its own convention)
#
#   Total combinations : 7 DR x 3 scripts x 3 thresholds = 63 runs
#                      + 7 DR x 1 MoE run at threshold=0.95 = 7 extra
#                      = 70 runs maximum
#
# USAGE
# -----
#   bash run_experiments.sh              # full 70-run sweep
#   bash run_experiments.sh --quick      # 8-run sanity check
#   bash run_experiments.sh --dry-run    # print commands, do not execute
#   PYTHON=python3.11 bash run_experiments.sh
#
# OUTPUT
# ------
#   results/<run_id>/          per-run results directory
#   logs/<run_id>.log          stdout + stderr per run
#   results/sweep_summary.csv  aggregated metrics across all runs
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${PYTHON:-python}"
CONFIG_TEMPLATE="${REPO_ROOT}/configs/experiment.yaml"
RESULTS_ROOT="${REPO_ROOT}/results"
LOGS_DIR="${REPO_ROOT}/logs"
SCRIPTS_DIR="${REPO_ROOT}/scripts"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
QUICK=false
DRY_RUN=false

for arg in "$@"; do
  case $arg in
    --quick)   QUICK=true  ;;
    --dry-run) DRY_RUN=true ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# DR condition registry (Bash 3.2 compatible — case statement, no declare -A)
# ---------------------------------------------------------------------------
get_dr_yaml() {
  local key="$1"
  case "$key" in
    none)
      echo "dimensionality_reduction: null"
      ;;
    mca)
      printf 'dimensionality_reduction:\n  mode: sequential\n  methods:\n    - type: mca\n      n_components: 50\n      categorical_threshold: 20\n'
      ;;
    famd)
      printf 'dimensionality_reduction:\n  mode: sequential\n  methods:\n    - type: famd\n      n_components: 80\n      categorical_threshold: 20\n'
      ;;
    catpca)
      printf 'dimensionality_reduction:\n  mode: sequential\n  methods:\n    - type: catpca\n      n_components: 60\n      max_iter: 30\n'
      ;;
    hellinger)
      printf 'dimensionality_reduction:\n  mode: sequential\n  methods:\n    - type: hellinger\n      n_features: 100\n      n_bins: 10\n      use_svm_refinement: true\n      svm_top_k: 300\n      svm_c: 0.5\n'
      ;;
    tfidf)
      printf 'dimensionality_reduction:\n  mode: sequential\n  methods:\n    - type: tfidf_embedding\n      n_components: 50\n      max_features: 3000\n      min_df: 3\n'
      ;;
    famd_hellinger)
      printf 'dimensionality_reduction:\n  mode: sequential\n  methods:\n    - type: famd\n      n_components: 80\n      categorical_threshold: 20\n    - type: hellinger\n      n_features: 60\n      n_bins: 10\n      use_svm_refinement: true\n      svm_top_k: 300\n      svm_c: 0.5\n'
      ;;
    *)
      echo "ERROR: unknown DR key '$key'" >&2; exit 1 ;;
  esac
}

# ---------------------------------------------------------------------------
# Sweep dimensions
# ---------------------------------------------------------------------------
MISSINGNESS_THRESHOLDS=()
for v in $(seq 0.1 0.1 1.0); do
  MISSINGNESS_THRESHOLDS+=("$v")
done
if $QUICK; then
  DR_KEYS=(none famd_hellinger)
  MISSINGNESS_THRESHOLDS=(0.65)
else
  DR_KEYS=(none mca famd catpca hellinger tfidf famd_hellinger)
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

run_id() { echo "${1}__dr-${2}__miss-${3}"; }

generate_config() {
  local dr_key="$1"
  local miss="$2"

  # Find a truly unique temporary file to avoid mkstemp collisions
  local tmp=""
  local tmp_attempts=0
  while :; do
    tmp="/tmp/copn_config_${RANDOM}_$$_${dr_key}_${miss}.yaml"
    if [ ! -e "$tmp" ]; then
      touch "$tmp"
      break
    fi
    tmp_attempts=$((tmp_attempts+1))
    if [ "$tmp_attempts" -ge 10 ]; then
      echo "ERROR: Could not create a unique temp file after 10 attempts." >&2
      exit 1
    fi
  done

  # Strip DR key, comments, and blank lines from the template
  grep -v '^dimensionality_reduction' "$CONFIG_TEMPLATE" \
    | grep -v '^#' \
    | grep -v '^[[:space:]]*$' \
    > "$tmp" || true

  # Append missingness override and DR block
  printf '\nmissingness_threshold: %s\n\n' "$miss" >> "$tmp"
  get_dr_yaml "$dr_key" >> "$tmp"

  echo "$tmp"
}

execute() {
  local rid="$1"; shift
  local log_file="${LOGS_DIR}/${rid}.log"
  mkdir -p "$LOGS_DIR"

  echo "[$(date '+%H:%M:%S')] START  $rid"
  if $DRY_RUN; then
    echo "  DRY-RUN: $*"
    return
  fi

  if "$@" >"$log_file" 2>&1; then
    echo "[$(date '+%H:%M:%S')] OK     $rid"
  else
    echo "[$(date '+%H:%M:%S')] FAIL   $rid  (see $log_file)"
  fi
}

# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
aggregate_results() {
  echo "Aggregating results..."
  "$PYTHON" - <<'PYEOF'
import csv, json, pathlib, sys

RESULTS_ROOT = pathlib.Path("results")
rows = []

for run_dir in sorted(RESULTS_ROOT.iterdir()):
    if not run_dir.is_dir():
        continue
    parts = run_dir.name.split("__")
    if len(parts) < 3:
        continue
    script_label = parts[0]
    dr_label     = parts[1].replace("dr-", "")
    miss_label   = parts[2].replace("miss-", "")

    for csv_name in [
        "model_comparison.csv",
        "sota_model_comparison.csv",
        "mixture_of_experts_comparison.csv",
    ]:
        csv_path = run_dir / csv_name
        if not csv_path.exists():
            continue
        with csv_path.open() as fh:
            for record in csv.DictReader(fh):
                rows.append({
                    "run_dir":               run_dir.name,
                    "script":                script_label,
                    "dr":                    dr_label,
                    "missingness":           miss_label,
                    "model":                 record.get("model"),
                    "accuracy":              record.get("accuracy"),
                    "balanced_accuracy":     record.get("balanced_accuracy"),
                    "auc_roc":               record.get("auc_roc"),
                    "sensitivity_recall_ap": record.get("sensitivity_recall_ap"),
                    "specificity_pd":        record.get("specificity_pd"),
                    "f1_ap":                 record.get("f1_ap"),
                })

    metrics_path = run_dir / "feature_selection_metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        rows.append({
            "run_dir":               run_dir.name,
            "script":                script_label,
            "dr":                    dr_label,
            "missingness":           miss_label,
            "model":                 "feature_selection_logistic",
            "accuracy":              m.get("accuracy"),
            "balanced_accuracy":     m.get("balanced_accuracy"),
            "auc_roc":               m.get("auc_roc"),
            "sensitivity_recall_ap": m.get("classification_report", {}).get("1", {}).get("recall"),
            "specificity_pd":        m.get("classification_report", {}).get("0", {}).get("recall"),
            "f1_ap":                 m.get("classification_report", {}).get("1", {}).get("f1-score"),
        })

if not rows:
    print("No result rows found.")
    sys.exit(0)

summary_path = RESULTS_ROOT / "sweep_summary.csv"
fieldnames = [
    "run_dir", "script", "dr", "missingness", "model",
    "accuracy", "balanced_accuracy", "auc_roc",
    "sensitivity_recall_ap", "specificity_pd", "f1_ap",
]
with summary_path.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {summary_path}")

def safe_float(v):
    try:    return float(v)
    except: return 0.0

rows.sort(key=lambda r: safe_float(r["balanced_accuracy"]), reverse=True)
print("\nTop 10 by balanced_accuracy:")
print(f"{'model':<30} {'dr':<18} {'miss':<6} {'bal_acc':<9} {'auc':<8} {'ap_rec'}")
print("-" * 79)
for r in rows[:10]:
    print(
        f"{str(r['model']):<30} {str(r['dr']):<18} {str(r['missingness']):<6} "
        f"{str(r['balanced_accuracy']):<9} {str(r['auc_roc']):<8} "
        f"{str(r['sensitivity_recall_ap'])}"
    )
PYEOF
}

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
mkdir -p "$RESULTS_ROOT" "$LOGS_DIR"
echo "=============================================="
echo " C-OPN Parkinson classification — full sweep"
echo " QUICK=${QUICK}   DRY_RUN=${DRY_RUN}"
echo " DR conditions : ${DR_KEYS[*]}"
echo " Thresholds    : ${MISSINGNESS_THRESHOLDS[*]}"
echo "=============================================="

TOTAL_RUNS=0

for dr_key in "${DR_KEYS[@]}"; do
  for miss in "${MISSINGNESS_THRESHOLDS[@]}"; do

    cfg=$(generate_config "$dr_key" "$miss")

    # 1. Baseline
    rid=$(run_id "baseline" "$dr_key" "$miss")
    mkdir -p "${RESULTS_ROOT}/${rid}"
    execute "$rid" \
      "$PYTHON" "${SCRIPTS_DIR}/train_models.py" \
        --config "$cfg" --output-dir "${RESULTS_ROOT}/${rid}"
    TOTAL_RUNS=$((TOTAL_RUNS + 1))

    # 2. SOTA
    rid=$(run_id "sota" "$dr_key" "$miss")
    mkdir -p "${RESULTS_ROOT}/${rid}"
    execute "$rid" \
      "$PYTHON" "${SCRIPTS_DIR}/train_state_of_the_art_models.py" \
        --config "$cfg" --output-dir "${RESULTS_ROOT}/${rid}"
    TOTAL_RUNS=$((TOTAL_RUNS + 1))

    # 3. Feature selection
    rid=$(run_id "feat_sel" "$dr_key" "$miss")
    mkdir -p "${RESULTS_ROOT}/${rid}"
    execute "$rid" \
      "$PYTHON" "${SCRIPTS_DIR}/feature_selection.py" \
        --config "$cfg" \
        --missingness-threshold "$miss" \
        --output-dir "${RESULTS_ROOT}/${rid}"
    TOTAL_RUNS=$((TOTAL_RUNS + 1))

    # Safely remove config, but only if it exists and is a file
    if [ -f "$cfg" ]; then
      rm -f "$cfg"
    fi
  done

  # 4. MoE — once per DR condition at its own threshold (DR ignored internally)
  cfg_moe=$(generate_config "$dr_key" "0.95")
  rid=$(run_id "moe" "$dr_key" "0.95")
  mkdir -p "${RESULTS_ROOT}/${rid}"
  execute "$rid" \
    "$PYTHON" "${SCRIPTS_DIR}/train_mixture_of_experts.py" \
      --missingness-threshold 0.95 \
      --output-dir "${RESULTS_ROOT}/${rid}"
  if [ -f "$cfg_moe" ]; then
    rm -f "$cfg_moe"
  fi
  TOTAL_RUNS=$((TOTAL_RUNS + 1))

done

echo ""
echo "=============================================="
echo " Sweep complete.  Total runs: ${TOTAL_RUNS}"
echo "=============================================="

if ! $DRY_RUN; then
  cd "$REPO_ROOT"
  aggregate_results
fi