#!/bin/bash
# ============================================================================
# The widened ground-truth lattice: every subset of the pool, at many
# federations, with replicates.
#
# Before first use, once:
#   mkdir -p ~/logs/slurm
#   bash slurm/01_setup_env.sh
#   bash slurm/00_preflight.sh
#
# Submit one task first, so a mistake costs one task instead of forty:
#   sbatch --array=0 slurm/10_groundtruth.sbatch
#   tail -f ~/logs/slurm/odrl-gt_*_0.out
#
# Then the whole array:
#   sbatch slurm/10_groundtruth.sbatch
#
# Campaign parameters can be overridden without editing anything:
#   sbatch --export=ALL,ODRL_SEEDS=2,ODRL_TRIALS=1 slurm/10_groundtruth.sbatch
#
# Replicates are the point of this campaign. The committed lattice has one
# training run per (subset, federation), which makes an argmax over a thousand
# candidates biased upward -- the winner is partly whichever candidate drew the
# kindest initialisation, so a regret measured against it is an upper bound.
# Five replicates let the shortlist be chosen from one run and scored on the
# others, which turns that into a measurement.
#
# No GPU is requested, deliberately. The model is a three-layer MLP over
# synthetic Gaussian mixtures and the per-run cost is Python and data handling,
# so a GPU would sit idle while the queue wait grew. The lever is array
# parallelism across CPUs: 102,300 runs is about 19 CPU-hours in total.
#
# No `set -e` either, also deliberately. A job that reports a problem and exits
# with a status is more useful than one that vanishes, which is how the first
# two attempts at this failed.
# ============================================================================
#SBATCH --job-name odrl-gt
#SBATCH --output /home/rustemd/logs/slurm/odrl-gt_%A_%a.out
#SBATCH --error /home/rustemd/logs/slurm/odrl-gt_%A_%a.err
#SBATCH --partition sintef
#SBATCH --array 0-39
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 4G
#SBATCH --time 00-23:30:00

REPO_DIR="${ODRL_REPO:-$HOME/odrl-fl}"

echo ""
echo "***** LAUNCHING *****"
date '+%F %H:%M:%S'
echo "Repo:  $REPO_DIR"
echo "Host:  $(hostname)"

if [ ! -d "$REPO_DIR" ]; then
    echo "ERROR: no checkout at $REPO_DIR (set ODRL_REPO)." >&2
    exit 1
fi

cd "$REPO_DIR" || exit 1
# shellcheck source=env.sh
. "$REPO_DIR/slurm/env.sh"

odrl_activate || exit 1
odrl_single_threaded
echo "Packages:"
odrl_report_versions

# The array variables as actually seen. SLURM_ARRAY_TASK_COUNT is not set by
# every Slurm version, and the previous version of this script defaulted it to
# 1 and then compared it against 40 -- which tripped its own guard and exited
# every task in under a second. The shard count now comes from env.sh, which is
# authoritative, and MIN/MAX are used only to check it.
task="${SLURM_ARRAY_TASK_ID:-0}"
shards="$ODRL_ARRAY_TASKS"
echo ""
echo "Array: task=${SLURM_ARRAY_TASK_ID:-<unset>}" \
     "min=${SLURM_ARRAY_TASK_MIN:-<unset>}" \
     "max=${SLURM_ARRAY_TASK_MAX:-<unset>}" \
     "count=${SLURM_ARRAY_TASK_COUNT:-<unset>}"

if [ -n "${SLURM_ARRAY_TASK_MIN:-}" ] && [ -n "${SLURM_ARRAY_TASK_MAX:-}" ]; then
    span=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))
    if [ "$span" -ne "$shards" ]; then
        # A single-task trial run is the expected case, not an error: it is
        # shard `task` of the full `shards`, and covers its stride only.
        if [ "$span" -eq 1 ]; then
            echo "Trial run: one task of $shards. It covers its own shard only."
        else
            echo "ERROR: --array spans $span tasks but ODRL_ARRAY_TASKS=$shards." >&2
            echo "The shards would not cover the work. Fix one or the other." >&2
            exit 1
        fi
    fi
else
    echo "Note: Slurm did not set SLURM_ARRAY_TASK_MIN/MAX, so the array size"
    echo "could not be cross-checked. Using ODRL_ARRAY_TASKS=$shards."
fi

mkdir -p "$ODRL_SHARDS_DIR" || exit 1
out="$ODRL_SHARDS_DIR/groundtruth.$(printf '%03d' "$task").csv"

echo ""
echo "Shard:   $task of $shards -> $out"
echo "Sweep:   pool $ODRL_POOL, $ODRL_SEEDS seeds, $ODRL_TRIALS trials," \
     "$ODRL_ROUNDS rounds, alpha $ODRL_ALPHA"
echo ""

# Resumable: the script skips (members, seed, trial) triples already in its own
# output file, so a requeued task picks up where it stopped.
python experiments/run_groundtruth.py \
    --pool "$ODRL_POOL" \
    --seeds "$ODRL_SEEDS" \
    --trials "$ODRL_TRIALS" \
    --rounds "$ODRL_ROUNDS" \
    --alpha "$ODRL_ALPHA" \
    --shard "$task" \
    --shards "$shards" \
    --out "$out"
STATUS=$?

rows=0
[ -f "$out" ] && rows=$(( $(wc -l < "$out") - 1 ))

echo ""
echo "***** DONE (exit $STATUS, $rows rows) *****"
date '+%F %H:%M:%S'
exit $STATUS
