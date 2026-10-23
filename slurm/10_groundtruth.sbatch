#!/usr/bin/env bash
#SBATCH --job-name=odrl-gt
#SBATCH --output=slurm/logs/gt-%A_%a.out
#SBATCH --error=slurm/logs/gt-%A_%a.err
#SBATCH --array=0-39
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=23:30:00
#
# The widened ground-truth lattice: every subset of the pool, at many
# federations, with replicates.
#
#   sbatch slurm/10_groundtruth.sbatch
#
# Replicates are the point of this campaign. The committed lattice has one
# training run per (subset, federation), which makes an argmax over a thousand
# candidates biased upward -- the winner is partly whichever candidate drew the
# kindest initialisation. Five replicates let the shortlist be chosen from one
# run and scored on the others, which turns every regret figure from an upper
# bound into a measurement.
#
# --array must match ODRL_ARRAY_TASKS in env.sh. Slurm reads #SBATCH lines
# before any shell runs, so it cannot be set from a variable; if you change one,
# change both. The guard below fails loudly rather than silently losing work.
#
# No GPU is requested, deliberately. The model is a three-layer MLP over
# synthetic Gaussian mixtures and the per-run cost is Python and data handling,
# so a GPU would sit idle while the queue wait grew.
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"
cd "$ODRL_REPO"
odrl_activate_conda
odrl_single_threaded

task="${SLURM_ARRAY_TASK_ID:-0}"
count="${SLURM_ARRAY_TASK_COUNT:-1}"

if (( count != ODRL_ARRAY_TASKS )); then
    echo "Array is $count tasks but env.sh says $ODRL_ARRAY_TASKS." >&2
    echo "The shards would not cover the work. Fix --array or env.sh." >&2
    exit 1
fi

mkdir -p "$ODRL_SHARDS_DIR"
out="$ODRL_SHARDS_DIR/groundtruth.$(printf '%03d' "$task").csv"

echo "shard $task of $count -> $out"
echo "pool $ODRL_POOL, $ODRL_SEEDS seeds, $ODRL_TRIALS trials, $ODRL_ROUNDS rounds"

# Resumable: the script skips (members, seed, trial) triples already in its own
# output file, so a requeued task picks up where it stopped.
srun python experiments/run_groundtruth.py \
    --pool "$ODRL_POOL" \
    --seeds "$ODRL_SEEDS" \
    --trials "$ODRL_TRIALS" \
    --rounds "$ODRL_ROUNDS" \
    --alpha "$ODRL_ALPHA" \
    --shard "$task" \
    --shards "$count" \
    --out "$out"

echo "shard $task done: $(( $(wc -l < "$out") - 1 )) rows"
