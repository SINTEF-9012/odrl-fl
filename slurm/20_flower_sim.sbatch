#!/usr/bin/env bash
#SBATCH --job-name=odrl-flwr
#SBATCH --output=slurm/logs/flwr-%j.out
#SBATCH --error=slurm/logs/flwr-%j.err
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#
# Run the policy-aware strategy as a real Flower federation.
#
#   sbatch slurm/20_flower_sim.sbatch
#
# This is the one thing in the project that has never happened. `strategy.py`
# has 33 tests over it, including one asserting that its aggregation matches
# Flower's own, but no federation has ever been started: Flower's simulation
# backend needs Ray, and Ray ships no Windows wheel for Python 3.13.
#
# Two things have to come out of it. First, that the Flower path and the
# in-process path of `odrl_fl.fedavg` reach the same accuracy on the same
# federation, seed and trial -- the claim `odrl_fl/fedavg.py` defers and this
# closes. Second, the timings: how long admission takes, how large a
# declaration is on the wire, and how the two-stage selection compares with
# training every candidate.
#
# Four hours rather than the full day. If it has not run in four it is broken,
# not slow, and a shorter limit gets it scheduled sooner.
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"
cd "$ODRL_REPO"
odrl_activate_conda

# Ray is the only reason this job needs the cluster at all.
python -c "import ray" 2>/dev/null || {
    echo "Ray is not installed; run 01_setup_env.sh first." >&2
    exit 1
}

# Ray writes a great deal to its session directory. Keep it on node-local
# scratch where one exists, so a shared home quota is not the thing that
# fails the run.
export RAY_TMPDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/ray-$SLURM_JOB_ID"
mkdir -p "$RAY_TMPDIR"
trap 'rm -rf "$RAY_TMPDIR"' EXIT

echo "Flower simulation: $(python -c 'import flwr; print(flwr.__version__)')"
echo "Ray: $(python -c 'import ray; print(ray.__version__)')"
echo "Population of $ODRL_POOL nodes, alpha $ODRL_ALPHA"
echo

# `require-purpose` is set deliberately. Left empty, the request names only
# actions, which nearly every contributor grants, so admission excludes almost
# nobody and the run demonstrates nothing. A real purpose is what makes the
# policy gate bite.
srun flwr run . local-sim \
    --run-config "num-server-rounds=$ODRL_ROUNDS alpha=$ODRL_ALPHA seed=0 require-purpose='ds-cancer' recheck-consent=true"

echo
echo "Now check the two paths agree:"
echo "  python experiments/run_flower_check.py"
