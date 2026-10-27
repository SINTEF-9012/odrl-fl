#!/bin/bash
# ============================================================================
# What the policy round costs as the federation grows.
#
#   sbatch slurm/40_scaling.sbatch
#   grep COST ~/logs/slurm/odrl-scale_*.out
#
# The paper claims that anti-monotone pruning is what makes admission
# affordable, and that claim has only ever been measured offline. This measures
# it on the path that actually runs: how long collecting declarations takes, how
# long the search takes, how many candidates it evaluated against the size of
# the powerset, and how large a declaration is on the wire.
#
# One training round per run, not twenty. Training is not what is being
# measured here, and a single round keeps each population cheap.
#
# Populations are powers of two up to 32. At 32 the powerset is 4.3 billion
# subsets, so if the pruning does not work this job is where that becomes
# obvious -- which is the point. ODRL_SCALE_MAX lowers the ceiling if a run
# does not return.
# ============================================================================
#SBATCH --job-name odrl-scale
#SBATCH --output /home/rustemd/logs/slurm/odrl-scale_%j.out
#SBATCH --error /home/rustemd/logs/slurm/odrl-scale_%j.err
#SBATCH --partition sintef
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 24G
#SBATCH --time 00-04:00:00

REPO_DIR="${ODRL_REPO:-$HOME/odrl-fl}"

echo ""
echo "***** LAUNCHING *****"
date '+%F %H:%M:%S'
cd "$REPO_DIR" || exit 1
# shellcheck source=env.sh
. "$REPO_DIR/slurm/env.sh"
odrl_activate || exit 1
echo "Packages:"
odrl_report_versions

if ! python -c "import ray" 2>/dev/null; then
    echo "ERROR: Ray is not installed; run 01_setup_env.sh first." >&2
    exit 1
fi

export RAY_TMPDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/ray-${SLURM_JOB_ID:-$$}"
mkdir -p "$RAY_TMPDIR"
trap 'rm -rf "$RAY_TMPDIR"' EXIT

SIZES="${ODRL_SCALE_SIZES:-4 8 16 24 32}"
MAX="${ODRL_SCALE_MAX:-32}"

echo ""
echo "Populations: $SIZES (ceiling $MAX)"
echo ""

STATUS=0
for n in $SIZES; do
    if [ "$n" -gt "$MAX" ]; then
        echo "--- skipping n=$n, above the ceiling"
        continue
    fi
    echo "=================================================="
    echo "--- population $n"
    date '+%F %H:%M:%S'

    out="$RAY_TMPDIR/run-$n.log"
    # The COST lines this prints are the result; see report_cost in
    # odrl_fl/server_app.py.
    flwr run . local-sim --stream \
        --federation-config "options.num-supernodes=$n" \
        --run-config \
        "num-server-rounds=1 alpha=$ODRL_ALPHA seed=0 require-purpose='ds-cancer' recheck-consent=false duty-evidence='obtainConsent=IRB-2026-0142'" \
        2>&1 | tee "$out"
    rc="${PIPESTATUS[0]}"

    # Check that the population we got is the one we asked for. The override
    # syntax for simulation settings changed in Flower 1.33 and the old form is
    # deprecated, so an override that is quietly ignored would leave this job
    # measuring the same ten nodes five times over and reporting it as a
    # scalability result.
    got="$(grep -oE '\(([0-9]+) simulated SuperNodes\)' "$out" \
           | grep -oE '[0-9]+' | head -1)"
    declared="$(grep -oE 'policy round: [0-9]+ nodes declared' "$out" \
                | grep -oE '[0-9]+' | head -1)"
    echo "--- asked for $n, federation reports ${got:-?}, declared ${declared:-?}"
    if [ -n "$got" ] && [ "$got" != "$n" ]; then
        echo "ERROR: the population override was ignored. Try instead:" >&2
        echo "  flwr federation simulation-config local-sim --num-supernodes $n" >&2
        STATUS=1
        break
    fi

    echo "--- population $n finished, exit $rc"
    [ "$rc" -ne 0 ] && STATUS=1
done

echo ""
echo "=================================================="
echo "The result, for the paper:"
grep -h "COST" "${ODRL_LOGS_DIR}/odrl-scale_${SLURM_JOB_ID:-}.out" 2>/dev/null \
    || echo "(COST lines are above in this log)"

echo ""
echo "***** DONE (exit $STATUS) *****"
date '+%F %H:%M:%S'
exit $STATUS
