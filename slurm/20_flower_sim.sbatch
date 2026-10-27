#!/bin/bash
# ============================================================================
# Run the policy-aware strategy as a real Flower federation.
#
#   sbatch slurm/20_flower_sim.sbatch
#   tail -f ~/logs/slurm/odrl-flwr_<jobid>.out
#
# This is the one thing in the project that has never happened.
# `odrl_fl/strategy.py` has 33 tests over it, including one asserting that its
# aggregation matches Flower's own, but no federation has ever been started:
# Flower's simulation backend needs Ray, and Ray ships no Windows wheel for
# Python 3.13.
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
# ============================================================================
#SBATCH --job-name odrl-flwr
#SBATCH --output /home/rustemd/logs/slurm/odrl-flwr_%j.out
#SBATCH --error /home/rustemd/logs/slurm/odrl-flwr_%j.err
#SBATCH --partition sintef
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 16G
#SBATCH --time 00-04:00:00

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
echo "Packages:"
odrl_report_versions

# Ray is the only reason this job needs the cluster at all.
if ! python -c "import ray" 2>/dev/null; then
    echo "" >&2
    echo "ERROR: Ray is not installed, so the simulation backend cannot run." >&2
    echo "  pip install 'ray[default]' $ODRL_PIP_TRUST" >&2
    exit 1
fi

# Ray writes a great deal to its session directory. Keep it on node-local
# scratch where one exists, so a shared home quota is not what fails the run.
export RAY_TMPDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/ray-${SLURM_JOB_ID:-$$}"
mkdir -p "$RAY_TMPDIR"
trap 'rm -rf "$RAY_TMPDIR"' EXIT

echo ""
echo "Federation: population of $ODRL_POOL nodes, alpha $ODRL_ALPHA," \
     "$ODRL_ROUNDS rounds"
echo "Ray tmp:    $RAY_TMPDIR"
echo ""

# --stream is not optional. Without it `flwr run` only *submits* the run and
# returns, the job then ends, Slurm tears down the SuperLink it started, and the
# federation dies before training a round -- while the job reports exit 0 after
# nine seconds. --stream follows the run and blocks until it finishes.
#
# `require-purpose` is set deliberately. Left empty, the request names only
# actions, which nearly every contributor grants, so admission excludes almost
# nobody and the run demonstrates nothing. A real purpose is what makes the
# policy gate bite.
# `duty-evidence` is not optional either, and the reason is the mechanism
# working rather than failing. Composition accumulates duties, and
# `obtainConsent` is a precondition: it cannot be discharged by the run that
# depends on it, so the server refuses to start without an assertion that
# somebody discharged it out of band. Run 3074 did exactly that and stopped
# with UnmetPrecondition before training a round. To get past the gate the
# assertion has to be made, which is the point -- nothing validates the
# reference, but a person has to put it there and it is recorded in the bundle.
EVIDENCE="${ODRL_EVIDENCE:-obtainConsent=IRB-2026-0142}"

flwr run . local-sim --stream --run-config \
    "num-server-rounds=$ODRL_ROUNDS alpha=$ODRL_ALPHA seed=0 require-purpose='ds-cancer' recheck-consent=true duty-evidence='$EVIDENCE'"
STATUS=$?

# `flwr run --stream` returns 0 even when the simulation it streamed raised:
# run 3074 printed "Exit Code: 700" and a traceback, and this script still
# reported exit 0. So the artefact is checked first and the banner reports what
# the check found, rather than the banner going out before anything is known.
bundle="$(find artifacts -name 'manifest*.json' -newermt '-1 day' 2>/dev/null | head -1)"
if [ -z "$bundle" ] && [ "$STATUS" -eq 0 ]; then
    echo "" >&2
    echo "WARNING: flwr exited 0 but no bundle was written, so the run did" >&2
    echo "  not finish training. Look above for 'Exit Code:' and a" >&2
    echo "  traceback -- flwr reports a failed simulation in its output" >&2
    echo "  rather than in its exit status." >&2
    STATUS=1
fi

echo ""
echo "***** DONE (exit $STATUS) *****"
date '+%F %H:%M:%S'
if [ -n "$bundle" ]; then
    echo "Bundle: $bundle"
    echo "Next, check the two paths agree on the admitted set:"
    echo "  python experiments/run_fedavg.py --contributors <admitted>"
fi
exit $STATUS
