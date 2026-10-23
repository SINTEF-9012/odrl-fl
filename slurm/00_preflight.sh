#!/bin/bash
# ============================================================================
# Check everything the campaigns need, before anything is queued.
#
#   bash slurm/00_preflight.sh
#
# A queued job that fails on a missing import wastes a place in the queue and
# is discovered whenever somebody next looks. With a deadline close that is the
# expensive kind of mistake, so every assumption is checked here instead.
#
# It must be impossible for this script to stop without saying something. An
# earlier version printed "Environment:" and then exited silently, because the
# environment check killed it -- so the one tool whose job was to prevent a
# wasted submission failed at exactly that, twice. Hence the EXIT trap below:
# every path out of here prints a verdict, and a run that ends without one is
# visibly truncated rather than ambiguously so.
#
# Exits non-zero if anything required is missing. Ray is reported but not
# required: only the Flower simulation needs it.
# ============================================================================
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
. "$here/env.sh"

fail=0
warn=0
ok()   { printf '  ok    %s\n' "$1"; }
bad()  { printf '  FAIL  %s\n' "$1"; fail=$((fail + 1)); }
note() { printf '  warn  %s\n' "$1"; warn=$((warn + 1)); }

verdict() {
    local status=$?
    echo ""
    if [ "$fail" -gt 0 ]; then
        echo "VERDICT: $fail check(s) failed, $warn warning(s)."
        echo "Fix those before queueing anything."
    elif [ "$status" -ne 0 ]; then
        echo "VERDICT: the preflight itself stopped early (exit $status)."
        echo "That is a bug in this script, not a result. Nothing was checked"
        echo "after the last line above."
    else
        echo "VERDICT: ready${warn:+, with $warn warning(s)}."
        echo ""
        echo "One task first, so a mistake costs one instead of forty:"
        echo "  sbatch --array=0 $here/10_groundtruth.sbatch"
        echo "  tail -f $ODRL_LOGS_DIR/odrl-gt_*_0.out"
    fi
}
trap verdict EXIT

echo "Preflight for the ODRL-FL campaigns"
echo ""

echo "Cluster:"
if command -v sbatch >/dev/null; then ok "sbatch found"; else bad "sbatch not on PATH"; fi
if command -v sinfo >/dev/null; then
    partitions="$(sinfo -h -o '%P' 2>/dev/null | tr -d '*' | sort -u | tr '\n' ' ')"
    if [ -n "$ODRL_PARTITION" ]; then
        if printf '%s' "$partitions" | tr ' ' '\n' | grep -qx "$ODRL_PARTITION"; then
            ok "partition '$ODRL_PARTITION' exists"
        else
            bad "partition '$ODRL_PARTITION' not among: $partitions"
        fi
    else
        note "ODRL_PARTITION empty; the default will be used. Available: $partitions"
    fi
fi

echo ""
echo "Paths:"
if [ -d "$ODRL_REPO" ]; then
    ok "checkout at $ODRL_REPO"
    if [ -f "$ODRL_REPO/results/groundtruth.csv" ]; then
        ok "committed ground truth present"
    else
        note "no results/groundtruth.csv; the offline analyses need it"
    fi
else
    bad "no checkout at $ODRL_REPO"
fi
# The failure that cost two submissions: Slurm does not create the parent of an
# --output path, and a job whose log directory is missing dies before running a
# line, with no log to say why.
if [ -d "$ODRL_LOGS_DIR" ]; then
    ok "log directory $ODRL_LOGS_DIR"
else
    bad "no log directory $ODRL_LOGS_DIR -- run: mkdir -p $ODRL_LOGS_DIR"
fi
if mkdir -p "$ODRL_SHARDS_DIR" 2>/dev/null; then
    ok "shard directory $ODRL_SHARDS_DIR writable"
else
    bad "cannot create $ODRL_SHARDS_DIR"
fi

echo ""
echo "Environment:"
# In a subshell, so that however activation fails it cannot end this script.
# That is what went wrong before.
env_report="$( (odrl_activate && odrl_report_versions) 2>&1 )"
env_ok=$?
printf '%s\n' "$env_report" | sed 's/^/  /'
if [ "$env_ok" -ne 0 ]; then
    bad "no usable Python environment (see above)"
else
    ok "environment activates"
    for pkg in torch numpy flwr; do
        if printf '%s' "$env_report" | grep -q "$pkg MISSING"; then
            bad "$pkg is not installed -- run 01_setup_env.sh"
        fi
    done
    if printf '%s' "$env_report" | grep -q "ray MISSING"; then
        note "ray missing: only 20_flower_sim.sbatch needs it"
    fi
fi

echo ""
echo "Project:"
# Metadata alone would not catch a half-finished `pip install -e`, and the
# `trial` parameter is what the whole replicate campaign depends on.
if project="$( (odrl_activate >/dev/null 2>&1; cd "$ODRL_REPO" && python - <<'PY'
import inspect

import odrl_fl.fedavg
import odrl_fl.shortlist  # noqa: F401
import odrl_fl.strategy  # noqa: F401

if "trial" not in inspect.signature(odrl_fl.fedavg.run_fedavg).parameters:
    raise SystemExit("run_fedavg has no 'trial' parameter: the checkout is old")
print("odrl_fl imports, and run_fedavg takes 'trial'")
PY
) 2>&1 )"; then
    ok "$project"
else
    bad "${project:-odrl_fl does not import}"
fi

echo ""
echo "Flower simulation backend:"
if (odrl_activate >/dev/null 2>&1; python -c "import ray" 2>/dev/null); then
    # Starting a local Ray is the thing that has never been tried in this
    # project, so it is tried here rather than inside a queued job.
    if (odrl_activate >/dev/null 2>&1; timeout 180 python - <<'PY'
import ray

ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
ray.shutdown()
PY
    ) 2>/dev/null; then
        ok "ray.init() succeeded"
    else
        note "ray.init() failed; the Flower simulation will not run"
    fi
else
    note "ray not importable; skip 20_flower_sim.sbatch"
fi

if [ "$fail" -gt 0 ]; then
    exit 1
fi
exit 0
