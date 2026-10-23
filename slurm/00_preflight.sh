#!/usr/bin/env bash
# Check everything the campaigns need, before anything is queued.
#
# A queued job that fails on a missing import wastes a place in the queue and
# is discovered whenever someone looks. With a deadline a week away that is the
# expensive kind of mistake, so every assumption is checked here instead.
#
# Exits non-zero if anything required is missing. Ray is reported but not
# required: only the Flower simulation needs it.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$here/env.sh"

fail=0
warn=0
ok()   { printf '  ok    %s\n' "$1"; }
bad()  { printf '  FAIL  %s\n' "$1"; fail=$((fail + 1)); }
note() { printf '  warn  %s\n' "$1"; warn=$((warn + 1)); }

echo "Preflight for the ODRL-FL campaigns"
echo

echo "Cluster:"
command -v sbatch >/dev/null && ok "sbatch found" || bad "sbatch not on PATH"
if command -v sinfo >/dev/null; then
    if [[ -n "$ODRL_PARTITION" ]]; then
        if sinfo -h -o '%P' | tr -d '*' | grep -qx "$ODRL_PARTITION"; then
            ok "partition '$ODRL_PARTITION' exists"
        else
            bad "partition '$ODRL_PARTITION' not in: $(sinfo -h -o '%P' | tr '\n' ' ')"
        fi
    else
        note "ODRL_PARTITION empty, the default partition will be used: $(sinfo -h -o '%P' | tr -d '*' | head -1)"
    fi
fi

echo
echo "Repository:"
if [[ -d "$ODRL_REPO" ]]; then
    ok "checkout at $ODRL_REPO"
    [[ -f "$ODRL_REPO/results/groundtruth.csv" ]] \
        && ok "committed ground truth present" \
        || note "no results/groundtruth.csv; the offline analyses need it"
else
    bad "no checkout at $ODRL_REPO"
fi

echo
echo "Environment:"
if odrl_activate_conda; then
    ok "conda environment '$ODRL_CONDA_ENV' activates"
    # Exits non-zero if anything required is missing, so the shell's own
    # counter sees it. Printing FAIL from a subprocess and exiting 0 would
    # report "Ready" with torch absent, which is worse than no check.
    if python - <<'PY'
import importlib.metadata as md
import sys

bad = 0
v = sys.version_info
if v >= (3, 13):
    print(f"  FAIL  Python {v.major}.{v.minor}: Ray has no wheel for 3.13")
    bad += 1
elif v < (3, 11):
    print(f"  FAIL  Python {v.major}.{v.minor}: the project needs 3.11 or later")
    bad += 1
else:
    print(f"  ok    Python {v.major}.{v.minor}.{v.micro}")

for name in ("torch", "numpy", "flwr"):
    try:
        print(f"  ok    {name} {md.version(name)}")
    except md.PackageNotFoundError:
        print(f"  FAIL  {name} is not installed")
        bad += 1

try:
    print(f"  ok    ray {md.version('ray')}")
except md.PackageNotFoundError:
    print("  warn  ray missing: only the Flower simulation needs it")

sys.exit(1 if bad else 0)
PY
    then :; else bad "the environment is missing a required package"; fi
    # Check the project itself imports, which catches a half-finished
    # `pip install -e` that metadata alone would not.
    if (cd "$ODRL_REPO" && python -c "
import odrl_fl.fedavg, odrl_fl.shortlist, odrl_fl.strategy
import inspect
assert 'trial' in inspect.signature(odrl_fl.fedavg.run_fedavg).parameters
" 2>/dev/null); then
        ok "odrl_fl imports, and run_fedavg takes 'trial'"
    else
        bad "odrl_fl does not import, or predates the 'trial' parameter"
    fi
else
    bad "conda environment '$ODRL_CONDA_ENV' does not activate"
fi

echo
echo "Flower simulation backend:"
if python -c "import ray" 2>/dev/null; then
    # Starting a local Ray is the thing that has never been tried in this
    # project, so it is tried here rather than inside a queued job.
    if timeout 180 python - <<'PY' 2>/dev/null
import ray
ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
ray.shutdown()
print("  ok    ray.init() succeeded")
PY
    then :; else note "ray.init() failed; the Flower simulation will not run"; fi
else
    note "ray not importable; skip 20_flower_sim.sbatch"
fi

echo
echo "Scratch directories:"
mkdir -p "$ODRL_SHARDS_DIR" "$ODRL_LOGS_DIR" 2>/dev/null \
    && ok "$ODRL_SHARDS_DIR and $ODRL_LOGS_DIR writable" \
    || bad "cannot create $ODRL_SHARDS_DIR"

echo
if (( fail )); then
    echo "$fail check(s) failed. Fix those before queueing anything."
    exit 1
fi
echo "Ready${warn:+ ($warn warning(s))}. Next:  sbatch $here/10_groundtruth.sbatch"
