# Cluster settings and shared helpers. This is the only file you should need
# to edit.
#
# The conventions here follow a campaign that already runs on this server --
# see `inTrust/benchmark/slurm/run_campaign.job`. The first version of these
# scripts was written against generic Slurm documentation instead, and every
# job died on its third line.

# --- Slurm -----------------------------------------------------------------
# `sinfo -s` lists partitions; `sacctmgr show assoc user=$USER` lists accounts.
# Leave a value empty and the corresponding #SBATCH line is simply absent.
export ODRL_PARTITION="${ODRL_PARTITION:-sintef}"
export ODRL_ACCOUNT="${ODRL_ACCOUNT:-}"

# The cluster caps jobs at 24 hours.
export ODRL_TIME="${ODRL_TIME:-00-23:30:00}"

# Logs go outside the repository, which is where the working campaign on this
# machine puts them and which means a fresh clone needs no directory created
# for it. Slurm does not create the parent of an --output path: a job whose log
# directory is missing fails before running a line, and leaves no log to say so.
export ODRL_LOGS_DIR="${ODRL_LOGS_DIR:-$HOME/logs/slurm}"

# --- The Python environment ------------------------------------------------
# A user-owned environment, as the cluster requires: only administrators can
# install into the global Anaconda.
#
# NOTE: the conda environment named `venv` on this machine belongs to another
# project and has neither torch nor flwr. This campaign needs its own, which
# `01_setup_env.sh` builds.
export ODRL_CONDA_ENV="${ODRL_CONDA_ENV:-odrl-fl}"

# Set ODRL_VENV to a path to force a plain virtualenv instead.
export ODRL_VENV="${ODRL_VENV:-}"

# Python 3.12, and this matters. The project needs 3.11 or later, and Ray,
# which Flower's simulation backend requires, publishes no wheel for 3.13 --
# the same wall that stops the Flower path running on Windows. 3.12 is also
# what the working campaign on this machine uses.
export ODRL_PYTHON="${ODRL_PYTHON:-3.12}"

# --- Paths -----------------------------------------------------------------
export ODRL_REPO="${ODRL_REPO:-$HOME/odrl-fl}"

# Per-task output. Many array tasks appending to one CSV interleave rows, so
# each writes its own file and `30_collect.sh` joins them.
export ODRL_SHARDS_DIR="${ODRL_SHARDS_DIR:-$ODRL_REPO/results/shards}"

# --- The campaign ----------------------------------------------------------
# All overridable at submission time, so a trial run needs no file edit:
#   sbatch --export=ALL,ODRL_SEEDS=2,ODRL_TRIALS=1 slurm/10_groundtruth.sbatch
export ODRL_POOL="${ODRL_POOL:-10}"
export ODRL_SEEDS="${ODRL_SEEDS:-20}"
export ODRL_TRIALS="${ODRL_TRIALS:-5}"
export ODRL_ROUNDS="${ODRL_ROUNDS:-20}"
export ODRL_ALPHA="${ODRL_ALPHA:-0.3}"

# Must match the `#SBATCH --array` line in 10_groundtruth.sbatch. Slurm reads
# that line before any shell runs, so it cannot be set from here; the job
# checks the two agree.
export ODRL_ARRAY_TASKS="${ODRL_ARRAY_TASKS:-40}"

# --- Helpers ---------------------------------------------------------------

# pip has to be told to trust the host behind the cluster's proxy.
export ODRL_PIP_TRUST="--trusted-host pypi.org --trusted-host files.pythonhosted.org"

# Activate a conda environment, surviving `set -u`.
#
# Two separate hazards, both of which have already cost a submission here:
#
#   - `conda` is not a function in a non-interactive shell, so `conda activate`
#     fails unless conda's profile script is sourced first;
#   - an environment's own `activate.d` hooks are not written for `set -u`.
#     This one has a `libblas_mkl_activate.sh` that reads MKL_INTERFACE_LAYER
#     without a default, which terminates the caller outright.
#
# So the flag is suspended across the whole activation and restored after.
# Every `conda activate` in this directory goes through here.
odrl_conda_activate() {
    local env_name="$1" base rc had_u=0
    case $- in *u*) had_u=1; set +u ;; esac

    if base="$(conda info --base 2>/dev/null)"; then
        # shellcheck disable=SC1091
        . "$base/etc/profile.d/conda.sh"
        conda activate "$env_name"
        rc=$?
    else
        rc=1
    fi

    [ "$had_u" = 1 ] && set -u
    return $rc
}

# Activate a Python environment and say which one it was.
#
# A batch job runs a non-interactive shell, where `conda` is not a function and
# `conda activate` therefore fails unless the profile script is sourced first.
# That is the usual reason a job dies immediately.
#
# Returns 0 and echoes what it found, or returns 1 after saying what to do.
odrl_activate() {
    # A venv `activate` script written for an older Python references
    # $PYTHONHOME and $PS1 without a default, which terminates the caller
    # outright under `set -u`. Suspend it across the source and restore it.
    local had_u=0
    case $- in *u*) had_u=1; set +u ;; esac

    local candidate found=""
    for candidate in \
        "$ODRL_VENV" \
        "${VIRTUAL_ENV:-}" \
        "$ODRL_REPO/venv" \
        "$ODRL_REPO/.venv"
    do
        if [ -n "$candidate" ] && [ -f "$candidate/bin/activate" ]; then
            # shellcheck disable=SC1091
            . "$candidate/bin/activate"
            found="venv at $candidate"
            break
        fi
    done

    [ "$had_u" = 1 ] && set -u

    if [ -z "$found" ] && odrl_conda_activate "$ODRL_CONDA_ENV" 2>/dev/null; then
        found="conda env '$ODRL_CONDA_ENV'"
    fi

    if [ -z "$found" ]; then
        echo "ERROR: no Python environment for this campaign." >&2
        echo "  Looked for a venv, then for the conda env '$ODRL_CONDA_ENV'." >&2
        echo "  The conda env 'venv' on this machine belongs to another" >&2
        echo "  project and has no torch. Build this one once with:" >&2
        echo "    bash $ODRL_REPO/slurm/01_setup_env.sh" >&2
        return 1
    fi

    echo "Environment: $found"
    echo "Using python: $(command -v python)"
    return 0
}

# What is actually installed. Absence of torch was the thing that could not be
# seen from either end of the last two failures, so every job now says.
odrl_report_versions() {
    python - <<'PY' 2>&1
import importlib.metadata as md
import sys

print(f"  python {sys.version.split()[0]}")
for name in ("torch", "numpy", "flwr", "ray"):
    try:
        print(f"  {name} {md.version(name)}")
    except md.PackageNotFoundError:
        print(f"  {name} MISSING")
PY
}

# A three-layer MLP over synthetic data does not benefit from intra-op
# threading, and forty array tasks each spawning one thread per core would
# contend for the node. One thread apiece is both correct and faster here.
odrl_single_threaded() {
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export TORCH_NUM_THREADS=1
}

# The old name, kept so nothing that calls it breaks.
odrl_activate_conda() { odrl_activate; }
