# Cluster settings. This is the only file you should need to edit.
#
# Sourced by every script in this directory. Nothing here is guessed from the
# environment, because a wrong guess inside a queued job fails hours later
# instead of immediately.

# --- Slurm -----------------------------------------------------------------
# Leave a value empty and the corresponding #SBATCH line is omitted, which is
# the right thing on a cluster with a single default partition. Run
# `sinfo -s` to list partitions and `sacctmgr show assoc user=$USER` for
# accounts.
export ODRL_PARTITION=""
export ODRL_ACCOUNT=""

# The cluster caps jobs at 24 hours. Half an hour of headroom, so a job that
# overruns is still killed by Slurm after writing most of its output rather
# than at an arbitrary point.
export ODRL_TIME="23:30:00"

# --- The Python environment ------------------------------------------------
# Either a venv or a conda environment works, and `odrl_activate` picks
# whichever is present. A venv is tried first, because if you already have one
# that runs the tests then it is the one that should run the jobs.
#
# Set ODRL_VENV to a path to force a venv; leave it empty and the usual
# locations inside the checkout are tried.
export ODRL_VENV="${ODRL_VENV:-}"

# A user-owned conda environment in $HOME, as the cluster documentation
# requires: only administrators can install into the global Anaconda. Used only
# if no venv is found.
export ODRL_CONDA_ENV="odrl-fl"

# Python 3.12, and this matters. The project needs 3.11 or later, and Ray,
# which Flower's simulation backend requires, publishes no wheel for 3.13 --
# the same wall that stops the Flower path running on the development machine.
# Do not follow the cluster documentation's suggestion of 3.6.
export ODRL_PYTHON="3.12"

# --- Paths -----------------------------------------------------------------
# Where this checkout lives on the cluster.
export ODRL_REPO="${ODRL_REPO:-$HOME/odrl-fl}"

# Per-task output. Many array tasks appending to one CSV interleave rows, so
# each writes its own file and `30_collect.sh` joins them.
export ODRL_SHARDS_DIR="$ODRL_REPO/results/shards"
export ODRL_LOGS_DIR="$ODRL_REPO/slurm/logs"

# --- The campaign ----------------------------------------------------------
export ODRL_POOL=10
export ODRL_SEEDS=20
export ODRL_TRIALS=5
export ODRL_ROUNDS=20
export ODRL_ALPHA=0.3
export ODRL_ARRAY_TASKS=40

# --- Helpers ---------------------------------------------------------------

# pip has to be told to trust the host behind the cluster's MITM proxy.
export ODRL_PIP_TRUST="--trusted-host pypi.org --trusted-host files.pythonhosted.org"

# Activate whichever Python environment exists, and say which one it was.
#
# A batch job runs a non-interactive shell, where `conda` is not a function and
# `conda activate` therefore fails unless the profile script is sourced first.
# That is the usual reason a job dies on its first line.
odrl_activate() {
    # $VIRTUAL_ENV first. Slurm exports the submitting environment into the
    # job, so a venv that was active when you ran sbatch names itself here and
    # nothing has to be configured at all.
    local candidate
    for candidate in \
        "$ODRL_VENV" \
        "${VIRTUAL_ENV:-}" \
        "$ODRL_REPO/venv" \
        "$ODRL_REPO/.venv" \
        "$HOME/venv"
    do
        if [[ -n "$candidate" && -f "$candidate/bin/activate" ]]; then
            # shellcheck disable=SC1091
            source "$candidate/bin/activate"
            echo "environment: venv at $candidate"
            echo "python:      $(command -v python)"
            return 0
        fi
    done

    local base
    if base="$(conda info --base 2>/dev/null)"; then
        # shellcheck disable=SC1091
        source "$base/etc/profile.d/conda.sh" || return 1
        if conda activate "$ODRL_CONDA_ENV" 2>/dev/null; then
            echo "environment: conda '$ODRL_CONDA_ENV'"
            echo "python:      $(command -v python)"
            return 0
        fi
        echo "no conda environment '$ODRL_CONDA_ENV' and no venv found" >&2
        return 1
    fi

    echo "no Python environment found. Set ODRL_VENV to your venv, or run" >&2
    echo "01_setup_env.sh to build a conda environment." >&2
    return 1
}

# The old name, kept so nothing that calls it breaks.
odrl_activate_conda() { odrl_activate; }

# A three-layer MLP over synthetic data does not benefit from intra-op
# threading, and 40 array tasks each spawning one thread per core would
# thrash. One thread apiece is both correct and faster here.
odrl_single_threaded() {
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export TORCH_NUM_THREADS=1
}
