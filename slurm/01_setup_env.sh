#!/bin/bash
# ============================================================================
# Build the conda environment. Run once, on a login node.
#
#   bash slurm/01_setup_env.sh
#
# This is the step that was skipped the first two times, and skipping it is why
# every job died: `odrl_activate` found no environment and exited.
#
# The campaign gets its own environment rather than reusing the conda env named
# `venv` on this machine. That one belongs to another project, has no torch, so
# sharing it would save nothing, and this campaign would add torch, flwr and
# Ray to something another project depends on.
#
# The compute nodes may not reach PyPI, and building the environment inside
# every array task would be forty redundant installs, so this runs once and the
# jobs only activate what it made.
# ============================================================================
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
. "$here/env.sh"

echo "Repository:  $ODRL_REPO"
echo "Environment: $ODRL_CONDA_ENV (Python $ODRL_PYTHON)"
echo "Logs:        $ODRL_LOGS_DIR"
echo ""

if [ ! -d "$ODRL_REPO" ]; then
    echo "ERROR: no checkout at $ODRL_REPO. Clone it there, or set ODRL_REPO." >&2
    exit 1
fi

# Slurm does not create the parent of an --output path, and a job whose log
# directory is missing fails before running a line and leaves no log to say so.
mkdir -p "$ODRL_LOGS_DIR" || exit 1
echo "Log directory ready."

if ! command -v conda >/dev/null; then
    echo "ERROR: conda is not on PATH." >&2
    exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ODRL_CONDA_ENV"; then
    echo "Environment '$ODRL_CONDA_ENV' exists, reusing it."
else
    echo "Creating '$ODRL_CONDA_ENV'..."
    conda create --yes --name "$ODRL_CONDA_ENV" "python=$ODRL_PYTHON" || exit 1
fi

# Through the helper, not `conda activate` directly. This environment has an
# `activate.d/libblas_mkl_activate.sh` that reads MKL_INTERFACE_LAYER without a
# default, and under the `set -u` at the top of this file that ends the script
# on the spot -- which is exactly how an earlier version of this script died
# after reporting "reusing it" and before installing anything.
odrl_conda_activate "$ODRL_CONDA_ENV" || {
    echo "ERROR: could not activate '$ODRL_CONDA_ENV'." >&2
    exit 1
}

# A guard against installing into the wrong place. Without it, a conda that
# silently failed to activate would have this script install torch into the
# base environment or into another project's.
if [ "${CONDA_DEFAULT_ENV:-}" != "$ODRL_CONDA_ENV" ]; then
    echo "ERROR: active environment is '${CONDA_DEFAULT_ENV:-<none>}', not" >&2
    echo "  '$ODRL_CONDA_ENV'. Refusing to install into it." >&2
    exit 1
fi
echo "Using python: $(command -v python)"
echo ""

# conda for the compiled dependencies, as the cluster documentation asks.
# CPU-only torch on purpose: the model is a three-layer MLP and the per-run
# cost is Python and data handling, not matrix multiplication, so a CUDA build
# would add gigabytes and buy nothing.
#
# Skipped when torch already imports. The conda solver takes minutes on an
# environment it has nothing to change, and re-solving a working install is a
# good way to break one.
if python -c "import torch, numpy" 2>/dev/null; then
    echo "torch and numpy already present, skipping the conda step."
else
    echo "Installing numpy and CPU torch through conda..."
    conda install --yes -c pytorch -c conda-forge numpy pytorch cpuonly || {
        echo "conda install failed; falling back to pip for torch." >&2
        # shellcheck disable=SC2086
        pip install torch --index-url https://download.pytorch.org/whl/cpu $ODRL_PIP_TRUST
    }
fi

# The rest through pip, since flwr is not packaged for conda. --trusted-host
# gets past the cluster's proxy.
echo ""
echo "Installing the project through pip..."
cd "$ODRL_REPO" || exit 1
# shellcheck disable=SC2086
pip install -e ".[dev]" $ODRL_PIP_TRUST || exit 1

# Ray is what Flower's simulation backend needs, and it is the one piece that
# has never run in this project. Installed separately so a failure here is
# obviously about Ray and not about the project, and is not fatal: only
# 20_flower_sim.sbatch needs it.
echo ""
echo "Installing Ray for the Flower simulation backend..."
# shellcheck disable=SC2086
pip install "ray[default]" $ODRL_PIP_TRUST || {
    echo "" >&2
    echo "Ray failed to install. The lattice campaign does not need it;" >&2
    echo "only 20_flower_sim.sbatch does. Continuing." >&2
}

mkdir -p "$ODRL_SHARDS_DIR"

echo ""
echo "Installed:"
odrl_report_versions

echo ""
echo "Done. Next:"
echo "  bash $here/00_preflight.sh"
