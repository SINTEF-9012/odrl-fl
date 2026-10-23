#!/usr/bin/env bash
# Build the conda environment. Run once, on a login node.
#
# The compute nodes may not reach PyPI, and in any case building the
# environment inside every array task would be forty redundant installs. So
# this runs interactively, once, and the jobs only activate what it made.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$here/env.sh"

echo "Repository: $ODRL_REPO"
echo "Environment: $ODRL_CONDA_ENV (Python $ODRL_PYTHON)"
echo

if [[ ! -d "$ODRL_REPO" ]]; then
    echo "No checkout at $ODRL_REPO. Clone it there, or set ODRL_REPO." >&2
    exit 1
fi

base="$(conda info --base)"
# shellcheck disable=SC1091
source "$base/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ODRL_CONDA_ENV"; then
    echo "Environment '$ODRL_CONDA_ENV' exists, reusing it."
else
    echo "Creating '$ODRL_CONDA_ENV'..."
    conda create --yes --name "$ODRL_CONDA_ENV" "python=$ODRL_PYTHON"
fi

conda activate "$ODRL_CONDA_ENV"

# conda for the compiled dependencies, as the cluster documentation asks.
# CPU-only torch on purpose: the model is a three-layer MLP and the per-run
# cost is Python and data handling, not matrix multiplication, so a CUDA build
# would add gigabytes and buy nothing.
echo
echo "Installing numpy and CPU torch through conda..."
conda install --yes -c pytorch -c conda-forge numpy "pytorch::pytorch" cpuonly

# The rest through pip, since flwr is not packaged for conda. --trusted-host
# gets past the cluster's proxy.
echo
echo "Installing the project through pip..."
cd "$ODRL_REPO"
# shellcheck disable=SC2086
pip install -e ".[dev]" $ODRL_PIP_TRUST

# Ray is what Flower's simulation backend needs, and it is the one piece that
# has never run in this project. Installed separately so a failure here is
# obviously about Ray and not about the project.
echo
echo "Installing Ray for the Flower simulation backend..."
# shellcheck disable=SC2086
pip install "ray[default]" $ODRL_PIP_TRUST || {
    echo
    echo "Ray failed to install. The offline sweeps do not need it; only" >&2
    echo "20_flower_sim.sbatch does. Continuing." >&2
}

mkdir -p "$ODRL_SHARDS_DIR" "$ODRL_LOGS_DIR"

echo
echo "Done. Now run:  bash $here/00_preflight.sh"
