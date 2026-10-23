# Running the campaigns

Four scripts, in order. Everything is driven by `env.sh`, which is the only
file you should need to edit.

```bash
# once, on a login node
vi slurm/env.sh                  # partition, account, repo path
bash slurm/01_setup_env.sh       # build the conda environment
bash slurm/00_preflight.sh       # check it before queueing anything

# the campaigns
sbatch slurm/10_groundtruth.sbatch   # array job, the widened lattice
sbatch slurm/20_flower_sim.sbatch    # the real Flower federation

# when the array finishes
bash slurm/30_collect.sh
```

## What each one is for

| script | what it does | writes |
|---|---|---|
| `env.sh` | cluster and campaign settings, sourced by the rest | — |
| `01_setup_env.sh` | builds the conda environment in `$HOME` | the environment |
| `00_preflight.sh` | checks Python, torch, flwr, Ray, `ray.init()`, the partition | nothing |
| `10_groundtruth.sbatch` | trains every subset, at 20 federations, 5 replicates each | `results/shards/groundtruth.NNN.csv` |
| `20_flower_sim.sbatch` | runs `PolicyAwareFedAvg` as a Flower federation | `artifacts/model-bundle/` |
| `30_collect.sh` | joins the shards and checks the lattice is complete | `results/groundtruth_wide.csv` |

Run `00_preflight.sh` before anything else. A queued job that fails on a
missing import is discovered whenever somebody next looks at the queue, and
with a deadline close that is the expensive kind of mistake.

## Why the campaigns are shaped this way

**Replicates are the point of `10_groundtruth.sbatch`.** The committed lattice
in `results/groundtruth.csv` has one training run per (subset, federation).
A maximum taken over a thousand such candidates is biased upward, because the
winner is partly whichever candidate drew the kindest initialisation, so a
regret measured against it flatters whatever picked it. Five replicates let the
shortlist be chosen from one run and scored on the others, which turns the
regret figures from an upper bound into a measurement.
`odrl_fl.shortlist.split_replicates` is where that split happens, and its
`holdout` flag is what records which of the two you are looking at.

**`trial=0` reproduces the committed rows exactly.** The seeding in
`odrl_fl.fedavg.run_fedavg` keeps `trial=0` on the original single seed, so the
widened campaign is comparable with the three hours of compute already in the
repository rather than replacing it.
`tests/test_fedavg.py::TestTrials` is the guard.

**No GPU is requested, and that is not an oversight.** `odrl_fl/task.py` trains
a three-layer MLP over synthetic Gaussian mixtures. At roughly 0.65 s a run the
cost is Python and data handling, not matrix multiplication, so a GPU would sit
idle while the queue wait grew. The lever is array parallelism across CPUs:
102,300 runs is about 19 CPU-hours in total, which is half an hour per task
across 40 tasks.

**Everything is single-threaded on purpose.** `odrl_single_threaded` in
`env.sh` pins the BLAS thread counts. Without it, forty array tasks each
spawning one thread per core would contend for the node and run slower than
one task would.

## If a task fails

The sweep is resumable. Each shard skips the `(members, seed, trial)` triples
already in its own output file, so requeueing a failed task picks up where it
stopped:

```bash
sbatch --array=7,19 slurm/10_groundtruth.sbatch
```

`30_collect.sh` refuses to write a combined file if the lattice has holes, and
reports which consortium sizes are short. That check matters more than it
looks: a maximum taken over a lattice with holes is not the maximum, and
nothing downstream could tell.

## Changing the array size

Slurm reads `#SBATCH` lines before any shell runs, so `--array` in
`10_groundtruth.sbatch` cannot be set from `env.sh`. If you change one, change
the other. The job checks the two agree and exits rather than silently losing
part of the work.
