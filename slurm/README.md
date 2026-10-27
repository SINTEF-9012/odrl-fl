# Running the campaigns on ds01

These follow the conventions of a campaign that already runs on this server,
`inTrust/benchmark/slurm/run_campaign.job`: the `sintef` partition, logs in
`~/logs/slurm`, a conda environment activated through conda's profile script,
and no `set -e` in the job itself. The first version of these scripts was
written against generic Slurm documentation instead, and every job died on its
third line twice over.

`env.sh` holds everything configurable and is the only file you should need to
edit.

## One-time setup

```bash
git clone <repo-url> ~/odrl-fl && cd ~/odrl-fl
```

```bash
bash slurm/01_setup_env.sh
```

That creates the conda environment `odrl-fl` with Python 3.12, installs CPU
torch and the project, makes `~/logs/slurm`, and tries Ray.

**Not the `venv` environment.** The conda environment named `venv` on this
machine belongs to another project. Its requirements are `google-adk` and
`litellm`; it has neither torch nor flwr, so there is nothing to be saved by
sharing it, and this campaign would add torch, flwr and Ray to something
another project depends on.

Python **3.12** specifically. The project needs 3.11 or later, and Ray, which
Flower's simulation backend requires, publishes no wheel for 3.13 — the same
wall that stops the Flower path running on Windows.

## Then, in order

```bash
bash slurm/00_preflight.sh
```

Read its verdict rather than skipping it. It checks the partition exists, that
the log directory exists, that torch, numpy and flwr are installed, that
`odrl_fl` imports with the `trial` parameter the replicate campaign depends on,
and that `ray.init()` actually works — which is the one thing in this project
that had never been tried. It always ends with a `VERDICT:` line; a run that
stops without one is a bug in the script, not a result.

**One task first**, so a mistake costs one task instead of forty:

```bash
sbatch --array=0 slurm/10_groundtruth.sbatch && sq
```

```bash
tail -f ~/logs/slurm/odrl-gt_*_0.out
```

Expect a launch banner, the package versions, a shard assignment, and
run-rate lines within a minute. Then the real thing:

```bash
sbatch slurm/10_groundtruth.sbatch && sbatch slurm/20_flower_sim.sbatch
```

```bash
bash slurm/30_collect.sh
```

## What each script is for

| script | what it does | writes |
|---|---|---|
| `env.sh` | cluster and campaign settings, and the shared helpers | — |
| `01_setup_env.sh` | builds the conda environment and `~/logs/slurm` | the environment |
| `00_preflight.sh` | checks everything, always ends with a verdict | nothing |
| `10_groundtruth.sbatch` | trains every subset, 20 federations, 5 replicates | `results/shards/groundtruth.NNN.csv` |
| `20_flower_sim.sbatch` | runs `PolicyAwareFedAvg` as a Flower federation | `artifacts/` |
| `30_collect.sh` | joins the shards and checks the lattice is complete | `results/groundtruth_wide.csv` |

Campaign parameters can be overridden at submission without editing anything:

```bash
sbatch --export=ALL,ODRL_SEEDS=2,ODRL_TRIALS=1 slurm/10_groundtruth.sbatch
```

## Why the campaigns are shaped this way

**Replicates are the point of `10_groundtruth.sbatch`.** The committed lattice
in `results/groundtruth.csv` has one training run per (subset, federation). A
maximum taken over a thousand such candidates is biased upward, because the
winner is partly whichever candidate drew the kindest initialisation, so a
regret measured against it flatters whatever picked it. Five replicates let the
shortlist be chosen from one run and scored on the others, which turns the
regret figures from an upper bound into a measurement.
`odrl_fl.shortlist.split_replicates` is where that split happens, and its
`holdout` flag records which of the two you are looking at.

**`trial=0` reproduces the committed rows exactly.** The seeding in
`odrl_fl.fedavg.run_fedavg` keeps `trial=0` on the original single seed, so the
widened campaign is comparable with the three hours of compute already in the
repository rather than replacing it.
`tests/test_fedavg.py::TestTrials` is the guard.

**No GPU is requested, and that is not an oversight.** `odrl_fl/task.py` trains
a three-layer MLP over synthetic Gaussian mixtures. At roughly 0.65 s a run the
cost is Python and data handling, not matrix multiplication, so a GPU would sit
idle while the queue wait grew. The lever is array parallelism across CPUs:
102,300 runs is about 19 CPU-hours, or half an hour per task across 40 tasks.

**Everything is single-threaded on purpose.** `odrl_single_threaded` in
`env.sh` pins the BLAS thread counts. Without it, forty array tasks each
spawning one thread per core would contend for the node and run slower than one
task would.

**No `set -e` in the jobs.** A job that reports a problem and exits with a
status is more useful than one that vanishes, which is how the first two
attempts failed. The scripts check their preconditions and say what is wrong.

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

To see what happened to a job that has already left the queue: `squeue` only
shows queued and running jobs, and `sacct` reports "Slurm accounting storage is
disabled" on this cluster, so the logs are the only record. Every job here ends
with a `***** DONE (exit N, M rows) *****` banner for exactly that reason — a
log that stops without one was killed rather than finished.

```bash
grep -h '\*\*\*\*\* DONE' ~/logs/slurm/odrl-gt_*.out | sort | uniq -c
```

```bash
wc -l results/shards/*.csv | tail -1
```

Those two answer what state the campaign is in: how many tasks finished and
with what exit code, and how many rows reached disk. A task that failed early
leaves a log whose last lines name the reason.

## Changing the array size

Slurm reads `#SBATCH` lines before any shell runs, so `--array` in
`10_groundtruth.sbatch` cannot be set from `env.sh`. If you change one, change
`ODRL_ARRAY_TASKS` too. The job cross-checks them against
`SLURM_ARRAY_TASK_MIN`/`MAX` and exits rather than silently covering part of
the work — except for a single-task run, which it recognises as a trial.
