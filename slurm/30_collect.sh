#!/usr/bin/env bash
# Join the array tasks' output and check it is complete.
#
#   bash slurm/30_collect.sh
#
# Checks rather than concatenates blindly. A silently short campaign -- one
# task killed by the time limit, say -- would produce a lattice with holes, and
# a maximum taken over a lattice with holes is not the maximum. The row count
# and the duplicate check are what catch that.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$here/env.sh"

cd "$ODRL_REPO"
odrl_activate

out="results/groundtruth_wide.csv"

python - "$ODRL_SHARDS_DIR" "$out" <<'PY'
import csv
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

shards_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
files = sorted(shards_dir.glob("groundtruth.*.csv"))
if not files:
    sys.exit(f"no shard files in {shards_dir}")

header, rows, keys = None, [], Counter()
for path in files:
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if header is None:
            header = reader.fieldnames
        elif reader.fieldnames != header:
            sys.exit(f"{path.name} has a different header to {files[0].name}")
        shard_rows = list(reader)
    print(f"  {path.name}: {len(shard_rows)} rows")
    for row in shard_rows:
        keys[(row["members"], row["seed"], row.get("trial", "0"))] += 1
    rows.extend(shard_rows)

print(f"\n{len(files)} shards, {len(rows)} rows, {len(keys)} unique keys")

dupes = [k for k, n in keys.items() if n > 1]
if dupes:
    sys.exit(f"{len(dupes)} duplicated keys, first {dupes[0]}: shards overlap")

# What a complete campaign looks like, derived from the rows themselves rather
# than from a number typed here, so it stays right if the campaign is resized.
pool = 1 + max(int(m) for k in keys for m in k[0].split("-"))
seeds = {k[1] for k in keys}
trials = {k[2] for k in keys}
expected = (2 ** pool - 1) * len(seeds) * len(trials)
print(f"pool {pool}, {len(seeds)} seeds, {len(trials)} trials")
print(f"expected {expected} rows")

if len(keys) != expected:
    missing = expected - len(keys)
    print(f"\nINCOMPLETE: {missing} runs missing ({100 * missing / expected:.1f}%).")
    print("Requeue the failed array tasks; the script resumes per shard.")
    for size in range(1, pool + 1):
        have = sum(1 for k in keys if k[0].count("-") + 1 == size)
        want = len(list(combinations(range(pool), size))) * len(seeds) * len(trials)
        if have != want:
            print(f"  size {size}: {have} of {want}")
    sys.exit(1)

with out_path.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows)
print(f"\nComplete. Wrote {out_path}")
PY

echo
echo "Now re-run the analyses against it:"
echo "  python experiments/run_shortlist.py --truth $out"
echo "  python experiments/make_figures.py"
