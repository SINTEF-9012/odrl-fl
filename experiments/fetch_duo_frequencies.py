"""Derive an empirical DUO code distribution from EGA's public metadata.

The corpus for the erosion study has to come from somewhere. Inventing a
distribution would leave the headline result unanswerable -- a reviewer asks
why erosion looks like *that* and the honest answer would be "because we chose
it". EGA publishes real data use conditions on its policy records, without
registration, so the distribution can be sampled from something real instead.

Run once; the result is committed to `vocab/duo_frequencies.json` and read from
there. The corpus generator must not depend on a live API, or the experiments
stop being reproducible.

    python experiments/fetch_duo_frequencies.py
    python experiments/fetch_duo_frequencies.py --max-policies 2000

Note what this also measures: how many policies carry structured conditions at
all. Sparse adoption is a finding about the state of practice, not an
inconvenience.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

API = "https://metadata.ega-archive.org/policies"
OUT = Path(__file__).resolve().parent.parent / "vocab" / "duo_frequencies.json"


def fetch_page(limit: int, offset: int, timeout: float = 30.0) -> list[dict]:
    url = f"{API}?limit={limit}&offset={offset}"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch DUO frequencies from EGA")
    parser.add_argument("--max-policies", type=int, default=5000)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--pause", type=float, default=0.2, help="Seconds between pages")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv)

    codes: Counter[str] = Counter()
    combinations: Counter[tuple[str, ...]] = Counter()
    seen = 0
    annotated = 0
    offset = 0

    print(f"Fetching from {API} ...")
    while seen < args.max_policies:
        try:
            page = fetch_page(args.page_size, offset)
        except Exception as exc:  # noqa: BLE001 - report and stop, do not crash
            print(f"  stopped at offset {offset}: {type(exc).__name__}: {exc}")
            break
        if not page:
            break

        for record in page:
            seen += 1
            duos = record.get("duos")
            if not duos:
                continue
            annotated += 1
            combination = tuple(sorted(duos))
            combinations[combination] += 1
            codes.update(duos)

        offset += len(page)
        if len(page) < args.page_size:
            break
        print(f"  {seen} policies, {annotated} annotated", end="\r", flush=True)
        time.sleep(args.pause)

    print(f"\n{seen} policies seen, {annotated} carry DUO codes "
          f"({annotated / max(1, seen):.1%} adoption)")

    if not annotated:
        print("No annotated policies found; not overwriting the committed table.")
        return 1

    payload = {
        "_provenance": {
            "source": "EGA metadata API",
            "url": API,
            "fetched": datetime.now(tz=UTC).date().isoformat(),
            "policies_seen": seen,
            "policies_annotated": annotated,
            "adoption_rate": round(annotated / seen, 4),
            "note": (
                "Most EGA policies carry no structured data use conditions and "
                "rely on free text or an external URL. The adoption rate is "
                "reported because sparse uptake of machine-readable conditions "
                "is itself a motivating observation."
            ),
        },
        "code_frequencies": dict(codes.most_common()),
        "combinations": [
            {"codes": list(combo), "count": count}
            for combo, count in combinations.most_common()
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    print(f"  {len(codes)} distinct codes, {len(combinations)} distinct combinations")
    for combo, count in combinations.most_common(10):
        print(f"    {count:>4}  {', '.join(combo)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
