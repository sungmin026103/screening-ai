"""Live smoke test for the search APIs. Run on a machine with internet access:

    python tools/smoke_apis.py [--mailto you@example.org] [--api-key NCBI_KEY]

Runs one tiny query per API (fetch=True, max_records=20) and prints count,
number of records and first year. Prints no titles. Exit code 1 on any failure.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feasibility.search_apis import europepmc_search, openalex_search, pubmed_search  # noqa: E402

QUERY = "vitamin D supplementation randomized controlled trial"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mailto")
    ap.add_argument("--api-key")
    a = ap.parse_args()
    runs = [
        ("pubmed", lambda: pubmed_search(QUERY, api_key=a.api_key, mailto=a.mailto, max_records=20)),
        ("europepmc", lambda: europepmc_search(QUERY, max_records=20)),
        ("openalex", lambda: openalex_search(QUERY, mailto=a.mailto, max_records=20)),
    ]
    failed = False
    for name, fn in runs:
        try:
            r = fn()
        except Exception as exc:  # noqa: BLE001
            print(f"{name}: FAIL {exc!r}")
            failed = True
            continue
        years = [y for y in r.records["year"].tolist() if y != ""]
        # max_records=20 truncation sets partial legitimately; a shortfall below it is a failure
        ok = r.count > 0 and len(r.records) >= min(r.count, 20)
        print(f"{name}: {'OK' if ok else 'FAIL'} count={r.count} n_records={len(r.records)} "
              f"first_year={years[0] if years else None} notes={r.notes}")
        failed |= not ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
