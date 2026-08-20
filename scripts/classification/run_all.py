#!/usr/bin/env python
"""Regenerate the graph-classification figures for every dataset config.

Keeps going when one dataset fails and reports a summary at the end; any
remaining CLI flags are forwarded to each dataset run, e.g.

    python scripts/graph_classification/run_all.py --cache-only
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import CONFIG_DIR, run  # noqa: E402


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    configs = sorted(CONFIG_DIR.glob("*.yaml"))
    if not configs:
        print(f"no configs found in {CONFIG_DIR}", file=sys.stderr)
        return 2

    results = {}
    for config in configs:
        print(f"\n{'=' * 70}\n{config.stem}\n{'=' * 70}")
        try:
            results[config.stem] = run(config, argv)
        except Exception as exc:  # keep going: one bad dataset must not stop the rest
            print(f"{config.stem}: {type(exc).__name__}: {exc}", file=sys.stderr)
            results[config.stem] = 1

    print(f"\n{'=' * 70}\nsummary")
    for name, code in results.items():
        print(f"  {'ok  ' if code == 0 else 'FAIL'} {name}")
    return 1 if any(results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
