#!/usr/bin/env python
"""Regenerate the classification figures for one config.

The config decides everything, including whether this is node- or graph-level
classification and therefore where the figures land:

    python scripts/classification/run.py --config chen22_node    # -> figures/node_classification/chen22/
    python scripts/classification/run.py --config chen22_graph   # -> figures/graph_classification/chen22/

`--config` takes a bare name from configs/ or a path. See --help for the rest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import run  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run())
