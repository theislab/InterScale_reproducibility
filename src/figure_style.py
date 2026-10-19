"""One place that reads `figures/config.yml`, so every figure in the repo looks the same.

`src/wandb.py::set_plot_configs` already did part of this but returns only the model palette and
hardcodes `font.size`, so it cannot style a figure that colours by cell type, zone or condition.
This module exposes the whole config instead:

    from figure_style import FigureStyle
    st = FigureStyle(BASE_DIR_REPO)      # applies the rcParams on construction
    st.cell_types("melton25")            # {'Acinar': '#028E95', ...}
    st.sequential(6)                     # ordered series colours, sampled from the config cmap
    st.continuous_cmap                   # 'viridis'
    st.divergent_cmap                    # 'bwr'
    ax.legend(**st.legend_kw)            # honours legend_fontsize / legend_fontweight

On `title_fontweight` and `legend_fontweight`: matplotlib has an rcParam for the title weight
(`axes.titleweight`) but none for legend text weight, so the latter is delivered through
``legend_kw`` and has to be passed at each call site.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import numpy as np
import yaml

# Neutral ink/grid greys. Not in config.yml -- they are figure furniture rather than data colours,
# and keeping them here avoids every script inventing its own.
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d9d8d4"


class FigureStyle:
    """Loads `figures/config.yml` and applies its `plot_configs.general` block to rcParams."""

    def __init__(self, base_dir_repo=None, config_path=None, apply=True, font_size=9):
        if config_path is None:
            base = Path(base_dir_repo or os.environ.get("BASE_DIR_REPO", "."))
            # Accept either the repo root or its parent, since notebooks set BASE_DIR_REPO to the
            # directory *containing* InterScale_reproducibility.
            for candidate in (base / "figures" / "config.yml",
                              base / "InterScale_reproducibility" / "figures" / "config.yml"):
                if candidate.is_file():
                    config_path = candidate
                    break
            else:
                raise FileNotFoundError(f"no figures/config.yml under {base}")
        self.config_path = Path(config_path)
        with self.config_path.open() as f:
            self.config = yaml.safe_load(f)

        self.general = self.config["plot_configs"]["general"]
        self.palettes = self.config["palettes"]
        self.slide_examples = self.config.get("slide_examples", {})
        # `font_size` is the base body size, which config.yml does not specify -- it gives only the
        # title and legend sizes. Kept as an argument so a dense multi-panel figure can go smaller
        # without editing the shared config.
        self.font_size = font_size
        if apply:
            self.apply()

    # -- rcParams ----------------------------------------------------------------------------

    def apply(self):
        g = self.general
        mpl.rcParams.update(
            {
                "figure.dpi": g["dpi"],
                "savefig.dpi": g["dpi_save"],
                "font.family": g["font_family"],
                "font.size": self.font_size,
                "axes.titlesize": g["title_fontsize"],
                "axes.titleweight": g["title_fontweight"],
                "legend.fontsize": g["legend_fontsize"],
                "legend.title_fontsize": g["legend_fontsize"],
                "image.cmap": g["cmap"],
                # Figure furniture, consistent across the repo.
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "axes.edgecolor": GRID,
                "axes.labelcolor": INK_2,
                "text.color": INK,
                "xtick.color": INK_2,
                "ytick.color": INK_2,
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )
        return self

    # -- text ---------------------------------------------------------------------------------

    @property
    def title_kw(self):
        return {"fontsize": self.general["title_fontsize"],
                "fontweight": self.general["title_fontweight"]}

    @property
    def legend_kw(self):
        """matplotlib has no rcParam for legend text weight, so it is passed explicitly."""
        return {"frameon": False,
                "prop": {"size": self.general["legend_fontsize"],
                         "weight": self.general["legend_fontweight"]},
                "title_fontproperties": {"size": self.general["legend_fontsize"],
                                         "weight": self.general["legend_fontweight"]}}

    # -- colours ------------------------------------------------------------------------------

    @property
    def continuous_cmap(self):
        return self.palettes.get("continuous", self.general["cmap"])

    @property
    def divergent_cmap(self):
        return self.palettes.get("divergent", "bwr")

    def palette(self, name):
        """A named palette from config.yml, e.g. 'melton25', 'Models', 'Condition_pancreas'."""
        if name not in self.palettes:
            raise KeyError(f"no palette '{name}' in {self.config_path}; "
                           f"available: {sorted(self.palettes)}")
        return self.palettes[name]

    def cell_types(self, dataset, keys=None):
        """Cell-type colours for a dataset, optionally restricted to (and ordered by) `keys`.

        Nested sub-palettes such as `melton25.condition` are dropped, so the result maps cell type
        to colour and nothing else.
        """
        pal = {k: v for k, v in self.palette(dataset).items() if isinstance(v, str)}
        if keys is None:
            return pal
        missing = [k for k in keys if k not in pal]
        if missing:
            raise KeyError(f"palette '{dataset}' has no colour for {missing}")
        return {k: pal[k] for k in keys}

    def sequential(self, n, cmap=None, lo=0.88, hi=0.06):
        """`n` colours for an ORDERED series, sampled from the config's continuous colormap.

        Runs light -> dark by default, so a larger value in the series reads as darker. `lo`/`hi`
        stop short of the extremes: viridis at 1.0 is a yellow that disappears against white.
        """
        cm = mpl.colormaps[cmap or self.continuous_cmap]
        return [cm(x) for x in (np.linspace(lo, hi, n) if n > 1 else [0.5])]

    def zones(self, dataset="legnini23", order=None):
        """SHH ring colours. config.yml numbers ring_1 as FURTHEST from the source and ring_4 as
        closest, so a zone list ordered by increasing distance maps to ring_4 ... ring_1.
        """
        pal = self.palette(dataset)
        rings = [pal[f"ring_{i}"] for i in (4, 3, 2, 1)]  # closest -> furthest
        if order is None:
            return rings
        if len(order) != len(rings):
            raise ValueError(f"expected {len(rings)} zones, got {len(order)}")
        return dict(zip(order, rings, strict=True))


def load_style(base_dir_repo=None, **kwargs):
    """Convenience wrapper: build a FigureStyle and return it with rcParams already applied."""
    return FigureStyle(base_dir_repo=base_dir_repo, **kwargs)
