"""Figure: does the masking scheme change how reproducibly the embeddings partition the cells?

Reads the `emb_cluster_*.csv` that `4e_embedding_and_rank_stability.py` writes for three legnini23
arms and puts them on one axis:

    published   cell masking @0.30, 100 epochs, MASK_VALUE = 0   (figures/.../seed_stability)
    cell        cell masking @0.30, 300 epochs, MASK_VALUE = -1  (.../seed_stability_cell)
    gene        gene masking @0.25, 300 epochs, MASK_VALUE = -1  (.../seed_stability_gene)

The middle arm is what makes this readable: published -> cell isolates training length and the
fill value, cell -> gene isolates masking granularity with everything else matched.

Bars are cross-seed AMI, i.e. how much two independently seeded models agree on the PARTITION.
A latent basis is defined only up to rotation, but a partition is not, so this needs no dimension
matching and no sign convention -- which is why it is readable where the loadings are not.
The crosses are the same clusterings scored against `SHH_zone`: they stay flat while the bars
triple, so the partitions became reproducible without becoming more biological. Read them
together or the bars overstate the result.

    python scripts/legnini_23/4f_plot_mask_cluster_stability.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FIG = REPO / "figures" / "legnini23"

# Arm label -> directory holding that arm's 4e output. Order is the ablation order and is what
# the x axis shows; it is a progression, not a ranking.
ARMS = [
    ("published\n(cell 0.30, 100ep, fill 0)", FIG / "seed_stability"),
    ("cell 0.30\n(300ep, fill -1)", FIG / "seed_stability_cell"),
    ("gene 0.25\n(300ep, fill -1)", FIG / "seed_stability_gene"),
]

# figure_style is imported BY FILE PATH on purpose: putting src/ on sys.path would let
# src/wandb.py shadow the real wandb package.
_spec = importlib.util.spec_from_file_location("figure_style", REPO / "src" / "figure_style.py")
figure_style = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(figure_style)
st = figure_style.FigureStyle(str(REPO) + "/")
INK, INK_2, GRID = figure_style.INK, figure_style.INK_2, figure_style.GRID

# Two hues from the house Models palette, not invented here. Checked rather than eyeballed:
# OKLab dE x100 = 34.1 normal vision and 33.7-34.1 under deuteran/protan/tritanopia, against a
# floor of 8 (15 for normal vision). #88C8B2 sits at 1.92:1 on white, so every bar carries an INK
# edge and the numbers are printed below the figure as well.
C_LOCAL = st.palettes["Models"]["InterScale"]
C_GLOBAL = st.palettes["Models"]["GCN"]


def load_arm(d: Path):
    cross = pd.read_csv(d / "emb_cluster_cross_seed.csv")
    bio = pd.read_csv(d / "emb_cluster_vs_biology.csv")
    return cross, bio


def main():
    data = {}
    for label, d in ARMS:
        if not (d / "emb_cluster_cross_seed.csv").exists():
            raise SystemExit(f"missing 4e output for {label!r}: {d}\nRun 4a then 4e for that spec.")
        data[label] = load_arm(d)

    ks = sorted(set.intersection(*(set(c["k"]) for c, _ in data.values())))
    labels = [lab for lab, _ in ARMS]
    x = np.arange(len(labels))
    width = 0.36

    # Wide enough that the house 14pt bold title and 12pt bold legend do not dominate the
    # panels; the legend goes BELOW the figure rather than inside an axes, where it landed on
    # top of the published arm's bars.
    fig, axes = plt.subplots(1, len(ks), figsize=(5.8 * len(ks), 4.6), sharey=True)
    axes = np.atleast_1d(axes)

    rows = []
    for ax, k in zip(axes, ks):
        for off, comp, colour in ((-width / 2, "local", C_LOCAL), (width / 2, "global", C_GLOBAL)):
            mean, sd, bio_mean = [], [], []
            for lab in labels:
                cross, bio = data[lab]
                c = cross[(cross.component == comp) & (cross.k == k)]["ami"]
                b = bio[(bio.component == comp) & (bio.k == k)]["ami_vs_biology"]
                mean.append(c.mean())
                sd.append(c.std(ddof=1))
                bio_mean.append(b.mean())
                rows.append({"k": k, "component": comp, "arm": lab.replace("\n", " "),
                             "cross_seed_ami": c.mean(), "sd": c.std(ddof=1),
                             "ami_vs_SHH_zone": b.mean()})
            ax.bar(x + off, mean, width, yerr=sd, capsize=3, color=colour, edgecolor=INK,
                   linewidth=0.7, label=f"{comp} embedding, seed vs seed",
                   error_kw={"ecolor": INK_2, "lw": 0.9})
            # zorder above the bars: these sit at ~0.01-0.02, i.e. right on the baseline, and
            # that they are invisibly small next to the bars IS the second half of the result.
            ax.plot(x + off, bio_mean, ls="none", marker="x", ms=8, mew=1.8, color=INK_2,
                    zorder=5,
                    label="the same clustering vs SHH_zone" if comp == "local" else None)

        ax.axhline(0, lw=0.6, color=INK_2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(f"k = {k} clusters", **st.title_kw)
        ax.grid(axis="y", alpha=0.3, ls="--")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("adjusted mutual information")
    axes[0].set_ylim(0, 0.85)

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02),
               **st.legend_kw)
    fig.suptitle("Gene masking makes the cell partition far more reproducible", **st.title_kw)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "emb_cluster_masking_comparison"
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")

    table = pd.DataFrame(rows)
    table.to_csv(f"{out}.csv", index=False)
    # The table view the low-contrast fill obliges, and the numbers behind every bar.
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out}.png / .pdf / .csv")


if __name__ == "__main__":
    main()
