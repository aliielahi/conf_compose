"""Raw ECE and Brier of log-odds Learned Weights against group size, consistency (solid) and sequence probability
(dotted), five datasets, in the style of the paper's appendix accuracy figure."""

import importlib.util
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
_spec = importlib.util.spec_from_file_location("paper_figures", ROOT / "paper_results/seq_vs_cons/paper_figures.py")
paper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(paper)

METRICS = {"ece": ("appendix_raw_ece", r"ECE ($\times 100$)"), "brier": ("appendix_raw_brier", r"Brier ($\times 100$)")}
# Same panel size, spacing and repeated x label as imgs/appendix_accuracy.tex in the paper.
WIDTH, SEP, XLABEL = 0.17, "0.3cm", r"xlabel={\# Agents in Group}"


def restyle(tex):
    """Repeat the short x label under every panel instead of one shared label under the middle panel."""
    tex = tex.replace(", xlabel={Number of Agents in Group}", "")
    return tex.replace("  title style=", f"  {XLABEL}, title style=", 1)


def main():
    frame = pd.read_csv(HERE / "raw.csv")
    for metric, (name, ylabel) in METRICS.items():
        series = paper.points(frame, paper.FIVE, ["cons", "seq_reasoned"], metric)
        tex = restyle(paper.figure(series, paper.FIVE, [("cons", False), ("seq_reasoned", True)], ylabel, WIDTH, SEP,
                                   line_legend=True, zero=False))
        for folder in (HERE, ROOT / "ARR-October/imgs"):
            (folder / f"{name}.tex").write_text(tex)
        print(f"wrote {name}.tex")


if __name__ == "__main__":
    main()
