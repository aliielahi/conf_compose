"""Draw every voting-protocol figure and the headline numbers from the saved atomic.csv and audit.json."""

import accuracy
import estimators
import group_size
import groups
import headlines
import judges
import panels
import ranks
import topline
import win_rates
from common import OUT, load, style

FIGURES = (topline, group_size, win_rates, groups, ranks, accuracy, estimators, judges, panels, headlines)


def main():
    style()
    rows, _ = load()
    for module in FIGURES:
        paths = module.write(rows) if module is headlines else module.draw(rows)
        print(f"{module.__name__:<12}{len(paths)} file(s)")
    print(f"all outputs in {OUT}")


if __name__ == "__main__":
    main()
